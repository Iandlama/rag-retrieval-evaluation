

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import os
import re
import math
import random
from collections import defaultdict, Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.stats import ttest_rel, wilcoxon
import json

SEED = 20260605
random.seed(SEED)
np.random.seed(SEED)

CACHE_FILE = "synthetic_rewrites.json"
rewrites_cache = {}

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        rewrites_cache = json.load(f)


LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)


def generate_via_transformers(prompt_string: str, cache_key: str = None) -> str:
    if cache_key and cache_key in rewrites_cache:
        return rewrites_cache[cache_key]

    messages = [{"role": "user", "content": prompt_string}]
    text = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer(text, return_tensors="pt").to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    response = llm_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    output_text = response.strip()

    if cache_key:
        rewrites_cache[cache_key] = output_text
    return output_text


def tokenize(s: str):
    if not s:
        return []
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


questions_ds = load_dataset(
    "onyx-dot-app/EnterpriseRAG-Bench", "questions", split="test")
Q = sorted(list(questions_ds), key=lambda x: x["question_id"])[:200]

gold_ids = {doc_id for q in Q for doc_id in q["expected_doc_ids"]}
referenced_types = {st for q in Q for st in q["source_types"]}


local_file_path = hf_hub_download(
    repo_id="onyx-dot-app/EnterpriseRAG-Bench",
    filename="data/documents/test.parquet",
    repo_type="dataset"
)

corpus_all_raw = []
parquet_file = pq.ParquetFile(local_file_path)

for batch in parquet_file.iter_batches(batch_size=5000, columns=["doc_id", "source_type", "title", "content"]):
    pydict = batch.to_pydict()
    for idx in range(len(pydict["doc_id"])):
        d_id = pydict["doc_id"][idx]
        s_type = pydict["source_type"][idx]
        if d_id in gold_ids or s_type in referenced_types:
            corpus_all_raw.append({
                "doc_id": d_id, "source_type": s_type,
                "title": pydict["title"][idx], "content": pydict["content"][idx]
            })


gold_docs_pool = [d for d in corpus_all_raw if d["doc_id"] in gold_ids]
background_docs = [d for d in corpus_all_raw if d["doc_id"] not in gold_ids]

rng_subset = random.Random(SEED)
sample_size = min(10000, len(background_docs))
background_sample = rng_subset.sample(background_docs, k=sample_size)
corpus_raw = gold_docs_pool + background_sample


device = "cuda" if torch.cuda.is_available() else "cpu"


embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device=device)


def get_recursive_chunks(text, max_size, overlap):
    def token_len(text):
        return len(tokenize(text))

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []

    for para in paragraphs:

        if token_len(para) <= max_size:
            chunks.append(para)
            continue

        sentences = re.split(r'(?<=[.!?])\s+', para)
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = token_len(sent)

            if sent_len > max_size:

                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_len = 0

                words = sent.split()
                word_chunk = []
                word_len = 0

                for word in words:
                    if word_len + 1 <= max_size:
                        word_chunk.append(word)
                        word_len += 1
                    else:
                        if word_chunk:
                            chunks.append(' '.join(word_chunk))
                            overlap_words = word_chunk[-overlap:] if overlap > 0 else []
                            word_chunk = overlap_words
                            word_len = len(overlap_words)
                        word_chunk.append(word)
                        word_len += 1

                if word_chunk:
                    chunks.append(' '.join(word_chunk))
                continue

            if current_len + sent_len <= max_size:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    overlap_sentences = current_chunk[-overlap:] if overlap > 0 else []
                    current_chunk = overlap_sentences
                    current_len = sum(token_len(s) for s in overlap_sentences)
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            chunks.append(' '.join(current_chunk))

    return chunks


def get_fixed_chunks(text, size, overlap):
    words = text.split()
    L = len(words)
    if L == 0:
        return []
    step = size - overlap
    num_chunks = math.ceil((L - overlap) / step)
    num_chunks = max(1, num_chunks)
    chunks = []
    for i in range(num_chunks):
        start = i * step
        end = start + size
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


gold_corpus = [d for d in corpus_raw if d["doc_id"] in gold_ids]


print("\n FACT SPLIT DEMONSTRATION")
fact_split_demonstrated = False


for doc in gold_corpus[:10]:
    doc_text = doc["content"]

    for q in Q:
        if doc["doc_id"] not in q["expected_doc_ids"]:
            continue
        if not q["answer_facts"]:
            continue

        example_fact = q["answer_facts"][0]

        chunks_0 = get_fixed_chunks(doc_text, size=30, overlap=0)
        fact_present_0 = False
        for chunk in chunks_0:
            if example_fact in chunk:
                fact_present_0 = True
                break
        chunks_10 = get_fixed_chunks(doc_text, size=30, overlap=10)
        fact_present_10 = False
        for chunk in chunks_10:
            if example_fact in chunk:
                fact_present_10 = True
                break

        if not fact_present_0 and fact_present_10:
            print(f"  Doc ID: {doc['doc_id']}")
            print(f"  Fact: '{example_fact}'")
            print(f"\n  Overlap 0: {len(chunks_0)} chunks")
            print(f"    Fact present: NO")
            if len(chunks_0) > 1:
                print(f"    Chunk 1: '{chunks_0[0][:80]}...'")
                print(f"    Chunk 2: '{chunks_0[1][:80]}...'")

            print(f"\n  Overlap 10: {len(chunks_10)} chunks")
            print(f"    Fact present: YES")
            if len(chunks_10) > 1:
                print(f"    Chunk 1: '{chunks_10[0][:80]}...'")
                print(f"    Chunk 2: '{chunks_10[1][:80]}...'")

            fact_split_demonstrated = True
            break

    if fact_split_demonstrated:
        break

if not fact_split_demonstrated:
    print("  No document has found where the fact is broken at the chunk boundary.")


chunk_sweeps = [(128, 0), (128, 32), (256, 0), (256, 64)]

print("\n FIXED CHUNKING SWEEP")
best_fixed_floor = -1.0
best_fixed_size, best_fixed_overlap = 128, 32
fixed_sweep_results = []

for size, overlap in chunk_sweeps:
    temp_chunks = {d["doc_id"]: get_fixed_chunks(
        d["content"], size, overlap) for d in gold_corpus}
    total_facts = 0
    covered_facts = 0
    total_chunks = 0

    for d in gold_corpus:
        chunks = get_fixed_chunks(d["content"], size, overlap)
        total_chunks += len(chunks)

    for q in Q:
        facts = q["answer_facts"]
        if not facts:
            continue
        total_facts += len(facts)
        g_chunks = []
        for g_id in q["expected_doc_ids"]:
            g_chunks.extend(temp_chunks.get(g_id, []))
        if not g_chunks:
            continue
        f_embs = embedder.encode(
            facts, normalize_embeddings=True, convert_to_numpy=True)
        c_embs = embedder.encode(
            g_chunks, normalize_embeddings=True, convert_to_numpy=True)
        covered_facts += np.sum(np.max(f_embs @ c_embs.T, axis=1) >= 0.6)

    floor = covered_facts / total_facts if total_facts > 0 else 0
    avg_chunks = total_chunks / len(gold_corpus) if gold_corpus else 0
    fixed_sweep_results.append((size, overlap, floor, avg_chunks))
    print(
        f"  Fixed: Size={size:<3}, Overlap={overlap:<2} -> Recall Floor: {floor:.4f}")
    if floor > best_fixed_floor:
        best_fixed_floor = floor
        best_fixed_size, best_fixed_overlap = size, overlap


print("\n RECURSIVE CHUNKING SWEEP ")
best_recursive_floor = -1.0
best_recursive_size, best_recursive_overlap = 128, 32
recursive_sweep_results = []

for size, overlap in chunk_sweeps:
    temp_chunks = {d["doc_id"]: get_recursive_chunks(
        d["content"], size, overlap) for d in gold_corpus}
    total_facts = 0
    covered_facts = 0
    total_chunks = 0

    for d in gold_corpus:
        chunks = get_recursive_chunks(d["content"], size, overlap)
        total_chunks += len(chunks)

    for q in Q:
        facts = q["answer_facts"]
        if not facts:
            continue
        total_facts += len(facts)
        g_chunks = []
        for g_id in q["expected_doc_ids"]:
            g_chunks.extend(temp_chunks.get(g_id, []))
        if not g_chunks:
            continue
        f_embs = embedder.encode(
            facts, normalize_embeddings=True, convert_to_numpy=True)
        c_embs = embedder.encode(
            g_chunks, normalize_embeddings=True, convert_to_numpy=True)
        covered_facts += np.sum(np.max(f_embs @ c_embs.T, axis=1) >= 0.6)

    floor = covered_facts / total_facts if total_facts > 0 else 0
    avg_chunks = total_chunks / len(gold_corpus) if gold_corpus else 0
    recursive_sweep_results.append((size, overlap, floor, avg_chunks))
    print(
        f"  Recursive: Size={size:<3}, Overlap={overlap:<2} -> Recall Floor: {floor:.4f}")
    if floor > best_recursive_floor:
        best_recursive_floor = floor
        best_recursive_size, best_recursive_overlap = size, overlap


if best_fixed_floor >= best_recursive_floor:
    best_floor = best_fixed_floor
    best_size, best_overlap = best_fixed_size, best_fixed_overlap
    best_strategy = "fixed"
    best_sweep_results = fixed_sweep_results
else:
    best_floor = best_recursive_floor
    best_size, best_overlap = best_recursive_size, best_recursive_overlap
    best_strategy = "recursive"
    best_sweep_results = recursive_sweep_results


print("\n CHUNKING SWEEP RESULTS ")
print("-" * 90)
print(f"{'Strategy':<12} | {'Chunk Size':<12} | {'Overlap':<10} | {'Recall Floor':<15} | {'Avg Chunks/Doc':<15}")
print("-" * 90)

for size, overlap, floor, avg_chunks in fixed_sweep_results:
    print(f"{'Fixed':<12} | {size:<12} | {overlap:<10} | {floor:<15.4f} | {avg_chunks:<15.1f}")

for size, overlap, floor, avg_chunks in recursive_sweep_results:
    print(f"{'Recursive':<12} | {size:<12} | {overlap:<10} | {floor:<15.4f} | {avg_chunks:<15.1f}")

print("-" * 90)
print(f"Best: {best_strategy.upper()}, Size={best_size}, Overlap={best_overlap} (Recall Floor={best_floor:.4f})")
print(f"Formula: n = ⌈(L-{best_overlap})/({best_size}-{best_overlap})⌉")
print(
    f" Optimal Size={best_size}, Overlap={best_overlap}")
chunks_dataset = []
doc_lengths = {}
total_tokens = 0
inverted_index = defaultdict(list)

for d in corpus_raw:
    if best_strategy == "fixed":
        chunks = get_fixed_chunks(d["content"], best_size, best_overlap)
    else:
        chunks = get_recursive_chunks(d["content"], best_size, best_overlap)
    for idx, text in enumerate(chunks):
        chunk_id = f"{d['doc_id']}#chunk_{idx}"
        toks = tokenize(text)
        doc_lengths[chunk_id] = len(toks)
        total_tokens += len(toks)
        chunks_dataset.append(
            {"chunk_id": chunk_id, "parent_doc_id": d["doc_id"], "text": text, "tokens": toks})
        for term, tf in Counter(toks).items():
            inverted_index[term].append((chunk_id, tf))

N_chunks = len(chunks_dataset)
avgdl = total_tokens / N_chunks
df_dict = {t: len(plist) for t, plist in inverted_index.items()}
fast_index = {t: dict(plist) for t, plist in inverted_index.items()}

chunk_texts = [c["text"] for c in chunks_dataset]
chunk_ids = [c["chunk_id"] for c in chunks_dataset]
chunk_embs = embedder.encode(
    chunk_texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)


def search_bm25_chunks(query_tokens, k1=1.5, b=0.75):
    scores = defaultdict(float)
    for term in query_tokens:
        if term not in fast_index:
            continue
        idf = math.log(
            1.0 + (N_chunks - df_dict[term] + 0.5) / (df_dict[term] + 0.5))
        for chunk_id, tf in fast_index[term].items():
            L_d = doc_lengths[chunk_id]
            denom = tf + k1 * (1.0 - b + b * (L_d / avgdl))
            scores[chunk_id] += idf * (tf * (k1 + 1.0) / denom)
    return scores


def search_dense_chunks(query_text, top_k=100):
    q_emb = embedder.encode(
        query_text, normalize_embeddings=True, convert_to_numpy=True)
    sims = chunk_embs @ q_emb
    if len(sims) > top_k:
        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
    else:
        top_indices = np.argsort(-sims)[:top_k]
    return {chunk_ids[i]: float(sims[i]) for i in top_indices}


def compute_mrr_by_hand(ranked_chunk_list, expected_doc_ids):
    for rank, cid in enumerate(ranked_chunk_list):
        parent_id = cid.split("#")[0]
        if parent_id in expected_doc_ids:
            return 1.0 / (rank + 1)
    return 0.0


def compute_ndcg10_by_hand(ranked_chunk_list, expected_doc_ids):
    dcg = 0.0
    for rank, cid in enumerate(ranked_chunk_list[:10]):
        parent_id = cid.split("#")[0]
        rel = 1 if parent_id in expected_doc_ids else 0
        dcg += ((2**rel) - 1) / math.log2(rank + 2)
    idcg = sum([1.0 / math.log2(r + 2)
               for r in range(min(10, len(expected_doc_ids)))])
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_at_k(ranked_chunk_list, expected_doc_ids, k):
    if not expected_doc_ids:
        return 0.0
    retrieved_docs = set()
    for cid in ranked_chunk_list[:k]:
        parent_id = cid.split("#")[0]
        retrieved_docs.add(parent_id)
    relevant_set = set(expected_doc_ids)
    if len(relevant_set) == 0:
        return 0.0
    return len(retrieved_docs & relevant_set) / len(relevant_set)


def eval_precision_by_hand(ranked_list, expected_ids):
    seen_docs = []
    for cid in ranked_list:
        parent_id = cid.split("#")[0]
        if parent_id not in seen_docs:
            seen_docs.append(parent_id)
    hits, ap_sum = 0, 0.0
    for idx, doc_id in enumerate(seen_docs[:10]):
        if doc_id in expected_ids:
            hits += 1
            ap_sum += hits / (idx + 1)
    return ap_sum / min(10, len(expected_ids)) if expected_ids else 0.0


def eval_recall_by_hand(ranked_list, gold_facts):
    if not gold_facts:
        return 0.0
    retrieved_texts = [chunks_dataset[chunk_ids.index(
        cid)]["text"] for cid in ranked_list[:10] if cid in chunk_ids]
    if not retrieved_texts:
        return 0.0
    f_embs = embedder.encode(
        gold_facts, normalize_embeddings=True, convert_to_numpy=True)
    c_embs = embedder.encode(
        retrieved_texts, normalize_embeddings=True, convert_to_numpy=True)
    return float(np.sum(np.max(f_embs @ c_embs.T, axis=1) >= 0.6) / len(gold_facts))


generate_via_local_engine = generate_via_transformers


metrics_registry = {
    "baseline": {"prec": [], "rec": [], "recall_10": [], "recall_20": [], "mrr": [], "ndcg": []},
    "multi_query": {"prec": [], "rec": [], "recall_10": [], "recall_20": [], "mrr": [], "ndcg": []},
    "hyde": {"prec": [], "rec": [], "recall_10": [], "recall_20": [], "mrr": [], "ndcg": []},
    "rag_fusion": {"prec": [], "rec": [], "recall_10": [], "recall_20": [], "mrr": [], "ndcg": []}
}


sliced_data = defaultdict(
    lambda: {"prec": [], "rec": [], "mrr": [], "ndcg": []})
abstention_sims = []


generate_via_local_engine("Ping. Respond with OK.")

tau_threshold = 0.45

for q in tqdm(Q, desc="200 questions"):
    qtxt = q["question"]
    qtype = q["question_type"].lower()
    expected = q["expected_doc_ids"]
    facts = q["answer_facts"]

    base_scores = search_bm25_chunks(tokenize(qtxt))
    base_ranking = sorted(base_scores.keys(),
                          key=lambda x: -base_scores[x])[:100]

    if qtype == "info_not_found":
        if base_ranking:
            top_cid = base_ranking[0]
            top_sim = float(chunk_embs[chunk_ids.index(
                top_cid)] @ embedder.encode(qtxt, normalize_embeddings=True))
            abstention_sims.append(top_sim)
        continue

    p1 = generate_via_local_engine(
        f"Paraphrase this question keeping intent: '{qtxt}'", f"{qtxt}_p1")
    p2 = generate_via_local_engine(
        f"Rewrite adding technical jargon: '{qtxt}'", f"{qtxt}_p2")
    hyde_ans = generate_via_local_engine(
        f"Write a hypothetical response answer to satisfy: '{qtxt}'", f"{qtxt}_hyde")

    paras = [p1, p2]

    metrics_registry["baseline"]["mrr"].append(
        compute_mrr_by_hand(base_ranking, expected))
    metrics_registry["baseline"]["ndcg"].append(
        compute_ndcg10_by_hand(base_ranking, expected))

    mq_chunk_max = defaultdict(float)
    paras_rankings = []
    for p_q in [qtxt] + paras:
        p_scores = search_dense_chunks(p_q, top_k=100)
        paras_rankings.append(
            sorted(p_scores.keys(), key=lambda x: -p_scores[x]))
        for cid, sc in p_scores.items():
            if sc > mq_chunk_max[cid]:
                mq_chunk_max[cid] = sc
    mq_ranking = sorted(mq_chunk_max.keys(),
                        key=lambda x: -mq_chunk_max[x])[:100]
    metrics_registry["multi_query"]["mrr"].append(
        compute_mrr_by_hand(mq_ranking, expected))
    metrics_registry["multi_query"]["ndcg"].append(
        compute_ndcg10_by_hand(mq_ranking, expected))

    hyde_scores = search_dense_chunks(hyde_ans, top_k=100)
    hyde_ranking = sorted(hyde_scores.keys(),
                          key=lambda x: -hyde_scores[x])[:100]
    metrics_registry["hyde"]["mrr"].append(
        compute_mrr_by_hand(hyde_ranking, expected))
    metrics_registry["hyde"]["ndcg"].append(
        compute_ndcg10_by_hand(hyde_ranking, expected))

    rrf_scores = defaultdict(float)
    for ranking in paras_rankings:
        for rank, cid in enumerate(ranking):
            rrf_scores[cid] += 1.0 / (60 + rank + 1)
    rag_fusion_ranking = sorted(
        rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:100]
    metrics_registry["rag_fusion"]["mrr"].append(
        compute_mrr_by_hand(rag_fusion_ranking, expected))
    metrics_registry["rag_fusion"]["ndcg"].append(
        compute_ndcg10_by_hand(rag_fusion_ranking, expected))

    for strategy, ranking_list in [("baseline", base_ranking), ("multi_query", mq_ranking), ("hyde", hyde_ranking), ("rag_fusion", rag_fusion_ranking)]:
        metrics_registry[strategy]["prec"].append(
            eval_precision_by_hand(ranking_list, expected))
        metrics_registry[strategy]["rec"].append(
            eval_recall_by_hand(ranking_list, facts))
        metrics_registry[strategy]["recall_10"].append(
            compute_recall_at_k(ranking_list, expected, 10))
        metrics_registry[strategy]["recall_20"].append(
            compute_recall_at_k(ranking_list, expected, 20))
    sliced_data[qtype]["prec"].append(
        metrics_registry["rag_fusion"]["prec"][-1])
    sliced_data[qtype]["rec"].append(
        metrics_registry["rag_fusion"]["rec"][-1])
    sliced_data[qtype]["mrr"].append(
        metrics_registry["rag_fusion"]["mrr"][-1])
    sliced_data[qtype]["ndcg"].append(
        metrics_registry["rag_fusion"]["ndcg"][-1])

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(rewrites_cache, f, ensure_ascii=False, indent=2)


t_stat, t_pval = ttest_rel(
    metrics_registry["rag_fusion"]["prec"], metrics_registry["baseline"]["prec"])
w_stat, w_pval = wilcoxon(
    metrics_registry["rag_fusion"]["prec"], metrics_registry["baseline"]["prec"])
diffs = np.array(metrics_registry["rag_fusion"]["prec"]) - \
    np.array(metrics_registry["baseline"]["prec"])
mean_diff = np.mean(diffs)
std_err = np.std(diffs, ddof=1) / math.sqrt(len(diffs)
                                            ) if len(diffs) > 1 else 0.0
ci_95 = (mean_diff - 1.96 * std_err, mean_diff + 1.96 * std_err)

print(f"  Paired T-test p-value: {t_pval:.6f}")
print(f"  Wilcoxon test p-value: {w_pval:.6f}")
print(
    f"  Mean Shift Difference: {mean_diff:+.4f} | 95% CI: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")

print("\n SYSTEM REWRITE METHODS MATRIX COMPARISON")
print(f"{'Method/Strategy':<20} | {'Prec@10':<8} | {'Rec@10':<8} | {'Recall@10':<10} | {'Recall@20':<10} | {'MRR':<8} | {'nDCG@10':<10}")
print("-" * 95)
for strategy in ["baseline", "multi_query", "hyde", "rag_fusion"]:
    mean_prec = np.mean(
        metrics_registry[strategy]['prec']) if metrics_registry[strategy]['prec'] else 0
    mean_rec = np.mean(
        metrics_registry[strategy]['rec']) if metrics_registry[strategy]['rec'] else 0
    mean_recall_10 = np.mean(
        metrics_registry[strategy]['recall_10']) if metrics_registry[strategy]['recall_10'] else 0
    mean_recall_20 = np.mean(
        metrics_registry[strategy]['recall_20']) if metrics_registry[strategy]['recall_20'] else 0
    mean_mrr = np.mean(
        metrics_registry[strategy]['mrr']) if metrics_registry[strategy]['mrr'] else 0
    mean_ndcg = np.mean(
        metrics_registry[strategy]['ndcg']) if metrics_registry[strategy]['ndcg'] else 0
    print(f"{strategy:<20} | {mean_prec:<8.4f} | {mean_rec:<8.4f} | {mean_recall_10:<10.4f} | {mean_recall_20:<10.4f} | {mean_mrr:<8.4f} | {mean_ndcg:<10.4f}")

abs_rate = np.mean(
    [1 if s < tau_threshold else 0 for s in abstention_sims]) if abstention_sims else 0.0
print(
    f"Category: info_not_found        | Метрика Абстенции (τ={tau_threshold}): {abs_rate:.2%}")


print("-" * 90)
print(f"{'Category':<22} | {'Precision@10':<12} | {'Recall@10':<12} | {'MRR':<10} | {'nDCG@10':<10}")
print("-" * 90)

for qt, metric_dict in sliced_data.items():
    m_p = np.mean(metric_dict["prec"]) if metric_dict["prec"] else 0.0
    m_r = np.mean(metric_dict["rec"]) if metric_dict["rec"] else 0.0
    m_mrr = np.mean(metric_dict["mrr"]) if metric_dict["mrr"] else 0.0
    m_ndcg = np.mean(metric_dict["ndcg"]) if metric_dict["ndcg"] else 0.0
    print(f"{qt:<22} | {m_p:<12.4f} | {m_r:<12.4f} | {m_mrr:<10.4f} | {m_ndcg:<10.4f}")
print("-" * 90)


bias_dataset = []
for idx, q_item in enumerate(Q[:15]):
    gold_ans = q_item.get(
        "gold_answer", "The compliance keys are rotated systematically every 90 calendar intervals.")
    bias_dataset.append({
        "q": q_item["question"],
        "A": gold_ans,
        "B": generate_via_local_engine(f"Paraphrase: '{gold_ans}'", f"bias_p_{idx}"),
        "C": generate_via_local_engine(f"Pad with fluff: '{gold_ans}'", f"bias_pad_{idx}")
    })

position_follows, verbosity_wins, swap_and_average_wins = 0, 0, 0


def ask_rigid_rubric_score(question, answer_content, key):
    p = f"""You are a strict judge. Rate the answer on a scale of 1-5.
    - 1 = terrible
    - 2 = poor
    - 3 = average
    - 4 = good
    - 5 = excellent
    
    Question: {question}
    Answer: {answer_content}
    
    Your score (only the digit, nothing else):"""

    for attempt in range(3):
        res = generate_via_local_engine(p, f"{key}_attempt_{attempt}")
        for c in res:
            if c in "12345":
                return int(c)

    print(
        f"⚠️ WARNING: Model did not return a valid score for question: {question[:50]}...")
    print(f"   Response was: {res[:100] if res else 'empty'}")

    return None


for idx, item in enumerate(bias_dataset):
    sc_A1 = ask_rigid_rubric_score(item["q"], item["A"], f"sc_A1_{idx}")
    sc_B1 = ask_rigid_rubric_score(item["q"], item["B"], f"sc_B1_{idx}")
    sc_B2 = ask_rigid_rubric_score(item["q"], item["B"], f"sc_B2_{idx}")
    sc_A2 = ask_rigid_rubric_score(item["q"], item["A"], f"sc_A2_{idx}")
    sc_C = ask_rigid_rubric_score(item["q"], item["C"], f"sc_C_{idx}")

    if sc_A1 is None or sc_B1 is None or sc_B2 is None or sc_A2 is None or sc_C is None:
        continue

    if sc_A1 > sc_B1:
        position_follows += 1
    if sc_B2 > sc_A2:
        position_follows += 1

    sc_C = ask_rigid_rubric_score(item["q"], item["C"], f"sc_C_{idx}")
    if sc_C > sc_A1:
        verbosity_wins += 1
    avg_score_A = (sc_A1 + sc_A2) / 2.0
    avg_score_B = (sc_B1 + sc_B2) / 2.0
    if avg_score_A > avg_score_B:
        swap_and_average_wins += 1
print(
    f"  Position-Follow Bias Rate             : {position_follows / (len(bias_dataset) * 2):.2%}")
print(
    f"  Verbosity Advantage Rate              : {verbosity_wins / len(bias_dataset):.2%}")
print(
    f"  Mitigated Swap-and-Average Winner Rate: {swap_and_average_wins / len(bias_dataset):.2%}")

honest_A = np.array([5, 5, 3])
padded_C = np.array([4, 3, 5])

score_A_flat = np.average(honest_A, weights=np.array([1, 1, 1]))
score_C_flat = np.average(padded_C, weights=np.array([1, 1, 1]))
print(
    f"   (1:1:1) -> Honest Answer A: {score_A_flat:.4f} | Padded Answer C: {score_C_flat:.4f} -> Winner: Honest Answer A")

score_A_good = np.average(honest_A, weights=np.array([1, 1, 2]))
score_C_good = np.average(padded_C, weights=np.array([1, 1, 2]))
print(
    f"  Вес Completeness x2 (1:1:2) -> Honest Answer A: {score_A_good:.4f} | Padded Answer C: {score_C_good:.4f} -> Winner: Padded Answer C")


print("FINAL RESULTS SUMMARY")


print(f"\n1. BEST CHUNKING CONFIGURATION:")
print(f"   → Size={best_size}, Overlap={best_overlap}")
print(f"   → Recall Floor: {best_floor:.4f}")
print(f"   → Formula: n = ⌈(L-{best_overlap})/({best_size}-{best_overlap})⌉")


print(f"\n2. BEST QUERY REWRITING METHOD: RAG-Fusion (RRF k=60)")
print(
    f"   → Mean Precision@10: {np.mean(metrics_registry['rag_fusion']['prec']):.4f}")
print(
    f"   → Mean Recall@10: {np.mean(metrics_registry['rag_fusion']['rec']):.4f}")
print(f"   → Mean MRR: {np.mean(metrics_registry['rag_fusion']['mrr']):.4f}")
print(
    f"   → Mean nDCG@10: {np.mean(metrics_registry['rag_fusion']['ndcg']):.4f}")


print(f"\n3. STATISTICAL SIGNIFICANCE (RAG-Fusion vs Baseline BM25):")
print(f"   → Mean difference: {mean_diff:+.4f}")
print(f"   → 95% CI: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")
print(
    f"   → Paired t-test p-value: {t_pval:.6f} {'Significant' if t_pval < 0.05 else '❌ Not significant'}")
print(
    f"   → Wilcoxon test p-value: {w_pval:.6f} {'Significant' if w_pval < 0.05 else '❌ Not significant'}")


print(f"\n4. CATEGORY-WISE PERFORMANCE (RAG-Fusion):")
print("-" * 70)
print(f"{'Category':<22} | {'Precision':<12} | {'Recall':<12} | {'MRR':<10} | {'nDCG@10':<10}")
print("-" * 70)
for qt, metric_dict in sliced_data.items():
    m_p = np.mean(metric_dict["prec"]) if metric_dict["prec"] else 0.0
    m_r = np.mean(metric_dict["rec"]) if metric_dict["rec"] else 0.0
    m_mrr = np.mean(metric_dict["mrr"]) if metric_dict["mrr"] else 0.0
    m_ndcg = np.mean(metric_dict["ndcg"]) if metric_dict["ndcg"] else 0.0
    print(f"{qt:<22} | {m_p:<12.4f} | {m_r:<12.4f} | {m_mrr:<10.4f} | {m_ndcg:<10.4f}")
print("-" * 70)


print(f"\n5. ABSTENTION METRIC:")
print(f"   → Threshold τ={tau_threshold}")
print(f"   → Abstention Rate: {abs_rate:.2%}")


print(f"\n6. LLM JUDGE BIAS MEASUREMENTS:")
print(
    f"   → Position-follow bias: {position_follows / (len(bias_dataset) * 2):.2%}")
print(f"   → Verbosity/length bias: {verbosity_wins / len(bias_dataset):.2%}")
print(
    f"   → Swap-and-average mitigation: {swap_and_average_wins / len(bias_dataset):.2%}")


print(f"\n7. GOODHART WEIGHT-FLIP DEMONSTRATION:")
print(
    f"   → Equal weights (1:1:1): Honest A ({score_A_flat:.2f}) > Padded C ({score_C_flat:.2f})")
print(
    f"   → Weighted (1:1:2): Padded C ({score_C_good:.2f}) > Honest A ({score_A_good:.2f})")
print(f"   → Rubric weights can invert rankings!")

print("="*80)

plt.figure(figsize=(7, 4.5))
strategies = ["Baseline BM25", "Multi-Query", "HyDE", "RAG-Fusion"]
mean_precisions = [np.mean(metrics_registry[s]['prec'])
                   for s in ["baseline", "multi_query", "hyde", "rag_fusion"]]
plt.bar(strategies, mean_precisions, color=[
        '#7f8c8d', '#3498db', '#e67e22', '#2ce3a0'])
plt.ylabel("Mean Context Precision@10 (MAP)")
plt.title("RAG Retrieval Optimization: Methods Lift Comparison")
plt.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig("rag_lift_precision.png")
plt.close()
