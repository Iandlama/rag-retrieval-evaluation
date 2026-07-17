
# RAG Retrieval and Evaluation


## 📋 Overview

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline on the **EnterpriseRAG-Bench** dataset. The implementation includes:

- **Chunking Study**: Fixed-size and recursive chunking with sweep over size/overlap parameters
- **Query Rewriting**: Multi-Query (MAX aggregation), HyDE, and RAG-Fusion (RRF k=60)
- **RAG Evaluation**: Context Precision (MAP), Context Recall (cosine ≥ 0.6), Abstention metric
- **LLM Judge Bias Lab**: Position bias, verbosity bias, swap-and-average, Goodhart weight-flip
- **Statistical Analysis**: Paired t-test, Wilcoxon signed-rank test, 95% Confidence Intervals

---

## 🛠️ Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

### System Requirements

- **Python**: 3.10+
- **RAM**: 8GB+ recommended
- **GPU**: Optional (for faster embedding encoding)
- **Storage**: ~20GB for dataset + models

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Iandlama/rag-retrieval-evaluation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python ass3.py
```

⚠️ **Note**: The script will download:
- EnterpriseRAG-Bench dataset (~1.4 GB)
- Qwen 2.5-7B-Instruct model (~15 GB in 4-bit)
- all-MiniLM-L6-v2 embedding model (~90 MB)

---

## 📊 Output

After execution, the following files will be generated:

| File | Description |
|------|-------------|
| `rag_lift_precision.png` | Plot comparing query rewriting methods |
| `synthetic_rewrites.json` | Cache of generated paraphrases (reproducibility) |
| Console output | All tables for the report |

### Key Results

| Component | Best Configuration |
|-----------|-------------------|
| **Chunking Strategy** | Recursive, Size=128, Overlap=0 |
| **Recall Floor** | 0.3483 |
| **Best Query Rewriting** | RAG-Fusion (RRF k=60) |
| **Statistical Significance** | Yes (p < 0.05) |

---


---


---

## 📝 Report

A full report with tables, figures, and analysis is available in `report.pdf`.

