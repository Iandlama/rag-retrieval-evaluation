# RAG Retrieval and Evaluation on EnterpriseRAG-Bench

This repository contains the implementation of **Homework 3: RAG Retrieval and Evaluation** as part of the Information Retrieval course.

## 📚 Overview

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline on the EnterpriseRAG-Bench dataset. The implementation includes:

- **Chunking Study**: Fixed-size and recursive chunking with sweep over size/overlap parameters
- **Query Rewriting**: Multi-Query (MAX aggregation), HyDE, and RAG-Fusion (RRF k=60)
- **RAG Evaluation**: Context Precision (MAP), Context Recall (with cosine ≥ 0.6), Abstention metric
- **LLM Judge Bias Lab**: Position bias, verbosity bias, swap-and-average, Goodhart weight-flip
- **Statistical Analysis**: Paired t-test, Wilcoxon signed-rank test, 95% Confidence Intervals

## 📊 Dataset

- **Dataset**: EnterpriseRAG-Bench (onyx-dot-app)
- **Questions**: First 200 questions by question_id
- **Corpus**: ~10,000 documents (all gold docs + background sample)
- **Relevance**: Binary (0/1)


### System Requirements
- **Python**: 3.10+
- **RAM**: 8GB+ recommended
- **GPU**: Optional (for faster embedding encoding)

### Optional: Ollama (for LLM Judge)
```bash
# Install Ollama from https://ollama.com/download
ollama pull llama3:8b
ollama serve
```

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-retrieval-evaluation.git
cd rag-retrieval-evaluation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python homework3.py
```

### 4. Output Files
After execution, the following files will be generated:
- `synthetic_rewrites.json` - Cache of generated paraphrases (reproducibility)
- `rag_lift_precision.png` - Plot comparing query rewriting methods
- `output.txt` - Full console output (save for report)


## 📝 Report

A full report with tables and figures is available in the PDF file included in this repository.



## 📅 Course
Information Retrieval, 2026
```
