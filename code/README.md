# HALO — Code

Runnable pipeline for the paper *"Does Hybrid Retrieval Help Offline Clinical RAG? An Empirical Study"*.

This code is motivated by [KaagapAI](https://github.com/jmomugtong/KaagapAI), an offline clinical AI assistant for remote rural health units in the Philippines. HALO empirically tests whether hybrid retrieval and reranking meaningfully reduce hallucination under the same CPU-only, offline constraints that KaagapAI operates under.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — runs the 12-condition ablation evaluation |
| `models.py` | HALO pipeline components (BM25, TF-IDF/FAISS, RRF, cross-encoder) |
| `requirements.txt` | Python dependencies |
