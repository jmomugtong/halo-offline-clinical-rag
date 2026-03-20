# Does Hybrid Retrieval Help Offline Clinical RAG? An Empirical Study

**Author:** Joshua Miguel DC. Omugtong — Independent Researcher (Makati City, Philippines)
**Contact:** jdomugtong@alumni.up.edu.ph | ORCID: [0009-0009-2271-5959](https://orcid.org/0009-0009-2271-5959)

---

## Inspiration

This research is directly motivated by **[KaagapAI](https://github.com/jmomugtong/KaagapAI)** — a clinical AI assistant built to serve remote rural health units in the Philippines (barangay health centers and rural health units) where:

- Internet connectivity is intermittent or absent
- GPU hardware is unavailable
- Patients lack access to specialist referral

KaagapAI is an offline-first clinical RAG application. The practical question it raised — *does a more sophisticated retrieval stack actually reduce hallucination on CPU-only hardware, or is simpler just as good?* — is what this paper sets out to answer empirically.

---

## Paper

**`paper.tex`** — full NeurIPS 2025 format paper. Compile with:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Requires the NeurIPS 2025 style file (`neurips_2025.sty`, included).

### Abstract

HALO (Hybrid Augmentation with Layered reranking for Offline clinical RAG) is a three-stage pipeline evaluated on BioASQ Task B across 12 ablation conditions and 5 seeds (60 total evaluations) with BioMistral-7B as the generation backbone. All conditions yield faithfulness-NLI = 0.84 and EM-F1 = 0.40 with zero variance — no retrieval configuration shows a statistically significant advantage over plain BM25 in this evaluation. The paper diagnoses why and what that means for offline clinical RAG benchmarking.

---

## Repository Structure

```
.
├── paper.tex               # Main paper (NeurIPS 2025 format)
├── references.bib          # Bibliography
├── neurips_2025.sty        # NeurIPS style file
├── charts/                 # All figures used in the paper
└── code/                   # Runnable HALO pipeline
    ├── main.py
    ├── models.py
    ├── requirements.txt
    └── README.md
```

---

## Code

See [`code/README.md`](code/README.md) for setup and usage instructions.

```bash
cd code
pip install -r requirements.txt
python main.py
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{omugtong2026halo,
  title={Does Hybrid Retrieval Help Offline Clinical RAG? An Empirical Study},
  author={Omugtong, Joshua Miguel DC.},
  year={2026},
  note={Independent Researcher, Makati City, Philippines}
}
```

---

## Related Project

**KaagapAI** — the clinical AI app that inspired this research:
[https://github.com/jmomugtong/KaagapAI](https://github.com/jmomugtong/KaagapAI)
