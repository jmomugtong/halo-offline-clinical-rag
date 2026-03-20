# Framework Diagram Prompt

**Paper**: HALO: An Empirical Study of Hybrid Retrieval for Offline Clinical Question Answering

## Image Generation Prompt

Academic architecture diagram for the HALO pipeline: a clean left-to-right flowchart on a white background in flat vector style with subtle drop shadows, suitable for an ML conference paper. Begin on the far left with a rounded rectangle labeled "Clinical Query" in dark charcoal (#333333), connected by a bold right-pointing arrow. The arrow forks into two parallel horizontal tracks. The top track contains a rounded rectangle in muted blue (#4477AA) labeled "BM25 Sparse Retrieval" with a small sub-label "Pyserini · k₁=0.9 · b=0.4"; the bottom track contains a rounded rectangle in teal (#44AA99) labeled "TF-IDF FAISS Dense Retrieval" with sub-label "Flat L₂ Index · CPU-only". Both tracks converge via angled arrows into a central rounded rectangle in soft purple (#AA3377) labeled "RRF Fusion" with sub-label "k=60 · Top-20". A right-pointing arrow leads to a warm-accent rounded rectangle (#CCBB44) labeled "Cross-Encoder Reranker" with sub-label "≤66M params · Top-5". Another arrow leads to a light-grey rounded rectangle labeled "Context Window C*". A final arrow points to a muted blue rounded rectangle labeled "BioMistral-7B SLM" with sub-label "7B · greedy · T=0.0". Two output boxes below it, connected by downward arrows: one teal box "Faithfulness-NLI (RAGAS DeBERTa)" and one purple box "EM-F1". A dashed-border annotation bracket spans the BM25, TF-IDF, and RRF boxes labeled "Conditionally Active (12 Ablation Conditions)". Thin horizontal guidelines separate the two retrieval tracks. All labels use a clean sans-serif font. No gradients, no photorealism, no decorative elements.

## Usage Instructions

1. Copy the prompt above into an AI image generator (DALL-E 3, Midjourney, Ideogram, etc.)
2. Generate the image at high resolution (2048x1024 or similar landscape)
3. Save as `framework_diagram.png` in the same `charts/` folder
4. Insert into the paper's Method section using:
   - LaTeX: `\includegraphics[width=\textwidth]{charts/framework_diagram.png}`
   - Markdown: `![Framework Overview](charts/framework_diagram.png)`
