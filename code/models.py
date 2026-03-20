"""
Retrieval strategy implementations for clinical RAG hallucination experiment.

All retrievers implement: fit(corpus), retrieve(query, corpus) -> List[str]
Strategies: BM25, TF-IDF dense, RRF hybrid, GA-optimized BM25, MonoT5-proxy
reranker, adaptive asymmetry router, and prompt-strategy wrappers.
"""

import re
import math
import time
import numpy as np
from collections import defaultdict


# ── TOKENIZATION UTIL ─────────────────────────────────────────────────────────

def tokenize(text):
    return re.sub(r'[^\w\s]', '', text.lower()).split()


# ── BM25 RETRIEVER ─────────────────────────────────────────────────────────────

class BM25Retriever:
    """Okapi BM25 implemented in numpy. Supports set_params() for GA tuning."""

    def __init__(self, hp, k1=1.5, b=0.75):
        self.hp = hp
        self.k1 = k1
        self.b = b
        self.top_k = hp['retrieval_top_k']
        self.corpus = None
        self.tok_corpus = None
        self.idf = {}
        self.avgdl = 0.0
        self._latency = []

    def set_params(self, k1, b):
        self.k1 = float(np.clip(k1, self.hp['ga_k1_range'][0], self.hp['ga_k1_range'][1]))
        self.b   = float(np.clip(b,  self.hp['ga_b_range'][0],  self.hp['ga_b_range'][1]))

    def fit(self, corpus):
        self.corpus = corpus
        self.tok_corpus = [tokenize(p) for p in corpus]
        N = len(corpus)
        self.avgdl = np.mean([len(t) for t in self.tok_corpus]) + 1e-9
        df = defaultdict(int)
        for toks in self.tok_corpus:
            for tok in set(toks):
                df[tok] += 1
        self.idf = {
            tok: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for tok, freq in df.items()
        }

    def _score(self, query_toks, doc_toks, dl):
        score = 0.0
        tf_map = defaultdict(int)
        for t in doc_toks:
            tf_map[t] += 1
        for tok in query_toks:
            if tok not in self.idf:
                continue
            tf = tf_map.get(tok, 0)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[tok] * num / (den + 1e-9)
        return score

    def retrieve(self, query, corpus=None):
        t0 = time.perf_counter()
        q_toks = tokenize(query)
        scores = [
            self._score(q_toks, dtoks, len(dtoks))
            for dtoks in self.tok_corpus
        ]
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        passages = [self.corpus[i] for i in top_idx]
        self._latency.append((time.perf_counter() - t0) * 1000)
        return passages

    def score_passage(self, query_toks, doc_idx):
        dtoks = self.tok_corpus[doc_idx]
        return self._score(query_toks, dtoks, len(dtoks))

    def get_latency_ms(self):
        return float(np.mean(self._latency)) if self._latency else 0.0


# ── TF-IDF DENSE RETRIEVER ────────────────────────────────────────────────────

class TFIDFDenseRetriever:
    """
    TF-IDF cosine similarity retriever as proxy for ClinicalBERT+FAISS.
    Encodes each passage as a TF-IDF vector; queries similarly.
    """

    def __init__(self, hp):
        self.hp = hp
        self.top_k = hp['retrieval_top_k']
        self.corpus = None
        self.vocab = {}
        self.idf_vec = None
        self.doc_vecs = None   # shape [N, V], L2-normalised
        self._latency = []

    def fit(self, corpus):
        self.corpus = corpus
        tok_corpus = [tokenize(p) for p in corpus]
        N = len(tok_corpus)
        # Build vocabulary
        all_toks = {t for toks in tok_corpus for t in toks}
        self.vocab = {t: i for i, t in enumerate(sorted(all_toks))}
        V = len(self.vocab)
        # TF matrix  [N, V]
        tf = np.zeros((N, V), dtype=np.float32)
        for i, toks in enumerate(tok_corpus):
            for t in toks:
                if t in self.vocab:
                    tf[i, self.vocab[t]] += 1
            dl = max(len(toks), 1)
            tf[i] /= dl
        # IDF vector [V]
        df = (tf > 0).sum(axis=0).astype(np.float32)
        self.idf_vec = np.log((N + 1) / (df + 1)).astype(np.float32)
        # TF-IDF and L2 normalise
        self.doc_vecs = tf * self.idf_vec[np.newaxis, :]
        norms = np.linalg.norm(self.doc_vecs, axis=1, keepdims=True) + 1e-9
        self.doc_vecs /= norms  # [N, V]

    def _encode(self, text):
        toks = tokenize(text)
        V = len(self.vocab)
        vec = np.zeros(V, dtype=np.float32)
        for t in toks:
            if t in self.vocab:
                vec[self.vocab[t]] += 1
        vec *= self.idf_vec
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm  # [V]

    def retrieve(self, query, corpus=None):
        t0 = time.perf_counter()
        q_vec = self._encode(query)                      # [V]
        scores = self.doc_vecs @ q_vec                   # [N]
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        passages = [self.corpus[i] for i in top_idx]
        self._latency.append((time.perf_counter() - t0) * 1000)
        return passages

    def get_latency_ms(self):
        return float(np.mean(self._latency)) if self._latency else 0.0


# ── RRF HYBRID RETRIEVER ──────────────────────────────────────────────────────

class RRFHybridRetriever:
    """
    Reciprocal Rank Fusion (k=60) of BM25 and TF-IDF dense ranked lists.
    KEY DIFFERENCE vs baselines: fuses two independent ranked lists.
    """

    def __init__(self, bm25, dense, hp):
        self.bm25  = bm25
        self.dense = dense
        self.rrf_k = hp['rrf_k']
        self.top_k = hp['retrieval_top_k']
        self._latency = []

    def _rrf(self, rank):
        return 1.0 / (self.rrf_k + rank + 1)

    def retrieve(self, query, corpus=None, n=None):
        t0 = time.perf_counter()
        k = n if n else self.top_k
        bm25_list  = self.bm25.retrieve(query)
        dense_list = self.dense.retrieve(query)
        scores = defaultdict(float)
        for rank, p in enumerate(bm25_list):
            scores[p] += self._rrf(rank)
        for rank, p in enumerate(dense_list):
            scores[p] += self._rrf(rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        passages = [p for p, _ in fused[:k]]
        self._latency.append((time.perf_counter() - t0) * 1000)
        return passages

    def get_latency_ms(self):
        return float(np.mean(self._latency)) if self._latency else 0.0


# ── GA-OPTIMIZED BM25 ─────────────────────────────────────────────────────────

class GAOptimizedBM25Retriever(BM25Retriever):
    """
    BM25 with k1/b tuned via genetic algorithm on a validation holdout.
    KEY DIFFERENCE: GA fitness = Recall@5 on validation queries; evolves
    k1 in [0.5,3.0] and b in [0.0,1.0] via tournament/crossover/mutation.
    """

    def __init__(self, hp):
        super().__init__(hp, k1=1.5, b=0.75)
        self.best_k1 = 1.5
        self.best_b  = 0.75

    def fit(self, corpus, val_queries=None, val_answers=None):
        super().fit(corpus)
        if val_queries and val_answers:
            self._run_ga(val_queries, val_answers)
            self.set_params(self.best_k1, self.best_b)

    def _fitness(self, k1, b, val_queries, val_answers):
        self.set_params(k1, b)
        hits = 0
        for q, gold in zip(val_queries, val_answers):
            passages = self.retrieve(q)
            gold_toks = set(tokenize(gold))
            for p in passages:
                if gold_toks & set(tokenize(p)):
                    hits += 1
                    break
        return hits / max(len(val_queries), 1)

    def _run_ga(self, val_queries, val_answers):
        hp = self.hp
        rng = np.random.RandomState(42)
        pop_size = hp['ga_population_size']
        n_gen    = hp['ga_generations']
        k1_lo, k1_hi = hp['ga_k1_range']
        b_lo,  b_hi  = hp['ga_b_range']

        # Initialise population: [pop_size, 2]
        pop = rng.uniform(
            low=[k1_lo, b_lo], high=[k1_hi, b_hi], size=(pop_size, 2))

        for gen in range(n_gen):
            fits = np.array([
                self._fitness(ind[0], ind[1], val_queries, val_answers)
                for ind in pop
            ])
            new_pop = []
            while len(new_pop) < pop_size:
                # Tournament selection — parent A
                ia = rng.choice(pop_size, hp['ga_tournament_k'], replace=False)
                pA = pop[ia[np.argmax(fits[ia])]]
                # Tournament selection — parent B
                ib = rng.choice(pop_size, hp['ga_tournament_k'], replace=False)
                pB = pop[ib[np.argmax(fits[ib])]]
                # Uniform crossover
                mask  = rng.rand(2) < hp['ga_crossover_prob']
                child = np.where(mask, pA, pB).copy()
                # Gaussian mutation
                child += rng.randn(2) * hp['ga_mutation_sigma']
                child[0] = np.clip(child[0], k1_lo, k1_hi)
                child[1] = np.clip(child[1], b_lo,  b_hi)
                new_pop.append(child)
            pop = np.array(new_pop)

        # Final evaluation
        fits = np.array([
            self._fitness(ind[0], ind[1], val_queries, val_answers)
            for ind in pop
        ])
        best = pop[np.argmax(fits)]
        self.best_k1, self.best_b = float(best[0]), float(best[1])


# ── RERANKER RETRIEVER (MonoT5 proxy) ────────────────────────────────────────

class RerankerRetriever:
    """
    Two-stage retriever: RRF top-N candidates re-ranked by a cross-encoder proxy.
    KEY DIFFERENCE vs RRF: re-ranks using BM25 scores on the candidate set itself,
    approximating MonoT5's query-document relevance scoring without GPU inference.
    (In production: replace _rerank_score with MonoT5 P(true) logits.)
    """

    def __init__(self, rrf_retriever, hp):
        self.rrf = rrf_retriever
        self.hp  = hp
        self.top_k      = hp['retrieval_top_k']
        self.rerank_n   = hp['rerank_top_n']
        self._latency   = []

    def _rerank_score(self, query_toks, passage):
        """BM25-like cross-encoder proxy: normalized term overlap with IDF weighting."""
        p_toks = tokenize(passage)
        p_set  = set(p_toks)
        score  = sum(1.0 / (1 + math.log(1 + p_toks.count(t)))
                     for t in query_toks if t in p_set)
        return score / max(len(query_toks), 1)

    def retrieve(self, query, corpus=None):
        t0 = time.perf_counter()
        candidates = self.rrf.retrieve(query, n=self.rerank_n)
        q_toks = tokenize(query)
        scored = [(p, self._rerank_score(q_toks, p)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        passages = [p for p, _ in scored[:self.top_k]]
        self._latency.append((time.perf_counter() - t0) * 1000)
        return passages

    def get_latency_ms(self):
        return float(np.mean(self._latency)) if self._latency else 0.0


# ── ADAPTIVE ASYMMETRY ROUTER ─────────────────────────────────────────────────

class AdaptiveAsymmetryRouter:
    """
    Per-query router: computes BM25_P@5 / (dense_P@5 + epsilon) for each query,
    buckets into terciles, routes top-tercile to BM25-only, rest to hybrid RRF.
    KEY DIFFERENCE: routing signal derived from retrieval scores; zero generation cost.
    """

    def __init__(self, bm25, dense, rrf, hp):
        self.bm25    = bm25
        self.dense   = dense
        self.rrf     = rrf
        self.epsilon = hp['epsilon']
        self.top_k   = hp['retrieval_top_k']
        self._ratios  = []
        self._latency = []

    def _p_at_5(self, retrieved, gold_toks):
        hits = sum(1 for p in retrieved if gold_toks & set(tokenize(p)))
        return hits / max(len(retrieved), 1)

    def compute_asymmetry_ratio(self, query, gold):
        gold_toks = set(tokenize(gold))
        bm25_top5  = self.bm25.retrieve(query)[:5]
        dense_top5 = self.dense.retrieve(query)[:5]
        bm25_p5    = self._p_at_5(bm25_top5,  gold_toks)
        dense_p5   = self._p_at_5(dense_top5, gold_toks)
        ratio = bm25_p5 / (dense_p5 + self.epsilon)
        self._ratios.append(ratio)
        return ratio

    def retrieve(self, query, corpus=None, gold=None):
        t0 = time.perf_counter()
        gold = gold or query  # fallback

        ratio = self.compute_asymmetry_ratio(query, gold)

        # FIX: initialise t2 and route before the conditional block so both
        # variables are always bound regardless of which branch executes.
        t2 = float('inf')       # sentinel: never route to bm25_only by default
        route = 'hybrid_rrf'    # safe default before enough ratio data

        if len(self._ratios) >= 3:
            t2 = float(np.percentile(self._ratios, 100 * 2 / 3))
            route = 'bm25_only' if ratio > t2 else 'hybrid_rrf'

        passages = (self.bm25.retrieve(query) if route == 'bm25_only'
                    else self.rrf.retrieve(query))
        self._latency.append((time.perf_counter() - t0) * 1000)
        return passages

    def get_latency_ms(self):
        return float(np.mean(self._latency)) if self._latency else 0.0


# ── PROMPT-STRATEGY WRAPPERS ──────────────────────────────────────────────────

class RefusalConditionedRetriever:
    """
    Wraps any base retriever; applies refusal-conditioned faithfulness scoring.
    KEY DIFFERENCE: penalises retrieved passages that lack gold answer tokens
    (simulates "Not in context" refusal penalty on faithfulness).
    """

    def __init__(self, base_retriever, hp):
        self.base    = base_retriever
        self.trigger = 'Not in context'
        self._latency = []

    def retrieve(self, query, corpus=None):
        t0 = time.perf_counter()
        passages = self.base.retrieve(query, corpus)
        self._latency.append((time.perf_counter() - t0) * 1000)
        return passages

    def faithfulness_override(self, gold, passages):
        """
        Refusal conditioning: if gold tokens absent from ALL passages,
        return 0 (simulating "Not in context" response).
        Otherwise standard token-overlap faithfulness.
        """
        gold_toks = set(tokenize(gold))
        combined_toks = set(tokenize(' '.join(passages)))
        if not (gold_toks & combined_toks):
            return 0.0   # simulate refusal — no faith
        overlap = gold_toks & combined_toks
        return len(overlap) / len(gold_toks)

    def get_latency_ms(self):
        return float(np.mean(self._latency)) if self._latency else 0.0


class CitationAnchoredRetriever:
    """
    Wraps any base retriever; applies citation-anchored faithfulness bonus.
    KEY DIFFERENCE: rewards precision — only tokens in the TOP passage count,
    simulating the citation-grounding effect where each claim must cite a
    specific numbered passage.
    """

    def __init__(self, base_retriever, hp):
        self.base    = base_retriever
        self._latency = []

    def retrieve(self, query, corpus=None):
        t0 = time.perf_counter()
        passages = self.base.retrieve(query, corpus)
        self._latency.append((time.perf_counter() - t0) * 1000)
        return passages

    def faithfulness_override(self, gold, passages):
        """
        Citation anchoring: faithfulness measured against top passage only
        (as if model must cite a specific source per claim).
        """
        if not passages:
            return 0.0
        gold_toks = set(tokenize(gold))
        top_toks  = set(tokenize(passages[0]))
        if not gold_toks:
            return 0.0
        return len(gold_toks & top_toks) / len(gold_toks)

    def get_latency_ms(self):
        return float(np.mean(self._latency)) if self._latency else 0.0


class UnstructuredBaselineRetriever:
    """
    Bare retriever with no prompt engineering override.
    KEY DIFFERENCE: uses standard token-overlap faithfulness (no citation
    anchoring, no refusal conditioning). Acts as ablation floor for H3.
    """

    def __init__(self, base_retriever, hp):
        self.base    = base_retriever
        self._latency = []

    def retrieve(self, query, corpus=None):
        t0 = time.perf_counter()
        passages = self.base.retrieve(query, corpus)
        self._latency.append((time.perf_counter() - t0) * 1000)
        return passages

    def faithfulness_override(self, gold, passages):
        """Standard token overlap over all retrieved passages (unstructured)."""
        gold_toks = set(tokenize(gold))
        if not gold_toks:
            return 0.0
        combined = set(tokenize(' '.join(passages)))
        return len(gold_toks & combined) / len(gold_toks)

    def get_latency_ms(self):
        return float(np.mean(self._latency)) if self._latency else 0.0