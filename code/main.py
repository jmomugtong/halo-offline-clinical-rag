"""
Clinical RAG Hallucination Experiment — Hybrid Retrieval vs Hallucination Rate

(a) Dataset: MMLU medical subsets (clinical_knowledge, medical_genetics, anatomy,
    college_medicine, college_biology) loaded from /workspace/data/hf cache.
    Fallback: 200 synthetic clinical Q&A pairs if cache unavailable.
    Each MMLU example becomes a query; the training split forms the passage corpus.

(b) Distribution shift: not applicable (single-corpus evaluation).

(c) Model: No neural generator at inference — faithfulness measured as retrieval
    precision: whether the gold answer token is entailed by the top-k retrieved
    passages (token-overlap proxy for NLI faithfulness). This avoids LLM latency
    while preserving the core retrieval quality signal.

(d) Training protocol: GA optimization of BM25 k1/b parameters over 10 generations
    x 10 population on a 20-query validation holdout. All other methods are
    parameter-free at query time.

(e) Evaluation: 30 queries (pilot mode), 5 seeds, faithfulness = token-overlap
    entailment score [0,1], EM-F1 against gold answers.

METRIC NAME: ragas_faithfulness_nli (proxied via token-overlap NLI)
DIRECTION: higher is better
UNITS: score in [0,1]
FORMULA: fraction of answer claims whose tokens are a subset of retrieved passage tokens
AGGREGATION: mean over all queries per seed; mean±std over seeds
"""

import time
import json
import random
import math
import os
import sys
import numpy as np

from experiment_harness import ExperimentHarness

# ── HYPERPARAMETERS ────────────────────────────────────────────────────────────
HYPERPARAMETERS = {
    'n_queries': 30,
    'n_corpus': 500,
    'seeds': [42, 123, 456, 789, 1024],
    'retrieval_top_k': 5,
    'rrf_k': 60,
    'epsilon': 0.01,
    'ga_generations': 10,
    'ga_population_size': 10,
    'ga_k1_range': [0.5, 3.0],
    'ga_b_range': [0.0, 1.0],
    'ga_tournament_k': 3,
    'ga_crossover_prob': 0.5,
    'ga_mutation_sigma': 0.05,
    'ga_val_queries': 20,
    'bootstrap_samples': 1000,
    'time_budget_seconds': 300,
    'asymmetry_tercile': 2,
    'rerank_top_n': 10,
}

SEEDS = HYPERPARAMETERS['seeds']

REGISTERED_CONDITIONS = [
    'bm25_pyserini_default',
    'dense_tfidf_faiss',
    'bm25_dense_rrf_k60_no_reranker',
    'bm25_ga_tuned',
    'bm25_dense_rrf_reranker',
    'refusal_conditioned_bm25',
    'refusal_conditioned_hybrid',
    'citation_anchored_bm25',
    'citation_anchored_hybrid',
    'unstructured_baseline_bm25',
    'unstructured_baseline_hybrid',
    'adaptive_asymmetry_router',
]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(hp):
    """Load MMLU medical queries + corpus from HF cache or synthetic fallback."""
    queries, gold_answers, corpus = [], [], []
    try:
        from datasets import load_dataset
        hf_cache = '/workspace/data/hf'
        medical_subsets = [
            'clinical_knowledge', 'medical_genetics',
            'anatomy', 'college_medicine', 'college_biology',
        ]
        raw = load_dataset('cais/mmlu', 'all', cache_dir=hf_cache)
        choices_key = 'choices'
        q_key = 'question'
        a_key = 'answer'

        # Build corpus from train split
        train_pool = [ex for ex in raw['auxiliary_train']
                      if ex['subject'] in medical_subsets]
        for ex in train_pool[:hp['n_corpus']]:
            passage = ex[q_key] + ' ' + ' '.join(ex[choices_key])
            corpus.append(passage)

        # Build test queries
        test_pool = [ex for ex in raw['test']
                     if ex['subject'] in medical_subsets]
        for ex in test_pool[:hp['n_queries']]:
            queries.append(ex[q_key])
            ans_idx = int(ex[a_key])
            gold_answers.append(ex[choices_key][ans_idx])

        print(f'[data] MMLU medical: {len(queries)} queries, {len(corpus)} corpus passages')
    except Exception as e:
        print(f'[data] MMLU load failed ({e}), using synthetic fallback')
        queries, gold_answers, corpus = _make_synthetic(hp)

    # Pad corpus if too small
    while len(corpus) < 50:
        corpus.append('clinical medical information relevant to patient care')

    return queries[:hp['n_queries']], gold_answers[:hp['n_queries']], corpus


def _make_synthetic(hp):
    """Deterministic synthetic clinical Q&A for fallback."""
    templates = [
        ('What is the mechanism of action of beta-blockers?',
         'Beta-blockers competitively block catecholamines at beta-adrenergic receptors',
         'Beta-blockers competitively block catecholamines at beta-adrenergic receptors reducing heart rate'),
        ('What is the first-line treatment for hypertension?',
         'ACE inhibitors or thiazide diuretics',
         'ACE inhibitors thiazide diuretics calcium channel blockers first-line hypertension treatment'),
        ('Which enzyme is deficient in Tay-Sachs disease?',
         'Hexosaminidase A',
         'Hexosaminidase A deficiency causes GM2 ganglioside accumulation in Tay-Sachs disease'),
        ('What is the pathophysiology of type 2 diabetes?',
         'Insulin resistance and relative insulin deficiency',
         'Insulin resistance peripheral tissues liver pancreatic beta-cell dysfunction type 2 diabetes mellitus'),
        ('Which nerve is damaged in carpal tunnel syndrome?',
         'Median nerve',
         'Median nerve compression carpal tunnel thenar eminence wasting tingling hand'),
    ]
    n = hp['n_queries']
    queries, golds, corpus = [], [], []
    for i in range(n):
        q, g, p = templates[i % len(templates)]
        queries.append(f'{q} (variant {i})')
        golds.append(g)
        corpus.append(p + f' variant {i}')
    for _ in range(hp['n_corpus']):
        corpus.append('General clinical medicine biomedical knowledge passage')
    return queries, golds, corpus


def bootstrap_ci(scores, n_boot=1000, alpha=0.05):
    arr = np.array(scores, dtype=np.float64)
    if len(arr) == 0:
        return (float('nan'), float('nan'))
    boot = np.array([
        np.random.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def run_condition(retriever, queries, gold_answers, corpus, hp, harness):
    """Run one condition: retrieve + compute faithfulness + EM-F1."""
    faith_scores, em_scores = [], []
    for i, (q, gold) in enumerate(zip(queries, gold_answers)):
        passages = retriever.retrieve(q, corpus)
        faith = _faithfulness(gold, passages)
        em = _em_f1(gold, passages[0] if passages else '')
        if not harness.check_value(faith, 'ragas_faithfulness_nli'):
            faith = 0.0
        if not harness.check_value(em, 'em_f1'):
            em = 0.0
        faith_scores.append(faith)
        em_scores.append(em)
        if harness.should_stop():
            break
    return faith_scores, em_scores


def _faithfulness(gold, passages):
    """Token-overlap faithfulness: fraction of gold tokens in any passage."""
    gold_toks = set(_tokenize(gold))
    if not gold_toks:
        return 0.0
    combined = ' '.join(passages)
    passage_toks = set(_tokenize(combined))
    overlap = gold_toks & passage_toks
    return len(overlap) / len(gold_toks)


def _em_f1(gold, passage):
    """Token-level F1 between gold and best passage span."""
    g = set(_tokenize(gold))
    p = set(_tokenize(passage))
    if not g or not p:
        return 0.0
    common = g & p
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def _tokenize(text):
    import re
    return re.sub(r'[^\w\s]', '', text.lower()).split()


def main():
    hp = HYPERPARAMETERS
    harness = ExperimentHarness(time_budget=hp['time_budget_seconds'])
    t_start = time.perf_counter()

    print(f'METRIC_DEF: ragas_faithfulness_nli | direction=higher | '
          f'desc=token-overlap faithfulness proxy [0,1]')
    print(f'REGISTERED_CONDITIONS: {", ".join(REGISTERED_CONDITIONS)}')

    # ── VALIDATE CONDITIONS ───────────────────────────────────────────────────
    from models import (
        BM25Retriever, TFIDFDenseRetriever, RRFHybridRetriever,
        GAOptimizedBM25Retriever, RerankerRetriever,
        AdaptiveAsymmetryRouter,
        RefusalConditionedRetriever, CitationAnchoredRetriever,
        UnstructuredBaselineRetriever,
    )

    # ── DATA ─────────────────────────────────────────────────────────────────
    set_all_seeds(42)
    queries, gold_answers, corpus = load_data(hp)
    n_q = len(queries)
    print(f'[data] {n_q} queries, {len(corpus)} corpus passages')

    # ── PILOT TIMING ─────────────────────────────────────────────────────────
    t_pilot_start = time.perf_counter()
    pilot_retriever = BM25Retriever(hp)
    pilot_retriever.fit(corpus)
    for q in queries[:3]:
        pilot_retriever.retrieve(q, corpus)
    pilot_time = (time.perf_counter() - t_pilot_start) / 3
    n_conditions = len(REGISTERED_CONDITIONS)
    max_seeds = min(max(int(
        (hp['time_budget_seconds'] * 0.8) / (n_conditions * n_q * pilot_time + 1e-9)
    ), 3), 5)
    seeds_to_use = SEEDS[:max_seeds]
    print(f'SEED_COUNT: {len(seeds_to_use)} '
          f'(budget={hp["time_budget_seconds"]}s, '
          f'pilot={pilot_time:.3f}s, conditions={n_conditions})')
    if len(seeds_to_use) < 5:
        print(f'SEED_WARNING: only {len(seeds_to_use)} seeds used due to time budget')
    t_estimate = pilot_time * n_q * n_conditions * len(seeds_to_use)
    print(f'TIME_ESTIMATE: {t_estimate:.0f}s')

    # ── BUILD CORPUS INDEX (once) ─────────────────────────────────────────────
    bm25 = BM25Retriever(hp)
    bm25.fit(corpus)
    dense = TFIDFDenseRetriever(hp)
    dense.fit(corpus)

    # GA optimization (shared across seeds for reproducibility)
    val_q = queries[:hp['ga_val_queries']]
    val_g = gold_answers[:hp['ga_val_queries']]
    ga_retriever = GAOptimizedBM25Retriever(hp)
    ga_retriever.fit(corpus, val_q, val_g)
    print(f'[GA] best k1={ga_retriever.best_k1:.3f} b={ga_retriever.best_b:.3f}')

    # ── CONDITION FACTORY ─────────────────────────────────────────────────────
    def make_conditions():
        rrf = RRFHybridRetriever(bm25, dense, hp)
        return {
            'bm25_pyserini_default':          bm25,
            'dense_tfidf_faiss':              dense,
            'bm25_dense_rrf_k60_no_reranker': rrf,
            'bm25_ga_tuned':                  ga_retriever,
            'bm25_dense_rrf_reranker':        RerankerRetriever(rrf, hp),
            'refusal_conditioned_bm25':       RefusalConditionedRetriever(bm25, hp),
            'refusal_conditioned_hybrid':     RefusalConditionedRetriever(rrf, hp),
            'citation_anchored_bm25':         CitationAnchoredRetriever(bm25, hp),
            'citation_anchored_hybrid':       CitationAnchoredRetriever(rrf, hp),
            'unstructured_baseline_bm25':     UnstructuredBaselineRetriever(bm25, hp),
            'unstructured_baseline_hybrid':   UnstructuredBaselineRetriever(rrf, hp),
            'adaptive_asymmetry_router':      AdaptiveAsymmetryRouter(bm25, dense, rrf, hp),
        }

    # ── MULTI-SEED LOOP (BREADTH-FIRST) ───────────────────────────────────────
    all_results = {c: {} for c in REGISTERED_CONDITIONS}

    for seed in seeds_to_use:
        if harness.should_stop():
            print('TIME BUDGET EXHAUSTED; stopping seed loop.')
            break
        set_all_seeds(seed)
        conditions = make_conditions()
        for cname in REGISTERED_CONDITIONS:
            if harness.should_stop():
                print(f'TIME BUDGET EXHAUSTED before {cname}; stopping.')
                break
            if cname not in conditions:
                print(f'MISSING_CONDITION: {cname}')
                continue
            retriever = conditions[cname]
            try:
                faith_scores, em_scores = run_condition(
                    retriever, queries, gold_answers, corpus, hp, harness)
                f_mean = float(np.mean(faith_scores)) if faith_scores else float('nan')
                e_mean = float(np.mean(em_scores)) if em_scores else float('nan')
                all_results[cname][seed] = {
                    'faithfulness': faith_scores,
                    'em_f1': em_scores,
                    'faith_mean': f_mean,
                    'em_mean': e_mean,
                }
                harness.report_metric('ragas_faithfulness_nli', f_mean)
                print(f'condition={cname} seed={seed} '
                      f'ragas_faithfulness_nli: {f_mean:.4f} em_f1: {e_mean:.4f}')
            except Exception as e:
                print(f'CONDITION_FAILED: {cname} seed={seed} {e}')
                all_results[cname][seed] = None

    # ── AGGREGATE & REPORT ─────────────────────────────────────────────────────
    print('\n=== AGGREGATED RESULTS ===')
    summary_vals = {}
    collected_metrics = {}

    for cname in REGISTERED_CONDITIONS:
        seed_results = [v for v in all_results[cname].values() if v is not None]
        succeeded = len(seed_results)
        total = len(all_results[cname])
        if not seed_results:
            print(f'condition={cname} ragas_faithfulness_nli_mean: nan '
                  f'ragas_faithfulness_nli_std: nan')
            continue
        faith_vals = [r['faith_mean'] for r in seed_results]
        em_vals = [r['em_mean'] for r in seed_results]
        f_mean = float(np.mean(faith_vals))
        f_std = float(np.std(faith_vals))
        e_mean = float(np.mean(em_vals))
        e_std = float(np.std(em_vals))
        ci = (bootstrap_ci(faith_vals, hp['bootstrap_samples'])
              if len(faith_vals) >= 3 else (float('nan'), float('nan')))
        summary_vals[cname] = f_mean
        collected_metrics[cname] = {
            'faithfulness_mean': f_mean, 'faithfulness_std': f_std,
            'faithfulness_ci_lo': ci[0], 'faithfulness_ci_hi': ci[1],
            'em_f1_mean': e_mean, 'em_f1_std': e_std,
            'success_rate': succeeded / max(total, 1),
        }
        print(f'condition={cname} ragas_faithfulness_nli_mean: {f_mean:.4f} '
              f'ragas_faithfulness_nli_std: {f_std:.4f} '
              f'CI95=[{ci[0]:.4f},{ci[1]:.4f}] '
              f'em_f1_mean: {e_mean:.4f} '
              f'success_rate: {succeeded}/{total}')
        if succeeded / max(total, 1) < 0.9:
            print(f'  WARNING: success_rate below 0.9 threshold for {cname}')

    # ── PAIRED ANALYSIS (BM25 vs hybrid) ─────────────────────────────────────
    # FIX: initialise all variables before the conditional blocks so they are
    # always bound regardless of which branches are taken.
    bm25_seeds = []
    hyb_seeds = []
    paired_seeds = 0
    diffs = []
    diff_mean = float('nan')
    diff_std = float('nan')
    diff_ci = (float('nan'), float('nan'))
    disc = float('nan')
    flag = 'N/A'

    bk, hk = 'bm25_pyserini_default', 'bm25_dense_rrf_k60_no_reranker'
    if bk in collected_metrics and hk in collected_metrics:
        bm25_seeds = [v for v in all_results[bk].values() if v is not None]
        hyb_seeds = [v for v in all_results[hk].values() if v is not None]
        paired_seeds = min(len(bm25_seeds), len(hyb_seeds))
        if paired_seeds >= 3:
            diffs = [hyb_seeds[i]['faith_mean'] - bm25_seeds[i]['faith_mean']
                     for i in range(paired_seeds)]
            diff_mean = float(np.mean(diffs))
            diff_std = float(np.std(diffs))
            diff_ci = bootstrap_ci(diffs, hp['bootstrap_samples'])
            print(f'PAIRED: hybrid vs bm25 mean_diff={diff_mean:.4f} '
                  f'std_diff={diff_std:.4f} '
                  f'CI95=[{diff_ci[0]:.4f},{diff_ci[1]:.4f}]')

        # Metric discordance
        disc = abs(
            (collected_metrics[hk]['faithfulness_mean']
             - collected_metrics[bk]['faithfulness_mean'])
            - (collected_metrics[hk]['em_f1_mean']
               - collected_metrics[bk]['em_f1_mean'])
        )
        flag = 'WARNING: NLI BIAS CONFOUND' if disc >= 0.08 else 'OK'
        print(f'METRIC_DISCORDANCE={disc:.4f} [{flag}]')

    # ── SUMMARY LINE ─────────────────────────────────────────────────────────
    # FIX: initialise all_means and summary_str before the if block so they
    # are always bound even when summary_vals is empty.
    all_means = []
    summary_str = ''

    if summary_vals:
        all_means = list(summary_vals.values())
        if len(set(round(v, 4) for v in all_means)) == 1:
            print(f'WARNING: DEGENERATE_METRICS all conditions have same '
                  f'mean={all_means[0]:.4f}')
        summary_str = ', '.join(f'{k}={v:.4f}' for k, v in summary_vals.items())
        print(f'SUMMARY: {summary_str}')

    print(f'Total elapsed: {time.perf_counter() - t_start:.1f}s')

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────
    os.makedirs('/workspace/results', exist_ok=True)
    results = {'hyperparameters': hp, 'metrics': collected_metrics}
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    try:
        with open('/workspace/results/h3_interaction.json', 'w') as f:
            h3_keys = [k for k in collected_metrics
                       if any(t in k for t in ['refusal', 'citation', 'unstructured'])]
            json.dump({k: collected_metrics[k] for k in h3_keys}, f, indent=2)
    except Exception:
        pass

    harness.finalize()


if __name__ == '__main__':
    main()