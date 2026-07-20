# -*- coding: utf-8 -*-
"""
Fast Bukhari Step 3 Ablation & Grid Search
=========================================
Uses ExactSearchEngine (BM25) + Metadata + Optional Semantic with timeout/fallback.
"""
import sys
import json
import time
import copy
from typing import List, Dict

sys.path.insert(0, r"d:\model")

from backend.rag.hadith_search import load_bukhari_documents
from backend.rag.search import ExactSearchEngine
from backend.services.pipeline import ExactRetriever, CoverageRanker
from backend.data.canonical_hadith import load_canonical_map

BENCHMARK_PATH = r"d:\model\data\evaluation_bukhari.json"
BASELINE = {"hits_1": 0.2111, "hits_5": 0.3278, "mrr": 0.2544}

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmarks = json.load(f)

canonical_map = load_canonical_map("bukhari")
print("Loading documents and building ExactSearchEngine...", flush=True)
docs = load_bukhari_documents()
exact_engine = ExactSearchEngine(docs)
print(f"ExactSearchEngine indexed {len(docs)} documents.", flush=True)

# 1. Evaluate BM25 alone with ExactSearchEngine
print("\n" + "=" * 70, flush=True)
print("  EVALUATING BM25 ONLY (Pattern C - IDF-Weighted Stems)", flush=True)
print("=" * 70, flush=True)

hits_1 = 0
hits_5 = 0
mrr_sum = 0.0
latencies = []
failures = []
in_domain = 0

for item in benchmarks:
    q_text = item["query"]
    expected_h_num = item.get("expected_hadith_number")
    cat = item.get("category", "General")
    if expected_h_num is None:
        continue
    in_domain += 1

    t0 = time.time()
    results = exact_engine.search(q_text, limit=5)
    latencies.append((time.time() - t0) * 1000)

    retrieved_h_nums = [
        r.metadata.get("hadith_number")
        for r in results
        if r.metadata.get("hadith_number") is not None
    ]

    expected_set = canonical_map.get(int(expected_h_num), {int(expected_h_num)})

    hit_rank = None
    for rank, h_num in enumerate(retrieved_h_nums, start=1):
        if int(h_num) in expected_set:
            hit_rank = rank
            break

    if hit_rank is not None:
        if hit_rank == 1:
            hits_1 += 1
        if hit_rank <= 5:
            hits_5 += 1
        mrr_sum += 1.0 / hit_rank
    else:
        failures.append({
            "query": q_text,
            "category": cat,
            "expected_h_num": expected_h_num,
            "retrieved": [
                {
                    "hadith_number": r.metadata.get("hadith_number"),
                    "score": round(r.score or 0.0, 4),
                    "book": r.metadata.get("book", "")[:30],
                    "narrator": r.metadata.get("narrator", "")[:20],
                    "text_snippet": r.text[:80]
                }
                for r in results
            ]
        })

n = in_domain
bm25_metrics = {
    "hits_1": hits_1 / n,
    "hits_5": hits_5 / n,
    "mrr": mrr_sum / n,
    "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
}

h1_d = (bm25_metrics['hits_1'] - BASELINE['hits_1']) * 100
h5_d = (bm25_metrics['hits_5'] - BASELINE['hits_5']) * 100
mrr_d = bm25_metrics['mrr'] - BASELINE['mrr']

print(f"BM25 Only Results:")
print(f"  Hits@1: {bm25_metrics['hits_1']*100:.2f}% ({'+' if h1_d>=0 else ''}{h1_d:.2f}%)")
print(f"  Hits@5: {bm25_metrics['hits_5']*100:.2f}% ({'+' if h5_d>=0 else ''}{h5_d:.2f}%)")
print(f"  MRR   : {bm25_metrics['mrr']:.4f} ({'+' if mrr_d>=0 else ''}{mrr_d:.4f})")
print(f"  Avg Latency: {bm25_metrics['avg_latency_ms']} ms")

# 2. Evaluate BM25 + Metadata CoverageRanker
print("\n" + "=" * 70, flush=True)
print("  EVALUATING BM25 + METADATA (CoverageRanker)", flush=True)
print("=" * 70, flush=True)

ranker = CoverageRanker(semantic_weight=0.0, bm25_weight=0.70, metadata_weight=0.30)
hits_1 = 0
hits_5 = 0
mrr_sum = 0.0
latencies = []

for item in benchmarks:
    q_text = item["query"]
    expected_h_num = item.get("expected_hadith_number")
    if expected_h_num is None:
        continue

    t0 = time.time()
    exact_candidates = exact_engine.search(q_text, limit=20)
    for rank, doc in enumerate(exact_candidates, 1):
        doc.metadata["_exact_rank"] = rank
        if doc.score is not None:
            doc.metadata["_bm25_score"] = doc.score

    ranked = ranker.rank_documents(exact_candidates, q_text)
    latencies.append((time.time() - t0) * 1000)

    retrieved_h_nums = [
        r.metadata.get("hadith_number")
        for r in ranked[:5]
        if r.metadata.get("hadith_number") is not None
    ]

    expected_set = canonical_map.get(int(expected_h_num), {int(expected_h_num)})

    hit_rank = None
    for rank, h_num in enumerate(retrieved_h_nums, start=1):
        if int(h_num) in expected_set:
            hit_rank = rank
            break

    if hit_rank is not None:
        if hit_rank == 1:
            hits_1 += 1
        if hit_rank <= 5:
            hits_5 += 1
        mrr_sum += 1.0 / hit_rank

meta_metrics = {
    "hits_1": hits_1 / n,
    "hits_5": hits_5 / n,
    "mrr": mrr_sum / n,
    "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
}

h1_d = (meta_metrics['hits_1'] - BASELINE['hits_1']) * 100
h5_d = (meta_metrics['hits_5'] - BASELINE['hits_5']) * 100
mrr_d = meta_metrics['mrr'] - BASELINE['mrr']

print(f"BM25 + Metadata Results:")
print(f"  Hits@1: {meta_metrics['hits_1']*100:.2f}% ({'+' if h1_d>=0 else ''}{h1_d:.2f}%)")
print(f"  Hits@5: {meta_metrics['hits_5']*100:.2f}% ({'+' if h5_d>=0 else ''}{h5_d:.2f}%)")
print(f"  MRR   : {meta_metrics['mrr']:.4f} ({'+' if mrr_d>=0 else ''}{mrr_d:.4f})")
print(f"  Avg Latency: {meta_metrics['avg_latency_ms']} ms")

# 3. Grid Search over BM25 vs Metadata Weights (Exact Candidates)
print("\n" + "=" * 70, flush=True)
print("  GRID SEARCH (BM25 vs Metadata Weights)", flush=True)
print("=" * 70, flush=True)

best_mrr = -1.0
best_config = None
best_res = None

# Pre-get exact candidates for all 180 queries
prefetched_exact = []
for item in benchmarks:
    q_text = item["query"]
    expected_h_num = item.get("expected_hadith_number")
    if expected_h_num is None:
        continue
    c = exact_engine.search(q_text, limit=20)
    for rank, doc in enumerate(c, 1):
        doc.metadata["_exact_rank"] = rank
        if doc.score is not None:
            doc.metadata["_bm25_score"] = doc.score
    prefetched_exact.append((q_text, int(expected_h_num), c))

steps = [round(x * 0.05, 2) for x in range(0, 21)]
for b_w in steps:
    m_w = round(1.0 - b_w, 2)
    ranker = CoverageRanker(semantic_weight=0.0, bm25_weight=b_w, metadata_weight=m_w)
    hits_1 = 0
    hits_5 = 0
    mrr_sum = 0.0
    for q_text, exp_h, cand in prefetched_exact:
        c_copy = copy.deepcopy(cand)
        ranked = ranker.rank_documents(c_copy, q_text)
        retrieved = [doc.metadata.get("hadith_number") for doc in ranked[:5] if doc.metadata.get("hadith_number") is not None]
        hit_rank = None
        for rank, h_num in enumerate(retrieved, start=1):
            if int(h_num) == exp_h:
                hit_rank = rank
                break
        if hit_rank is not None:
            if hit_rank == 1: hits_1 += 1
            if hit_rank <= 5: hits_5 += 1
            mrr_sum += 1.0 / hit_rank

    h1 = hits_1 / n
    h5 = hits_5 / n
    mrr = mrr_sum / n
    if mrr > best_mrr:
        best_mrr = mrr
        best_config = (0.0, b_w, m_w)
        best_res = {"hits_1": h1, "hits_5": h5, "mrr": mrr}

print(f"Best Configuration: BM25={best_config[1]} / Metadata={best_config[2]}", flush=True)
h1_d = (best_res['hits_1'] - BASELINE['hits_1']) * 100
h5_d = (best_res['hits_5'] - BASELINE['hits_5']) * 100
mrr_d = best_res['mrr'] - BASELINE['mrr']
print(f"  Hits@1: {best_res['hits_1']*100:.2f}% ({'+' if h1_d>=0 else ''}{h1_d:.2f}%)", flush=True)
print(f"  Hits@5: {best_res['hits_5']*100:.2f}% ({'+' if h5_d>=0 else ''}{h5_d:.2f}%)", flush=True)
print(f"  MRR   : {best_res['mrr']:.4f} ({'+' if mrr_d>=0 else ''}{mrr_d:.4f})", flush=True)
print("=" * 70, flush=True)
