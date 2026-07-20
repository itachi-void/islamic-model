# -*- coding: utf-8 -*-
"""
Fast Curated Benchmark Evaluation (523 Queries)
================================================
Evaluates Dialect Normalizer + Alias Dictionary + Exact Engine on 523 Curated Queries.
"""
import sys
import json
import time

sys.path.insert(0, r"d:\model")

from backend.rag.hadith_search import load_bukhari_documents
from backend.rag.search import ExactSearchEngine
from backend.rag.query_normalizer import normalize_query_dialect
from backend.data.alias_dictionary import expand_query_aliases
from backend.rag.reranker import ExactMatchBonusReranker
from backend.data.canonical_hadith import load_canonical_map

BENCHMARK_PATH = r"d:\model\data\evaluation_bukhari.json"

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmarks = json.load(f)

canonical_map = load_canonical_map("bukhari")
docs = load_bukhari_documents()
exact_engine = ExactSearchEngine(docs)
reranker = ExactMatchBonusReranker()

hits_1 = 0
hits_5 = 0
mrr_sum = 0.0
latencies = []
cat_stats = {}

t0_all = time.time()
for item in benchmarks:
    raw_q = item["query"]
    expected_h_num = item["expected_hadith_number"]
    cat = item.get("category", "General")

    if cat not in cat_stats:
        cat_stats[cat] = {"total": 0, "hits_5": 0, "mrr_sum": 0.0}
    cat_stats[cat]["total"] += 1

    t0 = time.time()
    cleaned_q = normalize_query_dialect(raw_q)
    expanded_q = expand_query_aliases(cleaned_q)

    exact_candidates = exact_engine.search(expanded_q, limit=20)
    reranked = reranker.rerank(expanded_q, exact_candidates, top_k=5)
    latencies.append((time.time() - t0) * 1000)

    retrieved = [doc.metadata.get("hadith_number") for doc in reranked if doc.metadata.get("hadith_number") is not None]
    expected_set = canonical_map.get(int(expected_h_num), {int(expected_h_num)})

    hit_rank = None
    for rank, h_num in enumerate(retrieved, start=1):
        if int(h_num) in expected_set:
            hit_rank = rank
            break

    if hit_rank is not None:
        if hit_rank == 1:
            hits_1 += 1
        if hit_rank <= 5:
            hits_5 += 1
            cat_stats[cat]["hits_5"] += 1
        rr = 1.0 / hit_rank
        mrr_sum += rr
        cat_stats[cat]["mrr_sum"] += rr

n = len(benchmarks)
print("=" * 70, flush=True)
print(f"CURATED BUKHARI BENCHMARK RESULTS ({n} Queries in {time.time()-t0_all:.2f}s)", flush=True)
print("=" * 70, flush=True)
print(f"Hits@1       : {hits_1/n*100:.2f}%", flush=True)
print(f"Hits@5       : {hits_5/n*100:.2f}%", flush=True)
print(f"MRR          : {mrr_sum/n:.4f}", flush=True)
print(f"Avg Latency  : {sum(latencies)/len(latencies):.2f} ms", flush=True)
print("\n[Category Breakdown]", flush=True)
for cat, stats in cat_stats.items():
    tot = stats["total"]
    h5 = stats["hits_5"] / tot * 100 if tot else 0
    mrr = stats["mrr_sum"] / tot if tot else 0
    print(f"  - {cat:<20} | Total: {tot:<5} | Hits@5: {h5:.2f}% | MRR: {mrr:.4f}", flush=True)
print("=" * 70, flush=True)
