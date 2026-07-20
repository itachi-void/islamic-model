# -*- coding: utf-8 -*-
import sys
import json
import time
import copy

sys.path.insert(0, r"d:\model")

from backend.rag.hadith_search import HadithSearchService
from backend.services.pipeline import CoverageRanker

BENCHMARK_PATH = r"d:\model\data\evaluation_bukhari.json"
BASELINE = {"hits_1": 0.2111, "hits_5": 0.3278, "mrr": 0.2544}

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmarks = json.load(f)

service = HadithSearchService()
print("Prefetching candidates...", flush=True)

cached_items = []
for item in benchmarks:
    q_text = item["query"]
    expected_h_num = item.get("expected_hadith_number")
    cat = item.get("category", "General")
    if expected_h_num is None:
        continue
    t0 = time.time()
    candidates = service.pipeline.retriever.retrieve_candidates(q_text, limit=5)
    lat = (time.time() - t0) * 1000
    cached_items.append({
        "query": q_text,
        "expected_hadith_number": int(expected_h_num),
        "category": cat,
        "candidates": candidates,
        "retrieval_latency": lat
    })

def eval_config(sw, bw, mw):
    ranker = CoverageRanker(semantic_weight=sw, bm25_weight=bw, metadata_weight=mw)
    hits_1 = 0
    hits_5 = 0
    mrr_sum = 0.0
    latencies = []
    n = len(cached_items)

    for item in cached_items:
        c_copy = copy.deepcopy(item["candidates"])
        t0 = time.time()
        ranked = ranker.rank_documents(c_copy, item["query"])
        latencies.append(item["retrieval_latency"] + (time.time() - t0) * 1000)

        retrieved_h_nums = [
            doc.metadata.get("hadith_number")
            for doc in ranked[:5]
            if doc.metadata.get("hadith_number") is not None
        ]

        hit_rank = None
        for rank, h_num in enumerate(retrieved_h_nums, start=1):
            if int(h_num) == item["expected_hadith_number"]:
                hit_rank = rank
                break

        if hit_rank is not None:
            if hit_rank == 1:
                hits_1 += 1
            if hit_rank <= 5:
                hits_5 += 1
            mrr_sum += 1.0 / hit_rank

    return {
        "hits_1": hits_1 / n,
        "hits_5": hits_5 / n,
        "mrr": mrr_sum / n,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2)
    }

print("\n" + "=" * 70)
print("  ABLATION STUDY (4 Modes + Combinations)")
print("=" * 70)

ablation_configs = [
    ("Semantic Only (1.00 / 0.00 / 0.00)",       1.00, 0.00, 0.00),
    ("BM25 Only (0.00 / 1.00 / 0.00)",           0.00, 1.00, 0.00),
    ("Semantic + BM25 (0.50 / 0.50 / 0.00)",     0.50, 0.50, 0.00),
    ("Semantic + Metadata (0.70 / 0.00 / 0.30)", 0.70, 0.00, 0.30),
    ("BM25 + Metadata (0.00 / 0.70 / 0.30)",     0.00, 0.70, 0.30),
    ("Full Hybrid (0.55 / 0.30 / 0.15)",         0.55, 0.30, 0.15),
]

for label, sw, bw, mw in ablation_configs:
    res = eval_config(sw, bw, mw)
    h1_d = (res['hits_1'] - BASELINE['hits_1']) * 100
    h5_d = (res['hits_5'] - BASELINE['hits_5']) * 100
    mrr_d = res['mrr'] - BASELINE['mrr']
    print(f"[{label}]")
    print(f"  Hits@1: {res['hits_1']*100:.2f}% ({'+' if h1_d>=0 else ''}{h1_d:.2f}%)")
    print(f"  Hits@5: {res['hits_5']*100:.2f}% ({'+' if h5_d>=0 else ''}{h5_d:.2f}%)")
    print(f"  MRR   : {res['mrr']:.4f} ({'+' if mrr_d>=0 else ''}{mrr_d:.4f})")
    print(f"  Latency: {res['avg_latency_ms']} ms\n")

print("=" * 70)
print("  GRID SEARCH (Exhaustive over S / B / M weights)")
print("=" * 70)

grid_results = []
steps = [round(x * 0.05, 2) for x in range(0, 21)]

for s in steps:
    for b in steps:
        m_w = round(1.0 - s - b, 2)
        if m_w < 0.0 or m_w > 1.0 or abs(s + b + m_w - 1.0) > 1e-4:
            continue
        res = eval_config(s, b, m_w)
        grid_results.append((s, b, m_w, res))

sorted_grid = sorted(grid_results, key=lambda x: (x[3]["mrr"], x[3]["hits_1"], x[3]["hits_5"]), reverse=True)

print("\nTOP 5 CONFIGURATIONS FOUND BY GRID SEARCH:")
for rank, (s, b, m_w, res) in enumerate(sorted_grid[:5], 1):
    h1_d = (res['hits_1'] - BASELINE['hits_1']) * 100
    h5_d = (res['hits_5'] - BASELINE['hits_5']) * 100
    mrr_d = res['mrr'] - BASELINE['mrr']
    print(f"  #{rank}: S={s:.2f} | B={b:.2f} | M={m_w:.2f}")
    print(f"      Hits@1: {res['hits_1']*100:.2f}% ({'+' if h1_d>=0 else ''}{h1_d:.2f}%) | Hits@5: {res['hits_5']*100:.2f}% ({'+' if h5_d>=0 else ''}{h5_d:.2f}%) | MRR: {res['mrr']:.4f} ({'+' if mrr_d>=0 else ''}{mrr_d:.4f})")

print("=" * 70)
