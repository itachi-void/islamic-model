# -*- coding: utf-8 -*-
"""
Bukhari Step 3 Ablation Study + Weight Grid Search (Optimized & Fast)
======================================================================
Pre-retrieves candidates once and evaluates ranker weights instantly.

Usage:
    python tests/step3_ablation.py
"""
import sys
import json
import time
import copy
import os
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, r"d:\model")

from backend.core.settings import settings
from backend.domain.document import BaseDocument
from backend.domain.query import SearchQuery
from backend.rag.hadith_search import HadithSearchService, load_bukhari_documents
from backend.rag.search import ExactSearchEngine, normalize_arabic, extract_stemmed_tokens
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.services.pipeline import (
    ExactRetriever, SemanticRetriever, HybridRetriever,
    CoverageRanker, MetadataFilter, ResponseBuilder, RetrievalPipeline
)

BENCHMARK_PATH = r"d:\model\data\evaluation_bukhari.json"
TOP_K = 5
BASELINE = {"hits_1": 0.2111, "hits_5": 0.3278, "mrr": 0.2544}


def load_benchmark() -> List[dict]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prefetch_candidates(service: HadithSearchService, benchmarks: List[dict]) -> List[Dict]:
    """Retrieves candidates for all benchmark queries once."""
    print("Prefetching candidates for all benchmark queries...", flush=True)
    cached_items = []
    t0_all = time.time()
    for idx, item in enumerate(benchmarks, 1):
        q_text = item["query"]
        expected_h_num = item.get("expected_hadith_number")
        cat = item.get("category", "General")
        if expected_h_num is None:
            continue

        t0 = time.time()
        # Get raw candidates from hybrid retriever
        candidates = service.pipeline.retriever.retrieve_candidates(q_text, limit=TOP_K)
        latency = (time.time() - t0) * 1000

        cached_items.append({
            "query": q_text,
            "expected_hadith_number": int(expected_h_num),
            "category": cat,
            "candidates": candidates,
            "retrieval_latency": latency
        })
        if len(cached_items) % 20 == 0 or len(cached_items) == 180:
            print(f"  [Prefetch Progress] {len(cached_items)}/180 queries retrieved...", flush=True)
    print(f"Prefetched candidates for {len(cached_items)} queries in {time.time()-t0_all:.2f}s", flush=True)
    return cached_items


def evaluate_candidates(ranker: CoverageRanker, cached_items: List[Dict], top_k: int = TOP_K) -> Dict:
    hits_1 = 0
    hits_5 = 0
    mrr_sum = 0.0
    latencies = []
    failures = []

    for item in cached_items:
        q_text = item["query"]
        expected_h_num = item["expected_hadith_number"]
        cat = item["category"]

        # Deepcopy candidates so ranker metadata cleanups don't mutate cache
        candidates_copy = copy.deepcopy(item["candidates"])

        t0 = time.time()
        ranked = ranker.rank_documents(candidates_copy, q_text)
        latency_ms = item["retrieval_latency"] + (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved = ranked[:top_k]
        retrieved_h_nums = [
            (doc.metadata.get("hadith_number"), doc.score, doc)
            for doc in retrieved
            if doc.metadata.get("hadith_number") is not None
        ]

        hit_rank = None
        for rank, (h_num, score, doc) in enumerate(retrieved_h_nums, start=1):
            if int(h_num) == int(expected_h_num):
                hit_rank = rank
                break

        if hit_rank is not None:
            if hit_rank == 1:
                hits_1 += 1
            if hit_rank <= top_k:
                hits_5 += 1
            mrr_sum += 1.0 / hit_rank
        else:
            failures.append({
                "query": q_text,
                "category": cat,
                "expected_h_num": expected_h_num,
                "retrieved": [
                    {
                        "hadith_number": h_num,
                        "score": round(score or 0.0, 4),
                        "book": doc.metadata.get("book", "")[:30],
                        "narrator": doc.metadata.get("narrator", "")[:20],
                        "text_snippet": doc.text[:80]
                    }
                    for h_num, score, doc in retrieved_h_nums
                ]
            })

    n = len(cached_items)
    return {
        "in_domain": n,
        "hits_1": hits_1 / n if n else 0,
        "hits_5": hits_5 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "failures": failures
    }


def print_metrics(label: str, m: Dict, baseline: Dict = None):
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    h1 = m['hits_1'] * 100
    h5 = m['hits_5'] * 100
    mrr = m['mrr']
    lat = m['avg_latency_ms']
    print(f"  Hits@1       : {h1:.2f}%")
    print(f"  Hits@5       : {h5:.2f}%")
    print(f"  MRR          : {mrr:.4f}")
    print(f"  Avg Latency  : {lat} ms")
    if baseline:
        d1 = h1 - baseline['hits_1'] * 100
        d5 = h5 - baseline['hits_5'] * 100
        dm = mrr - baseline['mrr']
        sign1 = "+" if d1 >= 0 else ""
        sign5 = "+" if d5 >= 0 else ""
        signm = "+" if dm >= 0 else ""
        print(f"  Δ Hits@1     : {sign1}{d1:.2f}%")
        print(f"  Δ Hits@5     : {sign5}{d5:.2f}%")
        print(f"  Δ MRR        : {signm}{dm:.4f}")


def print_failures(failures: List[Dict], max_show: int = 20):
    print(f"\n{'='*70}")
    print(f"  TOP {min(max_show, len(failures))} FAILED QUERIES")
    print(f"{'='*70}")
    for i, f in enumerate(failures[:max_show], 1):
        print(f"\n  [{i}] Category: {f['category']}")
        print(f"       Query    : {f['query']}")
        print(f"       Expected : Hadith #{f['expected_h_num']}")
        print(f"       Retrieved:")
        for r in f['retrieved'][:5]:
            print(f"         • #{r['hadith_number']} | Score={r['score']:.4f} | Book={r['book']} | Narrator={r['narrator']}")
            print(f"           Text: {r['text_snippet']}...")


def run_ablation(cached_items: List[Dict]):
    print(f"\n{'='*70}")
    print("  ABLATION STUDY")
    print(f"{'='*70}")

    configs = [
        ("1. Semantic Only",         1.00, 0.00, 0.00),
        ("2. BM25 Only",             0.00, 1.00, 0.00),
        ("3. Semantic + BM25",       0.55, 0.45, 0.00),
        ("4. Semantic + Metadata",   0.70, 0.00, 0.30),
        ("5. BM25 + Metadata",       0.00, 0.70, 0.30),
        ("6. Hybrid Default (0.55 / 0.30 / 0.15)", 0.55, 0.30, 0.15),
    ]

    ablation_results = {}
    for label, sw, bw, mw in configs:
        ranker = CoverageRanker(semantic_weight=sw, bm25_weight=bw, metadata_weight=mw)
        m = evaluate_candidates(ranker, cached_items)
        ablation_results[label] = m
        print_metrics(label, m, BASELINE)

    return ablation_results


def run_grid_search(cached_items: List[Dict]):
    print(f"\n{'='*70}")
    print("  WEIGHT GRID SEARCH (Exhaustive — 0.05 step)")
    print(f"{'='*70}")

    best_mrr = -1.0
    best_config = None
    best_metrics = None
    results_table = []

    # Grid: s + b + m = 1.0 (step 0.05)
    steps = [round(x * 0.05, 2) for x in range(0, 21)]

    tested = 0
    for s in steps:
        for b in steps:
            m_w = round(1.0 - s - b, 2)
            if m_w < 0.0 or m_w > 1.0:
                continue
            if abs(s + b + m_w - 1.0) > 1e-4:
                continue

            ranker = CoverageRanker(semantic_weight=s, bm25_weight=b, metadata_weight=m_w)
            metrics = evaluate_candidates(ranker, cached_items)
            results_table.append((s, b, m_w, metrics))
            tested += 1

            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                best_config = (s, b, m_w)
                best_metrics = metrics

    print(f"  Total valid configurations evaluated: {tested}")
    print(f"\n{'─'*70}")
    print(f"  BEST CONFIGURATION BY MRR")
    print(f"{'─'*70}")
    print(f"  Semantic Weight : {best_config[0]}")
    print(f"  BM25 Weight     : {best_config[1]}")
    print(f"  Metadata Weight : {best_config[2]}")
    print_metrics("Best Grid Search Config", best_metrics, BASELINE)

    # Top 5 configs by MRR
    print(f"\n{'─'*70}")
    print("  TOP 5 CONFIGURATIONS BY MRR")
    print(f"{'─'*70}")
    sorted_results = sorted(results_table, key=lambda x: (x[3]["mrr"], x[3]["hits_1"], x[3]["hits_5"]), reverse=True)
    for i, (s, b, m_w, met) in enumerate(sorted_results[:5], 1):
        print(f"  #{i}: S={s:.2f} / B={b:.2f} / M={m_w:.2f} | Hits@1={met['hits_1']*100:.2f}% | Hits@5={met['hits_5']*100:.2f}% | MRR={met['mrr']:.4f}")

    return best_config, best_metrics


def main():
    print(f"{'='*70}")
    print("  BUKHARI STEP 3 — ABLATION STUDY + WEIGHT GRID SEARCH")
    print(f"  Baseline (Step 2): Hits@1={BASELINE['hits_1']*100:.2f}% | Hits@5={BASELINE['hits_5']*100:.2f}% | MRR={BASELINE['mrr']:.4f}")
    print(f"{'='*70}")

    benchmarks = load_benchmark()
    service = HadithSearchService()

    # 1. Prefetch candidates once
    cached_items = prefetch_candidates(service, benchmarks)

    # 2. Step 3 default evaluation
    print("\n[1] Evaluating Step 3 Default (Semantic=0.55, BM25=0.30, Metadata=0.15)...")
    ranker_default = CoverageRanker(semantic_weight=0.55, bm25_weight=0.30, metadata_weight=0.15)
    step3_metrics = evaluate_candidates(ranker_default, cached_items)
    print_metrics("STEP 3 DEFAULT (vs Step 2 Baseline)", step3_metrics, BASELINE)

    # 3. Failures analysis if any
    if step3_metrics["failures"]:
        print_failures(step3_metrics["failures"], max_show=20)

    # 4. Ablation Study
    ablation_results = run_ablation(cached_items)

    # 5. Grid Search
    best_config, best_metrics = run_grid_search(cached_items)

    # 6. Final verdict & comparison
    print(f"\n{'='*70}")
    print("  FINAL VERDICT")
    print(f"{'='*70}")
    d1 = (step3_metrics['hits_1'] - BASELINE['hits_1']) * 100
    d5 = (step3_metrics['hits_5'] - BASELINE['hits_5']) * 100
    dm = step3_metrics['mrr'] - BASELINE['mrr']

    print(f"  Step 3 Default vs Baseline:")
    print(f"    Hits@1: {BASELINE['hits_1']*100:.2f}% → {step3_metrics['hits_1']*100:.2f}%  ({'+' if d1>=0 else ''}{d1:.2f}%)")
    print(f"    Hits@5: {BASELINE['hits_5']*100:.2f}% → {step3_metrics['hits_5']*100:.2f}%  ({'+' if d5>=0 else ''}{d5:.2f}%)")
    print(f"    MRR:    {BASELINE['mrr']:.4f} → {step3_metrics['mrr']:.4f}  ({'+' if dm>=0 else ''}{dm:.4f})")

    d1_best = (best_metrics['hits_1'] - BASELINE['hits_1']) * 100
    d5_best = (best_metrics['hits_5'] - BASELINE['hits_5']) * 100
    dm_best = best_metrics['mrr'] - BASELINE['mrr']

    print(f"\n  Empirically Best Config (S={best_config[0]:.2f} / B={best_config[1]:.2f} / M={best_config[2]:.2f}) vs Baseline:")
    print(f"    Hits@1: {BASELINE['hits_1']*100:.2f}% → {best_metrics['hits_1']*100:.2f}%  ({'+' if d1_best>=0 else ''}{d1_best:.2f}%)")
    print(f"    Hits@5: {BASELINE['hits_5']*100:.2f}% → {best_metrics['hits_5']*100:.2f}%  ({'+' if d5_best>=0 else ''}{d5_best:.2f}%)")
    print(f"    MRR:    {BASELINE['mrr']:.4f} → {best_metrics['mrr']:.4f}  ({'+' if dm_best>=0 else ''}{dm_best:.4f})")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
