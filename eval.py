# -*- coding: utf-8 -*-
"""
Unified Research Evaluation CLI for Islamic AI Engine (IRB-v1 Specification)
=============================================================================
Calculates Hits@1..5, Recall@1..10, Precision@5, MRR, nDCG@5, and OOD Accuracy.

Usage:
    python eval.py --collection irb --split dev
    python eval.py --collection irb --split test
    python eval.py --dashboard
"""
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, r"d:\model")

from backend.core.settings import settings
from backend.domain.query import SearchQuery
from backend.domain.document import BaseDocument
from backend.data.canonical_hadith import load_canonical_map
from backend.rag.search import ExactSearchEngine
from backend.rag.hadith_search import HadithSearchService, load_bukhari_documents

IRB_V1_DIR = r"d:\model\data\benchmarks\irb\v1"


def compute_ndcg_at_k(retrieved_ids: List[str], gold_set: set, k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@K)."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_dcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold_set), k) + 1))
    return (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def evaluate_irb_v1(benchmarks: List[dict], top_k: int = 5):
    service = HadithSearchService()
    canonical_map = load_canonical_map("bukhari")

    total = len(benchmarks)
    in_domain_total = 0
    ood_total = 0
    ood_correct = 0

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    precision_5_sum = 0.0
    ndcg_5_sum = 0.0
    mrr_sum = 0.0
    latencies = []

    category_stats = {}
    difficulty_stats = {}

    for item in benchmarks:
        q_text = item["query"]
        acceptable_answers = item.get("acceptable_answers", item.get("gold_evidence", []))
        cat = item.get("category", "General")
        diff = item.get("difficulty", "medium")
        answer_type = item.get("answer_type", "hadith")
        expected_h_num = item.get("expected_hadith_number")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "hits_5": 0, "mrr_sum": 0.0, "ndcg_sum": 0.0}
        category_stats[cat]["total"] += 1

        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "hits_5": 0, "mrr_sum": 0.0}
        difficulty_stats[diff]["total"] += 1

        # OOD Negative Evaluation
        if answer_type == "no_evidence" or cat == "OOD Negative":
            ood_total += 1
            t0 = time.time()
            resp = service.search(q_text, limit=top_k)
            latencies.append((time.time() - t0) * 1000)
            if not resp.documents or len(resp.documents) == 0:
                ood_correct += 1
                category_stats[cat]["hits_5"] += 1
            continue

        in_domain_total += 1

        # Build Gold ID set including multi-variant acceptable answers and canonical equivalents
        gold_set = set(acceptable_answers)
        if expected_h_num is not None:
            c_set = canonical_map.get(int(expected_h_num), {int(expected_h_num)})
            for h in c_set:
                gold_set.add(f"bukhari_{h}")
                gold_set.add(str(h))

        t0 = time.time()
        resp = service.search(q_text, limit=10)
        latencies.append((time.time() - t0) * 1000)

        retrieved_ids = [doc.id for doc in resp.documents]
        retrieved_h_nums = [str(doc.metadata.get("hadith_number")) for doc in resp.documents if doc.metadata.get("hadith_number") is not None]

        all_retrieved = retrieved_ids + retrieved_h_nums

        # Calculate Hits / Recalls / Precision / nDCG
        hit_rank = None
        for rank, item_id in enumerate(all_retrieved, start=1):
            if item_id in gold_set:
                hit_rank = rank
                break

        if hit_rank is not None:
            if hit_rank == 1:
                hits_1 += 1
                recall_1 += 1
            if hit_rank <= 3:
                hits_3 += 1
            if hit_rank <= 5:
                hits_5 += 1
                recall_5 += 1
                category_stats[cat]["hits_5"] += 1
                difficulty_stats[diff]["hits_5"] += 1
            if hit_rank <= 10:
                recall_10 += 1

            rr = 1.0 / hit_rank
            mrr_sum += rr
            category_stats[cat]["mrr_sum"] += rr
            difficulty_stats[diff]["mrr_sum"] += rr

        # Precision@5
        p5_matches = sum(1 for item_id in all_retrieved[:5] if item_id in gold_set)
        precision_5_sum += (p5_matches / 5.0)

        # nDCG@5
        ndcg5 = compute_ndcg_at_k(all_retrieved, gold_set, k=5)
        ndcg_5_sum += ndcg5
        category_stats[cat]["ndcg_sum"] += ndcg5

    return {
        "total": total,
        "in_domain_total": in_domain_total,
        "ood_total": ood_total,
        "ood_accuracy": (ood_correct / ood_total * 100) if ood_total > 0 else 100.0,
        "hits_1": (hits_1 / in_domain_total * 100) if in_domain_total > 0 else 0,
        "hits_3": (hits_3 / in_domain_total * 100) if in_domain_total > 0 else 0,
        "hits_5": (hits_5 / in_domain_total * 100) if in_domain_total > 0 else 0,
        "recall_1": (recall_1 / in_domain_total * 100) if in_domain_total > 0 else 0,
        "recall_5": (recall_5 / in_domain_total * 100) if in_domain_total > 0 else 0,
        "recall_10": (recall_10 / in_domain_total * 100) if in_domain_total > 0 else 0,
        "precision_5": (precision_5_sum / in_domain_total * 100) if in_domain_total > 0 else 0,
        "mrr": (mrr_sum / in_domain_total) if in_domain_total > 0 else 0,
        "ndcg_5": (ndcg_5_sum / in_domain_total) if in_domain_total > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "category_stats": category_stats,
        "difficulty_stats": difficulty_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Islamic Retrieval Benchmark (IRB-v1) Evaluation CLI")
    parser.add_argument("--collection", type=str, default="irb", choices=["irb", "bukhari", "quran"])
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"], help="IRB split")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K limit")
    parser.add_argument("--dashboard", action="store_true", help="View experiment history dashboard")
    args = parser.parse_args()

    if args.dashboard:
        from backend.eval.experiment_tracker import print_experiment_dashboard
        print_experiment_dashboard(args.collection)
        return

    benchmark_file = os.path.join(IRB_V1_DIR, f"{args.split}.json")
    if not os.path.exists(benchmark_file):
        benchmark_file = os.path.join("data", "bukhari", "benchmarks", "evaluation_bukhari.json")

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print("=" * 80)
    print(f"ISLAMIC RETRIEVAL BENCHMARK (IRB-v1) - [SPLIT: {args.split.upper()}]")
    print(f"Benchmark File : {benchmark_file}")
    print(f"Total Cases    : {len(benchmarks)}")
    print("=" * 80)

    metrics = evaluate_irb_v1(benchmarks, args.top_k)

    print("\n[Extended 11 IR Research Metrics]")
    print(f"Hits@1             : {metrics['hits_1']:.2f}%")
    print(f"Hits@3             : {metrics['hits_3']:.2f}%")
    print(f"Hits@5             : {metrics['hits_5']:.2f}%")
    print(f"Recall@1           : {metrics['recall_1']:.2f}%")
    print(f"Recall@5           : {metrics['recall_5']:.2f}%")
    print(f"Recall@10          : {metrics['recall_10']:.2f}%")
    print(f"Precision@5        : {metrics['precision_5']:.2f}%")
    print(f"MRR                : {metrics['mrr']:.4f}")
    print(f"nDCG@5             : {metrics['ndcg_5']:.4f}")
    print(f"OOD Guard Accuracy : {metrics['ood_accuracy']:.2f}%")
    print(f"Avg Latency        : {metrics['avg_latency_ms']} ms")

    print("\n[Breakdown by 15 Research Categories]")
    for cat, stats in metrics["category_stats"].items():
        tot = stats["total"]
        r5 = (stats["hits_5"] / tot * 100) if tot > 0 else 0
        mrr = (stats["mrr_sum"] / tot) if tot > 0 else 0
        ndcg = (stats["ndcg_sum"] / tot) if tot > 0 else 0
        print(f"  - {cat:<24} | Cases: {tot:<3} | Recall@5: {r5:>5.1f}% | MRR: {mrr:.4f} | nDCG@5: {ndcg:.4f}")

    print("\n[Breakdown by Difficulty Tier]")
    for diff, stats in metrics["difficulty_stats"].items():
        tot = stats["total"]
        r5 = (stats["hits_5"] / tot * 100) if tot > 0 else 0
        mrr = (stats["mrr_sum"] / tot) if tot > 0 else 0
        print(f"  - {diff.upper():<10} | Cases: {tot:<3} | Recall@5: {r5:>5.1f}% | MRR: {mrr:.4f}")

    print("\n" + "=" * 80)

    # Auto-log experiment
    from backend.eval.experiment_tracker import log_experiment
    rec = log_experiment(
        collection="IRB-v1",
        split_type=args.split,
        metrics={
            "hits_1": metrics["hits_1"] / 100.0,
            "hits_5": metrics["hits_5"] / 100.0,
            "mrr": metrics["mrr"],
            "avg_latency_ms": metrics["avg_latency_ms"],
            "in_domain_total": metrics["in_domain_total"]
        },
        notes=f"IRB-v1 11-Metric Suite on {args.split} split"
    )
    print(f"[EXPERIMENT LOGGED]: Run ID {rec['run_id']} logged to experiment_history.json")


if __name__ == "__main__":
    main()
