# -*- coding: utf-8 -*-
import json
import os
import time
from typing import Dict, Any, List
from backend.core.settings import settings
from backend.domain.query import SearchQuery
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.rag.hadith_search import HadithSearchService, load_bukhari_documents
from backend.rag.search import ExactSearchEngine
from backend.services.pipeline import (
    ExactRetriever,
    SemanticRetriever,
    HybridRetriever,
    CoverageRanker,
    MetadataFilter,
    ResponseBuilder,
    RetrievalPipeline
)

BUKHARI_EVAL_FILE = r"d:\model\data\evaluation_bukhari.json"
REPORT_OUTPUT_FILE = r"d:\model\data\bukhari_benchmark_report.json"


def run_benchmark():
    if not os.path.exists(BUKHARI_EVAL_FILE):
        print(f"Error: Evaluation file not found at {BUKHARI_EVAL_FILE}")
        return

    with open(BUKHARI_EVAL_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print("Initializing Bukhari Search Pipeline...")
    service = HadithSearchService()

    total_valid = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr_sum = 0.0

    category_stats: Dict[str, Dict[str, Any]] = {}
    details = []

    start_time = time.time()

    for item in benchmarks:
        q_id = item["id"]
        query_text = item["query"]
        category = item.get("category", "Uncategorized")
        expected_h_num = item.get("expected_hadith_number")

        if category not in category_stats:
            category_stats[category] = {"total": 0, "hits_5": 0, "mrr_sum": 0.0}

        category_stats[category]["total"] += 1

        if expected_h_num is None:
            # Out of domain test
            resp = service.search(query_text, limit=5)
            details.append({
                "id": q_id,
                "query": query_text,
                "category": category,
                "expected": None,
                "retrieved_count": len(resp.documents),
                "is_out_of_domain": True
            })
            continue

        total_valid += 1

        resp = service.search(query_text, limit=10)
        retrieved_h_nums = [
            doc.metadata.get("hadith_number")
            for doc in resp.documents
            if doc.metadata.get("hadith_number") is not None
        ]

        hit_rank = None
        for rank, h_num in enumerate(retrieved_h_nums, start=1):
            if int(h_num) == int(expected_h_num):
                hit_rank = rank
                break

        if hit_rank is not None:
            if hit_rank == 1:
                hits_at_1 += 1
            if hit_rank <= 3:
                hits_at_3 += 1
            if hit_rank <= 5:
                hits_at_5 += 1
                category_stats[category]["hits_5"] += 1
            if hit_rank <= 10:
                hits_at_10 += 1

            rr = 1.0 / hit_rank
            mrr_sum += rr
            category_stats[category]["mrr_sum"] += rr

        details.append({
            "id": q_id,
            "query": query_text,
            "category": category,
            "expected_hadith_number": expected_h_num,
            "hit_rank": hit_rank,
            "retrieved_hadith_numbers": retrieved_h_nums[:5]
        })

    elapsed = round(time.time() - start_time, 2)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": len(benchmarks),
        "valid_benchmark_cases": total_valid,
        "elapsed_seconds": elapsed,
        "metrics": {
            "hit_rate_at_1": round(hits_at_1 / total_valid, 4) if total_valid > 0 else 0.0,
            "hit_rate_at_3": round(hits_at_3 / total_valid, 4) if total_valid > 0 else 0.0,
            "hit_rate_at_5": round(hits_at_5 / total_valid, 4) if total_valid > 0 else 0.0,
            "hit_rate_at_10": round(hits_at_10 / total_valid, 4) if total_valid > 0 else 0.0,
            "mrr_at_10": round(mrr_sum / total_valid, 4) if total_valid > 0 else 0.0,
        },
        "category_breakdown": {
            cat: {
                "total": stats["total"],
                "hit_rate_at_5": round(stats["hits_5"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
                "mrr": round(stats["mrr_sum"] / stats["total"], 4) if stats["total"] > 0 else 0.0
            }
            for cat, stats in category_stats.items()
        },
        "details": details
    }

    os.makedirs(os.path.dirname(REPORT_OUTPUT_FILE), exist_ok=True)
    with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("SAHIH AL-BUKHARI BENCHMARK EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Test Cases   : {len(benchmarks)}")
    print(f"Valid Benchmark    : {total_valid}")
    print(f"Elapsed Time       : {elapsed}s")
    print(f"Hit Rate @ 1       : {summary['metrics']['hit_rate_at_1']:.4f}")
    print(f"Hit Rate @ 5       : {summary['metrics']['hit_rate_at_5']:.4f}")
    print(f"Hit Rate @ 10      : {summary['metrics']['hit_rate_at_10']:.4f}")
    print(f"MRR @ 10           : {summary['metrics']['mrr_at_10']:.4f}")
    print("=" * 70)
    print(f"Saved detailed report to: {REPORT_OUTPUT_FILE}\n")


if __name__ == "__main__":
    run_benchmark()
