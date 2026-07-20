# -*- coding: utf-8 -*-
"""
Retrieval Regression Test Suite Runner.
Verifies that all historically fixed retrieval failures remain fixed.
"""
import os
import sys
import json
import time
from typing import Dict, List, Any

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.data.loader import load_search_documents
from backend.rag.search import ExactSearchEngine
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.core.settings import settings
from backend.domain.query import SearchQuery
from backend.services.pipeline import (
    ExactRetriever,
    SemanticRetriever,
    HybridRetriever,
    CoverageRanker,
    MetadataFilter,
    ResponseBuilder,
    RetrievalPipeline
)
from backend.rag.hadith_search import HadithSearchService

REGRESSION_SUITE_PATH = os.path.join(PROJECT_ROOT, "data", "regression", "regression_suite.json")
REGRESSION_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "regression", "regression_report.json")


def load_regression_suite() -> List[Dict[str, Any]]:
    if not os.path.exists(REGRESSION_SUITE_PATH):
        print(f"Error: Regression suite not found at {REGRESSION_SUITE_PATH}")
        return []
    with open(REGRESSION_SUITE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_regression_tests():
    print("=" * 80)
    print("RUNNING ISLAMIC AI RETRIEVAL REGRESSION TEST SUITE")
    print("=" * 80)

    suite = load_regression_suite()
    if not suite:
        print("No regression test cases found.")
        return

    # 1. Initialize Quran Pipeline
    print("[1/2] Initializing Quran Hybrid Retrieval Pipeline...")
    search_documents = load_search_documents()
    exact_engine = ExactSearchEngine(search_documents)
    embedding_provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
    chroma_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PATH,
        embedding_provider=embedding_provider,
        collection_name="quran"
    )
    quran_pipeline = RetrievalPipeline(
        retriever=HybridRetriever(ExactRetriever(exact_engine), SemanticRetriever(chroma_store)),
        ranker=CoverageRanker(),
        filter_step=MetadataFilter(),
        response_builder=ResponseBuilder()
    )

    # 2. Initialize Hadith Service
    print("[2/2] Initializing Hadith Search Service...")
    hadith_service = HadithSearchService()

    results = []
    passed_count = 0
    failed_count = 0
    start_time = time.time()

    print("\n" + "-" * 80)
    print(f"{'ID':<15} | {'DOMAIN':<8} | {'STATUS':<6} | {'RANK':<6} | {'QUERY'}")
    print("-" * 80)

    for item in suite:
        t_id = item["id"]
        domain = item["domain"]
        query = item["query"]
        expected = [str(e) for e in item["expected_ids"]]
        target_k = item.get("target_hit_k", 5)

        retrieved_ids = []
        if domain == "quran":
            search_q = SearchQuery(text=query, limit=10)
            res = quran_pipeline.retrieve(search_q)
            retrieved_ids = [doc.id.split("_chunk_")[0] for doc in res.documents]
        elif domain == "bukhari":
            raw_res = hadith_service.search(query, limit=10)
            retrieved_ids = [str(doc.metadata.get("hadith_number", "")) for doc in raw_res.documents]

        # Evaluate rank of best expected hit
        best_rank = None
        for rank_idx, r_id in enumerate(retrieved_ids, 1):
            if r_id in expected:
                best_rank = rank_idx
                break

        passed = (best_rank is not None) and (best_rank <= target_k)
        if passed:
            passed_count += 1
            status_str = "PASS"
        else:
            failed_count += 1
            status_str = "FAIL"

        rank_display = str(best_rank) if best_rank is not None else "MISS"
        print(f"{t_id:<15} | {domain:<8} | {status_str:<6} | {rank_display:<6} | {query[:40]}")

        results.append({
            "id": t_id,
            "domain": domain,
            "category": item.get("category", ""),
            "query": query,
            "expected_ids": expected,
            "retrieved_ids": retrieved_ids[:5],
            "best_rank": best_rank,
            "passed": passed,
            "historical_failure": item.get("historical_failure", "")
        })

    total_time = round(time.time() - start_time, 2)
    pass_rate = round((passed_count / len(suite)) * 100, 2) if suite else 0.0

    print("-" * 80)
    print(f"REGRESSION SUITE SUMMARY: {passed_count}/{len(suite)} Passed ({pass_rate}%) in {total_time}s")
    print("=" * 80)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": len(suite),
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate_percent": pass_rate,
        "total_time_seconds": total_time,
        "details": results
    }

    os.makedirs(os.path.dirname(REGRESSION_REPORT_PATH), exist_ok=True)
    with open(REGRESSION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved regression test report -> {REGRESSION_REPORT_PATH}")
    return failed_count == 0


if __name__ == "__main__":
    success = run_regression_tests()
    sys.exit(0 if success else 1)
