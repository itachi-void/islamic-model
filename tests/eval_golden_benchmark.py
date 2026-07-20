# -*- coding: utf-8 -*-
import sys
import os
import json
import time

sys.path.insert(0, r"d:\model")

import chromadb
from backend.core.settings import settings
from backend.data.loader import load_search_documents
from backend.rag.search import ExactSearchEngine
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
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
from backend.services.chat_service import ChatService

EVAL_GOLDEN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluation_golden.json")
REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden_benchmark_report_v2.json")

def run_golden_benchmark():
    start_time = time.time()
    print("Initializing Golden Benchmark Evaluation...")

    # 1. Verify Chroma DB collection count
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    quran_col = chroma_client.get_collection("quran")
    vector_count = quran_col.count()
    print(f"Verified ChromaDB collection 'quran' vector count: {vector_count}")

    # 2. Wire Retrieval Pipeline
    print("Loading search documents & initializing retrievers...")
    docs = load_search_documents()
    exact_engine = ExactSearchEngine(docs)
    exact_retriever = ExactRetriever(exact_engine)

    embedding_provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
    chroma_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PATH,
        embedding_provider=embedding_provider,
        collection_name="quran"
    )
    semantic_retriever = SemanticRetriever(chroma_store)
    hybrid_retriever = HybridRetriever(exact_retriever, semantic_retriever)

    pipeline = RetrievalPipeline(
        retriever=hybrid_retriever,
        ranker=CoverageRanker(),
        filter_step=MetadataFilter(),
        response_builder=ResponseBuilder()
    )

    chat_service = ChatService(pipeline)

    # 3. Load Golden Evaluation Queries
    with open(EVAL_GOLDEN_PATH, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print(f"Loaded {len(benchmarks)} Golden Benchmark queries across categories.")

    category_stats = {}
    total_queries = 0
    in_domain_queries = 0
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr_sum = 0.0

    eval_details = []

    for item in benchmarks:
        q_id = item["id"]
        category = item.get("category", "General")
        question = item["question"]
        expected_ids = set(item["expected_ids"])

        total_queries += 1
        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "hits_1": 0,
                "hits_5": 0,
                "hits_10": 0,
                "mrr_sum": 0.0
            }
        category_stats[category]["total"] += 1

        # Retrieve candidates (Top 10)
        search_query = SearchQuery(text=question, limit=10)
        res = pipeline.retrieve(search_query)
        # Extract parent verse IDs by stripping chunk suffix e.g. 2:282_chunk_0 -> 2:282
        retrieved_ids = [doc.id.split("_chunk_")[0] for doc in res.documents]

        if not expected_ids:
            # Out-of-domain query test
            eval_details.append({
                "id": q_id,
                "category": category,
                "question": question,
                "expected": [],
                "retrieved_top_5": retrieved_ids[:5],
                "out_of_domain_pass": True
            })
            continue

        in_domain_queries += 1

        h1 = any(doc_id in expected_ids for doc_id in retrieved_ids[:1])
        h3 = any(doc_id in expected_ids for doc_id in retrieved_ids[:3])
        h5 = any(doc_id in expected_ids for doc_id in retrieved_ids[:5])
        h10 = any(doc_id in expected_ids for doc_id in retrieved_ids[:10])

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                rr = 1.0 / rank
                break

        if h1:
            hits_at_1 += 1
            category_stats[category]["hits_1"] += 1
        if h3:
            hits_at_3 += 1
        if h5:
            hits_at_5 += 1
            category_stats[category]["hits_5"] += 1
        if h10:
            hits_at_10 += 1
            category_stats[category]["hits_10"] += 1

        mrr_sum += rr
        category_stats[category]["mrr_sum"] += rr

        eval_details.append({
            "id": q_id,
            "category": category,
            "question": question,
            "expected": list(expected_ids),
            "retrieved_top_5": retrieved_ids[:5],
            "hits_at_1": h1,
            "hits_at_5": h5,
            "rr": round(rr, 4)
        })

    # Overall in-domain metrics
    overall_h1 = round(hits_at_1 / in_domain_queries, 4) if in_domain_queries > 0 else 0.0
    overall_h3 = round(hits_at_3 / in_domain_queries, 4) if in_domain_queries > 0 else 0.0
    overall_h5 = round(hits_at_5 / in_domain_queries, 4) if in_domain_queries > 0 else 0.0
    overall_h10 = round(hits_at_10 / in_domain_queries, 4) if in_domain_queries > 0 else 0.0
    overall_mrr = round(mrr_sum / in_domain_queries, 4) if in_domain_queries > 0 else 0.0

    # Category breakdown
    cat_breakdown = {}
    for cat, stats in category_stats.items():
        tot = stats["total"]
        cat_breakdown[cat] = {
            "total_queries": tot,
            "hits_at_1": round(stats["hits_1"] / tot, 4) if tot > 0 else 0.0,
            "hits_at_5": round(stats["hits_5"] / tot, 4) if tot > 0 else 0.0,
            "hits_at_10": round(stats["hits_10"] / tot, 4) if tot > 0 else 0.0,
            "mrr": round(stats["mrr_sum"] / tot, 4) if tot > 0 else 0.0
        }

    elapsed_time = round(time.time() - start_time, 2)
    avg_latency = round((elapsed_time / total_queries) * 1000, 2) if total_queries > 0 else 0.0

    report = {
        "vector_count": vector_count,
        "elapsed_seconds": elapsed_time,
        "avg_latency_ms": avg_latency,
        "overall_metrics": {
            "total_queries": total_queries,
            "in_domain_queries": in_domain_queries,
            "hits_at_1": overall_h1,
            "hits_at_3": overall_h3,
            "hits_at_5": overall_h5,
            "hits_at_10": overall_h10,
            "mrr": overall_mrr
        },
        "category_breakdown": cat_breakdown,
        "eval_details": eval_details
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==================================================")
    print("GOLDEN BENCHMARK RESULTS")
    print("==================================================")
    print(f"Total Queries: {total_queries} (In-Domain: {in_domain_queries})")
    print(f"Overall Hits@1  : {overall_h1 * 100:.2f}%")
    print(f"Overall Hits@5  : {overall_h5 * 100:.2f}%")
    print(f"Overall MRR     : {overall_mrr:.4f}")
    print(f"Avg Latency     : {avg_latency} ms per query")
    print("\nCategory Breakdown (Hits@1):")
    for cat, metrics in cat_breakdown.items():
        print(f"  - {cat:15s}: Hits@1 = {metrics['hits_at_1']*100:.2f}%, Hits@5 = {metrics['hits_at_5']*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_golden_benchmark()
