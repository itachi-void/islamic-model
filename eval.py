# -*- coding: utf-8 -*-
"""
Unified Evaluation CLI for Islamic AI Engine
============================================
Usage:
    python eval.py --collection quran
    python eval.py --collection bukhari
    python eval.py --collection muslim
    python eval.py --collection tafsir
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, r"d:\model")

from backend.core.settings import settings
from backend.domain.query import SearchQuery
from backend.domain.document import BaseDocument
from backend.data.loader import load_search_documents
from backend.data.muslim_loader import load_muslim_documents
from backend.data.tafsir_loader import TafsirLoader
from backend.data.canonical_hadith import load_canonical_map
from backend.rag.search import ExactSearchEngine
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.rag.hadith_search import HadithSearchService
from backend.services.pipeline import (
    ExactRetriever,
    SemanticRetriever,
    HybridRetriever,
    CoverageRanker,
    MetadataFilter,
    ResponseBuilder,
    RetrievalPipeline
)

TARGETS = {
    "quran": {"hits_1": 0.90, "hits_5": 0.98},
    "bukhari": {"hits_1": 0.88, "hits_5": 0.97},
    "muslim": {"hits_1": 0.88, "hits_5": 0.97},
    "tafsir": {"hits_1": 0.85, "hits_5": 0.95},
}


def get_benchmark_file(collection: str) -> str:
    candidates = [
        os.path.join("data", f"evaluation_{collection}.json"),
        os.path.join("data", "evaluation_golden.json") if collection == "quran" else "",
        os.path.join("data", "evaluation.json") if collection == "quran" else "",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"No benchmark file found for collection '{collection}'.")


def evaluate_quran_collection(benchmarks: List[dict], top_k: int):
    search_documents = load_search_documents()
    exact_engine = ExactSearchEngine(search_documents)
    embedding_provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
    chroma_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PATH,
        embedding_provider=embedding_provider,
        collection_name="quran"
    )
    exact_retriever = ExactRetriever(exact_engine)
    semantic_retriever = SemanticRetriever(chroma_store)
    pipeline = RetrievalPipeline(
        retriever=HybridRetriever(exact_retriever, semantic_retriever),
        ranker=CoverageRanker(semantic_weight=0.35, bm25_weight=0.45, metadata_weight=0.20),
        filter_step=MetadataFilter(),
        response_builder=ResponseBuilder()
    )

    total = len(benchmarks)
    in_domain_total = 0
    hits_1 = 0
    hits_5 = 0
    mrr_sum = 0.0
    latencies = []
    category_stats = {}

    for item in benchmarks:
        q_text = item.get("question") or item.get("query")
        expected_ids = set(item.get("expected_ids", []))
        cat = item.get("category", "General")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "hits_5": 0, "mrr_sum": 0.0}
        category_stats[cat]["total"] += 1

        if not expected_ids:
            continue

        in_domain_total += 1

        t0 = time.time()
        search_query = SearchQuery(text=q_text, limit=top_k)
        resp = pipeline.retrieve(search_query)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [doc.id.split("_chunk_")[0] for doc in resp.documents]

        hit_rank = None
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                hit_rank = rank
                break

        if hit_rank is not None:
            if hit_rank == 1:
                hits_1 += 1
            if hit_rank <= 5:
                hits_5 += 1
                category_stats[cat]["hits_5"] += 1

            rr = 1.0 / hit_rank
            mrr_sum += rr
            category_stats[cat]["mrr_sum"] += rr

    return {
        "total": total,
        "in_domain_total": in_domain_total,
        "hits_1": hits_1 / in_domain_total if in_domain_total > 0 else 0,
        "hits_5": hits_5 / in_domain_total if in_domain_total > 0 else 0,
        "mrr": mrr_sum / in_domain_total if in_domain_total > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "category_stats": category_stats
    }


def evaluate_bukhari_collection(benchmarks: List[dict], top_k: int, debug: bool = False):
    service = HadithSearchService()
    service.pipeline.ranker.debug = debug
    canonical_map = load_canonical_map("bukhari")
    total = len(benchmarks)
    in_domain_total = 0
    hits_1 = 0
    hits_5 = 0
    mrr_sum = 0.0
    latencies = []
    category_stats = {}

    for item in benchmarks:
        q_text = item["query"]
        expected_h_num = item.get("expected_hadith_number")
        cat = item.get("category", "General")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "hits_5": 0, "mrr_sum": 0.0}
        category_stats[cat]["total"] += 1

        if expected_h_num is None:
            continue

        in_domain_total += 1

        t0 = time.time()
        if debug:
            print(f"\n{'='*60}\n[DEBUG QUERY]: {q_text}\n[EXPECTED HADITH #]: {expected_h_num}")
        resp = service.search(q_text, limit=top_k)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_h_nums = [
            doc.metadata.get("hadith_number")
            for doc in resp.documents
            if doc.metadata.get("hadith_number") is not None
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
                category_stats[cat]["hits_5"] += 1

            rr = 1.0 / hit_rank
            mrr_sum += rr
            category_stats[cat]["mrr_sum"] += rr

    return {
        "total": total,
        "in_domain_total": in_domain_total,
        "hits_1": hits_1 / in_domain_total if in_domain_total > 0 else 0,
        "hits_5": hits_5 / in_domain_total if in_domain_total > 0 else 0,
        "mrr": mrr_sum / in_domain_total if in_domain_total > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "category_stats": category_stats
    }


def evaluate_muslim_collection(benchmarks: List[dict], top_k: int):
    docs = load_muslim_documents()
    exact_engine = ExactSearchEngine(docs)
    pipeline = RetrievalPipeline(
        retriever=ExactRetriever(exact_engine),
        ranker=CoverageRanker(semantic_weight=0.0, bm25_weight=0.50, metadata_weight=0.50),
        filter_step=MetadataFilter(),
        response_builder=ResponseBuilder()
    )

    total = len(benchmarks)
    in_domain_total = 0
    hits_1 = 0
    hits_5 = 0
    mrr_sum = 0.0
    latencies = []
    category_stats = {}

    for item in benchmarks:
        q_text = item["query"]
        expected_h_num = item.get("expected_hadith_number")
        cat = item.get("category", "General")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "hits_5": 0, "mrr_sum": 0.0}
        category_stats[cat]["total"] += 1

        if expected_h_num is None:
            continue

        in_domain_total += 1

        t0 = time.time()
        search_query = SearchQuery(text=q_text, limit=top_k)
        resp = pipeline.retrieve(search_query)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

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
                hits_1 += 1
            if hit_rank <= 5:
                hits_5 += 1
                category_stats[cat]["hits_5"] += 1

            rr = 1.0 / hit_rank
            mrr_sum += rr
            category_stats[cat]["mrr_sum"] += rr

    return {
        "total": total,
        "in_domain_total": in_domain_total,
        "hits_1": hits_1 / in_domain_total if in_domain_total > 0 else 0,
        "hits_5": hits_5 / in_domain_total if in_domain_total > 0 else 0,
        "mrr": mrr_sum / in_domain_total if in_domain_total > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "category_stats": category_stats
    }


def evaluate_tafsir_collection(benchmarks: List[dict], top_k: int):
    loader = TafsirLoader()
    records = loader.load_tafsir("ibnkathir")
    docs = [
        BaseDocument(
            id=r.id,
            type="tafsir",
            source=getattr(r, "source", "tafsir"),
            text=r.text,
            metadata={
                "surah_number": r.surah_number,
                "ayah_number": r.ayah_number,
                "mufassir": r.mufassir,
            }
        )
        for r in records
    ]
    exact_engine = ExactSearchEngine(docs)
    pipeline = RetrievalPipeline(
        retriever=ExactRetriever(exact_engine),
        ranker=CoverageRanker(semantic_weight=0.0, bm25_weight=0.50, metadata_weight=0.50),
        filter_step=MetadataFilter(),
        response_builder=ResponseBuilder()
    )

    total = len(benchmarks)
    in_domain_total = 0
    hits_1 = 0
    hits_5 = 0
    mrr_sum = 0.0
    latencies = []
    category_stats = {}

    for item in benchmarks:
        q_text = item.get("question") or item.get("query")
        expected_ids = set(item.get("expected_ids", []))
        cat = item.get("category", "General")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "hits_5": 0, "mrr_sum": 0.0}
        category_stats[cat]["total"] += 1

        if not expected_ids:
            continue

        in_domain_total += 1

        t0 = time.time()
        search_query = SearchQuery(text=q_text, limit=top_k)
        resp = pipeline.retrieve(search_query)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [doc.id for doc in resp.documents]

        hit_rank = None
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                hit_rank = rank
                break

        if hit_rank is not None:
            if hit_rank == 1:
                hits_1 += 1
            if hit_rank <= 5:
                hits_5 += 1
                category_stats[cat]["hits_5"] += 1

            rr = 1.0 / hit_rank
            mrr_sum += rr
            category_stats[cat]["mrr_sum"] += rr

    return {
        "total": total,
        "in_domain_total": in_domain_total,
        "hits_1": hits_1 / in_domain_total if in_domain_total > 0 else 0,
        "hits_5": hits_5 / in_domain_total if in_domain_total > 0 else 0,
        "mrr": mrr_sum / in_domain_total if in_domain_total > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "category_stats": category_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation CLI")
    parser.add_argument("--collection", type=str, required=True, choices=["quran", "bukhari", "muslim", "tafsir"])
    parser.add_argument("--benchmark_file", type=str, default=None, help="Custom benchmark file path (e.g. data/evaluation_bukhari_3000.json)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K candidates limit")
    parser.add_argument("--debug", action="store_true", help="Print per-document score breakdown (Semantic / BM25 / Metadata / Final)")
    args = parser.parse_args()

    collection = args.collection.lower()
    file_path = args.benchmark_file if args.benchmark_file and os.path.exists(args.benchmark_file) else get_benchmark_file(collection)

    with open(file_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print("=" * 70)
    print(f"UNIFIED EVALUATION BENCHMARK: [{collection.upper()}]")
    print(f"Benchmark File : {file_path}")
    print(f"Total Cases    : {len(benchmarks)}")
    print("=" * 70)

    if collection == "quran":
        metrics = evaluate_quran_collection(benchmarks, args.top_k)
    elif collection == "bukhari":
        metrics = evaluate_bukhari_collection(benchmarks, args.top_k, debug=args.debug)
    elif collection == "muslim":
        metrics = evaluate_muslim_collection(benchmarks, args.top_k)
    elif collection == "tafsir":
        metrics = evaluate_tafsir_collection(benchmarks, args.top_k)

    target = TARGETS.get(collection, {"hits_1": 0.85, "hits_5": 0.95})

    print(f"\n[Overall Results]")
    print(f"Hits@1       : {metrics['hits_1']*100:.2f}%  (Target: >= {target['hits_1']*100:.1f}%)")
    print(f"Hits@5       : {metrics['hits_5']*100:.2f}%  (Target: >= {target['hits_5']*100:.1f}%)")
    print(f"MRR          : {metrics['mrr']:.4f}")
    print(f"Avg Latency  : {metrics['avg_latency_ms']} ms")

    print("\n[Breakdown by Question Type]")
    for cat, stats in metrics["category_stats"].items():
        h5 = (stats["hits_5"] / stats["total"] * 100) if stats["total"] > 0 else 0
        mrr = (stats["mrr_sum"] / stats["total"]) if stats["total"] > 0 else 0
        print(f"  - {cat:<24} | Total: {stats['total']:<3} | Hits@5: {h5:.1f}% | MRR: {mrr:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
