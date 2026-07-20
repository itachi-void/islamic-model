# -*- coding: utf-8 -*-
import json
import os
import pytest
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.core.settings import settings
from backend.domain.query import SearchQuery
from backend.services.pipeline import (
    SemanticRetriever,
    CoverageRanker,
    MetadataFilter,
    ResponseBuilder,
    RetrievalPipeline
)

BUKHARI_EVAL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluation_bukhari.json")


def evaluate_bukhari_retrieval(pipeline: RetrievalPipeline, top_k: int = 5):
    """
    Evaluates retrieval recall on evaluation_bukhari.json benchmark set.
    Returns hit_rate and mean_reciprocal_rank (MRR).
    """
    if not os.path.exists(BUKHARI_EVAL_FILE):
        pytest.skip(f"Bukhari evaluation dataset not found at {BUKHARI_EVAL_FILE}")

    with open(BUKHARI_EVAL_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    valid_items = [b for b in benchmarks if b.get("expected_hadith_number") is not None]
    total = len(valid_items)
    hits = 0
    mrr_sum = 0.0

    for item in valid_items:
        query_text = item["query"]
        expected_h_num = item["expected_hadith_number"]

        search_query = SearchQuery(text=query_text, limit=top_k)
        response = pipeline.retrieve(search_query)

        retrieved_h_nums = [
            doc.metadata.get("hadith_number")
            for doc in response.documents
            if doc.metadata.get("hadith_number") is not None
        ]

        hit = False
        rr = 0.0
        for rank, h_num in enumerate(retrieved_h_nums, start=1):
            if int(h_num) == int(expected_h_num):
                hit = True
                if rr == 0.0:
                    rr = 1.0 / rank

        if hit:
            hits += 1
        mrr_sum += rr

    hit_rate = hits / total if total > 0 else 0.0
    mrr = mrr_sum / total if total > 0 else 0.0

    return {
        "total": total,
        "hit_rate": hit_rate,
        "mrr": mrr
    }


def test_bukhari_eval_benchmark():
    embedding_provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
    chroma_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PATH,
        embedding_provider=embedding_provider,
        collection_name="bukhari"
    )

    if chroma_store.collection.count() == 0:
        pytest.skip("Bukhari collection is empty or not yet indexed.")

    retriever = SemanticRetriever(chroma_store)
    pipeline = RetrievalPipeline(
        retriever=retriever,
        ranker=CoverageRanker(),
        filter_step=MetadataFilter(),
        response_builder=ResponseBuilder()
    )

    results = evaluate_bukhari_retrieval(pipeline, top_k=5)
    print(f"\n[Bukhari Evaluation Benchmark]")
    print(f"Total Test Cases : {results['total']}")
    print(f"Hit Rate @ 5     : {results['hit_rate']:.4f}")
    print(f"MRR @ 5          : {results['mrr']:.4f}")

    assert results["hit_rate"] >= 0.50, f"Expected Hit Rate >= 0.50, got {results['hit_rate']:.4f}"
