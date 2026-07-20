# -*- coding: utf-8 -*-
import json
import os
import pytest
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

EVALUATION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluation.json")

def evaluate_retrieval(pipeline: RetrievalPipeline, top_k: int = 5):
    """
    Evaluates retrieval recall on evaluation.json benchmark set.
    Returns hit_rate and mean_reciprocal_rank (MRR).
    """
    if not os.path.exists(EVALUATION_FILE):
        pytest.skip(f"Evaluation dataset not found at {EVALUATION_FILE}")

    with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    total = len(benchmarks)
    hits = 0
    mrr_sum = 0.0

    for item in benchmarks:
        query_text = item["question"]
        expected_ids = set(item["expected_ids"])

        search_query = SearchQuery(text=query_text, limit=top_k)
        response = pipeline.retrieve(search_query)

        retrieved_ids = [doc.id for doc in response.documents]

        hit = False
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
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
