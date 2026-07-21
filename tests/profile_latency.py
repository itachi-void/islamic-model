# -*- coding: utf-8 -*-
"""
Granular Profiling Script for Retrieval Latency Breakdown.
Measures execution time (in ms) for:
1. Query Normalization & Preprocessing
2. Exact BM25 Keyword Search
3. Vector Embedding & Chroma Similarity Search
4. Candidate Fusion & Ranking (RRF / Coverage)
"""
import sys
import time
import json
from typing import Dict, Any

sys.path.insert(0, r"d:\model")

from backend.rag.search import ExactSearchEngine, normalize_arabic, extract_stemmed_tokens
from backend.rag.hadith_search import load_bukhari_documents, HadithSearchService
from backend.data.loader import load_search_documents
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.core.settings import settings

def profile_retrieval(query: str, iterations: int = 5) -> Dict[str, float]:
    print(f"\n--- Profiling Query: '{query}' ({iterations} iterations) ---")
    
    # 1. Preprocessing & Normalization
    t0 = time.perf_counter()
    for _ in range(iterations):
        norm_text = normalize_arabic(query)
        tokens = extract_stemmed_tokens(norm_text)
    t_norm = ((time.perf_counter() - t0) / iterations) * 1000

    # 2. Exact Search Engine
    docs = load_search_documents()
    exact_engine = ExactSearchEngine(docs)
    t0 = time.perf_counter()
    for _ in range(iterations):
        exact_res = exact_engine.search(query, limit=10)
    t_exact = ((time.perf_counter() - t0) / iterations) * 1000

    # 3. Vector Similarity Search
    emb_provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
    chroma_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PATH,
        embedding_provider=emb_provider,
        collection_name="quran"
    )
    t0 = time.perf_counter()
    for _ in range(iterations):
        vec_res = chroma_store.similarity_search(query, limit=10)
    t_vector = ((time.perf_counter() - t0) / iterations) * 1000

    # 4. Hadith Search Service Pipeline
    hadith_service = HadithSearchService()
    t0 = time.perf_counter()
    for _ in range(iterations):
        h_res = hadith_service.search(query, limit=10)
    t_hadith = ((time.perf_counter() - t0) / iterations) * 1000

    total = t_norm + t_exact + t_vector + t_hadith
    results = {
        "1_normalization_ms": round(t_norm, 2),
        "2_bm25_exact_ms": round(t_exact, 2),
        "3_vector_embedding_search_ms": round(t_vector, 2),
        "4_hadith_hybrid_pipeline_ms": round(t_hadith, 2),
        "total_retrieval_ms": round(total, 2)
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results

if __name__ == "__main__":
    profile_retrieval("إنما الأعمال بالنيات")
    profile_retrieval("قال الله تعالى عن الصبر")
    profile_retrieval("حكم الصلاة في السفر وتقصيرها")
