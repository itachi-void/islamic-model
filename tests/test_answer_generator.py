# -*- coding: utf-8 -*-
"""
Test script for verifying AnswerGenerator post-retrieval pipeline.
"""
import sys
import json

sys.path.insert(0, r"d:\model")

from backend.data.loader import load_search_documents
from backend.rag.search import ExactSearchEngine
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.core.settings import settings
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
from backend.services.chat_service import ChatService


def test_answer_generator_e2e():
    print("=" * 70)
    print("TESTING ANSWER GENERATOR PIPELINE E2E")
    print("=" * 70)

    # 1. Wire Quran Pipeline
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

    # 2. Wire Hadith Service
    hadith_service = HadithSearchService()

    # 3. Wire ChatService
    chat_service = ChatService(quran_pipeline=quran_pipeline, hadith_service=hadith_service)

    test_queries = [
        ("لماذا حرم الله الربا وما الحكم الوارد فيه", "hybrid"),
        ("ما هو الفضل الوارد في حديث إنما الأعمال بالنيات", "hadith"),
        ("ما هي أسعار العملات في السوق اليوم", "hybrid"),  # Out of domain test
    ]

    for q_text, dom in test_queries:
        print(f"\n[USER QUESTION]: '{q_text}' (Target Domain: {dom})")
        resp = chat_service.chat(q_text, limit=3, domain=dom)
        
        print("\n--- GENERATED ANSWER ENVELOPE ---")
        print(resp["answer"])
        print("---------------------------------")
        print(f"Citations Count: {len(resp['citations'])}")
        for cit in resp["citations"]:
            print(f"  - [{cit['source'].upper()}] {cit['title']} ({cit['reference']})")
        print("=" * 70)


if __name__ == "__main__":
    test_answer_generator_e2e()
