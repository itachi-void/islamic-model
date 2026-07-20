# -*- coding: utf-8 -*-
"""
Validation Suite for Router, RRF Cross Search, and Structured LLM Prompting.
"""
import sys
import json
import os

sys.path.insert(0, r"d:\model")

from backend.rag.router import QueryRouter
from backend.rag.hadith_search import HadithSearchService
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
from backend.rag.cross_search import CrossCollectionRetriever
from backend.services.chat_service import ChatService


def validate_router():
    print("\n" + "=" * 70)
    print("VALIDATION 1: QUERY ROUTER (20 TEST QUERIES)")
    print("=" * 70)

    test_cases = [
        # Quran
        ("قال الله تعالى عن الصبر", "quran"),
        ("ما هي آية الكرسي من سورة البقرة", "quran"),
        ("ما معنى قوله تعالى وتوكل على الحي الذي لا يموت", "quran"),
        ("آيات الطلاق في القرآن الكريم", "quran"),
        
        # Hadith
        ("قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات", "hadith"),
        ("حديث بني الإسلام على خمس", "hadith"),
        ("ما الحديث الذي يرويه البخاري عن أبي هريرة", "hadith"),
        ("رواه البخاري في كتاب الإيمان", "hadith"),

        # Tafsir
        ("تفسير سورة الفاتحة لابن كثير", "tafsir"),
        ("شرح آية الكرسي والبيان عنها", "tafsir"),

        # Asbab Nuzul
        ("سبب نزول سورة الكوثر", "asbab_nuzul"),
        ("فيمن نزلت هذه الآية الكريمة", "asbab_nuzul"),

        # Fiqh
        ("حكم التيمم عند عدم وجود الماء", "fiqh"),
        ("ما حكم صلاة الجماعة في المسجد", "fiqh"),

        # Aqidah
        ("أسماء الله الحسنى ومفهوم التوحيد", "aqidah"),
        ("الإيمان بالقدر خيره وشره", "aqidah"),

        # Biography
        ("سيرة النبي في غزوة بدر الكبرى", "biography"),
        ("خلافة أبي بكر الصديق رضي الله عنه", "biography"),

        # Comparison
        ("الفرق بين الحنفي والشافعي في النية", "comparison"),
        ("اختلاف المذاهب في الطهارة", "comparison"),
    ]

    router = QueryRouter()
    passed = 0

    print(f"{'#':<3} | {'Query Text':<42} | {'Expected':<12} | {'Actual Route':<12} | {'Status'}")
    print("-" * 85)

    for i, (q, expected) in enumerate(test_cases, start=1):
        actual = router.route(q)
        is_ok = actual == expected
        if is_ok:
            passed += 1
        status = "[PASS]" if is_ok else "[FAIL]"
        print(f"{i:<3} | {q:<42} | {expected:<12} | {actual:<12} | {status}")

    print("-" * 85)
    print(f"Router Test Accuracy: {passed}/20 ({passed/20*100:.1f}%)\n")


def validate_cross_search():
    print("\n" + "=" * 70)
    print("VALIDATION 2: CROSS-COLLECTION PARALLEL RETRIEVAL & RRF")
    print("=" * 70)

    # Quran pipeline
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

    # Hadith service
    hadith_service = HadithSearchService()

    cross_retriever = CrossCollectionRetriever(quran_pipeline, hadith_service)

    query = "الصبر والاحتساب عند المصيبة"
    print(f"Hybrid Query: '{query}'")
    results = cross_retriever.retrieve(query, limit=6)

    print(f"\nRetrieved {len(results)} Fused Candidates using RRF (Reciprocal Rank Fusion):")
    print(f"{'Rank':<5} | {'Source':<10} | {'ID':<15} | {'RRF Score':<10} | {'Snippet'}")
    print("-" * 80)

    for rank, doc in enumerate(results, start=1):
        snippet = doc.text[:55].replace('\n', ' ') + "..."
        print(f"#{rank:<4} | {doc.source:<10} | {doc.id:<15} | {doc.score:<10.6f} | {snippet}")


def validate_rag_prompting():
    print("\n" + "=" * 70)
    print("VALIDATION 3: STRUCTURED LLM ANSWER GENERATION & CITATIONS")
    print("=" * 70)

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

    hadith_service = HadithSearchService()
    chat_service = ChatService(quran_pipeline=quran_pipeline, hadith_service=hadith_service)

    query = "ما هي أركان الإسلام وما الدليل عليها"
    print(f"User Question: '{query}'\n")

    response = chat_service.chat(query, limit=5, domain="hybrid")

    print("[GENERATED 5-SECTION STRUCTURED RESPONSE]")
    print("-" * 70)
    print(response["answer"])
    print("-" * 70)

    print(f"\n[EXTRACTED CITATIONS ENVELOPE] ({len(response['citations'])} citations):")
    for cit in response["citations"]:
        print(f"  - [{cit['source'].upper()}] {cit['title']} | Ref: {cit['reference']}")
        print(f"    Text: {cit['text'][:70]}...")


if __name__ == "__main__":
    validate_router()
    validate_cross_search()
    validate_rag_prompting()
