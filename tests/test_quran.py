# -*- coding: utf-8 -*-
import pytest
import os
import shutil
from backend.core.settings import settings
from backend.data.loader import load_quran, get_ayah, load_search_documents
from backend.rag.search import ExactSearchEngine
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.domain.document import BaseDocument
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

# Temporary directories for testing
TEST_CHROMA_PATH = "./data/test_chroma"

@pytest.fixture(scope="module")
def temp_vector_store():
    # Setup temporary Chroma store
    os.makedirs(TEST_CHROMA_PATH, exist_ok=True)
    provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
    store = ChromaVectorStore(TEST_CHROMA_PATH, provider, collection_name="test_quran")
    
    # Ingest a few sample documents for search testing
    samples = [
        BaseDocument(
            id="1:1",
            type="verse",
            source="quran",
            text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            metadata={"surah_number": 1, "ayah_number": 1, "surah_name_ar": "الفاتحة"}
        ),
        BaseDocument(
            id="1:2",
            type="verse",
            source="quran",
            text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
            metadata={"surah_number": 1, "ayah_number": 2, "surah_name_ar": "الفاتحة"}
        ),
        BaseDocument(
            id="2:255",
            type="verse",
            source="quran",
            text="اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ ۚ لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ",
            metadata={"surah_number": 2, "ayah_number": 255, "surah_name_ar": "البقرة"}
        )
    ]
    store.add_documents(samples)
    
    yield store
    
    # Teardown: delete test database directory
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH, ignore_errors=True)

def test_ayat_al_kursi_exists():
    """Verify Surah 2 Ayah 255 exists and contains the correct text."""
    ayah = get_ayah(2, 255)
    assert ayah is not None
    assert ayah["surah_name_ar"] == "البقرة"
    assert "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ" in ayah["text"]
    assert ayah["type"] == "verse"
    assert ayah["source"] == "quran"

def test_surah_1_count():
    """Verify Surah 1 has exactly 7 verses."""
    quran = load_quran()
    surah_1_verses = [v for v in quran if v["surah_number"] == 1]
    assert len(surah_1_verses) == 7

def test_no_duplicate_verses():
    """Verify there are no duplicate verse IDs."""
    quran = load_quran()
    ids = [v["id"] for v in quran]
    assert len(ids) == len(set(ids))

def test_total_verse_count():
    """Verify the Quran dataset has exactly 6236 verses."""
    quran = load_quran()
    assert len(quran) == 6236

def test_exact_search_retriever_returns_unscored():
    """Verify retriever returns raw BaseDocument objects."""
    docs = load_search_documents()[:10]
    engine = ExactSearchEngine(docs)
    retriever = ExactRetriever(engine)
    
    results = retriever.retrieve_candidates("الحمد لله", limit=5)
    if results:
        assert isinstance(results[0], BaseDocument)
        assert results[0].id == "1:2"

def test_semantic_search_and_hybrid_retrieval(temp_vector_store):
    """Verify semantic searches return matching results, and HybridRetriever merges them."""
    docs = [
        BaseDocument(
            id="1:2",
            type="verse",
            source="quran",
            text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
            metadata={"surah_number": 1, "ayah_number": 2, "surah_name_ar": "الفاتحة"}
        )
    ]
    exact_engine = ExactSearchEngine(docs)
    
    exact_retriever = ExactRetriever(exact_engine)
    semantic_retriever = SemanticRetriever(temp_vector_store)
    
    # 1. Test Semantic Search directly
    semantic_results = semantic_retriever.retrieve_candidates("الرحيم", limit=5)
    assert len(semantic_results) > 0
    retrieved_ids = [d.id for d in semantic_results]
    assert "1:1" in retrieved_ids
    
    # 2. Test Hybrid Retrieval (should merge exact and semantic candidates)
    hybrid_retriever = HybridRetriever(exact_retriever, semantic_retriever)
    hybrid_results = hybrid_retriever.retrieve_candidates("الحمد لله", limit=5)
    
    # Both exact and semantic should find 1:2, it should be deduplicated
    matching_ids = [d.id for d in hybrid_results]
    assert "1:2" in matching_ids
    # Duplicate check
    assert len(matching_ids) == len(set(matching_ids))

def test_chat_service_rag_flow(temp_vector_store):
    """Verify ChatService completes RAG generation and returns standard response envelope."""
    exact_engine = ExactSearchEngine([])
    exact_retriever = ExactRetriever(exact_engine)
    semantic_retriever = SemanticRetriever(temp_vector_store)
    
    pipeline = RetrievalPipeline(
        retriever=HybridRetriever(exact_retriever, semantic_retriever),
        ranker=CoverageRanker(),
        filter_step=MetadataFilter(),
        response_builder=ResponseBuilder()
    )
    
    chat_service = ChatService(pipeline)
    response = chat_service.chat("ما هي آية الكرسي؟")
    
    assert "answer" in response
    assert "citations" in response
    assert "retrieved_documents" in response
    assert response["model"] == settings.MODEL_NAME
    assert response["search_type"] in ["quran", "hybrid"]
    
    # Check citation structure
    if response["citations"]:
        citation = response["citations"][0]
        assert "id" in citation
        assert "source" in citation
        if citation["source"] == "quran":
            assert "surah" in citation
            assert "ayah" in citation
