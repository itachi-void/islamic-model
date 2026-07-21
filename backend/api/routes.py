# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Body
from typing import Optional, Dict, Any
from backend.core.settings import settings
from backend.llm.ollama_client import generate
from backend.data.loader import get_ayah, load_search_documents
from backend.rag.search import ExactSearchEngine
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.rag.hadith_search import HadithSearchService
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
from backend.data.ingest import run_ingestion

router = APIRouter()

# Global Lazy Singletons
_pipeline = None
_hadith_service = None
_chat_service = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
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
        hybrid_retriever = HybridRetriever(
            exact_retriever=exact_retriever,
            semantic_retriever=semantic_retriever
        )
        _pipeline = RetrievalPipeline(
            retriever=hybrid_retriever,
            ranker=CoverageRanker(),
            filter_step=MetadataFilter(),
            response_builder=ResponseBuilder()
        )
    return _pipeline

def get_hadith_service():
    global _hadith_service
    if _hadith_service is None:
        _hadith_service = HadithSearchService()
    return _hadith_service

def get_chat_service():
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(quran_pipeline=get_pipeline(), hadith_service=get_hadith_service())
    return _chat_service


@router.get("/test")
def test():
    answer = generate("قل السلام عليكم فقط")
    return {"response": answer}


@router.get("/quran/{surah_number}/{ayah_number}")
def quran_endpoint(surah_number: int, ayah_number: int):
    ayah_data = get_ayah(surah_number, ayah_number)
    if not ayah_data:
        raise HTTPException(status_code=404, detail="Ayah not found")
    return ayah_data


@router.get("/search")
def search_endpoint(
    q: str = Query(..., min_length=1),
    limit: int = Query(settings.TOP_K, ge=1, le=settings.MAX_RESULTS),
    surah: Optional[int] = Query(None, description="Filter by Surah number"),
    revelation_type: Optional[str] = Query(None, description="Filter by revelation type (Meccan/Madinan)")
):
    filters: Dict[str, Any] = {}
    if surah is not None:
        filters["surah_number"] = surah
    if revelation_type is not None:
        filters["revelation_type"] = revelation_type

    query = SearchQuery(text=q, limit=limit, filters=filters if filters else None)
    response = get_pipeline().retrieve(query)
    return response


@router.get("/hadith/search")
def hadith_search_get(
    q: str = Query(..., min_length=1),
    limit: int = Query(settings.TOP_K, ge=1, le=settings.MAX_RESULTS),
    book: Optional[str] = Query(None, description="Filter by Book name"),
    narrator: Optional[str] = Query(None, description="Filter by Narrator name")
):
    """
    Search Sahih Al-Bukhari hadith collection (GET).
    """
    return get_hadith_service().search(query_text=q, limit=limit, book=book, narrator=narrator)


@router.post("/hadith/search")
def hadith_search_post(payload: Dict[str, Any] = Body(...)):
    """
    Search Sahih Al-Bukhari hadith collection (POST).
    JSON body: {"q": "...", "limit": 5, "book": "...", "narrator": "..."}
    """
    q = payload.get("q")
    if not q:
        raise HTTPException(status_code=400, detail="Field 'q' is required in request body.")
    limit = payload.get("limit", settings.TOP_K)
    book = payload.get("book")
    narrator = payload.get("narrator")
    return get_hadith_service().search(query_text=q, limit=limit, book=book, narrator=narrator)


from fastapi.responses import StreamingResponse


@router.get("/chat")
def chat_endpoint(
    q: str = Query(..., min_length=1),
    limit: int = Query(settings.TOP_K, ge=1, le=settings.MAX_RESULTS),
    domain: Optional[str] = Query(None, description="Domain route: quran, hadith, or hybrid"),
    surah: Optional[int] = Query(None, description="Filter context by Surah number"),
    revelation_type: Optional[str] = Query(None, description="Filter context by revelation type")
):
    try:
        filters: Dict[str, Any] = {}
        if surah is not None:
            filters["surah_number"] = surah
        if revelation_type is not None:
            filters["revelation_type"] = revelation_type

        response = get_chat_service().chat(
            query_text=q,
            limit=limit,
            filters=filters if filters else None,
            domain=domain
        )
        return response
    except Exception as e:
        import logging
        logging.exception(f"Chat endpoint exception: {e}")
        return {
            "answer": f"تم البحث في المصادر الإسلامية المتاحة للسؤال: '{q}'",
            "route": domain or "hybrid",
            "citations": [],
            "retrieved_documents": [],
            "model": settings.MODEL_NAME,
            "search_type": domain or "hybrid"
        }


@router.get("/chat/stream")
def chat_stream_endpoint(
    q: str = Query(..., min_length=1),
    limit: int = Query(settings.TOP_K, ge=1, le=settings.MAX_RESULTS),
    domain: Optional[str] = Query(None, description="Domain route: quran, hadith, or hybrid"),
    surah: Optional[int] = Query(None, description="Filter context by Surah number"),
    revelation_type: Optional[str] = Query(None, description="Filter context by revelation type")
):
    """
    Stream real-time Server-Sent Events (SSE) chat response.
    """
    filters: Dict[str, Any] = {}
    if surah is not None:
        filters["surah_number"] = surah
    if revelation_type is not None:
        filters["revelation_type"] = revelation_type

    generator = get_chat_service().chat_stream(
        query_text=q,
        limit=limit,
        filters=filters if filters else None,
        domain=domain
    )
    return StreamingResponse(generator, media_type="text/event-stream")



@router.post("/reindex")
def reindex_endpoint(background_tasks: BackgroundTasks):
    """
    Triggers re-indexing of the Quran dataset in the background.
    """
    background_tasks.add_task(run_ingestion)
    return {
        "status": "success",
        "message": "Quran dataset reindexing triggered in the background."
    }

