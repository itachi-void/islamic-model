# -*- coding: utf-8 -*-
import json
import os
from typing import List, Optional, Dict, Any
from backend.core.settings import settings
from backend.domain.document import BaseDocument
from backend.domain.query import SearchQuery
from backend.domain.search_result import SearchResponse
from backend.rag.search import ExactSearchEngine
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.rag.isnad_index import IsnadIndex, build_isnad_index_from_file
from backend.services.pipeline import (
    ExactRetriever,
    SemanticRetriever,
    HybridRetriever,
    CoverageRanker,
    MetadataFilter,
    ResponseBuilder,
    RetrievalPipeline
)

BUKHARI_PROCESSED_PATH = r"d:\model\data\bukhari\bukhari_processed.json"


def load_bukhari_documents() -> List[BaseDocument]:
    """Loads processed Bukhari Hadith records as BaseDocument objects."""
    if not os.path.exists(BUKHARI_PROCESSED_PATH):
        return []

    with open(BUKHARI_PROCESSED_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    docs = []
    for r in records:
        meta = {
            "book": r.get("book", ""),
            "book_number": r.get("book_number", 0),
            "chapter": r.get("chapter", ""),
            "hadith_number": r.get("hadith_number", 0),
            "narrator": r.get("narrator", ""),
            "grade": r.get("grade", "صحيح"),
            "sanad": r.get("sanad", ""),
            "topics": r.get("topics", []),
            "keywords": r.get("keywords", []),
            "aliases": r.get("aliases", []),
            "original_matn": r.get("matn", "")
        }
        # Enriched searchable text containing matn + book + chapter + narrator + aliases
        aliases_str = " ".join(r.get("aliases", []))
        search_text = f"{r.get('matn', '')} {r.get('book', '')} {r.get('chapter', '')} {r.get('narrator', '')} {aliases_str}".strip()

        docs.append(BaseDocument(
            id=r["id"],
            type="hadith",
            source="bukhari",
            text=search_text,
            metadata=meta
        ))
    return docs


class HadithSearchService:
    def __init__(self):
        self.documents = load_bukhari_documents()
        self.exact_engine = ExactSearchEngine(self.documents)
        self.embedding_provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
        self.vector_store = ChromaVectorStore(
            persist_directory=settings.CHROMA_PATH,
            embedding_provider=self.embedding_provider,
            collection_name="bukhari"
        )

        # Build isnad index for direct chain-of-narration matching
        self.isnad_index = IsnadIndex()
        if os.path.exists(BUKHARI_PROCESSED_PATH):
            with open(BUKHARI_PROCESSED_PATH, "r", encoding="utf-8") as f:
                bukhari_data = json.load(f)
            self.isnad_index.build(bukhari_data)

        exact_retriever = ExactRetriever(self.exact_engine)
        semantic_retriever = SemanticRetriever(self.vector_store)
        hybrid_retriever = HybridRetriever(exact_retriever, semantic_retriever)

        self.pipeline = RetrievalPipeline(
            retriever=hybrid_retriever,
            ranker=CoverageRanker(
                semantic_weight=0.40,
                bm25_weight=0.35,
                metadata_weight=0.25
            ),
            filter_step=MetadataFilter(),
            response_builder=ResponseBuilder()
        )

    def search(
        self,
        query_text: str,
        limit: int = 5,
        book: Optional[str] = None,
        narrator: Optional[str] = None
    ) -> SearchResponse:
        from backend.rag.query_normalizer import normalize_query_dialect
        from backend.data.alias_dictionary import expand_query_aliases

        cleaned_query = normalize_query_dialect(query_text)
        expanded_query = expand_query_aliases(cleaned_query)

        # Stage 0: Direct Isnad Index Lookup (fastest path for chain queries)
        isnad_results = self.isnad_index.search(query_text, min_matches=1)
        if isnad_results:
            isnad_doc_map = {doc.id: doc for doc in self.documents}
            isnad_docs = []
            for h_id, score in isnad_results[:limit * 4]:
                if h_id in isnad_doc_map:
                    doc = isnad_doc_map[h_id]
                    isnad_docs.append(BaseDocument(
                        id=doc.id,
                        type=doc.type,
                        source=doc.source,
                        text=doc.text,
                        metadata=doc.metadata,
                        score=score
                    ))

            # If isnad found high-confidence match (score >= 0.5), boost it into pipeline candidates
            if isnad_docs and isnad_docs[0].score >= 0.5:
                # Run normal pipeline too, then merge with isnad results via RRF
                filters: Dict[str, Any] = {}
                if book:
                    filters["book"] = book
                if narrator:
                    filters["narrator"] = narrator

                query = SearchQuery(
                    text=expanded_query,
                    limit=limit,
                    filters=filters if filters else None
                )

                if self.vector_store.collection and self.vector_store.collection.count() > 0:
                    pipeline_response = self.pipeline.retrieve(query)
                    pipeline_docs = pipeline_response.documents

                    # Merge isnad + pipeline results using RRF
                    merged = {}
                    k = 60
                    for rank, doc in enumerate(isnad_docs, 1):
                        merged[doc.id] = doc
                        merged[doc.id].score = merged[doc.id].score or 0.0
                        merged[doc.id].metadata["_rrf"] = merged[doc.id].metadata.get("_rrf", 0.0) + 1.0 / (k + rank)

                    for rank, doc in enumerate(pipeline_docs, 1):
                        if doc.id in merged:
                            merged[doc.id].metadata["_rrf"] = merged[doc.id].metadata.get("_rrf", 0.0) + 1.0 / (k + rank)
                        else:
                            doc.score = doc.score or 0.0
                            doc.metadata["_rrf"] = 1.0 / (k + rank)
                            merged[doc.id] = doc

                    # Sort by RRF score
                    final = sorted(merged.values(), key=lambda d: d.metadata.get("_rrf", 0), reverse=True)[:limit]
                    for d in final:
                        d.score = d.metadata.pop("_rrf", 0.0)
                    return SearchResponse(query=query_text, count=len(final), documents=final)

        filters: Dict[str, Any] = {}
        if book:
            filters["book"] = book
        if narrator:
            filters["narrator"] = narrator

        query = SearchQuery(
            text=expanded_query,
            limit=limit,
            filters=filters if filters else None
        )

        # Fallback to exact BM25/Exact match if Chroma vector store is uninitialized or empty
        if not self.vector_store.collection or self.vector_store.collection.count() == 0:
            # Try isnad results first
            if isnad_results:
                isnad_doc_map = {doc.id: doc for doc in self.documents}
                isnad_docs = []
                for h_id, score in isnad_results[:limit]:
                    if h_id in isnad_doc_map:
                        doc = isnad_doc_map[h_id]
                        isnad_docs.append(BaseDocument(
                            id=doc.id,
                            type=doc.type,
                            source=doc.source,
                            text=doc.text,
                            metadata=doc.metadata,
                            score=score
                        ))
                return SearchResponse(query=query_text, count=len(isnad_docs), documents=isnad_docs)
            results = self.exact_engine.search(query_text, limit=limit)
            return SearchResponse(query=query_text, count=len(results), documents=results)

        return self.pipeline.retrieve(query)
