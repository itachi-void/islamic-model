# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Any, Optional
from backend.domain.document import BaseDocument
from backend.domain.query import SearchQuery
from backend.domain.search_result import SearchResponse
from backend.rag.search import BaseSearchEngine, normalize_arabic, extract_stemmed_tokens, strip_dialectal_phrases
from backend.rag.vector_store import BaseVectorStore


class BaseRetriever:
    def retrieve_candidates(self, query: str, limit: int = 5) -> List[BaseDocument]:
        raise NotImplementedError("Subclasses must implement retrieve_candidates")


class ExactRetriever(BaseRetriever):
    def __init__(self, search_engine: BaseSearchEngine):
        self.search_engine = search_engine

    def retrieve_candidates(self, query: str, limit: int = 20) -> List[BaseDocument]:
        docs = self.search_engine.search(query, limit=limit)
        # Attach BM25 score from ExactSearchEngine into metadata for CoverageRanker
        for doc in docs:
            if doc.score is not None:
                doc.metadata["_bm25_score"] = doc.score
        return docs


class SemanticRetriever(BaseRetriever):
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    def retrieve_candidates(self, query: str, limit: int = 50) -> List[BaseDocument]:
        return self.vector_store.similarity_search(query, limit=limit)


class HybridRetriever(BaseRetriever):
    def __init__(self, exact_retriever: BaseRetriever, semantic_retriever: BaseRetriever):
        self.exact_retriever = exact_retriever
        self.semantic_retriever = semantic_retriever

    def retrieve_candidates(self, query: str, limit: int = 5) -> List[BaseDocument]:
        exact_candidates = self.exact_retriever.retrieve_candidates(query, limit=20)
        semantic_candidates = self.semantic_retriever.retrieve_candidates(query, limit=50)

        for rank, doc in enumerate(exact_candidates, 1):
            doc.metadata["_exact_rank"] = rank

        for rank, doc in enumerate(semantic_candidates, 1):
            doc.metadata["_semantic_rank"] = rank

        candidate_map: Dict[str, BaseDocument] = {}

        # Merge: exact first (they carry _bm25_score), then semantic
        for doc in exact_candidates:
            if doc.id not in candidate_map:
                candidate_map[doc.id] = doc

        for doc in semantic_candidates:
            if doc.id not in candidate_map:
                candidate_map[doc.id] = doc
            else:
                # Merge semantic rank into existing doc from exact retrieval
                existing = candidate_map[doc.id]
                if "_semantic_rank" in doc.metadata and "_semantic_rank" not in existing.metadata:
                    existing.metadata["_semantic_rank"] = doc.metadata["_semantic_rank"]

        return list(candidate_map.values())


class BaseRanker:
    def rank_documents(self, documents: List[BaseDocument], query: str) -> List[BaseDocument]:
        raise NotImplementedError("Subclasses must implement rank_documents")


def _extract_narrator_hint(query: str) -> Optional[str]:
    """
    Generic narrator extraction from query.
    Matches patterns like: عن عمر بن الخطاب / رواه أبو هريرة / حديث عائشة
    """
    norm_q = normalize_arabic(query)
    patterns = [
        r'عن\s+([\w\s]{3,25}?)(?:\s+قال|\s+روى|\s+حدث|\s+أن|\s+رضي|$)',
        r'رواه?\s+([\w\s]{3,20}?)(?:\s+في|$)',
        r'حديث\s+(?:أبي?\s+|ابن\s+|أم\s+)?([\w\s]{3,20}?)(?:\s+في|$)',
    ]
    for p in patterns:
        m = re.search(p, norm_q)
        if m:
            return normalize_arabic(m.group(1).strip())
    return None


class CoverageRanker(BaseRanker):
    """
    Hybrid Ranker: 0.55 * Semantic + 0.30 * BM25 + 0.15 * Metadata.
    Supports per-document debug output for diagnostics.
    """
    def __init__(
        self,
        k: int = 60,
        semantic_weight: float = 0.55,
        bm25_weight: float = 0.30,
        metadata_weight: float = 0.15,
        debug: bool = False
    ):
        self.k = k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.metadata_weight = metadata_weight
        self.debug = debug

    def rank_documents(self, documents: List[BaseDocument], query: str, debug: Optional[bool] = None) -> List[BaseDocument]:
        emit_debug = debug if debug is not None else self.debug
        ranked = []
        clean_q = strip_dialectal_phrases(query)
        norm_query = normalize_arabic(clean_q)
        query_tokens = extract_stemmed_tokens(clean_q)

        # Extract narrator hint from query for metadata boosting
        narrator_hint = _extract_narrator_hint(query)
        narrator_hint_tokens = extract_stemmed_tokens(narrator_hint) if narrator_hint else set()

        for doc in documents:
            exact_rank = doc.metadata.get("_exact_rank")
            semantic_rank = doc.metadata.get("_semantic_rank")
            bm25_score = doc.metadata.get("_bm25_score", 0.0)  # Raw IDF-weighted BM25 score

            # --- Component 1: Semantic Score (normalized RRF → [0, 1]) ---
            semantic_rrf = (1.0 / (self.k + semantic_rank)) if semantic_rank else 0.0
            semantic_norm = semantic_rrf * (self.k + 1)  # max possible = 1.0 when rank=1

            # --- Component 2: BM25 Score (already [0, 1] as IDF ratio) ---
            bm25_norm = min(1.0, float(bm25_score))

            # --- Component 3: Metadata Score (narrator + book + topic + text overlap) ---
            metadata_score = 0.0
            metadata_reasons = []

            # 3a. Narrator exact match (highest weight: 0.50)
            doc_narrator = normalize_arabic(str(doc.metadata.get("narrator", "")))
            doc_narrator_tokens = extract_stemmed_tokens(doc_narrator)
            if narrator_hint_tokens and doc_narrator_tokens and (set(narrator_hint_tokens) & set(doc_narrator_tokens)):
                metadata_score += 0.50
                metadata_reasons.append("Matched narrator")
            elif query_tokens and doc_narrator_tokens and (set(query_tokens) & set(doc_narrator_tokens)):
                metadata_score += 0.30
                metadata_reasons.append("Partial narrator match")

            # 3b. Book name match in query (weight: 0.25)
            doc_book = normalize_arabic(str(doc.metadata.get("book", "")))
            doc_book_tokens = extract_stemmed_tokens(doc_book)
            if doc_book_tokens and query_tokens and (set(query_tokens) & set(doc_book_tokens)):
                metadata_score += 0.25
                metadata_reasons.append("Matched book")

            # 3c. Title / aliases match
            title = str(doc.metadata.get("title_ar", ""))
            aliases = doc.metadata.get("aliases", [])
            title_match = title and norm_query in normalize_arabic(title)
            alias_match = any(norm_query in normalize_arabic(str(a)) for a in aliases)
            if title_match or alias_match:
                metadata_score += 0.20
                metadata_reasons.append("Matched title/alias")

            # 3d. Topics match
            topics = doc.metadata.get("topics", [])
            if any(norm_query in normalize_arabic(str(t)) for t in topics):
                metadata_score += 0.15
                metadata_reasons.append("Matched topic")

            # 3e. Text overlap (fallback)
            if metadata_score == 0.0 and query_tokens:
                doc_text_norm = normalize_arabic(doc.text)
                doc_tokens = extract_stemmed_tokens(doc.text)
                if norm_query in doc_text_norm:
                    metadata_score += 0.15
                    metadata_reasons.append("Exact phrase in text")
                elif set(query_tokens) & set(doc_tokens):
                    overlap = len(set(query_tokens) & set(doc_tokens)) / float(len(query_tokens))
                    metadata_score += round(0.10 * overlap, 4)
                    metadata_reasons.append(f"Text overlap ({overlap:.0%})")

            metadata_norm = min(1.0, metadata_score)

            # --- Exact Rank Bonus (independent component) ---
            exact_bonus = 0.0
            if exact_rank:
                if exact_rank == 1:
                    exact_bonus = 0.25
                    metadata_reasons.append("Exact rank #1 bonus")
                elif exact_rank <= 3:
                    exact_bonus = 0.15
                    metadata_reasons.append("Exact rank top-3 bonus")

            # --- Final Hybrid Score ---
            final_score = round(
                self.semantic_weight * semantic_norm +
                self.bm25_weight * bm25_norm +
                self.metadata_weight * metadata_norm +
                exact_bonus,
                4
            )

            # Clean internal rank metadata
            doc.metadata.pop("_exact_rank", None)
            doc.metadata.pop("_semantic_rank", None)
            doc.metadata.pop("_bm25_score", None)

            scored_doc = BaseDocument(
                id=doc.id,
                type=doc.type,
                source=doc.source,
                text=doc.text,
                metadata=doc.metadata,
                score=final_score
            )
            ranked.append(scored_doc)

            if emit_debug:
                h_num = doc.metadata.get("hadith_number", "?")
                book = doc.metadata.get("book", "?")[:20]
                print(
                    f"  [DEBUG] #{h_num:<6} | Sem={semantic_norm:.3f} | BM25={bm25_norm:.3f} | Meta={metadata_norm:.3f} | Bonus={exact_bonus:.2f} | Final={final_score:.4f}"
                    f"  | Reasons: {', '.join(metadata_reasons) or 'none'} | Book: {book}"
                )

        ranked.sort(key=lambda x: x.score or 0.0, reverse=True)
        return ranked


class BaseFilter:
    def filter_documents(self, documents: List[BaseDocument], filters: Optional[Dict[str, Any]]) -> List[BaseDocument]:
        raise NotImplementedError("Subclasses must implement filter_documents")


class MetadataFilter(BaseFilter):
    def filter_documents(self, documents: List[BaseDocument], filters: Optional[Dict[str, Any]]) -> List[BaseDocument]:
        if not filters:
            return documents

        filtered = []
        for doc in documents:
            match = True
            for key, val in filters.items():
                doc_val = doc.metadata.get(key)
                if doc_val is None:
                    doc_val = getattr(doc, key, None)

                if doc_val != val:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered


class ResponseBuilder:
    def build_response(self, query: str, documents: List[BaseDocument]) -> SearchResponse:
        return SearchResponse(
            query=query,
            count=len(documents),
            documents=documents
        )


class MetadataLookupStep:
    """
    Explicit Metadata Lookup Pipeline for Surah/Ayah references (e.g. سورة البقرة آية 94).
    """
    def parse_query(self, query_text: str):
        m = re.search(r'سور[ةه]\s+(.+?)\s+(?:آية|اية|الآية|الرقم)\s+(\d+)', query_text)
        if m:
            surah_name = normalize_arabic(m.group(1).strip())
            ayah_num = int(m.group(2))
            return surah_name, ayah_num
        return None, None

    def lookup(self, query_text: str, documents: List[BaseDocument]) -> Optional[BaseDocument]:
        surah_name, ayah_num = self.parse_query(query_text)
        if not surah_name or ayah_num is None:
            return None

        for doc in documents:
            s_name = doc.metadata.get("surah_name_ar")
            a_num = doc.metadata.get("ayah_number")
            if s_name and a_num is not None:
                if normalize_arabic(s_name) == surah_name and int(a_num) == ayah_num:
                    return BaseDocument(
                        id=doc.id,
                        type=doc.type,
                        source=doc.source,
                        text=doc.text,
                        metadata=doc.metadata,
                        score=1.0
                    )
        return None


class RetrievalPipeline:
    def __init__(self, retriever: BaseRetriever, ranker: BaseRanker, filter_step: BaseFilter, response_builder: ResponseBuilder):
        self.retriever = retriever
        self.ranker = ranker
        self.filter_step = filter_step
        self.response_builder = response_builder
        self.metadata_lookup = MetadataLookupStep()

    def retrieve(self, query: SearchQuery, debug: bool = False) -> SearchResponse:
        if hasattr(self.retriever, "exact_retriever") and hasattr(self.retriever.exact_retriever, "search_engine"):
            search_docs = getattr(self.retriever.exact_retriever.search_engine, "documents", [])
            direct_match = self.metadata_lookup.lookup(query.text, search_docs)
            if direct_match:
                return self.response_builder.build_response(query.text, [direct_match])

        candidates = self.retriever.retrieve_candidates(query.text, limit=query.limit)
        filtered = self.filter_step.filter_documents(candidates, query.filters)
        ranked = self.ranker.rank_documents(filtered, query.text, debug=debug)
        final_docs = ranked[:query.limit]
        return self.response_builder.build_response(query.text, final_docs)
