# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
from backend.domain.document import BaseDocument
from backend.domain.query import SearchQuery


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) algorithm for combining ranked result lists
    from multiple distinct knowledge collections (Quran, Bukhari, Tafsir, Fiqh).
    Formula: RRF(d) = sum_{m in Systems} (w_m / (k + r_m(d)))
    """
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, ranked_lists: List[List[BaseDocument]], weights: Optional[List[float]] = None) -> List[BaseDocument]:
        if not ranked_lists:
            return []

        if weights is None:
            weights = [1.0] * len(ranked_lists)

        scores: Dict[str, float] = {}
        doc_map: Dict[str, BaseDocument] = {}

        for list_idx, doc_list in enumerate(ranked_lists):
            w = weights[list_idx]
            for rank, doc in enumerate(doc_list, start=1):
                doc_map[doc.id] = doc
                rrf_val = w / (self.k + rank)
                scores[doc.id] = scores.get(doc.id, 0.0) + rrf_val

        fused_docs = []
        for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            orig_doc = doc_map[doc_id]
            fused_doc = BaseDocument(
                id=orig_doc.id,
                type=orig_doc.type,
                source=orig_doc.source,
                text=orig_doc.text,
                metadata=orig_doc.metadata,
                score=round(score, 6)
            )
            fused_docs.append(fused_doc)

        return fused_docs


class CrossCollectionRetriever:
    """
    Parallel Cross-Collection Engine.
    Executes searches across Quran, Hadith, Tafsir, and Fiqh engines,
    fusing candidates using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, quran_pipeline, hadith_service):
        self.quran_pipeline = quran_pipeline
        self.hadith_service = hadith_service
        self.rrf = ReciprocalRankFusion(k=60)

    def retrieve(self, query_text: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[BaseDocument]:
        search_query = SearchQuery(text=query_text, limit=limit * 2, filters=filters)

        # 1. Candidate retrieval from Quran
        quran_resp = self.quran_pipeline.retrieve(search_query)
        quran_candidates = quran_resp.documents

        # 2. Candidate retrieval from Bukhari
        hadith_resp = self.hadith_service.search(query_text, limit=limit * 2)
        hadith_candidates = hadith_resp.documents

        # 3. Fuse using RRF
        fused = self.rrf.fuse([quran_candidates, hadith_candidates], weights=[1.0, 1.0])
        return fused[:limit]
