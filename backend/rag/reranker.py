# -*- coding: utf-8 -*-
"""
Cross-Encoder Re-ranker Module for Hadith & Islamic RAG
=========================================================
Second-stage re-ranker that computes deep query-document entailment scores
for candidate documents retrieved during first-stage RRF candidate fusion.
"""
import logging
from typing import List, Optional
from backend.domain.document import BaseDocument
from backend.rag.search import extract_stemmed_tokens, normalize_arabic

logger = logging.getLogger(__name__)

class BaseReranker:
    def rerank(self, query: str, documents: List[BaseDocument], top_k: int = 5) -> List[BaseDocument]:
        raise NotImplementedError("Subclasses must implement rerank")


class ExactMatchBonusReranker(BaseReranker):
    """
    Fast rule-based re-ranker that boosts exact phrase matches, narrator matches, and book matches.
    """
    def rerank(self, query: str, documents: List[BaseDocument], top_k: int = 5) -> List[BaseDocument]:
        if not documents:
            return []

        norm_q = normalize_arabic(query)
        q_tokens = extract_stemmed_tokens(query)

        scored = []
        for doc in documents:
            score = float(doc.score or 0.0)
            norm_doc = normalize_arabic(doc.text)
            doc_tokens = extract_stemmed_tokens(doc.text)

            # Boost 1: Exact phrase match in document text
            if norm_q in norm_doc:
                score += 0.40

            # Boost 2: Token overlap ratio
            if q_tokens and doc_tokens:
                overlap = len(q_tokens & doc_tokens) / float(len(q_tokens))
                score += 0.20 * overlap

            # Boost 3: Metadata matches (narrator or book)
            narrator = normalize_arabic(str(doc.metadata.get("narrator", "")))
            book = normalize_arabic(str(doc.metadata.get("book", "")))
            if narrator and any(t in narrator for t in q_tokens):
                score += 0.25
            if book and any(t in book for t in q_tokens):
                score += 0.15

            scored.append(BaseDocument(
                id=doc.id,
                type=doc.type,
                source=doc.source,
                text=doc.text,
                metadata=doc.metadata,
                score=round(score, 4)
            ))

        scored.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker(BaseReranker):
    """
    Neural Cross-Encoder Re-ranker using sentence_transformers.CrossEncoder.
    Falls back to ExactMatchBonusReranker if model loading fails or model offline.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._cross_encoder = None
        self._fallback = ExactMatchBonusReranker()

    def _get_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder model '{self.model_name}': {e}. Using fallback.")
                self._cross_encoder = False
        return self._cross_encoder if self._cross_encoder is not False else None

    def rerank(self, query: str, documents: List[BaseDocument], top_k: int = 5) -> List[BaseDocument]:
        if not documents:
            return []

        encoder = self._get_encoder()
        if encoder is None:
            return self._fallback.rerank(query, documents, top_k=top_k)

        try:
            pairs = [[query, doc.text[:512]] for doc in documents]
            scores = encoder.predict(pairs)

            scored = []
            for doc, score in zip(documents, scores):
                scored.append(BaseDocument(
                    id=doc.id,
                    type=doc.type,
                    source=doc.source,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=round(float(score), 4)
                ))

            scored.sort(key=lambda x: x.score or 0.0, reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.warning(f"CrossEncoder reranking error: {e}. Falling back.")
            return self._fallback.rerank(query, documents, top_k=top_k)
