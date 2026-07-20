# -*- coding: utf-8 -*-
"""
Neural Cross-Encoder Reranker for Hadith & Islamic RAG
======================================================
Deep Neural Cross-Encoder runner that scores (query, candidate_document) pairs
using Sentence-Transformers / PyTorch sequence classification models.
"""
import logging
import time
from typing import List, Optional
from backend.domain.document import BaseDocument
from backend.rag.reranker import BaseReranker, ExactMatchBonusReranker

logger = logging.getLogger(__name__)

# Pre-trained Arabic / Multilingual Reranker models
DEFAULT_RERANKER_MODELS = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
]


class NeuralCrossEncoderReranker(BaseReranker):
    """
    Neural Cross-Encoder Re-ranker.
    Calculates query-document cross-attention logits to rank retrieved candidates.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._fallback = ExactMatchBonusReranker()

    def _init_model(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
                device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
                logger.info(f"Loading Neural CrossEncoder '{self.model_name}' on device '{device}'...")
                self._model = CrossEncoder(self.model_name, device=device, max_length=512)
                logger.info(f"CrossEncoder '{self.model_name}' loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder '{self.model_name}': {e}. Using ExactMatch fallback.")
                self._model = False
        return self._model if self._model is not False else None

    def rerank(self, query: str, documents: List[BaseDocument], top_k: int = 5) -> List[BaseDocument]:
        if not documents:
            return []

        model = self._init_model()
        if model is None:
            return self._fallback.rerank(query, documents, top_k=top_k)

        try:
            t0 = time.time()
            pairs = [[query, doc.text[:512]] for doc in documents]
            scores = model.predict(pairs)

            scored_docs = []
            for doc, score in zip(documents, scores):
                scored_docs.append(BaseDocument(
                    id=doc.id,
                    type=doc.type,
                    source=doc.source,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=round(float(score), 4)
                ))

            scored_docs.sort(key=lambda x: x.score or 0.0, reverse=True)
            logger.debug(f"CrossEncoder reranked {len(documents)} docs in {(time.time()-t0)*1000:.2f}ms")
            return scored_docs[:top_k]

        except Exception as e:
            logger.warning(f"NeuralCrossEncoder execution error: {e}. Falling back.")
            return self._fallback.rerank(query, documents, top_k=top_k)
