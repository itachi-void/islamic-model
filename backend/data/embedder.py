# -*- coding: utf-8 -*-
from typing import List
from backend.domain.document import BaseDocument
from backend.rag.embeddings import BaseEmbeddingProvider

class DocumentEmbedder:
    def __init__(self, embedding_provider: BaseEmbeddingProvider):
        self.embedding_provider = embedding_provider

    def embed_documents(self, documents: List[BaseDocument]) -> List[List[float]]:
        """
        Computes vectors for a list of documents using the embedding provider.
        """
        texts = [doc.text for doc in documents]
        return self.embedding_provider.embed_documents(texts)
