# -*- coding: utf-8 -*-
from typing import List
from backend.domain.document import BaseDocument
from backend.rag.vector_store import BaseVectorStore

class DocumentIndexer:
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    def index_documents(self, documents: List[BaseDocument]) -> None:
        """
        Indexes a list of documents by putting them into the vector store.
        """
        self.vector_store.add_documents(documents)
