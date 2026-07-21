# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Any, Optional
try:
    import chromadb
except ImportError:
    chromadb = None

from backend.domain.document import BaseDocument
from backend.rag.embeddings import BaseEmbeddingProvider

class BaseVectorStore:
    def add_documents(self, documents: List[BaseDocument]) -> None:
        raise NotImplementedError("Subclasses must implement add_documents")

    def similarity_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[BaseDocument]:
        raise NotImplementedError("Subclasses must implement similarity_search")


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_directory: str, embedding_provider: BaseEmbeddingProvider, collection_name: str = "quran"):
        self.persist_directory = persist_directory
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        
        os.makedirs(persist_directory, exist_ok=True)
        if not chromadb:
            self.client = None
            self.collection = None
            return

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            print(f"ChromaDB initialization fallback: {e}")
            self.client = None
            self.collection = None

    def reset_collection(self) -> None:
        """
        Deletes and recreates the collection to ensure clean ingestion.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, documents: List[BaseDocument]) -> None:
        """
        Ingests documents into Chroma collection using enriched texts for vector embeddings
        while retaining clean verse text in metadata.
        """
        if not documents:
            return

        ids = []
        embedding_texts = []
        metadatas = []
        stored_documents = []
        
        for doc in documents:
            ids.append(doc.id)
            embedding_texts.append(doc.text)
            stored_documents.append(doc.text)
            
            meta = doc.metadata.copy()
            meta["source"] = doc.source
            meta["type"] = doc.type
            meta["raw_text"] = doc.metadata.get("original_text", doc.text)
            
            # Serialize list/dict fields if necessary for Chroma DB compatibility
            if "aliases" in meta and isinstance(meta["aliases"], list):
                meta["aliases"] = " | ".join(meta["aliases"])
            if "topics" in meta and isinstance(meta["topics"], list):
                meta["topics"] = " | ".join(meta["topics"])

            metadatas.append(meta)

        # Compute embeddings for all document texts
        embeddings = self.embedding_provider.embed_documents(embedding_texts)
        
        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=stored_documents
        )

    def similarity_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[BaseDocument]:
        """
        Performs vector similarity search on Chroma collection using the query embedding.
        """
        if not self.collection:
            return []

        query_embedding = self.embedding_provider.embed_query(query)
        if not query_embedding:
            return []

        where_clause = self._build_where_clause(filters)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause
        )

        documents = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        for idx in range(len(results["ids"][0])):
            doc_id = results["ids"][0][idx]
            text = results["documents"][0][idx]
            meta = results["metadatas"][0][idx] or {}
            
            distance = results["distances"][0][idx] if results["distances"] else 0.0
            similarity = round(max(0.0, 1.0 - (distance / 2)), 4)

            doc_source = meta.pop("source", self.collection_name)
            doc_type = meta.pop("type", "verse")

            doc = BaseDocument(
                id=doc_id,
                type=doc_type,
                source=doc_source,
                text=text,
                metadata=meta,
                score=similarity
            )
            documents.append(doc)

        return documents

    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
            
        if len(filters) == 1:
            key, val = list(filters.items())[0]
            return {key: val}
            
        return {"$and": [{key: val} for key, val in filters.items()]}
