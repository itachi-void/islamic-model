# -*- coding: utf-8 -*-
from typing import List
import ollama

class BaseEmbeddingProvider:
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement embed_query")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement embed_documents")


class BGEEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "bge-m3"):
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        try:
            res = ollama.embed(model=self.model_name, input=text)
            if hasattr(res, "embeddings") and res.embeddings:
                return res.embeddings[0]
            elif isinstance(res, dict) and "embeddings" in res and res["embeddings"]:
                return res["embeddings"][0]
        except Exception:
            pass
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            res = ollama.embed(model=self.model_name, input=texts)
            if hasattr(res, "embeddings") and len(res.embeddings) == len(texts):
                return res.embeddings
            elif isinstance(res, dict) and "embeddings" in res and len(res["embeddings"]) == len(texts):
                return res["embeddings"]
        except Exception as e:
            print(f"Batch embedding fallback due to error: {e}")

        # Fallback to sequential calls if batching fails
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings
