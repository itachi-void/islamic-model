# -*- coding: utf-8 -*-
from typing import List
try:
    import ollama
except ImportError:
    ollama = None

import socket

def _is_ollama_running() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.1):
            return True
    except Exception:
        return False


class BaseEmbeddingProvider:
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("Subclasses must implement embed_query")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement embed_documents")


class BGEEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "bge-m3"):
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        if not _is_ollama_running():
            return []
        import requests
        try:
            url = "http://127.0.0.1:11434/api/embed"
            payload = {"model": self.model_name, "input": text}
            res = requests.post(url, json=payload, timeout=0.8)
            if res.status_code == 200:
                data = res.json()
                embs = data.get("embeddings", [])
                if embs and len(embs) > 0:
                    return embs[0]
        except Exception:
            pass
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts or not ollama or not _is_ollama_running():
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
