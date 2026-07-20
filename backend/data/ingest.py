# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, r"d:\model")

from backend.core.settings import settings
from backend.data.loader import load_search_documents
from backend.data.cleaner import DataCleaner
from backend.data.chunker import DocumentChunker
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.data.indexer import DocumentIndexer

def run_ingestion() -> None:
    print("Starting ingestion pipeline...")
    
    # 1. Load documents
    print("Loading search documents...")
    raw_documents = load_search_documents()

    # 2. Clean documents
    cleaner = DataCleaner()
    for doc in raw_documents:
        doc.text = cleaner.clean_text(doc.text)

    # 3. Chunk documents
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
    chunked_documents = []
    for doc in raw_documents:
        chunked_documents.extend(chunker.chunk_document(doc))
    
    print(f"Total chunks to index: {len(chunked_documents)}")

    # 4. Initialize embedding provider & vector store
    embedding_provider = BGEEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)
    vector_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PATH,
        embedding_provider=embedding_provider,
        collection_name="quran"
    )

    current_count = vector_store.collection.count()
    print(f"Current collection count: {current_count}/{len(chunked_documents)}")

    if current_count >= len(chunked_documents):
        print("Collection is already fully indexed!")
        return

    # Resume from current_count
    remaining = chunked_documents[current_count:]
    batch_size = 50
    indexer = DocumentIndexer(vector_store)
    
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    print(f"Indexing remaining {len(remaining)} chunks in {total_batches} batches...")

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Batch {batch_num}/{total_batches} (items {current_count + i} to {current_count + i + len(batch)})...")
        indexer.index_documents(batch)
        
    print("Ingestion pipeline successfully completed!")

if __name__ == "__main__":
    run_ingestion()
