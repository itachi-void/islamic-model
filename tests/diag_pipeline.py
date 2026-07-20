# -*- coding: utf-8 -*-
"""Quick diagnostic: test pipeline components in isolation."""
import sys
import time
sys.path.insert(0, r"d:\model")

print("Step 1: importing ExactSearchEngine...", flush=True)
from backend.rag.search import ExactSearchEngine
print("OK", flush=True)

print("Step 2: loading bukhari docs...", flush=True)
from backend.rag.hadith_search import load_bukhari_documents
docs = load_bukhari_documents()
print(f"OK — {len(docs)} docs", flush=True)

print("Step 3: building ExactSearchEngine index...", flush=True)
t0 = time.time()
engine = ExactSearchEngine(docs)
print(f"OK — {time.time()-t0:.2f}s", flush=True)

print("Step 4: ExactSearchEngine.search()...", flush=True)
t0 = time.time()
results = engine.search("إنما الأعمال بالنيات", limit=3)
print(f"OK — {len(results)} results in {time.time()-t0:.3f}s", flush=True)
for r in results:
    h = r.metadata.get("hadith_number")
    print(f"  hadith_number={h}  score={r.score}", flush=True)

print("Step 5: importing BGEEmbeddingProvider (this may be slow)...", flush=True)
t0 = time.time()
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.core.settings import settings
emb = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
print(f"OK — model loaded in {time.time()-t0:.2f}s", flush=True)

print("Step 6: embedding one query...", flush=True)
t0 = time.time()
vec = emb.embed("إنما الأعمال بالنيات")
print(f"OK — embedding in {time.time()-t0:.3f}s, dim={len(vec)}", flush=True)

print("Step 7: importing ChromaVectorStore...", flush=True)
t0 = time.time()
from backend.rag.vector_store import ChromaVectorStore
store = ChromaVectorStore(
    persist_directory=settings.CHROMA_PATH,
    embedding_provider=emb,
    collection_name="bukhari"
)
print(f"OK — {store.collection.count()} vectors loaded in {time.time()-t0:.2f}s", flush=True)

print("Step 8: similarity_search...", flush=True)
t0 = time.time()
sem_results = store.similarity_search("إنما الأعمال بالنيات", limit=5)
print(f"OK — {len(sem_results)} results in {time.time()-t0:.3f}s", flush=True)
for r in sem_results:
    print(f"  hadith_number={r.metadata.get('hadith_number')}  score={r.score}", flush=True)

print("ALL STEPS PASSED", flush=True)
