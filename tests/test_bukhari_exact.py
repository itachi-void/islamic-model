# -*- coding: utf-8 -*-
import json
import sys

sys.path.insert(0, r"d:\model")

from backend.rag.hadith_search import load_bukhari_documents
from backend.rag.search import ExactSearchEngine

docs = load_bukhari_documents()
exact_engine = ExactSearchEngine(docs)

test_queries = [
    ("bk008", "من يرد الله به خيرا يفقهه في الدين", 71),
    ("bk009", "بلغوا عني ولو آية", 3461),
    ("bk010", "لا تقبل صلاة من أحدث حتى يتوضأ", 135),
    ("bk013", "صلاة الجماعة أفضل من صلاة الفذ بسبع وعشرين درجة", 645),
    ("bk014", "من شهد الجنازة حتى يصلى عليها فله قيراط", 1253),
]

print("=" * 70)
print("TESTING BUKHARI EXACT SEARCH ENGINE DIRECTLY")
print("=" * 70)

for q_id, q_text, expected in test_queries:
    results = exact_engine.search(q_text, limit=10)
    retrieved_hnums = [int(d.metadata.get("hadith_number", 0)) for d in results]
    rank = (retrieved_hnums.index(expected) + 1) if expected in retrieved_hnums else None
    print(f"[{q_id}] Query: '{q_text}' | Expected Hadith: #{expected}")
    print(f"       Exact Engine Hits: {retrieved_hnums[:5]} | Rank: {rank}")
    print("-" * 70)
