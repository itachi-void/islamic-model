# -*- coding: utf-8 -*-
"""
Large-Scale Benchmark Generator (3,000+ Queries)
=================================================
Generates large-scale evaluation benchmarks for Hadith (Bukhari/Muslim) and Quran
across multiple query types (Exact Quote, Narrator, Topic/Ruling, Book, Colloquial).

Usage:
    python -m backend.data.generate_large_benchmark
"""
import os
import json
import random
from typing import List, Dict
from backend.rag.search import normalize_arabic, extract_stemmed_tokens

BUKHARI_PATH = r"d:\model\data\bukhari\bukhari_processed.json"
QURAN_PATH = r"d:\model\data\quran\quran.json"

EVAL_BUKHARI_3000_PATH = r"d:\model\data\evaluation_bukhari_3000.json"
EVAL_QURAN_3000_PATH = r"d:\model\data\evaluation_quran_3000.json"


def generate_bukhari_3000() -> List[Dict]:
    """Generates 3,000 realistic Bukhari Hadith queries."""
    if not os.path.exists(BUKHARI_PATH):
        print(f"Warning: {BUKHARI_PATH} not found.")
        return []

    with open(BUKHARI_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    valid_records = [r for r in records if r.get("matn") and len(r.get("matn", "")) >= 30 and r.get("hadith_number")]

    queries = []
    query_id = 1

    # Distribution:
    # 1. Exact Quote Snippets (1,000 queries)
    # 2. Narrator Search (700 queries)
    # 3. Topic / Ruling Search (700 queries)
    # 4. Book & Chapter Search (300 queries)
    # 5. Colloquial Egyptian Variations (300 queries)

    # 1. Exact Quote Snippets
    for r in valid_records[:1000]:
        matn = r["matn"]
        words = matn.split()
        if len(words) >= 8:
            start_idx = random.randint(0, max(0, len(words) - 7))
            snippet = " ".join(words[start_idx:start_idx + 7])
            queries.append({
                "id": f"bk_large_{query_id:04d}",
                "query": snippet,
                "category": "Exact Quote",
                "expected_hadith_number": r["hadith_number"]
            })
            query_id += 1

    # 2. Narrator Search
    for r in valid_records[:700]:
        narrator = r.get("narrator", "").strip()
        matn = r["matn"]
        words = matn.split()
        snippet = " ".join(words[:5]) if len(words) >= 5 else matn
        if narrator:
            q_str = f"عن {narrator} {snippet}"
        else:
            q_str = f"حديث {snippet}"
        queries.append({
            "id": f"bk_large_{query_id:04d}",
            "query": q_str,
            "category": "Narrator",
            "expected_hadith_number": r["hadith_number"]
        })
        query_id += 1

    # 3. Topic / Ruling Search
    for r in valid_records[:700]:
        topics = r.get("topics", [])
        book = r.get("book", "")
        if topics:
            q_str = f"حديث في {topics[0]} {book}"
        else:
            q_str = f"حديث في {book}"
        queries.append({
            "id": f"bk_large_{query_id:04d}",
            "query": q_str,
            "category": "Topic/Ruling",
            "expected_hadith_number": r["hadith_number"]
        })
        query_id += 1

    # 4. Book & Chapter Search
    for r in valid_records[:300]:
        book = r.get("book", "")
        chapter = r.get("chapter", "")
        queries.append({
            "id": f"bk_large_{query_id:04d}",
            "query": f"{book} {chapter}".strip(),
            "category": "Book Reference",
            "expected_hadith_number": r["hadith_number"]
        })
        query_id += 1

    # 5. Colloquial Egyptian Variations
    colloquial_prefixes = [
        "عايز حديث عن",
        "إيه الحديث اللي بيقول",
        "هو في حديث عن",
        "ما هو الحديث الوارد في"
    ]
    for r in valid_records[:300]:
        matn = r["matn"]
        words = matn.split()
        snippet = " ".join(words[:6]) if len(words) >= 6 else matn
        prefix = random.choice(colloquial_prefixes)
        queries.append({
            "id": f"bk_large_{query_id:04d}",
            "query": f"{prefix} {snippet}",
            "category": "Colloquial",
            "expected_hadith_number": r["hadith_number"]
        })
        query_id += 1

    os.makedirs(os.path.dirname(EVAL_BUKHARI_3000_PATH), exist_ok=True)
    with open(EVAL_BUKHARI_3000_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"Generated Bukhari 3,000 Benchmark at: {EVAL_BUKHARI_3000_PATH} ({len(queries)} cases)")
    return queries


def generate_quran_3000() -> List[Dict]:
    """Generates 3,000 Quran queries."""
    if not os.path.exists(QURAN_PATH):
        print(f"Warning: {QURAN_PATH} not found.")
        return []

    with open(QURAN_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    queries = []
    query_id = 1

    # Generate 3,000 queries from 6,236 Ayahs
    sample_records = random.sample(records, min(3000, len(records)))
    for r in sample_records:
        text = r.get("text", "")
        surah_name = r.get("surah_name_ar", "")
        doc_id = r["id"]

        words = text.split()
        if len(words) >= 5:
            snippet = " ".join(words[:5])
        else:
            snippet = text

        queries.append({
            "id": f"qr_large_{query_id:04d}",
            "question": f"آية {snippet} في سورة {surah_name}",
            "category": "Exact Quote",
            "expected_ids": [doc_id]
        })
        query_id += 1

    with open(EVAL_QURAN_3000_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"Generated Quran 3,000 Benchmark at: {EVAL_QURAN_3000_PATH} ({len(queries)} cases)")
    return queries


if __name__ == "__main__":
    generate_bukhari_3000()
    generate_quran_3000()
