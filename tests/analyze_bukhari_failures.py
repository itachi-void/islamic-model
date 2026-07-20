# -*- coding: utf-8 -*-
"""
Step 1: Sahih Al-Bukhari Failure Diagnostics & Categorization
"""
import json
import os
import re
import sys

sys.path.insert(0, r"d:\model")

from backend.rag.hadith_search import HadithSearchService

EVAL_FILE = r"d:\model\data\evaluation_bukhari.json"
OUTPUT_FILE = r"d:\model\data\bukhari_failure_analysis.json"


def categorize_bukhari_failure(query: str, category: str, rank: int, top20_hnums: list, expected_hnum: int) -> str:
    if expected_hnum in top20_hnums:
        return "Ranking"
    
    if any(k in query for k in ["عن أبي", "عن عمر", "عن عائشة", "عن ابن", "عن انس", "راوي"]):
        return "Metadata (Narrator)"
    
    if any(k in query for k in ["كتاب", "باب"]):
        return "Metadata (Book/Chapter)"
    
    if category == "Exact Quote":
        return "Exact Match"

    if category == "Colloquial":
        return "Semantic"

    return "Semantic / Synonym"


def run_failure_analysis():
    if not os.path.exists(EVAL_FILE):
        print(f"File not found: {EVAL_FILE}")
        return

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    service = HadithSearchService()
    valid_items = [b for b in benchmarks if b.get("expected_hadith_number") is not None]

    category_counts = {
        "Exact Match": 0,
        "Metadata (Narrator)": 0,
        "Metadata (Book/Chapter)": 0,
        "Ranking": 0,
        "Semantic": 0,
        "Missing Knowledge": 0
    }

    failed_details = []

    for item in valid_items:
        q_id = item["id"]
        q_text = item["query"]
        expected_hnum = int(item["expected_hadith_number"])
        q_cat = item.get("category", "General")

        resp = service.search(q_text, limit=20)
        retrieved_hnums = [
            int(doc.metadata.get("hadith_number"))
            for doc in resp.documents
            if doc.metadata.get("hadith_number") is not None
        ]

        hit_rank = None
        for rank, h_num in enumerate(retrieved_hnums, start=1):
            if h_num == expected_hnum:
                hit_rank = rank
                break

        if hit_rank is None or hit_rank > 5:
            fail_type = categorize_bukhari_failure(q_text, q_cat, hit_rank, retrieved_hnums, expected_hnum)
            category_counts[fail_type] = category_counts.get(fail_type, 0) + 1

            failed_details.append({
                "id": q_id,
                "query": q_text,
                "category": q_cat,
                "expected_hadith_number": expected_hnum,
                "hit_rank": hit_rank,
                "top5_retrieved": retrieved_hnums[:5],
                "failure_type": fail_type
            })

    output_data = {
        "total_valid_test_cases": len(valid_items),
        "total_failures_top5": len(failed_details),
        "failure_breakdown": category_counts,
        "failures": failed_details
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("STEP 1: SAHIH AL-BUKHARI FAILURE DIAGNOSTICS REPORT")
    print("=" * 70)
    print(f"Total In-Domain Test Cases: {len(valid_items)}")
    print(f"Total Failures (Outside Top-5): {len(failed_details)} ({(len(failed_details)/len(valid_items))*100:.1f}%)")
    print("\nCategorized Failure Breakdown:")
    for cat, count in category_counts.items():
        pct = (count / len(failed_details) * 100) if failed_details else 0
        print(f"  - {cat:<30} : {count:<3} ({pct:.1f}%)")
    print("=" * 70)
    print(f"Saved diagnostics to: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    run_failure_analysis()
