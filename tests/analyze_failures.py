# -*- coding: utf-8 -*-
import json
import os
import re
from typing import Dict, Any, List

REPORT_FILE = r"d:\model\data\bukhari_benchmark_report.json"
FAILED_OUTPUT_FILE = r"d:\model\data\eval_failed_queries.json"


def categorize_failure(detail: dict) -> str:
    """Categorizes a failed benchmark query into an actionable error bucket."""
    hit_rank = detail.get("hit_rank")
    query = detail.get("query", "")

    if hit_rank is not None and hit_rank > 5:
        return "Ranking Error (Ranked outside Top-5)"

    # Check for normalization indicators
    if re.search(r"[أإآىئؤة]", query):
        return "Normalization Error (Arabic Character Variants)"

    # Check for metadata indicators
    if any(k in query for k in ["عن أبي", "عن عمر", "كتاب", "باب"]):
        return "Metadata Error (Narrator/Book Filter)"

    return "Missing Knowledge / Insufficient Context"


def analyze_benchmark_failures():
    if not os.path.exists(REPORT_FILE):
        print(f"Report file not found at {REPORT_FILE}. Please run benchmark first.")
        return

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        report = json.load(f)

    details = report.get("details", [])
    failures = []
    bucket_counts: Dict[str, int] = {}

    for d in details:
        if d.get("is_out_of_domain"):
            continue

        rank = d.get("hit_rank")
        if rank is None or rank > 5:
            category = categorize_failure(d)
            bucket_counts[category] = bucket_counts.get(category, 0) + 1

            failures.append({
                "id": d.get("id"),
                "query": d.get("query"),
                "expected_hadith_number": d.get("expected_hadith_number"),
                "hit_rank": rank,
                "failure_category": category
            })

    output_data = {
        "total_failures": len(failures),
        "category_counts": bucket_counts,
        "failures_backlog": failures
    }

    os.makedirs(os.path.dirname(FAILED_OUTPUT_FILE), exist_ok=True)
    with open(FAILED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK FAILURE ANALYSIS & BACKLOG CATEGORIZATION")
    print("=" * 70)
    print(f"Total Failures Tracked: {len(failures)}")
    print("\nFailure Categorization Breakdown:")
    for cat, count in bucket_counts.items():
        print(f"  - {cat:<48} : {count}")
    print("=" * 70)
    print(f"Saved failed queries backlog to: {FAILED_OUTPUT_FILE}\n")


if __name__ == "__main__":
    analyze_benchmark_failures()
