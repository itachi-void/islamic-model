# -*- coding: utf-8 -*-
"""
Scientific 4-Bucket Failure Analysis & Taxonomy Module
======================================================
Classifies retrieval misses into 4 distinct failure categories:
1. Retrieval Failure (Document absent from first-stage Top-50 candidates)
2. Ranking Failure (Document present in Top-50 candidates, but ranked > 5)
3. Knowledge / Metadata Failure (Unresolved narrator/book kunya term)
4. Benchmark Failure (Ambiguous query wording or gold answer defect)
"""
import os
import json
from typing import Dict, List, Set

FAILURE_REPORT_PATH = r"d:\model\data\experiments\failure_report_v1.json"


def analyze_and_classify_failures(
    failed_items: List[Dict],
    top50_candidates_map: Dict[str, List[str]],
    gold_sets_map: Dict[str, Set[str]]
) -> Dict:
    """Classifies failed queries into 4 taxonomy buckets and exports report."""
    retrieval_failures = []
    ranking_failures = []
    knowledge_failures = []
    benchmark_failures = []

    for item in failed_items:
        item_id = item["id"]
        query = item["query"]
        cat = item.get("category", "General")
        gold_set = gold_sets_map.get(item_id, set())
        candidates = top50_candidates_map.get(item_id, [])

        # Check if gold doc is present in Top-50 candidates
        in_top50 = any(c in gold_set for c in candidates)

        record = {
            "id": item_id,
            "query": query,
            "category": cat,
            "difficulty": item.get("difficulty", "medium"),
            "expected_hadith_number": item.get("expected_hadith_number"),
            "acceptable_answers": list(gold_set),
            "retrieved_candidates_top5": candidates[:5],
            "in_top50": in_top50
        }

        if not in_top50:
            if cat in ["Narrator", "Book", "Chapter"]:
                knowledge_failures.append(record)
            elif cat in ["Misspelling/Synonyms", "Partial Quote"]:
                benchmark_failures.append(record)
            else:
                retrieval_failures.append(record)
        else:
            ranking_failures.append(record)

    summary = {
        "total_failures": len(failed_items),
        "taxonomy_breakdown": {
            "retrieval_failure": len(retrieval_failures),
            "ranking_failure": len(ranking_failures),
            "knowledge_failure": len(knowledge_failures),
            "benchmark_failure": len(benchmark_failures)
        },
        "details": {
            "retrieval_failures": retrieval_failures,
            "ranking_failures": ranking_failures,
            "knowledge_failures": knowledge_failures,
            "benchmark_failures": benchmark_failures
        }
    }

    os.makedirs(os.path.dirname(FAILURE_REPORT_PATH), exist_ok=True)
    with open(FAILURE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary
