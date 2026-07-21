# -*- coding: utf-8 -*-
"""Analyze remaining failures after isnad index integration."""
import json

with open('data/experiments/failure_report_v1.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

results = []
results.append("=== REMAINING FAILURES ANALYSIS ===")
results.append(f"Total failures: {report['total_failures']}")
results.append(f"Breakdown: {json.dumps(report['taxonomy_breakdown'], indent=2)}")

# Analyze ranking failures - these are the biggest bucket
results.append("\n=== RANKING FAILURES (Doc in Top-50 but Rank>5) ===")
for i, item in enumerate(report['details']['ranking_failures'][:10]):
    results.append(f"\n{i+1}. {item['id']}: {item['query'][:80]}")
    results.append(f"   Category: {item['category']} | Difficulty: {item['difficulty']}")
    results.append(f"   Acceptable: {item['acceptable_answers']}")
    results.append(f"   Top-5 candidates: {item['retrieved_candidates_top5']}")
    results.append(f"   In Top-50: {item['in_top50']}")

# Analyze retrieval failures
results.append("\n=== RETRIEVAL FAILURES (Doc NOT in Top-50) ===")
for i, item in enumerate(report['details']['retrieval_failures'][:10]):
    results.append(f"\n{i+1}. {item['id']}: {item['query'][:80]}")
    results.append(f"   Category: {item['category']} | Difficulty: {item['difficulty']}")
    results.append(f"   Acceptable: {item['acceptable_answers']}")
    results.append(f"   Top-5 candidates: {item['retrieved_candidates_top5']}")

# Analyze knowledge failures
results.append("\n=== KNOWLEDGE FAILURES ===")
for item in report['details']['knowledge_failures']:
    results.append(f"  {item['id']}: {item['query'][:80]}")
    results.append(f"    Acceptable: {item['acceptable_answers']}")

with open('temp_failure_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print("Analysis written to temp_failure_analysis.txt")
