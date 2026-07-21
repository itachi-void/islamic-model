# -*- coding: utf-8 -*-
"""Check rank distribution of failing queries."""
import sys
sys.path.insert(0, r'd:\model')

from backend.rag.hadith_search import HadithSearchService
import json

with open('data/benchmarks/irb/v1/test.json', 'r', encoding='utf-8') as f:
    test = json.load(f)

with open('data/experiments/failure_report_v1.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

# Collect failed IDs
failed_ids = set()
for bucket in ['ranking_failures', 'retrieval_failures', 'knowledge_failures']:
    for item in report['details'].get(bucket, []):
        failed_ids.add(item['id'])

service = HadithSearchService()
lines = []

rank_counts = {i: 0 for i in range(1, 51)}
rank_counts['not_found'] = 0
per_query = []

for i, item in enumerate(test):
    if item['id'] not in failed_ids:
        continue
    
    q = item['query']
    acc = item.get('acceptable_answers', item.get('gold_evidence', []))
    expected = acc[0] if acc else 'N/A'
    
    resp = service.search(q, limit=50)
    retrieved_ids = [d.id for d in resp.documents]
    
    if expected in retrieved_ids:
        r = retrieved_ids.index(expected) + 1
        rank_counts[r] = rank_counts.get(r, 0) + 1
        per_query.append((item['id'], r, expected[:30], q[:60]))
    else:
        rank_counts['not_found'] += 1
        per_query.append((item['id'], 'NF', expected[:30], q[:60]))

lines.append("=== RANK DISTRIBUTION OF FAILING QUERIES ===")
lines.append(f"Total failing: {len(failed_ids)}")
lines.append(f"Not found: {rank_counts['not_found']}")
close = sum(v for k, v in rank_counts.items() if isinstance(k, int) and 6 <= k <= 10)
far = sum(v for k, v in rank_counts.items() if isinstance(k, int) and k > 10)
lines.append(f"Ranks 6-10 (close): {close}")
lines.append(f"Ranks 11-50 (far): {far}")
lines.append("")
lines.append("Rank distribution:")
for r in sorted([k for k in rank_counts if isinstance(k, int)]):
    if rank_counts[r] > 0:
        lines.append(f"  Rank {r}: {rank_counts[r]} queries")

lines.append("\n=== PER QUERY RANK ===")
for qid, rank, expected, qtext in sorted(per_query, key=lambda x: (0 if isinstance(x[1], int) and x[1] <= 10 else 1, x[1] if isinstance(x[1], int) else 99)):
    lines.append(f"  {qid}: rank={rank} | expected={expected[:20]}")

with open('temp_rank_dist.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Done")
