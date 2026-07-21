# -*- coding: utf-8 -*-
"""Check exact failure status of key queries."""
import json

with open('data/benchmarks/irb/v1/test.json', 'r', encoding='utf-8') as f:
    test = json.load(f)

with open('data/experiments/failure_report_v1.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

# Collect failed IDs
failed_ids = set()
for bucket in ['retrieval_failures', 'ranking_failures', 'knowledge_failures']:
    for item in report['details'].get(bucket, []):
        failed_ids.add(item['id'])

# Check specific queries
queries_to_check = ['irb_v1_00050', 'irb_v1_00004', 'irb_v1_00288', 'irb_v1_00013']
for item in test:
    if item['id'] in queries_to_check:
        status = "FAILING" if item['id'] in failed_ids else "PASSING"
        acc = item.get('acceptable_answers', item.get('gold_evidence', []))
        cat = item.get('category', '?')
        # Get first 100 chars of query without diacritics
        q = item['query']
        print(f"{item['id']}: {status} | Cat={cat} | Ans={acc}")

# Count all non-isnad queries
isnad_count = 0
non_isnad = []
for item in test:
    qc = item['query']
    if 'حدثنا' in qc or 'حدثني' in qc or 'أخبرنا' in qc or 'اخبرنا' in qc:
        isnad_count += 1
    else:
        non_isnad.append(item['id'])
        
print(f"\nIsnad queries: {isnad_count}")
print(f"Non-isnad queries: {len(non_isnad)}")
for nid in non_isnad:
    status = "FAIL" if nid in failed_ids else "PASS"
    print(f"  {nid}: {status}")
