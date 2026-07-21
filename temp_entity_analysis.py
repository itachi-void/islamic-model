# -*- coding: utf-8 -*-
"""Analyze test queries for entity resolution opportunities."""
import json
import re

def strip_diacritics(text):
    return re.sub(r'[\u064B-\u0652\u0670]', '', text) if text else ''

with open('data/benchmarks/irb/v1/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open('data/experiments/failure_report_v1.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

results = []
results.append("=== QUERIES WITH RESOLVABLE ENTITIES ===")

# Entity patterns to look for
entity_patterns = {
    'prophet_names': ['النبي', 'رسول الله', 'نبي الله', 'رسول'],
    'companion_kunyas': ['أبو', 'أبي', 'ابن', 'بنت'],
    'places': ['المدينة', 'مكة', 'طيبة', 'البيت الحرام', 'المسجد', 'الكعبة', 'غار'],
    'honorifics': ['رضي الله', 'صلى الله', 'عليه السلام'],
    'hadith_titles': ['حديث النية', 'حديث جبريل', 'حديث الإفك', 'حديث الشفاعة'],
}

hits = {k: 0 for k in entity_patterns}
queries_with_patterns = []

for item in test_data:
    q = item['query']
    q_clean = strip_diacritics(q)
    matched_patterns = []
    
    for pattern_type, patterns in entity_patterns.items():
        for p in patterns:
            if p in q_clean:
                hits[pattern_type] += 1
                if p not in matched_patterns:
                    matched_patterns.append(p)
    
    if matched_patterns:
        queries_with_patterns.append((item['id'], q[:80], matched_patterns))

for pattern_type, count in sorted(hits.items(), key=lambda x: -x[1]):
    results.append(f"  {pattern_type}: {count} queries")

results.append(f"\nTotal queries with resolvable entities: {len(queries_with_patterns)}")

# Show the queries
results.append("\n=== QUERIES THAT COULD BENEFIT FROM ENTITY RESOLUTION ===")
for qid, q, patterns in queries_with_patterns[:20]:
    results.append(f"  {qid}: {q}")

# Check which of these are in the failure list
failed_ids = set()
for bucket in ['retrieval_failures', 'ranking_failures', 'knowledge_failures', 'benchmark_failures']:
    for item in report['details'].get(bucket, []):
        failed_ids.add(item['id'])

results.append(f"\n=== ENTITY QUERIES THAT ARE FAILING ===")
for qid, q, patterns in queries_with_patterns:
    if qid in failed_ids:
        results.append(f"  {qid}: {q}")

# Check non-isnad queries
results.append("\n=== NON-ISNAD QUERIES (entity resolution has most impact here) ===")
for item in test_data:
    q = item['query']
    q_clean = strip_diacritics(q)
    if 'حدثنا' not in q_clean and 'حدثني' not in q_clean and 'اخبرنا' not in q_clean:
        results.append(f"  {item['id']}: {q[:100]}")
        results.append(f"    Acceptable: {item.get('acceptable_answers', [])}")
        if item['id'] in failed_ids:
            results.append(f"    **FAILING**")

with open('temp_entity_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print("Analysis written to temp_entity_analysis.txt")
