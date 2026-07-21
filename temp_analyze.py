# -*- coding: utf-8 -*-
import json
import re
from collections import Counter

def strip_diacritics(text):
    """Remove Arabic diacritics (tashkeel)"""
    if not text:
        return ""
    diacritics = re.compile(r'[\u064B-\u0652\u0670]')
    return diacritics.sub('', text)

with open('data/bukhari/bukhari_processed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('data/benchmarks/irb/v1/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

results = []
results.append(f"Total hadiths: {len(data)}")
results.append(f"Total test queries: {len(test_data)}")

# Show first 3 isnad patterns
results.append("\n=== SAMPLE MATN TEXTS ===")
for r in data[:3]:
    matn = r.get('matn', '')[:300]
    results.append(f"ID: {r['id']} | HNum: {r.get('hadith_number')}")
    results.append(f"Narrator: {r.get('narrator', '')}")
    results.append(f"Matn: {matn}")
    results.append("---")

# Extract isnad patterns from matn - use diacritics-stripped text
isnad_pattern = re.compile(r'(?:حدثنا|حدثني|اخبرنا|اخبرني|قال حدثنا|سمعت)\s+([^،,]+)')
isnad_names = Counter()
for r in data:
    matn = r.get('matn', '')
    matn_clean = strip_diacritics(matn)
    matches = isnad_pattern.findall(matn_clean)
    for m in matches:
        name = m.strip()
        if len(name) > 3:
            isnad_names[name] += 1

results.append(f"\nUnique isnad narrators (cleaned): {len(isnad_names)}")
results.append("Top 30 most common isnad narrators:")
for name, count in isnad_names.most_common(30):
    results.append(f"  {count:>4}x  {name}")

# Check test queries
results.append("\n=== TEST QUERY ANALYSIS ===")
isnad_queries = 0
for item in test_data:
    q = item['query']
    q_clean = strip_diacritics(q)
    if 'حدثنا' in q_clean or 'حدثني' in q_clean or 'اخبرنا' in q_clean or 'اخبرني' in q_clean:
        isnad_queries += 1

results.append(f"Test queries with isnad pattern: {isnad_queries}/{len(test_data)}")

# Show sample test queries
results.append("\nSample test queries (cleaned):")
for item in test_data[:5]:
    q = item['query']
    q_clean = strip_diacritics(q)
    results.append(f"  {item['id']}: {q_clean[:100]}")
    results.append(f"    Acceptable: {item.get('acceptable_answers', [])}")

# Check how many hadiths have the query isnad names in their matn
results.append("\n=== ISNAD MATCHING ANALYSIS ===")
hits = 0
misses = 0
for item in test_data:
    q = item['query']
    q_clean = strip_diacritics(q)
    q_names = isnad_pattern.findall(q_clean)
    if not q_names:
        results.append(f"  SKIP (no isnad pattern): {item['id']}: {q_clean[:80]}")
        continue
    
    acceptable = set(item.get('acceptable_answers', []))
    found = False
    for r in data:
        h_id = r['id']
        h_num = str(r.get('hadith_number', ''))
        if h_id in acceptable or h_num in acceptable:
            matn = r.get('matn', '')
            matn_clean = strip_diacritics(matn)
            for qn in q_names:
                qn_clean = qn.strip()
                if qn_clean in matn_clean:
                    found = True
                    break
            break
    
    if found:
        hits += 1
    else:
        misses += 1
        results.append(f"  MISS: {item['id']}: {q_clean[:80]}...")
        results.append(f"    Looking for: {[qn.strip() for qn in q_names]}")
        # Find the actual hadith and show its isnad
        for r in data:
            h_id = r['id']
            h_num = str(r.get('hadith_number', ''))
            if h_id in acceptable or h_num in acceptable:
                matn = r.get('matn', '')[:250]
                matn_c = strip_diacritics(matn)
                results.append(f"    Actual matn (cleaned): {matn_c}")
                # Show what isnad names are in this matn
                matn_isnad = isnad_pattern.findall(matn_c)
                results.append(f"    Actual isnad names: {matn_isnad[:5]}")
                break

total_checked = hits + misses
results.append(f"\nMatch rate: {hits}/{total_checked} = {hits/total_checked*100:.1f}%" if total_checked > 0 else "No matches")

# Build isnad -> hadith mapping for the top narrators
results.append("\n=== ISNAD INVERTED INDEX SAMPLE ===")
isnad_to_hadiths = {}
for r in data:
    matn = r.get('matn', '')
    matn_clean = strip_diacritics(matn)
    matches = isnad_pattern.findall(matn_clean)
    h_id = r['id']
    for m in matches:
        name = m.strip()
        if len(name) > 3:
            if name not in isnad_to_hadiths:
                isnad_to_hadiths[name] = []
            isnad_to_hadiths[name].append(h_id)

results.append(f"Total isnad entries: {sum(len(v) for v in isnad_to_hadiths.values())}")
results.append(f"Unique isnad narrators: {len(isnad_to_hadiths)}")

# Show some example mappings
for name in list(isnad_names.keys())[:5]:
    h_list = isnad_to_hadiths.get(name, [])
    results.append(f"  '{name}' -> {len(h_list)} hadiths: {h_list[:5]}")

with open('temp_analysis_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print("Analysis written to temp_analysis_output.txt")
