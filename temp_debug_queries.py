# -*- coding: utf-8 -*-
"""Debug specific failing queries through the pipeline."""
import sys, json, re
sys.path.insert(0, r'd:\model')

from backend.rag.hadith_search import HadithSearchService
from backend.rag.isnad_index import extract_isnad_narrators, strip_diacritics
from backend.rag.search import normalize_arabic
from backend.data.entity_resolver import EntityResolver

resolver = EntityResolver()
service = HadithSearchService()

with open('data/benchmarks/irb/v1/test.json', 'r', encoding='utf-8') as f:
    test = json.load(f)

with open('data/bukhari/bukhari_processed.json', 'r', encoding='utf-8') as f:
    bukhari = json.load(f)

b_map = {r['id']: r for r in bukhari}

lines = []
queries_to_check = ['irb_v1_00050', 'irb_v1_00288', 'irb_v1_00013']

for item in test:
    if item['id'] not in queries_to_check:
        continue
    
    q = item['query']
    acc = item.get('acceptable_answers', item.get('gold_evidence', []))
    expected = acc[0] if acc else 'N/A'
    
    lines.append(f"\n=== {item['id']} ===")
    lines.append(f"Expected: {expected}")
    lines.append(f"Query narrators: {extract_isnad_narrators(q)}")
    
    resolved = resolver.expand_query(q)
    expanded_diff = q != resolved
    if expanded_diff:
        lines.append(f"Entity expanded -> added terms")
    
    resp = service.search(q, limit=50)
    retrieved_ids = [d.id for d in resp.documents]
    
    if expected in retrieved_ids:
        rank = retrieved_ids.index(expected) + 1
        doc = resp.documents[retrieved_ids.index(expected)]
        lines.append(f"Found at rank #{rank}, score={doc.score:.4f}")
    else:
        lines.append(f"NOT in top-50")
        for i, d in enumerate(resp.documents[:5]):
            lines.append(f"  #{i+1}: {d.id} score={d.score:.4f}")
    
    if expected in b_map:
        h = b_map[expected]
        h_narrators = extract_isnad_narrators(h.get('matn', ''))
        lines.append(f"Hadith narrators: {h_narrators[:5]}")

with open('temp_debug_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Done")
