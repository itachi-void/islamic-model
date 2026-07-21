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

# Load test data
with open('data/benchmarks/irb/v1/test.json', 'r', encoding='utf-8') as f:
    test = json.load(f)

with open('data/bukhari/bukhari_processed.json', 'r', encoding='utf-8') as f:
    bukhari = json.load(f)

# Build hadith map
b_map = {r['id']: r for r in bukhari}

# Check specific queries
queries_to_check = ['irb_v1_00050', 'irb_v1_00288', 'irb_v1_00013', 'irb_v1_00480']

for item in test:
    if item['id'] not in queries_to_check:
        continue
    
    q = item['query']
    acc = item.get('acceptable_answers', item.get('gold_evidence', []))
    expected = acc[0] if acc else 'N/A'
    
    print(f"\n=== {item['id']} ===")
    print(f"Query: {q[:100]}")
    print(f"Expected: {expected}")
    
    # 1. Check entity resolution
    resolved = resolver.expand_query(q)
    expanded_diff = q != resolved
    print(f"Entity expanded: {expanded_diff} -> {resolved[:120]}" if expanded_diff else "Entity expanded: No")
    
    # 2. Check isnad extraction
    narrators = extract_isnad_narrators(q)
    print(f"Isnad narrators: {narrators}")
    
    # 3. Search with limit=50
    resp = service.search(q, limit=50)
    retrieved_ids = [d.id for d in resp.documents]
    
    if expected in retrieved_ids:
        rank = retrieved_ids.index(expected) + 1
        doc = resp.documents[retrieved_ids.index(expected)]
        print(f"Found at rank #{rank}, score={doc.score:.4f}")
    else:
        print(f"NOT in top-50 results")
        # Show first 5 results
        for i, d in enumerate(resp.documents[:5]):
            print(f"  #{i+1}: {d.id} score={d.score:.4f}")
    
    # 4. Check if expected hadith's isnad has the query narrators
    if expected in b_map:
        h = b_map[expected]
        h_narrators = extract_isnad_narrators(h.get('matn', ''))
        print(f"Expected hadith isnad: {h_narrators[:5]}")
        match = [n for n in narrators if any(n in hn for hn in h_narrators)]
        print(f"Match with query narrators: {match}")
