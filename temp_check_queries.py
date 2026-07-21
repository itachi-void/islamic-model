# -*- coding: utf-8 -*-
import json

with open('data/benchmarks/irb/v1/test.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for item in d:
    if item['id'] in ['irb_v1_00050','irb_v1_00004','irb_v1_00288','irb_v1_00013']:
        print(f"ID: {item['id']}")
        print(f"Query: {item['query'][:120]}")
        print(f"Category: {item.get('category')}")
        print(f"Answer type: {item.get('answer_type')}")
        print(f"Acceptable: {item.get('acceptable_answers', item.get('gold_evidence', []))}")
        print(f"Expected H#: {item.get('expected_hadith_number')}")
        print(f"Expected book: {item.get('expected_book_id')}")
        print("---")
