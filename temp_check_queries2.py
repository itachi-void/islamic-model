# -*- coding: utf-8 -*-
"""Check specific queries"""
import json

with open('data/benchmarks/irb/v1/test.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for item in d:
    if item['id'] in ['irb_v1_00050','irb_v1_00004','irb_v1_00288','irb_v1_00013']:
        q = item['query']
        acc = item.get('acceptable_answers', item.get('gold_evidence', []))
        print(f"ID: {item['id']}")
        print(f"Cat: {item.get('category')}")
        with open('temp_q_'+item['id']+'.txt', 'w', encoding='utf-8') as f2:
            f2.write(q + '\n---\n' + str(acc))
        print(f" -> wrote to temp_q_{item['id']}.txt")
