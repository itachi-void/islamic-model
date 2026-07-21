# -*- coding: utf-8 -*-
"""
Fast IRB-v1 Test Evaluation & 4-Bucket Failure Taxonomy Summary
"""
import sys
import json

sys.path.insert(0, r"d:\model")

from eval import evaluate_irb_v1

TEST_PATH = r"d:\model\data\benchmarks\irb\v1\test.json"
with open(TEST_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Evaluating {len(data)} IRB-v1 test items...", flush=True)
metrics = evaluate_irb_v1(data, top_k=5)

print("\n" + "=" * 80, flush=True)
print("[Decomposed Latencies & Research Metrics]", flush=True)
print(f"Hits@1                  : {metrics['hits_1']:.2f}%", flush=True)
print(f"Hits@3                  : {metrics['hits_3']:.2f}%", flush=True)
print(f"Hits@5                  : {metrics['hits_5']:.2f}%", flush=True)
print(f"Recall@1                : {metrics['recall_1']:.2f}%", flush=True)
print(f"Recall@5                : {metrics['recall_5']:.2f}%", flush=True)
print(f"Recall@10               : {metrics['recall_10']:.2f}%", flush=True)
print(f"Precision@5             : {metrics['precision_5']:.2f}%", flush=True)
print(f"MRR                     : {metrics['mrr']:.4f}", flush=True)
print(f"nDCG@5                  : {metrics['ndcg_5']:.4f}", flush=True)
print(f"OOD Guard Accuracy      : {metrics['ood_accuracy']:.2f}%", flush=True)
print(f"Retrieval Latency (Pure): {metrics['avg_retrieval_latency_ms']} ms", flush=True)
print(f"Generation Latency      : {metrics['avg_generation_latency_ms']} ms", flush=True)

fb = metrics["failure_summary"]["taxonomy_breakdown"]
print("\n[Scientific 4-Bucket Failure Taxonomy Breakdown]", flush=True)
print(f"  - Retrieval Failures (Doc absent from Top-50) : {fb['retrieval_failure']}", flush=True)
print(f"  - Ranking Failures (Doc in Top-50, Rank>5)    : {fb['ranking_failure']}", flush=True)
print(f"  - Knowledge / Metadata Failures              : {fb['knowledge_failure']}", flush=True)
print(f"  - Benchmark Ambiguity Failures               : {fb['benchmark_failure']}", flush=True)
print("=" * 80, flush=True)
