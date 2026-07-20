# -*- coding: utf-8 -*-
"""
Experiment Tracker & Dashboard Reporting Module
=================================================
Automates logging of evaluation runs, tracks score progression over time,
and generates Markdown progress reports and ASCII progress charts for README / Graduation Reports.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

EXPERIMENTS_LOG_PATH = r"d:\model\data\experiments\experiment_history.json"


def _load_history() -> List[Dict]:
    if not os.path.exists(EXPERIMENTS_LOG_PATH):
        return []
    try:
        with open(EXPERIMENTS_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def log_experiment(
    collection: str,
    split_type: str,
    metrics: Dict,
    model_info: str = "BGE-M3 + BM25 + CrossEncoder",
    notes: str = ""
) -> Dict:
    """Logs an evaluation run into experiment_history.json."""
    history = _load_history()
    run_id = f"run_{len(history) + 1:03d}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        "run_id": run_id,
        "timestamp": now_str,
        "collection": collection,
        "split_type": split_type,
        "model_info": model_info,
        "hits_1": round(metrics.get("hits_1", 0.0) * 100, 2),
        "hits_5": round(metrics.get("hits_5", 0.0) * 100, 2),
        "mrr": round(metrics.get("mrr", 0.0), 4),
        "avg_latency_ms": round(metrics.get("avg_latency_ms", 0.0), 2),
        "total_queries": metrics.get("in_domain_total", metrics.get("total", 0)),
        "notes": notes
    }

    history.append(record)
    os.makedirs(os.path.dirname(EXPERIMENTS_LOG_PATH), exist_ok=True)
    with open(EXPERIMENTS_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return record


def print_experiment_dashboard(collection: Optional[str] = None):
    """Prints a Markdown dashboard and ASCII progression chart of all past experiments."""
    history = _load_history()
    if collection:
        history = [r for r in history if r["collection"].lower() == collection.lower()]

    if not history:
        print("No experiment history found.")
        return

    print("=" * 80)
    print("EXPERIMENT PROGRESSION DASHBOARD")
    print("=" * 80)
    print(f"{'Run ID':<8} | {'Timestamp':<19} | {'Collection':<10} | {'Split':<8} | {'Hits@1':<8} | {'Hits@5':<8} | {'MRR':<8} | {'Latency':<8}")
    print("-" * 80)

    for r in history:
        print(f"{r['run_id']:<8} | {r['timestamp']:<19} | {r['collection']:<10} | {r['split_type']:<8} | {r['hits_1']:>6.2f}% | {r['hits_5']:>6.2f}% | {r['mrr']:>7.4f} | {r['avg_latency_ms']:>6.1f}ms")

    print("=" * 80)
    print("\n[Hits@5 Progression Chart]")
    for r in history:
        bar_len = int(r['hits_5'] / 2)
        bar = "█" * bar_len
        print(f"  {r['run_id']} ({r['collection']}/{r['split_type']}): {bar} {r['hits_5']:.2f}%")
    print("=" * 80)
