# -*- coding: utf-8 -*-
"""
Train / Dev / Test Benchmark Splitter
======================================
Splits benchmark datasets into reproducible Train (60%), Dev (20%), and Held-out Test (20%) splits
using a fixed random seed (42).

Usage:
    python -m backend.data.split_benchmarks
"""
import os
import json
import random
from typing import Dict, List, Tuple

BASE_DATA_DIR = r"d:\model\data"

TARGET_BENCHMARKS = [
    ("bukhari", os.path.join(BASE_DATA_DIR, "bukhari", "benchmarks", "evaluation_bukhari.json")),
    ("bukhari_3000", os.path.join(BASE_DATA_DIR, "bukhari", "benchmarks", "evaluation_bukhari_3000.json")),
    ("quran", os.path.join(BASE_DATA_DIR, "quran", "benchmarks", "evaluation_golden.json")),
    ("quran_3000", os.path.join(BASE_DATA_DIR, "quran", "benchmarks", "evaluation_quran_3000.json")),
]


def split_dataset(items: List[Dict], seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Splits items into 60% Train, 20% Dev, and 20% Held-out Test."""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * 0.60)
    dev_end = int(n * 0.80)

    train_set = shuffled[:train_end]
    dev_set = shuffled[train_end:dev_end]
    test_set = shuffled[dev_end:]

    return train_set, dev_set, test_set


def process_splits():
    print("=" * 70)
    print("SPLITTING BENCHMARKS INTO REPRODUCIBLE TRAIN / DEV / TEST SETS (Seed=42)")
    print("=" * 70)

    for name, file_path in TARGET_BENCHMARKS:
        if not os.path.exists(file_path):
            print(f"Skipping {name}: file not found at {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        train_set, dev_set, test_set = split_dataset(items, seed=42)

        base_dir = os.path.dirname(file_path)
        prefix = name

        train_path = os.path.join(base_dir, f"{prefix}_train.json")
        dev_path = os.path.join(base_dir, f"{prefix}_dev.json")
        test_path = os.path.join(base_dir, f"{prefix}_test.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_set, f, ensure_ascii=False, indent=2)

        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump(dev_set, f, ensure_ascii=False, indent=2)

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)

        print(f"[{name.upper()}] Total: {len(items)} | Train: {len(train_set)} | Dev: {len(dev_set)} | Held-out Test: {len(test_set)}")

    print("=" * 70)


if __name__ == "__main__":
    process_splits()
