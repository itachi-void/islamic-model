# -*- coding: utf-8 -*-
"""
Scientific Data Directory Organizer
=====================================
Organizes data/ directory into standardized, clean subfolders:
- data/<collection>/raw/
- data/<collection>/processed/
- data/<collection>/aliases/
- data/<collection>/benchmarks/
- data/experiments/
"""
import os
import shutil
import json

BASE_DATA_DIR = r"d:\model\data"

COLLECTIONS = ["bukhari", "quran", "muslim", "tafsir"]
SUBDIRS = ["raw", "processed", "aliases", "benchmarks"]

def setup_directory_structure():
    # 1. Create subdirectories
    for col in COLLECTIONS:
        for sub in SUBDIRS:
            path = os.path.join(BASE_DATA_DIR, col, sub)
            os.makedirs(path, exist_ok=True)

    os.makedirs(os.path.join(BASE_DATA_DIR, "experiments"), exist_ok=True)

    # 2. Copy/Move benchmarks into collection benchmarks/ directory
    copies = [
        (os.path.join(BASE_DATA_DIR, "evaluation_bukhari.json"), os.path.join(BASE_DATA_DIR, "bukhari", "benchmarks", "evaluation_bukhari.json")),
        (os.path.join(BASE_DATA_DIR, "evaluation_bukhari_3000.json"), os.path.join(BASE_DATA_DIR, "bukhari", "benchmarks", "evaluation_bukhari_3000.json")),
        (os.path.join(BASE_DATA_DIR, "evaluation_quran_3000.json"), os.path.join(BASE_DATA_DIR, "quran", "benchmarks", "evaluation_quran_3000.json")),
        (os.path.join(BASE_DATA_DIR, "evaluation_golden.json"), os.path.join(BASE_DATA_DIR, "quran", "benchmarks", "evaluation_golden.json")),
        (os.path.join(BASE_DATA_DIR, "evaluation_muslim.json"), os.path.join(BASE_DATA_DIR, "muslim", "benchmarks", "evaluation_muslim.json")),
        (os.path.join(BASE_DATA_DIR, "evaluation_tafsir.json"), os.path.join(BASE_DATA_DIR, "tafsir", "benchmarks", "evaluation_tafsir.json")),
        (os.path.join(BASE_DATA_DIR, "bukhari_canonical_map.json"), os.path.join(BASE_DATA_DIR, "bukhari", "aliases", "bukhari_canonical_map.json")),
    ]

    for src, dst in copies:
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # 3. Create empty experiment history log if not exists
    exp_log = os.path.join(BASE_DATA_DIR, "experiments", "experiment_history.json")
    if not os.path.exists(exp_log):
        with open(exp_log, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    print("Data directory structure successfully organized.")

if __name__ == "__main__":
    setup_directory_structure()
