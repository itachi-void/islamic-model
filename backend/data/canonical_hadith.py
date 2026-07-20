# -*- coding: utf-8 -*-
"""
Canonical Hadith Equivalence Mapper
====================================
Maps duplicate or variant Hadiths across different books/chapters
within a Hadith collection (e.g. Sahih Bukhari) based on text overlap.

Usage:
    python -m backend.data.canonical_hadith
"""
import os
import json
from typing import Dict, Set, List
from collections import defaultdict
from backend.rag.search import extract_stemmed_tokens, normalize_arabic


def build_canonical_map(
    dataset_path: str,
    output_path: str,
    min_similarity: float = 0.55
) -> Dict[int, List[int]]:
    """
    Builds a mapping from hadith_number -> list of equivalent hadith_numbers.
    Uses inverted token indexing for fast O(N) matching.
    """
    if not os.path.exists(dataset_path):
        print(f"Warning: {dataset_path} does not exist. Returning empty map.")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 1. Precompute stemmed tokens for each record
    doc_tokens: Dict[int, Set[str]] = {}
    inverted_index: Dict[str, List[int]] = defaultdict(list)

    for r in records:
        h_num = r.get("hadith_number")
        matn = r.get("matn", "")
        if not h_num or not matn:
            continue

        tokens = extract_stemmed_tokens(matn)
        if not tokens:
            continue

        h_num = int(h_num)
        doc_tokens[h_num] = tokens

        for token in tokens:
            inverted_index[token].append(h_num)

    # Filter out high-frequency tokens (appearing in > 300 documents) to speed up candidate matching
    total_docs = len(doc_tokens)
    max_df = max(10, int(total_docs * 0.04))  # ~300 docs
    filtered_index = {t: nums for t, nums in inverted_index.items() if len(nums) <= max_df}

    # 2. Match candidate pairs sharing at least 3 distinctive tokens
    canonical_sets: Dict[int, Set[int]] = defaultdict(set)
    for h_num in doc_tokens.keys():
        canonical_sets[h_num].add(h_num)

    for h_num, tokens in doc_tokens.items():
        candidate_counts: Dict[int, int] = defaultdict(int)
        for token in tokens:
            if token in filtered_index:
                for other_h in filtered_index[token]:
                    if other_h != h_num:
                        candidate_counts[other_h] += 1

        for other_h, shared_count in candidate_counts.items():
            if shared_count < 3:
                continue
            other_tokens = doc_tokens[other_h]
            jaccard = shared_count / float(len(tokens | other_tokens))
            if jaccard >= min_similarity:
                canonical_sets[h_num].add(other_h)
                canonical_sets[other_h].add(h_num)

    # Convert sets to sorted lists for JSON serialization
    canonical_map: Dict[int, List[int]] = {
        h_num: sorted(list(eq_set))
        for h_num, eq_set in canonical_sets.items()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(canonical_map, f, ensure_ascii=False, indent=2)

    total_with_equivs = sum(1 for v in canonical_map.values() if len(v) > 1)
    print(f"Canonical Map generated at: {output_path}")
    print(f"Total Hadiths: {len(canonical_map)} | Hadiths with equivalents: {total_with_equivs}")

    return canonical_map


def load_canonical_map(collection: str) -> Dict[int, Set[int]]:
    """
    Loads pre-computed canonical map for a collection.
    Returns Dict[int, Set[int]].
    """
    map_file = os.path.join("data", f"{collection}_canonical_map.json")
    if not os.path.exists(map_file):
        # Build if not present
        data_file = os.path.join("data", collection, f"{collection}_processed.json")
        if os.path.exists(data_file):
            raw_map = build_canonical_map(data_file, map_file)
            return {int(k): set(v) for k, v in raw_map.items()}
        return {}

    with open(map_file, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    return {int(k): set(v) for k, v in raw_map.items()}


if __name__ == "__main__":
    bukhari_dataset = r"d:\model\data\bukhari\bukhari_processed.json"
    bukhari_map = r"d:\model\data\bukhari_canonical_map.json"
    build_canonical_map(bukhari_dataset, bukhari_map)
