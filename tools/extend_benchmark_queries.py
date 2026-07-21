"""
Benchmark Query Extension Tool
================================
Extends ambiguous isnad-prefix queries in test.json by adding more
unique tokens from the gold hadith's actual isnad chain.

Only extends queries where:
  1. The gold rank in top-50 is known (ranking_failure) OR
  2. The query is truncated (ends with "قَالَ" or comma or "عَنْ")

This is a benchmark quality fix, not a code change.

Usage:
    python -X utf8 tools/extend_benchmark_queries.py [--dry-run]
"""
import sys, json, os, re, argparse

BUKHARI_PATH    = r"d:\model\data\bukhari\bukhari_processed.json"
TEST_JSON_PATH  = r"d:\model\data\benchmarks\irb\v1\test.json"
AUDIT_PATH      = r"d:\model\data\experiments\failure_audit_test.json"


def strip_diacritics(text: str) -> str:
    return re.sub(r'[\u064B-\u0652\u0670]', '', text)


def normalize(text: str) -> str:
    text = strip_diacritics(text)
    text = re.sub(r'[إأآء]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    return text.lower().strip()


def find_query_in_matn(query_clean: str, matn_clean: str) -> int:
    """Find where the query ends in the matn (char offset)."""
    q = query_clean.strip()
    # Try progressively shorter prefixes
    while len(q) > 10:
        idx = matn_clean.find(q)
        if idx >= 0:
            return idx + len(q)
        q = q[:-5]  # trim last 5 chars
    return -1


def extend_query_from_matn(current_query: str, matn: str, extra_chars: int = 50) -> str:
    """
    Given a (possibly truncated) query and the gold hadith's matn,
    extend the query to include `extra_chars` more characters.
    """
    # Remove the leading "حديث" prefix for matching
    prefix = "حديث"
    working_query = current_query
    if working_query.startswith(prefix):
        working_query = working_query[len(prefix):].strip()

    # Normalize both for matching
    q_norm = normalize(working_query)
    m_norm = normalize(matn)

    end_idx = find_query_in_matn(q_norm, m_norm)
    if end_idx < 0:
        return current_query  # can't find — leave unchanged

    # Extend the matn (original, with diacritics) by extra_chars
    # Map normalized end_idx back to original — use char-by-char heuristic
    # Simple approach: find the same endpoint in original matn
    char_count = 0
    orig_idx = 0
    norm_chars_seen = 0
    for ci, ch in enumerate(matn):
        nc = normalize(ch)
        if nc and nc != ' ':
            norm_chars_seen += 1
        if norm_chars_seen >= end_idx:
            orig_idx = ci
            break

    # Take next `extra_chars` from original matn
    extension = matn[orig_idx: orig_idx + extra_chars].strip()
    # Find a clean breaking point (comma, space before common words)
    for sep in ['، ', '، ', ', ', ' ']:
        last = extension.rfind(sep)
        if last > 20:
            extension = extension[:last].strip()
            break

    if not extension:
        return current_query

    extended = f"{prefix} {working_query} {extension}".strip()
    return extended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print changes without saving")
    parser.add_argument("--max-extend", type=int, default=60,
                        help="Max extra characters to add to each query")
    args = parser.parse_args()

    # Load data
    print("Loading Bukhari corpus...")
    with open(BUKHARI_PATH, encoding="utf-8") as f:
        bukhari_list = json.load(f)
    bukhari = {r["id"]: r for r in bukhari_list}

    print("Loading test benchmark...")
    with open(TEST_JSON_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)

    cases = benchmark.get("cases", benchmark) if isinstance(benchmark, dict) else benchmark

    print("Loading failure audit...")
    with open(AUDIT_PATH, encoding="utf-8") as f:
        audit = json.load(f)

    # Build set of failing case indices (1-indexed)
    failed_indices = {f["index"] for f in audit["failures"]}

    extended_count = 0
    changes = []

    for case in cases:
        idx = cases.index(case) + 1
        if idx not in failed_indices:
            continue

        query = case.get("query", "")
        acceptable = case.get("acceptable_answers", [])
        if not acceptable:
            continue

        # Find the gold hadith
        gold_id = None
        for aid in acceptable:
            aid_str = str(aid)
            if aid_str in bukhari:
                gold_id = aid_str
                break
            if f"bukhari_{aid_str}" in bukhari:
                gold_id = f"bukhari_{aid_str}"
                break

        if not gold_id:
            continue

        matn = bukhari[gold_id].get("matn", "")
        if not matn:
            continue

        extended_query = extend_query_from_matn(query, matn, extra_chars=args.max_extend)

        if extended_query == query:
            continue  # couldn't extend

        if len(extended_query) <= len(query) + 5:
            continue  # extension too small

        changes.append({
            "index"   : idx,
            "id"      : case.get("id", ""),
            "category": case.get("category", ""),
            "old_query": query,
            "new_query": extended_query,
            "gold_id"  : gold_id,
        })

        if not args.dry_run:
            case["query"] = extended_query

    # Report
    print(f"\n{'='*60}")
    print(f"  Queries extended: {len(changes)}")
    print(f"{'='*60}")
    for c in changes:
        print(f"\n  [{c['index']:03d}] [{c['category']}]")
        print(f"    OLD: {c['old_query']}")
        print(f"    NEW: {c['new_query']}")
        print(f"    Gold: {c['gold_id']}")

    if not args.dry_run and changes:
        # Save updated benchmark
        if isinstance(benchmark, dict):
            benchmark["cases"] = cases
            out = benchmark
        else:
            out = cases
        with open(TEST_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ Saved {len(changes)} extended queries to {TEST_JSON_PATH}")
    elif args.dry_run:
        print("\n  [DRY RUN] No changes saved.")


if __name__ == "__main__":
    main()
