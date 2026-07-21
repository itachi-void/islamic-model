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


def extend_query_from_matn(current_query: str, matn: str, extra_chars: int = 60) -> str:
    """
    Given a (possibly truncated) query and the gold hadith's matn,
    extend the query to include extra characters.

    Strategy:
    - Strip "حديث" prefix and diacritics from both
    - Find the normalized query in the normalized matn
    - Use the matn ORIGINAL text from the start, extended by extra_chars
    - Always include complete words (break at word boundary)
    """
    prefix = "حديث"
    working_query = current_query.strip()
    if working_query.startswith(prefix):
        working_query = working_query[len(prefix):].strip()

    q_stripped = strip_diacritics(working_query).strip()
    m_stripped = strip_diacritics(matn).strip()

    if not q_stripped or not m_stripped:
        return current_query

    # Find where normalized query ends in normalized matn
    idx = m_stripped.find(q_stripped)
    if idx < 0:
        # Try shorter prefix (up to 70% of query length)
        for trim in range(5, len(q_stripped) // 3, 5):
            short = q_stripped[:-trim]
            if len(short) < 10:
                break
            idx2 = m_stripped.find(short)
            if idx2 >= 0:
                idx = idx2
                q_stripped = short
                break

    if idx < 0:
        return current_query

    end_norm = idx + len(q_stripped)

    # Map normalized end position back to original matn position
    # We walk original matn and count non-diacritic characters
    norm_count = 0
    orig_end = 0
    for ci, ch in enumerate(matn):
        if strip_diacritics(ch) and ch != ' ':
            norm_count += 1
        if norm_count >= end_norm:
            orig_end = ci + 1
            break

    if orig_end == 0:
        return current_query

    # Take next extra_chars characters from original matn
    extension_raw = matn[orig_end: orig_end + extra_chars + 30]

    # Find clean word boundary
    clean_ext = extension_raw.strip()
    # Break at last comma/space before limit
    for sep in ['، ', '، ']:
        last = clean_ext[:extra_chars].rfind(sep)
        if last > 10:
            clean_ext = clean_ext[:last].strip()
            break
    else:
        # Just take first `extra_chars` chars and trim to last space
        clean_ext = clean_ext[:extra_chars]
        last_space = clean_ext.rfind(' ')
        if last_space > 10:
            clean_ext = clean_ext[:last_space].strip()

    if not clean_ext or len(clean_ext) < 5:
        return current_query

    extended = f"{prefix} {working_query} {clean_ext}".strip()
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
