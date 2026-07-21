# -*- coding: utf-8 -*-
"""
Failure Audit Tool — IRB-v1 Test Split
=======================================
Uses the same HadithSearchService as eval.py to ensure identical
retrieval behaviour.

Usage:
    python -X utf8 tools/failure_audit.py
"""
import sys, json, os
sys.path.insert(0, r"d:\model")

from backend.rag.hadith_search import HadithSearchService

BENCHMARK_PATH = r"d:\model\data\benchmarks\irb\v1\test.json"
TOP_K = 50


def load_benchmark(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", data) if isinstance(data, dict) else data


def main():
    print("Loading HadithSearchService...")
    service = HadithSearchService()

    print(f"Loading benchmark: {BENCHMARK_PATH}")
    cases = load_benchmark(BENCHMARK_PATH)
    print(f"Total cases: {len(cases)}\n")

    failures = []
    successes = []

    for i, case in enumerate(cases):
        query       = case.get("query", "")
        acceptable  = list(case.get("acceptable_answers", []))
        category    = case.get("category", "unknown")
        difficulty  = case.get("difficulty", "unknown")
        is_ood      = (case.get("answer_type", "") == "no_evidence"
                       or case.get("category", "") == "OOD Negative")

        if is_ood:
            continue

        try:
            resp = service.search(query, limit=TOP_K)
            retrieved_ids  = [str(doc.id) for doc in resp.documents]
            retrieved_hnums = [str(doc.metadata.get("hadith_number"))
                               for doc in resp.documents
                               if doc.metadata.get("hadith_number") is not None]
            all_retrieved  = retrieved_ids + retrieved_hnums
        except Exception as e:
            all_retrieved = []

        gold_set = set(str(a) for a in acceptable)

        hit5   = any(g in all_retrieved[:5] for g in gold_set)
        rank50 = next((all_retrieved.index(g) + 1
                       for g in gold_set if g in all_retrieved), None)

        if hit5:
            successes.append({"query": query, "category": category})
            continue

        # Categorize failure
        if rank50 is None:
            bucket = "retrieval_failure"
        else:
            bucket = "ranking_failure"

        # Top-5 for inspection
        top5_texts = []
        try:
            resp5 = service.search(query, limit=5)
            for doc in resp5.documents:
                top5_texts.append({
                    "id"   : str(doc.id),
                    "score": round(getattr(doc, "score", 0) or 0, 4),
                    "text" : doc.text[:140].strip(),
                    "book" : doc.metadata.get("book", ""),
                    "narrator": doc.metadata.get("narrator", ""),
                })
        except Exception:
            pass

        failures.append({
            "index"      : i + 1,
            "query"      : query,
            "gold_set"   : list(gold_set)[:5],
            "category"   : category,
            "difficulty" : difficulty,
            "bucket"     : bucket,
            "rank_in_50" : rank50,
            "top5"       : top5_texts,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    retrieval_fails = [f for f in failures if f["bucket"] == "retrieval_failure"]
    ranking_fails   = [f for f in failures if f["bucket"] == "ranking_failure"]

    print(f"\n{'='*72}")
    print(f"  FAILURE AUDIT REPORT — IRB-v1 Test Split")
    print(f"{'='*72}")
    print(f"  Total in-domain cases : {len(successes) + len(failures)}")
    print(f"  Hits@5 (successes)    : {len(successes)}")
    print(f"  Failures              : {len(failures)}")
    print(f"  ├─ Retrieval Failures : {len(retrieval_fails)}")
    print(f"  └─ Ranking Failures   : {len(ranking_fails)}")
    print()

    from collections import defaultdict
    by_cat = defaultdict(list)
    for f in failures:
        by_cat[f["category"]].append(f)

    print("  Failures by Category:")
    for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        r = sum(1 for i in items if i["bucket"] == "retrieval_failure")
        k = sum(1 for i in items if i["bucket"] == "ranking_failure")
        print(f"    {cat:<30} total={len(items):2d}  retrieval={r}  ranking={k}")

    print()
    print("  ── RETRIEVAL FAILURES (doc absent from Top-50) ──")
    for f in retrieval_fails:
        print(f"\n  [{f['index']:03d}] [{f['category']}] [{f['difficulty']}]")
        print(f"       Q    : {f['query']}")
        print(f"       Gold : {f['gold_set']}")
        if f["top5"]:
            t = f["top5"][0]
            print(f"       Top1 : {t['text'][:100]}  (score={t['score']})")

    print()
    print("  ── RANKING FAILURES (doc in Top-50 but rank>5) ──")
    for f in ranking_fails:
        print(f"\n  [{f['index']:03d}] [{f['category']}] [{f['difficulty']}]")
        print(f"       Q    : {f['query']}")
        print(f"       Gold : {f['gold_set']}  → rank #{f['rank_in_50']}")
        if f["top5"]:
            t = f["top5"][0]
            print(f"       Top1 : {t['text'][:100]}")

    # Save JSON
    report = {
        "split"           : "test",
        "total_in_domain" : len(successes) + len(failures),
        "hits_at_5"       : len(successes),
        "failure_count"   : len(failures),
        "retrieval_count" : len(retrieval_fails),
        "ranking_count"   : len(ranking_fails),
        "failures"        : failures,
    }
    out_path = r"d:\model\data\experiments\failure_audit_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Full report saved → {out_path}")


if __name__ == "__main__":
    main()
