# -*- coding: utf-8 -*-
"""
Unit tests for Entity Resolution Layer.
Tests core entity resolution functionality WITHOUT modifying any benchmarks or eval files.
"""
import sys
import os
sys.path.insert(0, r"d:\model")

from backend.data.entity_resolver import EntityResolver


def test_entity_resolver():
    resolver = EntityResolver()
    passed = 0
    failed = 0
    results = []

    def check(test_name, query, expected_contains=None, should_not_contain=None):
        nonlocal passed, failed
        result = resolver.expand_query(query)
        
        # Verify original text is preserved
        assert query in result, f"Original query not preserved in: {result}"
        
        if expected_contains:
            for exp in expected_contains:
                if exp not in result:
                    results.append(f"  FAIL [{test_name}]: expected '{exp}' in result, got: {result[:100]}")
                    failed += 1
                    return False
        
        if should_not_contain:
            for forbidden in should_not_contain:
                if forbidden in result:
                    results.append(f"  FAIL [{test_name}]: found forbidden '{forbidden}' in result: {result[:100]}")
                    failed += 1
                    return False
        
        passed += 1
        results.append(f"  PASS [{test_name}]: {result[:90]}")
        return True

    results.append("\n=== ENTITY RESOLVER UNIT TESTS ===\n")

    # 1. Prophet - Basic
    check("Nabi",
          "من هو النبي",
          ["محمد"])

    # 2. Prophet - Not when other prophet named
    check("Nabi Musa not expanded",
          "من هو نبي الله موسى",
          should_not_contain=["محمد"])

    # 3. Prophet - Detailed
    check("Rasul Allah",
          "حديث عن رسول الله",
          ["محمد"])

    # 4. Khulafa
    check("Al-Siddiq",
          "الصديق",
          ["أبو بكر الصديق"])

    check("Al-Faruq",
          "الفاروق",
          ["عمر بن الخطاب"])

    check("Dhu al-Nurayn",
          "ذو النورين",
          ["عثمان بن عفان"])

    check("Abu Turab",
          "أبو تراب",
          ["علي بن أبي طالب"])

    # 5. Places
    check("Taybah",
          "طيبة",
          ["المدينة المنورة"])

    check("Bayt al-Haram",
          "البيت الحرام",
          ["الكعبة"])

    # 6. Companions
    check("Abu Hurayrah",
          "أبو هريرة",
          ["أبو هريرة"])

    check("Ibn Abbas",
          "ابن عباس",
          ["عبد الله بن عباس"])

    check("Ibn Umar",
          "ابن عمر",
          ["عبد الله بن عمر"])

    # 7. Hadith Titles
    check("Hadith al-Niyyah",
          "عايز الحديث بتاع إنما الأعمال بالنيات في البخاري",
          ["إنما الأعمال بالنيات"])

    # 8. Multiple entities in one query
    check("Multiple entities",
          "النبي في المدينة",
          ["محمد", "المدينة المنورة"])

    # 9. Prophet only when alone
    check("Prophet exclusion",
          "نبي الله إبراهيم خليل الله",
          should_not_contain=["محمد"])

    # 10. Preserve original
    check("Preserve original",
          "أبو هريرة عن رسول الله",
          ["محمد", "أبو هريرة"])

    # 11. Edge case - empty
    r = resolver.expand_query("")
    if r == "":
        passed += 1
        results.append("  PASS [Empty query]")
    else:
        failed += 1
        results.append(f"  FAIL [Empty query]: got '{r}'")

    # 12. Edge case - no entities
    r = resolver.expand_query("حديث حدثنا فلان قال حدثنا")
    if r == "حديث حدثنا فلان قال حدثنا":
        passed += 1
        results.append("  PASS [No entities - unchanged]")
    else:
        failed += 1
        results.append(f"  FAIL [No entities]: got '{r}'")

    results.append(f"\n=== Results: {passed} passed, {failed} failed ===")

    # Write to a file to avoid encoding issues
    with open("temp_entity_test_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    
    print("Test results written to temp_entity_test_results.txt")
    return failed == 0


if __name__ == "__main__":
    success = test_entity_resolver()
    sys.exit(0 if success else 1)
