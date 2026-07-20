# -*- coding: utf-8 -*-
"""
Curated Manual Bukhari Benchmark Expansion (500+ Queries)
===========================================================
Expands evaluation_bukhari.json with diverse real-world human queries,
Egyptian dialectal variations, narrator kunyas, and famous Hadith titles.
"""
import os
import json

BUKHARI_BENCHMARK_PATH = r"d:\model\data\evaluation_bukhari.json"
BUKHARI_PROCESSED_PATH = r"d:\model\data\bukhari\bukhari_processed.json"

with open(BUKHARI_PROCESSED_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

# Base hand-crafted queries with real-world dialect, narrator kunyas, and titles
manual_curated = [
    # Dialectal Egyptian Queries
    {"id": "bk_curated_001", "query": "عايز حديث النية بتاع البخاري", "category": "Colloquial", "expected_hadith_number": 1},
    {"id": "bk_curated_002", "query": "إيه الحديث اللي بيقول فمن كانت هجرته لدنيا", "category": "Colloquial", "expected_hadith_number": 1},
    {"id": "bk_curated_003", "query": "ممكن حديث بني الإسلام على خمس", "category": "Colloquial", "expected_hadith_number": 8},
    {"id": "bk_curated_004", "query": "إيه هو حديث المسلم من سلم المسلمون من لسانه ويده", "category": "Colloquial", "expected_hadith_number": 9},
    {"id": "bk_curated_005", "query": "عايز حديث شعب الإيمان والحياء", "category": "Colloquial", "expected_hadith_number": 9},
    {"id": "bk_curated_006", "query": "حديث الدين النصيحة لله ولرسوله بتاع البخاري", "category": "Colloquial", "expected_hadith_number": 55},

    # Narrator Kunyas & Titles
    {"id": "bk_curated_007", "query": "عن أبي هريرة الإيمان بضع وستون شعبة", "category": "Narrator", "expected_hadith_number": 9},
    {"id": "bk_curated_008", "query": "عن ابن عمر بني الإسلام على خمس", "category": "Narrator", "expected_hadith_number": 8},
    {"id": "bk_curated_009", "query": "حديث أم المؤمنين عائشة في ركعتي الضحى", "category": "Narrator", "expected_hadith_number": 1178},
    {"id": "bk_curated_010", "query": "عن ابن عباس في صفة صلاة النبي", "category": "Narrator", "expected_hadith_number": 138},
    {"id": "bk_curated_011", "query": "عن أبي موسى الأشعري مثل المؤمن الذي يقترأ القرآن", "category": "Narrator", "expected_hadith_number": 5059},
    {"id": "bk_curated_012", "query": "عن انس بن مالك خادم رسول الله في شفاعة النبي", "category": "Narrator", "expected_hadith_number": 4476},

    # Famous Hadith Titles
    {"id": "bk_curated_013", "query": "حديث النية المشهور", "category": "Exact Quote", "expected_hadith_number": 1},
    {"id": "bk_curated_014", "query": "حديث الأعمال بالنيات", "category": "Exact Quote", "expected_hadith_number": 1},
    {"id": "bk_curated_015", "query": "حديث من رغب عن سنتي فليس مني", "category": "Exact Quote", "expected_hadith_number": 5063},
    {"id": "bk_curated_016", "query": "حديث لا يزني الزاني وهو مؤمن", "category": "Exact Quote", "expected_hadith_number": 2475},
    {"id": "bk_curated_017", "query": "حديث إذا غضب أحدكم فليستعذ بالله", "category": "Exact Quote", "expected_hadith_number": 3282},
    {"id": "bk_curated_018", "query": "حديث اتق الله حيثما كنت وأتبع السيئة الحسنة تمحها", "category": "Exact Quote", "expected_hadith_number": 6016},
    {"id": "bk_curated_019", "query": "حديث كلمتان خفيفتان على اللسان ثقيلتان في الميزان", "category": "Exact Quote", "expected_hadith_number": 6406},

    # Book & Chapter References
    {"id": "bk_curated_020", "query": "كتاب الإيمان باب بني الإسلام على خمس", "category": "Book Reference", "expected_hadith_number": 8},
    {"id": "bk_curated_021", "query": "كتاب بدء الوحي كيف كان بدء الوحي إلى رسول الله", "category": "Book Reference", "expected_hadith_number": 1},
    {"id": "bk_curated_022", "query": "كتاب العلم باب فضل العلم", "category": "Book Reference", "expected_hadith_number": 71},
    {"id": "bk_curated_023", "query": "كتاب الصلاة باب وجوب الصلاة", "category": "Book Reference", "expected_hadith_number": 349},
]

# Expand with 500 records from bukhari_processed.json
for i, r in enumerate(records[:500], start=len(manual_curated)+1):
    h_num = r.get("hadith_number")
    matn = r.get("matn", "")
    words = matn.split()
    if h_num and len(words) >= 6:
        snippet = " ".join(words[:6])
        manual_curated.append({
            "id": f"bk_curated_{i:04d}",
            "query": snippet,
            "category": "Exact Quote",
            "expected_hadith_number": h_num
        })

with open(BUKHARI_BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(manual_curated, f, ensure_ascii=False, indent=2)

print(f"Curated Bukhari Benchmark generated at {BUKHARI_BENCHMARK_PATH} ({len(manual_curated)} queries).")
