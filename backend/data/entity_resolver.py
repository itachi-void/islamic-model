# -*- coding: utf-8 -*-
"""
Entity Resolution Layer for Islamic Queries
============================================
Resolves kunyas, titles, honorifics, aliases, and entity references to their
canonical searchable forms BEFORE retrieval. Uses the Islamic Knowledge Graph
(IKG) as the primary source, with extensive fallback mappings.

Designed to improve BM25 and semantic search by expanding entity references
with canonical names and alternate aliases.

Usage:
    resolver = EntityResolver()
    resolved = resolver.expand_query("من هو النبي؟")
    # Returns: "من هو النبي؟ محمد ﷺ محمد رسول الله"
"""
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Set
from backend.rag.search import normalize_arabic

IKG_PATH = r"d:\model\data\islamic_knowledge_graph.json"

# Diacritics removal for matching
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")


def strip_diacritics(text: str) -> str:
    if not text:
        return ""
    return ARABIC_DIACRITICS.sub("", text)


class EntityResolver:
    """
    Multi-source entity resolution for Islamic Arabic queries.
    
    Resolution strategy:
    1. Load the Islamic Knowledge Graph (IKG) for structured entities
    2. Apply hard-coded comprehensive mappings for major entities (prophets, 
       companions, narrators, places, books, battles, hadith titles)
    3. For each query, find all matching entities and append their canonical
       searchable forms
    4. Special rules: "النبي" → "محمد ﷺ" only if no other prophet is mentioned
    """

    def __init__(self, ikg_path: str = IKG_PATH):
        self.ikg_path = ikg_path
        # Primary mapping: alias → (canonical_name, entity_type)
        self.mappings: Dict[str, Tuple[str, str]] = {}
        self._load_ikg()
        self._load_hardcoded_mappings()

    def _load_ikg(self):
        """Load entities from the Islamic Knowledge Graph JSON."""
        if not os.path.exists(self.ikg_path):
            return

        try:
            with open(self.ikg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        entities = data.get("entities", {})

        for category, items in entities.items():
            for entity_id, record in items.items():
                canonical = record.get("canonical_name") or record.get("canonical_title", "")
                aliases = record.get("aliases", [])

                if not canonical:
                    continue

                for alias in aliases:
                    normalized_alias = normalize_arabic(strip_diacritics(alias)).strip()
                    if normalized_alias and len(normalized_alias) >= 2:
                        existing = self.mappings.get(normalized_alias)
                        if existing is None or len(canonical) > len(existing[0]):
                            self.mappings[normalized_alias] = (canonical, category)

    def _load_hardcoded_mappings(self):
        """
        Hard-coded comprehensive entity mappings covering major Islamic entities
        beyond what the IKG currently contains.
        
        Format: {normalized_alias: (canonical_name, entity_type)}
        
        These are loaded AFTER the IKG so they serve as fallback/override.
        """
        hardcoded: Dict[str, Tuple[str, str]] = {
            # ===== PROPHETS =====
            # Muhammad ﷺ
            "النبي": ("محمد ﷺ", "prophet"),
            "رسول الله": ("محمد ﷺ", "prophet"),
            "رسول اللّه": ("محمد ﷺ", "prophet"),
            "محمد": ("محمد ﷺ", "prophet"),
            "مصطفى": ("محمد ﷺ", "prophet"),
            "أحمد": ("محمد ﷺ", "prophet"),
            "حبيب الله": ("محمد ﷺ", "prophet"),
            "خاتم الانبياء": ("محمد ﷺ", "prophet"),
            "خاتم النبيين": ("محمد ﷺ", "prophet"),
            "سيد المرسلين": ("محمد ﷺ", "prophet"),
            "سيد ولد ادم": ("محمد ﷺ", "prophet"),
            # Other prophets (for exclusion logic)
            "نبي الله موسى": ("موسى عليه السلام", "prophet"),
            "موسى بن عمران": ("موسى عليه السلام", "prophet"),
            "كليم الله": ("موسى عليه السلام", "prophet"),
            "نبي الله عيسى": ("عيسى عليه السلام", "prophet"),
            "عيسى بن مريم": ("عيسى عليه السلام", "prophet"),
            "المسيح": ("عيسى عليه السلام", "prophet"),
            "نبي الله إبراهيم": ("إبراهيم عليه السلام", "prophet"),
            "خليل الله": ("إبراهيم عليه السلام", "prophet"),
            "أبو الانبياء": ("إبراهيم عليه السلام", "prophet"),
            "نبي الله نوح": ("نوح عليه السلام", "prophet"),
            "نبي الله يوسف": ("يوسف عليه السلام", "prophet"),
            "نبي الله سليمان": ("سليمان عليه السلام", "prophet"),
            "نبي الله داود": ("داود عليه السلام", "prophet"),
            "نبي الله آدم": ("آدم عليه السلام", "prophet"),
            "ابو البشر": ("آدم عليه السلام", "prophet"),
            "نبي الله صالح": ("صالح عليه السلام", "prophet"),
            "نبي الله هود": ("هود عليه السلام", "prophet"),
            "نبي الله لوط": ("لوط عليه السلام", "prophet"),
            "نبي الله شعيب": ("شعيب عليه السلام", "prophet"),
            "نبي الله إسماعيل": ("إسماعيل عليه السلام", "prophet"),
            "نبي الله إسحاق": ("إسحاق عليه السلام", "prophet"),
            "نبي الله يعقوب": ("يعقوب عليه السلام", "prophet"),
            "إسرائيل": ("يعقوب عليه السلام", "prophet"),
            "نبي الله يحيى": ("يحيى عليه السلام", "prophet"),
            "نبي الله زكريا": ("زكريا عليه السلام", "prophet"),
            "نبي الله أيوب": ("أيوب عليه السلام", "prophet"),
            "نبي الله يونس": ("يونس عليه السلام", "prophet"),
            "ذو النون": ("يونس عليه السلام", "prophet"),
            "نبي الله إلياس": ("إلياس عليه السلام", "prophet"),
            "نبي الله اليسع": ("اليسع عليه السلام", "prophet"),
            "نبي الله ذو الكفل": ("ذو الكفل عليه السلام", "prophet"),

            # ===== AL-KHULAFA' AL-RASHIDUN =====
            "الصديق": ("أبو بكر الصديق", "companion"),
            "أبو بكر": ("أبو بكر الصديق", "companion"),
            "ابو بكر": ("أبو بكر الصديق", "companion"),
            "أبي بكر": ("أبو بكر الصديق", "companion"),
            "ابن أبي قحافة": ("أبو بكر الصديق", "companion"),
            "عتيق": ("أبو بكر الصديق", "companion"),
            "الفاروق": ("عمر بن الخطاب", "companion"),
            "عمر": ("عمر بن الخطاب", "companion"),
            "عمر بن الخطاب": ("عمر بن الخطاب", "companion"),
            "أبو حفص": ("عمر بن الخطاب", "companion"),
            "ذو النورين": ("عثمان بن عفان", "companion"),
            "عثمان": ("عثمان بن عفان", "companion"),
            "عثمان بن عفان": ("عثمان بن عفان", "companion"),
            "أبو عبد الله": ("عثمان بن عفان", "companion"),
            "علي": ("علي بن أبي طالب", "companion"),
            "علي بن أبي طالب": ("علي بن أبي طالب", "companion"),
            "علي بن ابي طالب": ("علي بن أبي طالب", "companion"),
            "أبو تراب": ("علي بن أبي طالب", "companion"),
            "أسد الله": ("علي بن أبي طالب", "companion"),
            "إمام المتقين": ("علي بن أبي طالب", "companion"),
            "أبو الحسن": ("علي بن أبي طالب", "companion"),

            # ===== MAJOR SAHABA (COMPANIONS) =====
            # Abu Hurayrah
            "أبو هريرة": ("أبو هريرة", "companion"),
            "ابو هريرة": ("أبو هريرة", "companion"),
            "أبي هريرة": ("أبو هريرة", "companion"),
            "عبد الرحمن بن صخر": ("أبو هريرة", "companion"),
            "الدوسي": ("أبو هريرة", "companion"),
            # Aisha
            "عائشة": ("عائشة بنت أبي بكر", "companion"),
            "أم المؤمنين": ("عائشة بنت أبي بكر", "companion"),
            "الصديقة بنت الصديق": ("عائشة بنت أبي بكر", "companion"),
            "أم عبد الله": ("عائشة بنت أبي بكر", "companion"),
            # Ibn Abbas
            "ابن عباس": ("عبد الله بن عباس", "companion"),
            "عبد الله بن عباس": ("عبد الله بن عباس", "companion"),
            "حبر الأمة": ("عبد الله بن عباس", "companion"),
            "ترجمان القرآن": ("عبد الله بن عباس", "companion"),
            "بحر العلم": ("عبد الله بن عباس", "companion"),
            # Ibn Umar
            "ابن عمر": ("عبد الله بن عمر", "companion"),
            "عبد الله بن عمر": ("عبد الله بن عمر", "companion"),
            # Anas
            "أنس بن مالك": ("أنس بن مالك", "companion"),
            "انس بن مالك": ("أنس بن مالك", "companion"),
            "خادم رسول الله": ("أنس بن مالك", "companion"),
            # Abu Musa
            "أبو موسى الأشعري": ("أبو موسى الأشعري", "companion"),
            "ابو موسى الاشعري": ("أبو موسى الأشعري", "companion"),
            "عبد الله بن قيس": ("أبو موسى الأشعري", "companion"),
            # Abu Said al-Khudri
            "أبو سعيد الخدري": ("أبو سعيد الخدري", "companion"),
            "ابو سعيد الخدري": ("أبو سعيد الخدري", "companion"),
            "سعد بن مالك بن سنان": ("أبو سعيد الخدري", "companion"),
            # Jabir
            "جابر بن عبد الله": ("جابر بن عبد الله", "companion"),
            "جابر بن عبد الله الأنصاري": ("جابر بن عبد الله", "companion"),
            # Abu Dharr
            "أبو ذر الغفاري": ("أبو ذر الغفاري", "companion"),
            "ابو ذر الغفاري": ("أبو ذر الغفاري", "companion"),
            "جندب بن جنادة": ("أبو ذر الغفاري", "companion"),
            # Ibn Masud
            "عبد الله بن مسعود": ("عبد الله بن مسعود", "companion"),
            "ابن مسعود": ("عبد الله بن مسعود", "companion"),
            "ابن أم عبد": ("عبد الله بن مسعود", "companion"),
            # Muadh
            "معاذ بن جبل": ("معاذ بن جبل", "companion"),
            "أعلم الأمة بالحلال والحرام": ("معاذ بن جبل", "companion"),
            # Bilal
            "بلال": ("بلال بن رباح", "companion"),
            "بلال بن رباح": ("بلال بن رباح", "companion"),
            "مؤذن الرسول": ("بلال بن رباح", "companion"),
            # Hamza
            "حمزة": ("حمزة بن عبد المطلب", "companion"),
            "حمزة بن عبد المطلب": ("حمزة بن عبد المطلب", "companion"),
            "أسد الله": ("حمزة بن عبد المطلب", "companion"),
            "سيد الشهداء": ("حمزة بن عبد المطلب", "companion"),
            # Khalid
            "خالد بن الوليد": ("خالد بن الوليد", "companion"),
            "سيف الله المسلول": ("خالد بن الوليد", "companion"),
            # Others
            "أبو أيوب الأنصاري": ("أبو أيوب الأنصاري", "companion"),
            "خالد بن زيد": ("أبو أيوب الأنصاري", "companion"),
            "سلمان الفارسي": ("سلمان الفارسي", "companion"),
            "سلمان": ("سلمان الفارسي", "companion"),
            "صهيب": ("صهيب الرومي", "companion"),
            "صهيب الرومي": ("صهيب الرومي", "companion"),
            "أبو الدرداء": ("أبو الدرداء", "companion"),
            "عويمر بن زيد": ("أبو الدرداء", "companion"),
            "أبو عبيدة بن الجراح": ("أبو عبيدة بن الجراح", "companion"),
            "أمين الأمة": ("أبو عبيدة بن الجراح", "companion"),
            "الزبير بن العوام": ("الزبير بن العوام", "companion"),
            "حواري الرسول": ("الزبير بن العوام", "companion"),
            "طلحة بن عبيد الله": ("طلحة بن عبيد الله", "companion"),
            "طلحة": ("طلحة بن عبيد الله", "companion"),
            "سعد بن أبي وقاص": ("سعد بن أبي وقاص", "companion"),
            "سعد": ("سعد بن أبي وقاص", "companion"),
            "أبو طلحة": ("أبو طلحة الأنصاري", "companion"),
            "زيد بن حارثة": ("زيد بن حارثة", "companion"),
            "أسامة بن زيد": ("أسامة بن زيد", "companion"),
            "حبيب بن حبيب بن مروان": ("أسامة بن زيد", "companion"),

            # ===== WOMEN (SAHABIYYAT) =====
            "فاطمة": ("فاطمة الزهراء", "companion"),
            "فاطمة الزهراء": ("فاطمة الزهراء", "companion"),
            "سيدة نساء العالمين": ("فاطمة الزهراء", "companion"),
            "خديجة": ("خديجة بنت خويلد", "companion"),
            "خديجة بنت خويلد": ("خديجة بنت خويلد", "companion"),
            "حفصة": ("حفصة بنت عمر", "companion"),
            "حفصة بنت عمر": ("حفصة بنت عمر", "companion"),
            "أم سلمة": ("أم سلمة", "companion"),
            "هند بنت أبي أمية": ("أم سلمة", "companion"),
            "أم حبيبة": ("أم حبيبة", "companion"),
            "رملة بنت أبي سفيان": ("أم حبيبة", "companion"),
            "ماريا القبطية": ("ماريا القبطية", "companion"),

            # ===== PLACES =====
            "مكة": ("مكة المكرمة", "place"),
            "مكة المكرمة": ("مكة المكرمة", "place"),
            "أم القرى": ("مكة المكرمة", "place"),
            "المدينة": ("المدينة المنورة", "place"),
            "المدينة المنورة": ("المدينة المنورة", "place"),
            "طيبة": ("المدينة المنورة", "place"),
            "يثرب": ("المدينة المنورة", "place"),
            "بيت المقدس": ("بيت المقدس", "place"),
            "القدس": ("بيت المقدس", "place"),
            "المسجد الأقصى": ("المسجد الأقصى", "place"),
            "البيت الحرام": ("الكعبة", "place"),
            "الكعبة": ("الكعبة", "place"),
            "الكعبة المشرفة": ("الكعبة", "place"),
            "المسجد الحرام": ("المسجد الحرام", "place"),
            "المسجد النبوي": ("المسجد النبوي", "place"),
            "مسجد النبي": ("المسجد النبوي", "place"),
            "غار حراء": ("غار حراء", "place"),
            "حراء": ("غار حراء", "place"),
            "جبل النور": ("غار حراء", "place"),
            "غار ثور": ("غار ثور", "place"),
            "جبل أحد": ("جبل أحد", "place"),
            "أحد": ("جبل أحد", "place"),
            "جبل عرفات": ("جبل عرفات", "place"),
            "عرفات": ("جبل عرفات", "place"),
            "منى": ("منى", "place"),
            "مزدلفة": ("مزدلفة", "place"),
            "بدر": ("بدر", "place"),
            "صفا": ("الصفا", "place"),
            "الصفا": ("الصفا", "place"),
            "مروة": ("المروة", "place"),
            "المروة": ("المروة", "place"),
            "الطائف": ("الطائف", "place"),
            "تبوك": ("تبوك", "place"),
            "خيبر": ("خيبر", "place"),
            "الحجاز": ("الحجاز", "place"),
            "نجد": ("نجد", "place"),
            "الشام": ("الشام", "place"),
            "مصر": ("مصر", "place"),
            "اليمن": ("اليمن", "place"),
            "العراق": ("العراق", "place"),
            "فلسطين": ("فلسطين", "place"),

            # ===== FAMOUS MOSQUES =====
            "مسجد قباء": ("مسجد قباء", "place"),
            "مسجد الضرار": ("مسجد الضرار", "place"),
            "مسجد القبلتين": ("مسجد القبلتين", "place"),

            # ===== BATTLES (GHAZAWAT) =====
            "غزوة بدر": ("غزوة بدر الكبرى", "battle"),
            "غزوة بدر الكبرى": ("غزوة بدر الكبرى", "battle"),
            "بدر": ("غزوة بدر الكبرى", "battle"),
            "يوم الفرقان": ("غزوة بدر الكبرى", "battle"),
            "غزوة أحد": ("غزوة أحد", "battle"),
            "غزوة احد": ("غزوة أحد", "battle"),
            "غزوة الخندق": ("غزوة الخندق", "battle"),
            "غزوة الأحزاب": ("غزوة الخندق", "battle"),
            "غزوة بني المصطلق": ("غزوة بني المصطلق", "battle"),
            "غزوة خيبر": ("غزوة خيبر", "battle"),
            "غزوة حنين": ("غزوة حنين", "battle"),
            "غزوة الطائف": ("غزوة الطائف", "battle"),
            "غزوة تبوك": ("غزوة تبوك", "battle"),
            "فتح مكة": ("فتح مكة", "battle"),
            "الفتح": ("فتح مكة", "battle"),
            "غزوة بدر الثانية": ("غزوة بدر الصغرى", "battle"),
            "غزوة بني قريظة": ("غزوة بني قريظة", "battle"),
            "غزوة بني النضير": ("غزوة بني النضير", "battle"),
            "سرية مؤتة": ("سرية مؤتة", "battle"),
            "مؤتة": ("سرية مؤتة", "battle"),

            # ===== HADITH TITLES =====
            "حديث النية": ("إنما الأعمال بالنيات", "hadith_title"),
            "حديث الأعمال بالنيات": ("إنما الأعمال بالنيات", "hadith_title"),
            "حديث إنما الأعمال بالنيات": ("إنما الأعمال بالنيات", "hadith_title"),
            "إنما الأعمال بالنيات": ("إنما الأعمال بالنيات", "hadith_title"),
            "حديث جبريل": ("حديث جبريل عليه السلام", "hadith_title"),
            "حديث سؤال جبريل": ("حديث جبريل عليه السلام", "hadith_title"),
            "بني الإسلام على خمس": ("بني الإسلام على خمس", "hadith_title"),
            "حديث بني الإسلام على خمس": ("بني الإسلام على خمس", "hadith_title"),
            "أركان الإسلام": ("بني الإسلام على خمس", "hadith_title"),
            "حديث الدين النصيحة": ("الدين النصيحة", "hadith_title"),
            "الدين النصيحة": ("الدين النصيحة", "hadith_title"),
            "حديث من رأى منكم منكرا": ("من رأى منكم منكرا", "hadith_title"),
            "حديث الإفك": ("حديث الإفك", "hadith_title"),
            "حديث الشفاعة": ("حديث الشفاعة", "hadith_title"),
            "حديث البطاقة": ("حديث البطاقة", "hadith_title"),
            "حديث صلاة الضحى": ("صلاة الضحى", "hadith_title"),
            "حديث شعب الإيمان": ("شعب الإيمان", "hadith_title"),
            "حديث شعب الايمان": ("شعب الإيمان", "hadith_title"),
            "حديث النزول": ("حديث النزول", "hadith_title"),
            "حديث الغار": ("حديث الغار", "hadith_title"),

            # ===== QURAN REFERENCES =====
            "القرآن": ("القرآن الكريم", "quran"),
            "القرآن الكريم": ("القرآن الكريم", "quran"),
            "الكتاب": ("القرآن الكريم", "quran"),
            "الفرقان": ("القرآن الكريم", "quran"),
            "سورة": ("سورة", "quran"),
            "آية": ("آية", "quran"),
            "اية": ("آية", "quran"),
            "آيات": ("آيات", "quran"),
        }

        for alias, (canonical, entity_type) in hardcoded.items():
            normalized = normalize_arabic(strip_diacritics(alias)).strip()
            if normalized and len(normalized) >= 2:
                # Only set if not already set by IKG with a longer canonical
                existing = self.mappings.get(normalized)
                if existing is None:
                    self.mappings[normalized] = (canonical, entity_type)

    def resolve(self, term: str) -> Optional[str]:
        """Resolve a single entity term to its canonical name."""
        if not term:
            return None
        norm = normalize_arabic(strip_diacritics(term.strip()))
        if not norm:
            return None
        result = self.mappings.get(norm)
        return result[0] if result else None

    def expand_query(self, query: str) -> str:
        """
        Expand a query string by appending canonical entity names.
        
        Strategy:
        1. Normalize query for matching
        2. Find all matching entities (sorted by longest match first)
        3. Apply special rules:
           - "النبي"/"رسول الله" → "محمد ﷺ" only if no other prophet named
           - Don't duplicate terms already in query
        4. Append unique canonical names to the original query
        
        Returns the original query with appended expansions.
        """
        if not query:
            return query

        norm_q = normalize_arabic(strip_diacritics(query))

        # Check if another prophet is explicitly named (for the "النبي" rule)
        other_prophet_names = [
            "موسى", "عيسى", "ابراهيم", "نوح", "يوسف", "سليمان",
            "داود", "ادم", "صالح", "هود", "لوط", "شعيب", "اسماعيل",
            "اسحاق", "يعقوب", "يحيى", "زكريا", "ايوب", "يونس",
            "الياس", "اليسع", "ذو الكفل"
        ]
        # Only count as "other prophet" if preceded by a prophet indicator
        has_other_prophet = False
        for p_name in other_prophet_names:
            # Check for "نبي الله X" or "نبي X" patterns
            if f"نبي {p_name}" in norm_q or f"نبي الله {p_name}" in norm_q:
                has_other_prophet = True
                break
            # Check for the name preceded by "الله" or at start
            if p_name in norm_q:
                has_other_prophet = True
                break

        # Sort keys by length (longest first) for greedy matching
        sorted_keys = sorted(self.mappings.keys(), key=len, reverse=True)

        matched_values: List[str] = []
        matched_keys: Set[str] = set()
        already_in_query: Set[str] = set()

        # Pre-populate with terms already in the query (normalized)
        for word in norm_q.split():
            if len(word) >= 3:
                already_in_query.add(word)

        for key in sorted_keys:
            # Special rule: don't expand "النبي"/"رسول الله" if another prophet is mentioned
            if key in ["النبي", "رسول الله", "محمد", "أحمد", "مصطفى", "حبيب الله"]:
                if has_other_prophet:
                    continue

            # Check if key is in normalized query
            if key in norm_q:
                # Avoid overlapping matches
                is_subsumed = False
                for mk in matched_keys:
                    if key in mk or mk in key:
                        is_subsumed = True
                        break
                if not is_subsumed:
                    val, etype = self.mappings[key]
                    # Only add if value not already in query
                    val_norm = normalize_arabic(strip_diacritics(val))
                    if val_norm not in norm_q and val not in already_in_query:
                        matched_values.append(val)
                        already_in_query.add(val_norm)
                    matched_keys.add(key)

        if matched_values:
            # Deduplicate while preserving order
            unique_values = []
            seen = set()
            for v in matched_values:
                v_norm = normalize_arabic(strip_diacritics(v))
                if v_norm not in seen:
                    seen.add(v_norm)
                    unique_values.append(v)
            
            if unique_values:
                return query + " " + " ".join(unique_values)

        return query
