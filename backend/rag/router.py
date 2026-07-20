# -*- coding: utf-8 -*-
import re
from typing import Literal

DomainRoute = Literal[
    "quran",
    "hadith",
    "tafsir",
    "asbab_nuzul",
    "fiqh",
    "aqidah",
    "biography",
    "general",
    "comparison",
    "hybrid"
]


class QueryRouter:
    """
    Multi-Domain Intent Classifier & Router.
    Classifies incoming user queries into 10 structured Islamic knowledge intents.
    """

    PATTERNS = {
        "asbab_nuzul": [r"سبب\s+النزول", r"لماذا\s+نزلت", r"فيمن\s+نزلت", r"أسباب\s+النزول"],
        "comparison": [r"الفرق\s+بين", r"مقارنة", r"اختلاف\s+المذاهب", r"بين\s+الحنفي\s+والشافعي"],
        "tafsir": [r"تفسير", r"شرح\s+آية", r"معنى\s+قوله\s+تعالى", r"تفسير\s+سورة", r"ابن\s+كثير"],
        "fiqh": [r"حكم", r"حلال", r"حرام", r"واجب", r"مستحب", r"مكروه", r"فتوى", r"المذهب", r"طهارة", r"وضوء", r"تيمم"],
        "aqidah": [r"عقيدة", r"توحيد", r"أسماء\s+الله", r"صفات\s+الله", r"إيمان", r"شرك", r"قدر"],
        "biography": [r"سيرة", r"غزوة", r"صحابي", r"صحابة", r"هجرة", r"خلافة"],
        "quran": [r"قال\s+الله", r"الله\s+تعالى", r"سورة", r"آية", r"اية", r"القرآن", r"قرآن", r"كتاب\s+الله"],
        "hadith": [r"قال\s+رسول\s+الله", r"قال\s+النبي", r"عن\s+النبي", r"حديث", r"البخاري", r"صحيح", r"رواه", r"مسلم", r"عن\s+أبي"],
    }

    def route(self, query: str) -> DomainRoute:
        if not query or not query.strip():
            return "hybrid"

        text = query.strip()

        # Check in priority order
        for intent in ["asbab_nuzul", "comparison", "tafsir", "fiqh", "aqidah", "biography"]:
            patterns = self.PATTERNS[intent]
            if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
                return intent

        is_quran = any(re.search(pat, text, re.IGNORECASE) for pat in self.PATTERNS["quran"])
        is_hadith = any(re.search(pat, text, re.IGNORECASE) for pat in self.PATTERNS["hadith"])

        if is_quran and not is_hadith:
            return "quran"
        if is_hadith and not is_quran:
            return "hadith"

        return "hybrid"
