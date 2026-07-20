# -*- coding: utf-8 -*-
"""
Dialectal Query Normalizer & Phrase Converter
=============================================
Transforms Egyptian colloquial queries, conversational filler words,
and dialectal expressions into clean, standardized Modern Standard Arabic (MSA)
search queries before entering the retrieval pipeline.
"""
import re
from typing import Dict, List
from backend.rag.search import normalize_arabic


# Colloquial Egyptian and conversational prefixes to strip or convert
DIALECTAL_PREFIXES = [
    r'^(?:عايز|عاوز|ممكن|ممكن تطلعلي|ممكن تجيبلي|قولي|قول لي|أبسط|شرح|اكتبلي|جيبلي)\s+',
    r'^(?:إيه|ايه|ايه هو|شنو|وش|ما هو|ماهو)\s+(?:الحديث|حديث)?\s*',
    r'^(?:الحديث|حديث)\s+(?:اللي|اللي بيقول|اللي بيتكلم عن|بتاع|عن|في|المشهور)\s+',
    r'^(?:هل ورد|هل فيه|في|هل هناك)\s+(?:حديث|آية|ايه)?\s*',
    r'^(?:حكم|كيف|طريقة|شرح|فضل|ثواب|أجر)\s+',
]

# Colloquial word mappings to MSA
DIALECT_MAPPINGS: Dict[str, str] = {
    "بتاع": "",
    "اللي": "",
    "عايز": "",
    "عاوز": "",
    "ممكن": "",
    "ربنا": "الله",
    "الرسول": "رسول الله",
    "النبي": "رسول الله",
    "سيدنا": "",
    "رضي الله عنه": "",
    "رضي الله عنها": "",
    "صلى الله عليه وسلم": "",
    "عليه الصلاة والسلام": "",
    "عليه السلام": "",
    "متفق عليه": "",
    "صحيح": "",
    "البخاري": "",
    "مسلم": "",
}


def normalize_query_dialect(query: str) -> str:
    """
    Strips dialectal prefixes, removes honorific fillers, and maps colloquial words
    to produce a clean canonical search query string.
    """
    if not query:
        return ""

    cleaned = query.strip()

    # Step 1: Strip dialectal prefixes
    for pattern in DIALECTAL_PREFIXES:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

    # Step 2: Replace dialect words with MSA equivalents
    words = cleaned.split()
    normalized_words = []
    for w in words:
        norm_w = normalize_arabic(w)
        if norm_w in DIALECT_MAPPINGS:
            mapped = DIALECT_MAPPINGS[norm_w]
            if mapped:
                normalized_words.append(mapped)
        else:
            normalized_words.append(w)

    result = " ".join(normalized_words).strip()
    return result if result else query.strip()
