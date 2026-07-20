# -*- coding: utf-8 -*-
"""
Sahih Muslim Dataset Loader & Document Builder
================================================
Loads processed Sahih Muslim Hadith records as BaseDocument objects.
"""
import json
import os
from typing import List, Optional, Dict, Any
from backend.domain.document import BaseDocument

MUSLIM_DATA_DIR = r"d:\model\data\muslim"
MUSLIM_PROCESSED_PATH = os.path.join(MUSLIM_DATA_DIR, "muslim_processed.json")


def load_muslim_documents() -> List[BaseDocument]:
    """Loads processed Sahih Muslim Hadith records as BaseDocument objects."""
    if not os.path.exists(MUSLIM_PROCESSED_PATH):
        return []

    with open(MUSLIM_PROCESSED_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    docs = []
    for r in records:
        meta = {
            "book": r.get("book", ""),
            "book_number": r.get("book_number", 0),
            "chapter": r.get("chapter", ""),
            "hadith_number": r.get("hadith_number", 0),
            "narrator": r.get("narrator", ""),
            "grade": r.get("grade", "صحيح"),
            "sanad": r.get("sanad", ""),
            "topics": r.get("topics", []),
            "keywords": r.get("keywords", []),
            "aliases": r.get("aliases", []),
        }
        docs.append(BaseDocument(
            id=r["id"],
            type="hadith",
            source="muslim",
            text=r["matn"],
            metadata=meta
        ))
    return docs
