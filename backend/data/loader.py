# -*- coding: utf-8 -*-
import os
import json
from typing import List, Dict, Optional
from backend.domain.document import BaseDocument

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QURAN_JSON_PATH = os.path.join(BASE_DIR, "data", "quran", "quran.json")
METADATA_JSON_PATH = os.path.join(BASE_DIR, "data", "quran", "metadata.json")
FAMOUS_VERSES_PATH = os.path.join(BASE_DIR, "data", "quran", "famous_verses.json")

_quran_cache: Optional[List[Dict]] = None
_metadata_cache: Optional[List[Dict]] = None
_famous_verses_cache: Optional[Dict] = None

def load_quran() -> List[Dict]:
    """
    Loads the Quran records from quran.json and caches them in memory.
    """
    global _quran_cache
    if _quran_cache is not None:
        return _quran_cache
    
    if not os.path.exists(QURAN_JSON_PATH):
        raise FileNotFoundError(f"Quran data file not found at {QURAN_JSON_PATH}")
        
    with open(QURAN_JSON_PATH, "r", encoding="utf-8") as f:
        _quran_cache = json.load(f)
        
    return _quran_cache

def load_metadata() -> List[Dict]:
    """
    Loads the Surah metadata from metadata.json and caches it in memory.
    """
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache
    
    if not os.path.exists(METADATA_JSON_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_JSON_PATH}")
        
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        _metadata_cache = json.load(f)
        
    return _metadata_cache

def load_famous_verses() -> Dict:
    """
    Loads famous verses, titles, aliases, and topics from famous_verses.json.
    """
    global _famous_verses_cache
    if _famous_verses_cache is not None:
        return _famous_verses_cache
    
    if os.path.exists(FAMOUS_VERSES_PATH):
        with open(FAMOUS_VERSES_PATH, "r", encoding="utf-8") as f:
            _famous_verses_cache = json.load(f)
    else:
        _famous_verses_cache = {}
        
    return _famous_verses_cache

def get_ayah(surah_number: int, ayah_number: int) -> Optional[Dict]:
    """
    Finds a specific ayah by surah and ayah number.
    """
    quran = load_quran()
    for record in quran:
        if record["surah_number"] == surah_number and record["ayah_number"] == ayah_number:
            return record
    return None

def load_search_documents() -> List[BaseDocument]:
    """
    Loads Quran verses and populates title_ar, aliases, and topics into metadata.
    Keeps doc.text as pure verse text for optimal vector embeddings and exact match ranking.
    """
    raw_quran = load_quran()
    famous_map = load_famous_verses()
    docs = []
    
    for record in raw_quran:
        doc_id = record["id"]
        famous_info = famous_map.get(doc_id, {})
        
        meta = {
            "surah_number": record["surah_number"],
            "surah_name_ar": record["surah_name_ar"],
            "surah_name_en": record["surah_name_en"],
            "ayah_number": record["ayah_number"],
            "revelation_type": record["revelation_type"],
            "original_text": record["text"]
        }
        if famous_info:
            if "title" in famous_info:
                meta["title_ar"] = famous_info["title"]
            if "aliases" in famous_info:
                meta["aliases"] = famous_info["aliases"]
            if "topics" in famous_info:
                meta["topics"] = famous_info["topics"]
                
        docs.append(BaseDocument(
            id=doc_id,
            type=record["type"],
            source=record["source"],
            text=record["text"],
            metadata=meta
        ))
    return docs
