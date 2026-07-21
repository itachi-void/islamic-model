# -*- coding: utf-8 -*-
"""
Knowledge Graph Parser & Entity Resolution Engine
===================================================
Loads and queries data/bukhari/knowledge_graph.json to resolve narrators,
aliases, eras, and famous Hadith titles for enhanced entity search.
"""
import os
import json
from typing import Dict, List, Optional
from backend.rag.search import normalize_arabic

KG_PATH = r"d:\model\data\bukhari\knowledge_graph.json"


class KnowledgeGraph:
    """Parsed Knowledge Graph for Islamic entity resolution."""
    def __init__(self, path: str = KG_PATH):
        self.path = path
        self.narrators: Dict[str, Dict] = {}
        self.famous_hadiths: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.narrators = data.get("narrators", {})
            self.famous_hadiths = data.get("famous_hadiths", {})

    def resolve_narrator(self, query_term: str) -> Optional[Dict]:
        """Resolves any narrator alias (e.g. 'أبو هريرة') to its full Knowledge Graph entity record."""
        norm_term = normalize_arabic(query_term)
        for canonical_name, entity in self.narrators.items():
            aliases = [normalize_arabic(a) for a in entity.get("aliases", [])]
            if norm_term == normalize_arabic(canonical_name) or norm_term in aliases:
                return entity
        return None

    def expand_narrator(self, query_term: str) -> List[str]:
        """Returns all canonical aliases for a narrator query term."""
        entity = self.resolve_narrator(query_term)
        if entity:
            return entity.get("aliases", [])
        return []
