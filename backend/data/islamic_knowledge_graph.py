# -*- coding: utf-8 -*-
"""
Unified Islamic Knowledge Graph Engine
=======================================
Entity resolution engine using canonical entity IDs for narrators, books, and Hadith titles.
"""
import os
import json
from typing import Dict, List, Optional
from backend.rag.search import normalize_arabic

IKG_PATH = r"d:\model\data\islamic_knowledge_graph.json"


class IslamicKnowledgeGraph:
    """Unified Islamic Knowledge Graph parser and resolver."""
    def __init__(self, path: str = IKG_PATH):
        self.path = path
        self.entities: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.entities = data.get("entities", {})

    def resolve_narrator(self, term: str) -> Optional[Dict]:
        norm_term = normalize_arabic(term)
        narrators = self.entities.get("narrators", {})
        for entity_id, record in narrators.items():
            aliases = [normalize_arabic(a) for a in record.get("aliases", [])]
            c_name = normalize_arabic(record.get("canonical_name", ""))
            if norm_term == c_name or norm_term in aliases:
                return record
        return None

    def resolve_hadith_title(self, term: str) -> Optional[Dict]:
        norm_term = normalize_arabic(term)
        titles = self.entities.get("hadith_titles", {})
        for entity_id, record in titles.items():
            aliases = [normalize_arabic(a) for a in record.get("aliases", [])]
            c_title = normalize_arabic(record.get("canonical_title", ""))
            if norm_term == c_title or norm_term in aliases:
                return record
        return None

    def expand_query(self, query: str) -> str:
        """Expands query with canonical entity names and gold evidence keys if matched."""
        if not query:
            return query

        norm_q = normalize_arabic(query)
        expanded = query

        # Hadith titles match
        titles = self.entities.get("hadith_titles", {})
        for entity_id, record in titles.items():
            for alias in record.get("aliases", []):
                if normalize_arabic(alias) in norm_q:
                    expanded += f" {record.get('canonical_title', '')}"
                    break

        # Narrators match
        narrators = self.entities.get("narrators", {})
        for entity_id, record in narrators.items():
            for alias in record.get("aliases", []):
                if normalize_arabic(alias) in norm_q:
                    expanded += f" {record.get('canonical_name', '')}"
                    break

        return expanded.strip()
