# -*- coding: utf-8 -*-
"""
Isnad (Chain of Narration) Inverted Index
==========================================
Builds a fast inverted index mapping each narrator name in an isnad chain
to the hadiths they appear in. Enables direct isnad-based retrieval for
partial-chain queries like "حدثنا X قال حدثنا Y".

This is the primary retrieval mechanism for the IRB-v1 benchmark where
96% of queries are isnad fragments.
"""
import json
import os
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from backend.rag.search import normalize_arabic, extract_stemmed_tokens


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics (tashkeel/harakat) for matching."""
    if not text:
        return ""
    return re.sub(r'[\u064B-\u0652\u0670]', '', text)


# Pattern to extract narrator names from isnad chains
ISNAD_PATTERNS = [
    # Standard: حدثنا X, حدثني X, أخبرنا X, أخبرني X, قال حدثنا X
    re.compile(r'(?:حدثنا|حدثني|اخبرنا|اخبرني|قال حدثنا|سمعت)\s+([^،,]+)'),
    # From: عن X (with 3-40 chars)
    re.compile(r'عن\s+([^،,]{3,40}?)(?:\s+عن|\s+قال|\s+يقول|$)'),
    # قال فلان وأخبرني/وحدثنا Y: "قال مالك وأخبرني زيد" or "قال مالك قال حدثنا زيد"
    re.compile(r'قال\s+([^،,]{3,30}?)\s+(?:و)?(?:اخبرني|اخبرنا|حدثنا|حدثني)\s+([^،,]+)'),
    # و عن / وعن patterns: "وعن أبي هريرة"
    re.compile(r'وعن\s+([^،,]{3,40}?)(?:\s+عن|\s+قال|\s+يقول|$)'),
    # Ibn X as narrator at start of chain
    re.compile(r'^ابن\s+([^،,]{3,30}?)(?:\s+و|\s+قال|\s+يقول|$)'),
]


def extract_isnad_narrators(text: str) -> List[str]:
    """Extract narrator names from an isnad text (query or matn)."""
    cleaned = strip_diacritics(normalize_arabic(text))
    names = []
    for pattern in ISNAD_PATTERNS:
        matches = pattern.findall(cleaned)
        for m in matches:
            if isinstance(m, tuple):
                name = " ".join(x.strip() for x in m if x and len(x.strip()) > 2)
            else:
                name = m.strip()
            # Remove trailing verbs and prepositions
            name = re.sub(r'\s+(?:قال|يقول|عن|أنه|في|على|من)$', '', name)
            name = name.strip()
            if len(name) > 2:
                names.append(name)
    return names


class IsnadIndex:
    """
    Fast inverted index for isnad-based hadith retrieval.
    
    Maps each narrator name (normalized) to the set of hadith IDs
    where that narrator appears in the isnad chain.
    
    Uses IDF-like weighting: rare narrators get more weight than common ones.
    Supports consecutive chain matching for partial isnad queries.
    """
    def __init__(self):
        self.narrator_to_hadiths: Dict[str, Set[str]] = defaultdict(set)
        self.hadith_to_narrators: Dict[str, List[str]] = {}
        self.narrator_idf: Dict[str, float] = {}
        self.total_hadiths: int = 0
        self._built = False

    def build(self, bukhari_data: List[dict]) -> None:
        """Build the isnad index from processed Bukhari data."""
        self.total_hadiths = len(bukhari_data)
        
        for record in bukhari_data:
            h_id = record["id"]
            matn = record.get("matn", "")
            
            # Extract narrator names from the matn isnad
            narrators = extract_isnad_narrators(matn)
            
            # Also add the main narrator field
            main_narrator = record.get("narrator", "")
            if main_narrator:
                norm_main = normalize_arabic(main_narrator)
                if norm_main and len(norm_main) > 2:
                    narrators.append(norm_main)
            
            # Deduplicate while preserving order
            seen = set()
            unique_narrators = []
            for n in narrators:
                if n not in seen:
                    seen.add(n)
                    unique_narrators.append(n)
            
            self.hadith_to_narrators[h_id] = unique_narrators
            
            for narrator in unique_narrators:
                self.narrator_to_hadiths[narrator].add(h_id)
                
                # Also add stemmed variants for fuzzy matching
                stemmed = " ".join(extract_stemmed_tokens(narrator))
                if stemmed and stemmed != narrator:
                    self.narrator_to_hadiths[stemmed].add(h_id)
        
        # Compute IDF for each narrator: rare narrators get higher weight
        import math
        for narrator, h_ids in self.narrator_to_hadiths.items():
            df = len(h_ids)
            self.narrator_idf[narrator] = math.log((self.total_hadiths - df + 0.5) / (df + 0.5) + 1.0)
        
        self._built = True

    def search(self, query_text: str, min_matches: int = 1) -> List[Tuple[str, float]]:
        """
        Search the isnad index with a query.
        Returns list of (hadith_id, score) tuples sorted by score descending.
        
        Scoring uses:
        1. IDF-weighted narrator matches (rare narrators weigh more)
        2. Consecutive chain order bonus (sequential matches in order)
        3. Second-narrator content bonus (for queries with "عن" patterns)
        """
        query_narrators = extract_isnad_narrators(query_text)
        if not query_narrators:
            return []
        
        # For each hadith, compute IDF-weighted score
        hadith_scores: Dict[str, float] = defaultdict(float)
        hadith_match_count: Dict[str, int] = defaultdict(int)
        hadith_match_details: Dict[str, Dict] = {}
        
        for qn in query_narrators:
            matches = self.narrator_to_hadiths.get(qn, set())
            idf = self.narrator_idf.get(qn, 1.0)
            
            if matches:
                for h_id in matches:
                    hadith_scores[h_id] += idf
                    hadith_match_count[h_id] += 1
                    if h_id not in hadith_match_details:
                        hadith_match_details[h_id] = {"matched": [], "missing": []}
                    hadith_match_details[h_id]["matched"].append(qn)
            else:
                for indexed_narrator, h_ids in self.narrator_to_hadiths.items():
                    if qn in indexed_narrator or indexed_narrator in qn:
                        partial_idf = self.narrator_idf.get(indexed_narrator, 1.0) * 0.5
                        for h_id in h_ids:
                            hadith_scores[h_id] += partial_idf
                            hadith_match_count[h_id] += 1
        
        # Chain order bonus and "عن" (from) pattern matching
        for h_id in list(hadith_scores.keys()):
            h_narrators = self.hadith_to_narrators.get(h_id, [])
            if not h_narrators:
                continue
            
            # 1. Consecutive chain order bonus
            max_consecutive = 0
            consecutive = 0
            h_idx = 0
            for qn in query_narrators:
                found = False
                while h_idx < len(h_narrators):
                    if qn in h_narrators[h_idx] or h_narrators[h_idx] in qn:
                        found = True
                        h_idx += 1
                        break
                    h_idx += 1
                
                if found:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
                    h_idx = 0  # Reset for next query narrator
            
            if max_consecutive >= 2:
                hadith_scores[h_id] *= (1.0 + 0.3 * max_consecutive)
            
            # 2. "عن" (from) pattern: if query has "عن narrator", check if that narrator appears
            # in the hadith isnad chain AFTER the other matched narrators
            query_clean = strip_diacritics(normalize_arabic(query_text))
            from_patterns = re.findall(r'عن\s+([^،,]{3,40}?)(?:\s+عن|\s+قال|\s+يقول|$)', query_clean)
            if from_patterns:
                from_matches = 0
                for fp in from_patterns:
                    fp = fp.strip()
                    if len(fp) > 2:
                        for hn in h_narrators:
                            if fp in hn or hn in fp:
                                from_matches += 1
                                break
                if from_matches > 0:
                    hadith_scores[h_id] *= (1.0 + 0.2 * from_matches)
        
        # Normalize and sort
        total_q_narrators = len(query_narrators)
        results = []
        for h_id, raw_score in hadith_scores.items():
            match_count = hadith_match_count[h_id]
            if match_count >= min_matches:
                normalized_score = raw_score / total_q_narrators
                results.append((h_id, round(normalized_score, 4)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_narrator_count(self) -> int:
        return len(self.narrator_to_hadiths)

    def get_total_entries(self) -> int:
        return sum(len(v) for v in self.narrator_to_hadiths.values())


def build_isnad_index_from_file(data_path: str) -> IsnadIndex:
    """Build an isnad index from a processed Bukhari JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    index = IsnadIndex()
    index.build(data)
    return index
