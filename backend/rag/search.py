# -*- coding: utf-8 -*-
"""
Exact Match & BM25 Inverted Search Engine
==========================================
High-performance inverted index search engine with exact token matching and IDF token weighting.

IMPORTANT: Sequential O(N) scan over 7000+ documents causes 15-minute timeout on PythonAnywhere.
           This file uses an Inverted Index for O(1) candidate lookup. DO NOT revert to sequential scan.
"""
import re
import math
from collections import defaultdict
from typing import List, Dict, Set, Tuple

from backend.domain.document import BaseDocument

ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670]")


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآء]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def strip_dialectal_phrases(query: str) -> str:
    if not query:
        return query
    prefixes = [
        "عايز حديث", "عايزة حديث", "اريد حديث", "أريد حديث",
        "ما هو حديث", "ايه الحديث اللي بيقول", "ممكن تقولي حديث",
        "حديث يشرح", "معنى حديث", "ما معنى", "ما هو",
        "عايز", "عاوز", "محتاج", "ابغى", "ودي",
    ]
    norm = query.strip()
    for p in prefixes:
        if norm.startswith(p):
            norm = norm[len(p):].strip()
            break
    return norm or query


def light_stem_arabic(word: str) -> str:
    """Light Arabic stemmer — strips common prefixes and suffixes."""
    if len(word) <= 3:
        return word
    prefixes = ['فبال', 'وبال', 'بال', 'فال', 'وال', 'ال', 'ولل', 'فلل', 'لل', 'وا', 'فا', 'فس']
    for p in prefixes:
        if word.startswith(p) and len(word) - len(p) >= 3:
            word = word[len(p):]
            break
    suffixes = ['اتهم', 'اتكم', 'اتنا', 'ينهم', 'ينكم', 'ونهم', 'ونكم', 'ات', 'ين', 'ون', 'هم', 'كم', 'نا', 'ها', 'ان']
    for s in suffixes:
        if word.endswith(s) and len(word) - len(s) >= 3:
            word = word[:-len(s)]
            break
    return word


def extract_search_tokens(text: str) -> List[str]:
    norm = strip_dialectal_phrases(text)
    norm = normalize_arabic(norm)
    words = norm.split()
    tokens = []
    stop_words = {
        "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "أن", "إن",
        "كان", "كانت", "هو", "هي", "لا", "ما", "وا", "ثم", "قد",
    }
    for w in words:
        if len(w) > 1 and w not in stop_words:
            tokens.append(w)
            stem = light_stem_arabic(w)
            if stem and stem != w and len(stem) >= 2:
                tokens.append(stem)
    return tokens


def extract_stemmed_tokens(text: str) -> Set[str]:
    """Returns a Set[str] for use in set-intersection operations in answer_generator."""
    return set(extract_search_tokens(text))


class ExactSearchEngine:
    """
    Inverted Index BM25 Search Engine for Hadith and Quranic text.

    Uses an inverted index for O(1) candidate lookup instead of O(N) sequential scan.
    Sequential scan over 7000+ documents causes 15-minute timeouts on single-worker WSGI.
    """

    def __init__(self, documents: List[BaseDocument]):
        self.documents = documents
        # doc_id -> BaseDocument (O(1) lookup)
        self.doc_map: Dict[str, BaseDocument] = {doc.id: doc for doc in documents}
        # doc_id -> Set[tokens] (O(1) token lookup per doc)
        self.doc_tokens_map: Dict[str, Set[str]] = {}
        # token -> Set[doc_id] (inverted index for O(1) candidate retrieval)
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self._idf: Dict[str, float] = {}
        self._build_index()

    def _build_index(self):
        total_docs = len(self.documents)
        if total_docs == 0:
            return

        df: Dict[str, int] = defaultdict(int)

        for doc in self.documents:
            meta_str = (
                f"{doc.metadata.get('book', '')} "
                f"{doc.metadata.get('chapter', '')} "
                f"{doc.metadata.get('narrator', '')} "
                f"{doc.metadata.get('title_ar', '')} "
                f"{doc.metadata.get('aliases', '')} "
                f"{doc.metadata.get('topics', '')}"
            )
            full_text = f"{doc.text} {meta_str}"
            tokens: Set[str] = set(extract_search_tokens(full_text))
            self.doc_tokens_map[doc.id] = tokens
            for token in tokens:
                self.inverted_index[token].add(doc.id)
                df[token] += 1

        for token, doc_freq in df.items():
            self._idf[token] = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def search(self, query_text: str, limit: int = 50) -> List[BaseDocument]:
        """
        Fast BM25/exact overlap search using inverted index.
        Complexity: O(|query_tokens| * avg_postings) — NOT O(N) sequential scan.
        """
        q_tokens: Set[str] = set(extract_search_tokens(query_text))
        if not q_tokens:
            return []

        total_q_weight = sum(self._idf.get(t, 1.0) for t in q_tokens) or 1.0

        # --- O(1) candidate retrieval via inverted index ---
        candidate_ids: Set[str] = set()
        for qt in q_tokens:
            candidate_ids.update(self.inverted_index.get(qt, set()))

        # Fallback: prefix matching when no exact hit found
        if not candidate_ids:
            for qt in q_tokens:
                if len(qt) >= 4:
                    prefix = qt[:4]
                    for term, ids in self.inverted_index.items():
                        if term.startswith(prefix):
                            candidate_ids.update(ids)

        # Prepare normalized query words for prefix sequence alignment bonus
        q_norm = normalize_arabic(re.sub(r'حديث\s*', '', query_text or '').strip())
        q_words = q_norm.split()

        # --- Score only the candidates, not all documents ---
        scored_docs = []
        for doc_id in candidate_ids:
            doc = self.doc_map[doc_id]
            d_tokens = self.doc_tokens_map.get(doc_id, set())
            matched = q_tokens & d_tokens
            if matched:
                score = round(sum(self._idf.get(t, 1.0) for t in matched) / total_q_weight, 4)

                # Sequence alignment bonus: reward word-for-word prefix match
                if q_words:
                    d_norm = normalize_arabic(doc.text)
                    d_words = d_norm.split()
                    prefix_matches = 0
                    for qw, dw in zip(q_words, d_words):
                        if qw == dw:
                            prefix_matches += 1
                        else:
                            break
                    bonus = (prefix_matches / float(len(q_words))) * 0.40
                    score = round(score + bonus, 4)

                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [
            BaseDocument(
                id=doc.id,
                type=doc.type,
                source=doc.source,
                text=doc.text,
                metadata=doc.metadata,
                score=score,
            )
            for doc, score in scored_docs[:limit]
        ]


# Backward compatibility alias
BaseSearchEngine = ExactSearchEngine
