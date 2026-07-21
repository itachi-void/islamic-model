# -*- coding: utf-8 -*-
"""
Exact Match & BM25 Search Engine (Restored Original + Bug Fixes)
=================================================================
Iterated document scoring using TF-IDF weighted token overlap.
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
    ]
    norm = query.strip()
    for p in prefixes:
        if norm.startswith(p):
            norm = norm[len(p):].strip()
            break
    return norm or query


def extract_search_tokens(text: str) -> List[str]:
    norm = normalize_arabic(text)
    words = norm.split()
    tokens = []
    stop_words = {"من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "أن", "إن", "كان", "كانت", "هو", "هي", "لا", "ما", "وا"}
    for w in words:
        if len(w) > 1 and w not in stop_words:
            tokens.append(w)
            # Also add de-articled form
            if w.startswith("ال") and len(w) > 3:
                tokens.append(w[2:])
    return tokens


def extract_stemmed_tokens(text: str) -> List[str]:
    return extract_search_tokens(text)


class ExactSearchEngine:
    """
    Sequential BM25-weighted token overlap search engine.
    Uses List[Tuple[doc, Set[tokens]]] for IDF-weighted exact matching.
    """

    def __init__(self, documents: List[BaseDocument]):
        self.documents = documents
        self.doc_tokens: List[Tuple[BaseDocument, Set[str]]] = []
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
                f"{doc.metadata.get('narrator', '')}"
            )
            full_text = f"{doc.text} {meta_str}"
            tokens: Set[str] = set(extract_search_tokens(full_text))
            self.doc_tokens.append((doc, tokens))
            for t in tokens:
                df[t] += 1

        for token, freq in df.items():
            self._idf[token] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query_text: str, limit: int = 20) -> List[BaseDocument]:
        # Convert to set for fast intersection
        q_tokens: Set[str] = set(extract_search_tokens(query_text))
        if not q_tokens:
            return []

        total_q_weight = sum(self._idf.get(t, 1.0) for t in q_tokens) or 1.0

        scored_docs = []
        for doc, d_tokens in self.doc_tokens:
            matched_tokens = q_tokens & d_tokens
            if matched_tokens:
                weighted_overlap = sum(self._idf.get(t, 1.0) for t in matched_tokens)
                score = round(weighted_overlap / total_q_weight, 4)
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
