# -*- coding: utf-8 -*-
"""
Exact Match & BM25 Inverted Search Engine
==========================================
High-performance inverted index search engine with exact token matching and IDF token weighting.
"""
import re
from collections import defaultdict
import math
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
        "عايز حديث", "اريد حديث", "ما هو حديث", "ايه الحديث اللي بيقول",
        "ممكن حديث", "حديث عن", "حديث يشرح", "معنى حديث"
    ]
    norm = query.strip()
    for p in prefixes:
        if norm.startswith(p):
            norm = norm[len(p):].strip()
            break
    return norm or query


def light_stem_arabic(word: str) -> str:
    word = normalize_arabic(word)
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
    stop_words = {"في", "على", "إلى", "عن", "مع", "هذا", "هذه", "أن", "إن", "كان", "كانت", "هو", "هي"}
    for w in words:
        if len(w) > 1 and w not in stop_words:
            tokens.append(w)
            stem = light_stem_arabic(w)
            if stem and stem != w and len(stem) >= 2:
                tokens.append(stem)
    return tokens


def extract_stemmed_tokens(text: str) -> Set[str]:
    return set(extract_search_tokens(text))


class ExactSearchEngine:
    """Inverted Index Search Engine for Hadith and Quranic text."""
    def __init__(self, documents: List[BaseDocument]):
        self.documents = documents
        self.doc_map: Dict[str, BaseDocument] = {doc.id: doc for doc in documents}
        self.doc_tokens_map: Dict[str, Set[str]] = {}
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self._idf: Dict[str, float] = {}

        self._build_index()

    def _build_index(self):
        total_docs = len(self.documents)
        if total_docs == 0:
            return

        df: Dict[str, int] = defaultdict(int)

        for doc in self.documents:
            meta_str = f"{doc.metadata.get('book', '')} {doc.metadata.get('chapter', '')} {doc.metadata.get('narrator', '')} {doc.metadata.get('title_ar', '')} {doc.metadata.get('aliases', '')} {doc.metadata.get('topics', '')}"
            full_text = f"{doc.text} {meta_str}"
            tokens = set(extract_search_tokens(full_text))

            self.doc_tokens_map[doc.id] = tokens
            for token in tokens:
                self.inverted_index[token].add(doc.id)
                df[token] += 1

        for token, doc_freq in df.items():
            self._idf[token] = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def search(self, query_text: str, limit: int = 50) -> List[BaseDocument]:
        """Fast BM25 / exact overlap search."""
        q_tokens = set(extract_search_tokens(query_text))
        if not q_tokens:
            return []

        total_q_weight = sum(self._idf.get(t, 1.0) for t in q_tokens) or 1.0

        # Candidate lookup via inverted index
        candidate_ids = set()
        for qt in q_tokens:
            candidate_ids.update(self.inverted_index.get(qt, set()))

        # Fallback if no exact candidate found
        if not candidate_ids:
            for qt in q_tokens:
                if len(qt) >= 4:
                    prefix = qt[:4]
                    for term, ids in self.inverted_index.items():
                        if term.startswith(prefix):
                            candidate_ids.update(ids)

        scored_docs = []
        for doc_id in candidate_ids:
            doc = self.doc_map[doc_id]
            d_tokens = self.doc_tokens_map.get(doc_id, set())
            matched = q_tokens & d_tokens
            if matched:
                score = round(sum(self._idf.get(t, 1.0) for t in matched) / total_q_weight, 4)
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored_docs[:max(limit, 50)]:
            results.append(BaseDocument(
                id=doc.id,
                type=doc.type,
                source=doc.source,
                text=doc.text,
                metadata=doc.metadata,
                score=score
            ))

        return results


# Backward compatibility alias
BaseSearchEngine = ExactSearchEngine
