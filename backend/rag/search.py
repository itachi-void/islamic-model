# -*- coding: utf-8 -*-
import re
import math
from typing import List, Set, Any, Dict
from collections import defaultdict
from backend.domain.document import BaseDocument

def strip_dialectal_phrases(text: str) -> str:
    """
    Strips dialectal Egyptian/Levantine query prefixes to extract core intent keywords.
    """
    patterns = [
        r'^(?:عايز|عاوز|محتاج|ابغى|ودي)\s+(?:اية|آية|سورة|حديث|كلام|معنى)?\s*(?:عن|بتتكلم عن|بتقول|فيها|بتاعة)?\s*',
        r'^(?:فين|وين|اين)\s+(?:ربنا قال عن|ربنا ذكر|ربنا قال|الجملة اللي|كلمة)?\s*',
        r'^(?:ايه|ايه هي|ايه هي الـ)\s+(?:الآية|الاية|السورة)\s+(?:اللي فيها|اللي بتقول|بتاعة)?\s*',
        r'^(?:آية عن|اية عن|سورة عن)\s*'
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned if len(cleaned) >= 3 else text


def normalize_arabic(text: str) -> str:
    """
    Normalizes Arabic text by removing diacritics (tashkeel), tatweel (kashida),
    and unifying characters.
    """
    text = re.sub(r'[\u064B-\u0652\u0670]', '', text)
    text = re.sub(r'\u0640', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[ؤئ]', 'ء', text)
    return text.strip()


def light_stem_arabic(word: str) -> str:
    """
    Applies Arabic light stemming to strip common prefixes and suffixes.
    """
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


def extract_stemmed_tokens(text: str) -> Set[str]:
    """
    Tokenizes and light-stems words from Arabic text after stripping dialectal query prefixes.
    """
    text_clean = strip_dialectal_phrases(text)
    words = re.findall(r'\w+', normalize_arabic(text_clean))
    tokens = set()
    for w in words:
        stemmed = light_stem_arabic(w)
        if len(stemmed) >= 2:
            tokens.add(stemmed)
    return tokens


class BaseSearchEngine:
    def search(self, query: str, limit: int = 5) -> List[BaseDocument]:
        raise NotImplementedError("Subclasses must implement search")


def _safe_str(val: Any) -> str:
    if isinstance(val, list):
        return " ".join([str(v) for v in val if v])
    elif val is None:
        return ""
    return str(val)


class ExactSearchEngine(BaseSearchEngine):
    """
    BM25/Exact Keyword Match Search Engine with IDF-Weighted Root/Stem Coverage Scoring.
    Pattern C: Weights each matched token by its inverse document frequency (IDF),
    so rare domain-specific stems (e.g. 'نعلين', 'البقره') score higher than
    common function words (e.g. 'موسى', 'قال', 'الله') that appear in many verses.
    """
    def __init__(self, documents: List[BaseDocument]):
        self.documents = documents
        self.doc_tokens = []
        token_doc_freq: Dict[str, int] = defaultdict(int)

        # First pass: build token sets per document and count document frequencies
        all_doc_tokens = []
        for doc in documents:
            title = _safe_str(doc.metadata.get("title_ar"))
            aliases = _safe_str(doc.metadata.get("aliases"))
            topics = _safe_str(doc.metadata.get("topics"))
            book = _safe_str(doc.metadata.get("book"))
            narrator = _safe_str(doc.metadata.get("narrator"))
            chapter = _safe_str(doc.metadata.get("chapter"))
            sanad = _safe_str(doc.metadata.get("sanad"))
            meta_text = f"{doc.text} {title} {aliases} {topics} {book} {narrator} {chapter} {sanad}"
            tokens = extract_stemmed_tokens(meta_text)
            all_doc_tokens.append((doc, tokens))
            for t in tokens:
                token_doc_freq[t] += 1

        # Second pass: compute IDF for each token
        N = len(documents)
        self._idf: Dict[str, float] = {}
        for token, df in token_doc_freq.items():
            self._idf[token] = math.log((N + 1) / (df + 1)) + 1.0

        for doc, tokens in all_doc_tokens:
            self.doc_tokens.append((doc, tokens))

    def search(self, query: str, limit: int = 5) -> List[BaseDocument]:
        cleaned_query = strip_dialectal_phrases(query)
        q_tokens = extract_stemmed_tokens(cleaned_query)
        if not q_tokens:
            return []

        # Pattern C: IDF-weighted query token score
        # Total query weight = sum of IDF weights of all query tokens
        total_q_weight = sum(self._idf.get(t, 1.0) for t in q_tokens)
        if total_q_weight == 0:
            return []

        scored_docs = []
        for doc, d_tokens in self.doc_tokens:
            matched_tokens = q_tokens & d_tokens
            if matched_tokens:
                # Weight matched tokens by their IDF — rare specific tokens score more
                weighted_overlap = sum(self._idf.get(t, 1.0) for t in matched_tokens)
                score = round(weighted_overlap / total_q_weight, 4)
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return BaseDocument objects with scores attached
        results = []
        for doc, score in scored_docs[:limit]:
            results.append(BaseDocument(
                id=doc.id,
                type=doc.type,
                source=doc.source,
                text=doc.text,
                metadata=doc.metadata,
                score=score
            ))
        return results
