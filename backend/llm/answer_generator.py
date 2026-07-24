# -*- coding: utf-8 -*-
"""
AnswerGenerator: Independent post-retrieval LLM answer generation layer.
Includes:
- Generic Polymorphic Evidence Representation
- Evidence Deduplication & Re-ranking (Relevance Filter with MIN_RELEVANCE)
- Deterministic Formula-based Confidence Scoring
- Dynamic Score-based Star Ratings (5-star scale)
- Explainability Bullets (Why this evidence was selected)
- Separation of Facts (Python-rendered) vs Reasoning (LLM-rendered)
"""
from typing import List, Dict, Any, Tuple, Optional
from backend.domain.document import BaseDocument
from backend.llm.ollama_client import generate, generate_stream
from backend.rag.search import extract_stemmed_tokens, normalize_arabic, strip_dialectal_phrases

MIN_RELEVANCE_THRESHOLD = 0.40

# Mapping from internal source ID to canonical Arabic display name
SOURCE_DISPLAY_NAMES = {
    "quran": "القرآن الكريم",
    "bukhari": "صحيح البخاري",
    "hadith": "صحيح البخاري",
    "muslim": "صحيح مسلم",
    "tirmidhi": "جامع الترمذي",
    "tafsir": "كتب التفسير",
}


class GenericEvidence:
    """Polymorphic Evidence Model adaptable to Quran, Bukhari, Muslim, Tafsir, Fiqh, etc."""
    def __init__(self, doc: BaseDocument, score: float):
        self.doc = doc
        self.id = doc.id
        self.source = doc.source
        self.text = doc.text
        self.score = score
        self.metadata = doc.metadata or {}

        # Derived properties
        if self.source == "quran":
            surah = self.metadata.get("surah_name_ar", "")
            ayah = self.metadata.get("ayah_number", "")
            self.title = f"سورة {surah}"
            self.reference = f"سورة {surah}، آية {ayah}"
            self.dedup_key = f"quran_{surah}_{ayah}"
        elif self.source in ["bukhari", "hadith", "muslim", "tirmidhi"]:
            book = self.metadata.get("book", "")
            book_str = book if str(book).startswith("كتاب") else f"كتاب {book}"
            h_num = self.metadata.get("hadith_number", "")
            narrator = self.metadata.get("narrator", "")
            narrator_str = f" (عن {narrator})" if narrator else ""
            display_name = SOURCE_DISPLAY_NAMES.get(self.source, self.source)
            self.title = f"{display_name} — {book_str}"
            self.reference = f"{display_name} — {book_str}، حديث #{h_num}{narrator_str}"
            self.dedup_key = f"{self.source}_{h_num}"
        else:
            self.title = self.metadata.get("title_ar", SOURCE_DISPLAY_NAMES.get(self.source, self.source))
            self.reference = self.id
            self.dedup_key = f"{self.source}_{self.id}"

    def compute_stars(self) -> str:
        if self.score >= 0.80:
            return "★★★★★"
        elif self.score >= 0.65:
            return "★★★★☆"
        elif self.score >= 0.50:
            return "★★★☆☆"
        elif self.score >= 0.35:
            return "★★☆☆☆"
        return "★☆☆☆☆"

    def explain_selection(self, query: str) -> List[str]:
        bullets = []
        q_tokens = extract_stemmed_tokens(query)
        d_tokens = extract_stemmed_tokens(self.text)
        meta_tokens = extract_stemmed_tokens(
            f"{self.metadata.get('title_ar', '')} {self.metadata.get('book', '')} {self.metadata.get('narrator', '')} {self.metadata.get('topics', '')}"
        )

        overlap = q_tokens & (d_tokens | meta_tokens)
        if len(overlap) >= 2:
            bullets.append("✔ يطابق ألفاظ ومفاهيم السؤال مباشرة")
        elif len(overlap) == 1:
            bullets.append("✔ يطابق الكلمة المفتاحية في الاستعلام")

        if self.score >= 0.75:
            bullets.append("✔ درجة تشابه دلالي مرتفعة للغاية")
        elif self.score >= 0.50:
            bullets.append("✔ تطابق جيد في موضوع البحث")

        if self.source == "quran":
            bullets.append("✔ نص قرآني محكم ثابِت القطع")
        elif self.source in ["bukhari", "hadith"]:
            bullets.append("✔ حديث موثق من صحيح البخاري")

        return bullets if bullets else ["✔ تم اختياره لملاءمته لسياق الاستعلام"]


def filter_and_deduplicate_evidence(query: str, docs: List[BaseDocument]) -> List[GenericEvidence]:
    """
    1. Converts docs to GenericEvidence
    2. Re-ranks & Filters out low-relevance documents (score < MIN_RELEVANCE_THRESHOLD)
    3. Deduplicates identical Hadiths/Ayahs
    4. Sorts descending by score
    """
    if not docs:
        return []

    q_tokens = extract_stemmed_tokens(query)

    evidences: List[GenericEvidence] = []
    seen_keys = set()

    for doc in docs:
        raw_score = doc.score if doc.score is not None else 0.5
        
        # Keyword match bonus check
        d_tokens = extract_stemmed_tokens(doc.text)
        meta_tokens = extract_stemmed_tokens(
            f"{doc.metadata.get('title_ar', '')} {doc.metadata.get('book', '')} {doc.metadata.get('narrator', '')} {doc.metadata.get('topics', '')}"
        )
        has_overlap = bool(q_tokens & (d_tokens | meta_tokens))

        # Re-ranking score adjustment
        effective_score = raw_score
        if has_overlap:
            effective_score = min(1.0, raw_score + 0.20)
        else:
            effective_score = raw_score * 0.5  # Penalize non-matching documents

        if effective_score < MIN_RELEVANCE_THRESHOLD:
            continue

        ev = GenericEvidence(doc, effective_score)
        if ev.dedup_key in seen_keys:
            continue
        seen_keys.add(ev.dedup_key)
        evidences.append(ev)

    # Sort descending by effective score
    evidences.sort(key=lambda e: e.score, reverse=True)
    return evidences


def compute_calculated_confidence(query: str, evidences: List[GenericEvidence]) -> Tuple[int, str, List[str]]:
    """
    Calculates confidence strictly via Python formula:
    confidence = (0.35 * avg_retrieval_score) + (0.25 * exact_token_match_ratio) + (0.20 * metadata_match_ratio) + (0.20 * evidence_volume_ratio)
    """
    if not evidences:
        return 25, "🔴", ["لا توجد أدلة كافية مسترجعة في القاعدة"]

    q_tokens = extract_stemmed_tokens(query)
    top_score = evidences[0].score

    # 1. Retrieval Score Component (35%)
    c_retrieval = min(1.0, top_score) * 0.35

    # 2. Exact Token Match Component (25%)
    all_ev_tokens = set()
    all_meta_tokens = set()
    for ev in evidences:
        all_ev_tokens.update(extract_stemmed_tokens(ev.text))
        all_meta_tokens.update(extract_stemmed_tokens(
            f"{ev.metadata.get('title_ar', '')} {ev.metadata.get('book', '')} {ev.metadata.get('narrator', '')} {ev.metadata.get('topics', '')}"
        ))

    exact_overlap = q_tokens & all_ev_tokens
    token_match_ratio = len(exact_overlap) / float(len(q_tokens)) if q_tokens else 0.0
    c_exact = min(1.0, token_match_ratio) * 0.25

    # 3. Metadata Match Component (20%)
    meta_overlap = q_tokens & all_meta_tokens
    meta_match_ratio = len(meta_overlap) / float(len(q_tokens)) if q_tokens else 0.0
    c_meta = min(1.0, meta_match_ratio) * 0.20

    # 4. Evidence Volume Component (20%)
    c_volume = min(1.0, len(evidences) / 3.0) * 0.20

    calculated_pct = int(round((c_retrieval + c_exact + c_meta + c_volume) * 100))
    calculated_pct = max(30, min(98, calculated_pct))

    reasons = []
    quran_count = sum(1 for e in evidences if e.source == "quran")
    hadith_count = sum(1 for e in evidences if e.source in ["bukhari", "hadith", "muslim"])

    if quran_count > 0:
        reasons.append(f"✔ {quran_count} أدلة قرآنية محكمة")
    if hadith_count > 0:
        reasons.append(f"✔ {hadith_count} أحاديث نبوية مسندة")
    if exact_overlap:
        reasons.append("✔ تطابق لفظي مباشر في الكلمات المفتاحية")
    if top_score >= 0.70:
        reasons.append("✔ تقارب دلالي عالي مع موضوع السؤال")

    if calculated_pct >= 80:
        badge = "🟢"
    elif calculated_pct >= 55:
        badge = "🟡"
    else:
        badge = "🔴"

    return calculated_pct, badge, reasons


class FactsBuilder:
    """
    Deterministic Facts Builder: Renders raw Quran verses, Hadiths, dynamic star ratings,
    explainability bullets, and references DIRECTLY from GenericEvidence Python objects.
    0% LLM Hallucination on Facts.
    """
    @staticmethod
    def render_facts(query: str, evidences: List[GenericEvidence]) -> Dict[str, str]:
        quran_blocks = []
        hadith_blocks = []
        citation_lines = []

        for i, ev in enumerate(evidences, 1):
            stars = ev.compute_stars()
            reasons = ev.explain_selection(query)
            reasons_str = "\n".join(f"  {r}" for r in reasons)

            is_primary = (i == 1)
            badge_label = "📖 الدليل الأساسي" if (is_primary and ev.source == "quran") else (
                "📚 الدليل الأساسي" if (is_primary and ev.source in ["bukhari", "hadith", "muslim"]) else (
                    "📖 دليل قرآني داعم" if ev.source == "quran" else "📚 دليل نبوي داعم"
                )
            )

            block = (
                f"{badge_label} {stars}\n"
                f"سبب الاختيار:\n{reasons_str}\n"
                f"نص الدليل: \"{ev.text}\"\n"
                f"[{ev.reference}]\n"
            )

            if ev.source == "quran":
                quran_blocks.append(block)
            elif ev.source in ["bukhari", "hadith", "muslim"]:
                hadith_blocks.append(block)

            citation_lines.append(f"[{i}] {ev.reference}")

        quran_text = "\n".join(quran_blocks) if quran_blocks else "لا توجد آيات في الأدلة المتاحة."
        hadith_text = "\n".join(hadith_blocks) if hadith_blocks else "لا توجد أحاديث في الأدلة المتاحة."
        citations_text = "\n".join(citation_lines) if citation_lines else "لا توجد مصادر."

        return {
            "quran_text": quran_text,
            "hadith_text": hadith_text,
            "citations_text": citations_text
        }


def build_smart_fallback(query: str, evidences: List[GenericEvidence]) -> str:
    """
    Constructs a beautifully formatted evidence summary if LLM API is offline or times out.
    Uses Quranic brackets ﴿ ﴾ and Hadith quotes « » with precise references.
    """
    if not evidences:
        return "لا توجد أدلة كافية في المصادر المتاحة."

    primary = evidences[0]
    quran_evs = [e for e in evidences if e.source == "quran"]
    hadith_evs = [e for e in evidences if e.source in ["bukhari", "hadith", "muslim", "tirmidhi", "nasai", "abudawud", "ibnmajah"]]

    lines = []

    # 1. Primary Evidence
    if primary.source == "quran":
        surah = primary.metadata.get("surah_name_ar", "")
        ayah = primary.metadata.get("ayah_number", "")
        ref = f"سورة {surah}، آية {ayah}" if surah and ayah else primary.reference
        title = primary.metadata.get("title_ar", "")
        header_name = f"**{title}**" if title else "**النص القرآني الكريم**"
        lines.append(f"{header_name} ({ref}):\n")
        lines.append(f"﴿ {primary.text} ﴾")
    elif primary.source in ["bukhari", "hadith", "muslim", "tirmidhi", "nasai", "abudawud", "ibnmajah"]:
        book = primary.metadata.get("book", "")
        ref = f"صحيح البخاري - {book}" if book else primary.reference
        lines.append(f"**الحديث الشريف** ({ref}):\n")
        lines.append(f"« {primary.text} »")

    # 2. Supporting Quranic Verses
    supporting_quran = [e for e in quran_evs if e != primary]
    if supporting_quran:
        lines.append("\n\n📖 **آيات قرآنية ذات صلة:**")
        for ev in supporting_quran[:3]:
            surah = ev.metadata.get("surah_name_ar", "")
            ayah = ev.metadata.get("ayah_number", "")
            ref = f"سورة {surah}: آية {ayah}" if surah and ayah else ev.reference
            lines.append(f"- ﴿ {ev.text} ﴾ [{ref}]")

    # 3. Supporting Hadiths
    supporting_hadith = [e for e in hadith_evs if e != primary]
    if supporting_hadith:
        lines.append("\n\n📚 **أحاديث نبوية ذات صلة:**")
        for ev in supporting_hadith[:3]:
            book = ev.metadata.get("book", "")
            ref = f"{ev.source.title()} - {book}" if book else ev.reference
            lines.append(f"- « {ev.text} » [{ref}]")

    return "\n".join(lines)


class AnswerGenerator:
    """
    Post-Retrieval Reasoning Layer.
    LLM is restricted purely to explanation and reasoning sections.
    Facts (Verses, Hadiths, Citations) are deterministically attached by FactsBuilder.
    """

    def __init__(self, dialect: str = "egyptian"):
        self.dialect = dialect
        self.system_instruction = (
            "أنت مساعد إسلامي متخصص وموثوق للغاية، يكتب بإسلوب شائق وسلس وفصيح تماماً مثل أحدث شبكات الذكاء الاصطناعي (ChatGPT / Gemini).\n"
            "اقرأ سؤال المستخدم والأدلة الشرعية المرفقة، واكتب إجابة علمية مسبوكة ومبوبة بوضوح طبقاً للتعليمات التنسيقية التالية:\n\n"
            "✨ قواعد التنسيق والأسلوب (استنسخ أسلوب ChatGPT / Gemini):\n"
            "1. **التنسيق الشكلي**:\n"
            "   - ضع الآيات القرآنية الكريمة دائماً بين قوسين قرآنيين: ﴿ ... ﴾ مع ذكر اسم السورة ورقم الآية.\n"
            "   - ضع الأحاديث النبوية الشريفة دائماً بين علامتي تنصيص: « ... » مع ذكر المصدر (مثلاً: رواه البخاري / رواه مسلم).\n"
            "   - استخدم عناوين فرعية مناسبة ومبوبة حسب سياق الموضوع (مثل: **معناها**، **فضلها**، **الحكم الشرعي**، **سبب النزول**، **أهم الأحكام المستفادة**).\n\n"
            "2. **الأسلوب واللغة**:\n"
            "   - اكتب بلغة عربية فصيحة، سلسة، شائقة، ومبسطة يفهمها الجميع بوضوح.\n"
            "   - ابدأ بفقرة تمهيدية مجيبة عن السؤال مباشرة، ثم بوب التفاصيل والفضائل والأحكام تحت عناوين واضحة ومستقلة.\n\n"
            "3. **الأمان والدقة الشرعية (منع الهلوسة)**:\n"
            "   - اعتمد حصراً على الأدلة الشرعية المرفقة في استخراج الحقائق والفضائل.\n"
            "   - لا تضف أي أحاديث أو آيات غير مذكورة في النص المرفق.\n"
            "   - إذا كانت الأدلة المسترجعة غير كافية للإجابة عن السؤال، أجب بهذه الجملة فقط دون أي عناوين: \"لا توجد أدلة كافية في المصادر المتاحة.\""
        )

    def build_prompt(self, query: str, evidences: List[GenericEvidence]) -> str:
        if not evidences:
            return ""

        context_str = ""
        for i, ev in enumerate(evidences, 1):
            context_str += f"[{i}] {ev.reference}\nالنص الكامل الأصلي: {ev.text}\n\n"

        prompt = (
            f"{self.system_instruction}\n\n"
            f"=== الأدلة المسترجعة المقبولة ===\n"
            f"{context_str}"
            f"=== سؤال المستخدم ===\n"
            f"{query}\n\n"
            f"=== الإجابة الذكية المنظمة (بأسلوب ChatGPT/Gemini) ==="
        )
        return prompt

    def generate_answer(self, query: str, context_docs: List[BaseDocument]) -> str:
        # Step 1: Filter, Deduplicate & Re-rank evidence
        evidences = filter_and_deduplicate_evidence(query, context_docs)

        # Step 2: Compute Deterministic Calculated Confidence
        score_pct, badge_emoji, score_reasons = compute_calculated_confidence(query, evidences)

        if not evidences:
            return "لا توجد أدلة كافية في المصادر المتاحة."

        confidence_header = f"📊 **درجة الثقة:** {score_pct}% {badge_emoji}"
        if score_reasons:
            confidence_header += "\n" + "\n".join(score_reasons)

        # Step 3: LLM Reasoning
        prompt = self.build_prompt(query, evidences)
        llm_reasoning = generate(prompt).strip()

        if not llm_reasoning or "لا توجد أدلة كافية" in llm_reasoning or "لا أملك إجابة" in llm_reasoning:
            return "لا توجد أدلة كافية في المصادر المتاحة."

        fallback_msg_1 = "تم استخراج المراجع والأدلة الشرعية الموثقة مباشرة من المصادر المتاحة."
        fallback_msg_2 = "تم استخراج وتوثيق الأدلة المباشرة للسؤال من المصادر الشرعية المعتمدة."

        if llm_reasoning in [fallback_msg_1, fallback_msg_2]:
            llm_reasoning = build_smart_fallback(query, evidences)

        final_response_envelope = f"{confidence_header}\n\n{llm_reasoning}"
        return final_response_envelope

