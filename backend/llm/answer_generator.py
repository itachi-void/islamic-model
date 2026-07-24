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


class AnswerGenerator:
    """
    Post-Retrieval Reasoning Layer.
    LLM is restricted purely to explanation and reasoning sections.
    Facts (Verses, Hadiths, Citations) are deterministically attached by FactsBuilder.
    """

    def __init__(self, dialect: str = "egyptian"):
        self.dialect = dialect
        self.system_instruction = (
            "أنت مساعد إسلامي متخصص وموثوق للغاية، تعتمد حصراً على الأدلة الشرعية المسترجعة لمنع الهلوسة.\n"
            "اقرأ سؤال المستخدم والأدلة المرفقة، واكتب فقط الأقسام التفسيرية والاستدلالية التالية بالترتيب الصارم دون إعادة كتابة آيات أو أحاديث أو مراجع كأقسام مستقلة:\n\n"
            "📌 الإجابة المختصرة\n"
            "(إجابة علمية مقتضبة بدقة باللغة العربية الفصحى تعتمد حصراً على الأدلة المسترجعة دون أي زيادة أو ادعاءات غير مذكورة)\n\n"
            "💡 شرح مبسط\n"
            "الفصحى المبسطة:\n"
            "(شرح ميسر بلغة عربية مبسطة يوضح الفكرة العامة من الدليل الشرعي دون استخدام كلمات معقدة)\n\n"
            "🇪🇬 بالمصري:\n"
            "(شرح سلس جداً بالعامية المصرية البسيطة يفهمه الجميع بوضوح دون تغيير المعنى الشرعي أو إضافة معلومات خارجية)\n\n"
            "🧠 كيف استنتجنا الإجابة؟\n"
            "(توضيح كيفية الاستدلال بالنص وكيف يدل على الإجابة طبقاً لنوع السؤال: إن كان \"لماذا\" يوضح الحكمة والعلة المذكورة، وإن كان \"ما حكم\" يوضح وجه الدلالة المباشر)\n\n"
            "⚠️ ملاحظات مهمة\n"
            "(ذكر أي تنبيهات شرعية أو سياق نزول أو سبب ورود مذكور في الأدلة، أو كتابة \"لا توجد ملاحظات إضافية.\" إذا لم تتوافر ملاحظات)\n\n"
            "🚫 قواعد صارمة يمنع مخالفتها حتماً:\n"
            "1. إذا كانت الأدلة متعددة، فلا تعتمد على دليل واحد وتتجاهل الباقي. استخلص الإجابة والوجه الاستدلالي من جميع الأدلة المرفقة، وإذا تعارضت الأدلة أو لم تكفِ فاذكر ذلك صراحة.\n"
            "2. يمنع منعاً باتاً استخدام مصطلحات مثل: \"متفق عليه\" (إلا إذا ورد البخاري ومسلم معاً صراحة في الأدلة المرفقة)، أو \"أجمع العلماء\"، أو \"جمهور العلماء\"، أو \"أصح الأقوال\"، أو ادعاء \"الأجر الكامل\" مالم يرد ذلك بنصه في الأدلة.\n"
            "3. إذا كان الحديث من صحيح البخاري اذكر صراحة أنه في صحيح البخاري فقط ولا تقل \"متفق عليه\".\n"
            "4. لا تكرر كتابة النصوص الأصلية للآيات أو الأحاديث أو قائمة المراجع المطبوعة لاحقاً، فقط صغ الشرح والاستدلال الإنساني.\n"
            "5. إذا كانت الأدلة المسترجعة غير كافية أو خارج الموضوع، أجب بهذه الجملة فقط دون أي عناوين أو إضافات أخرى: \"لا توجد أدلة كافية في المصادر المتاحة.\""
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
            f"=== الشرح والاستدلال المنظم ==="
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

        fallback_msg = "تم استخراج المراجع والأدلة الشرعية الموثقة مباشرة من المصادر المتاحة."
        fallback_msg_2 = "تم استخراج وتوثيق الأدلة المباشرة للسؤال من المصادر الشرعية المعتمدة."
        if llm_reasoning == fallback_msg:
            llm_reasoning = fallback_msg_2

        final_response_envelope = f"{confidence_header}\n\n{llm_reasoning}"
        return final_response_envelope

