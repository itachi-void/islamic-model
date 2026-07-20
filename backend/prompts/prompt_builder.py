# -*- coding: utf-8 -*-
from typing import List
from backend.domain.document import BaseDocument

class BasePromptBuilder:
    def build_prompt(self, query: str, context: List[BaseDocument]) -> str:
        """
        Builds a combined LLM prompt from the query and retrieved context documents.
        """
        raise NotImplementedError("Subclasses must implement build_prompt")


class IslamicPromptBuilder(BasePromptBuilder):
    def __init__(self, system_instruction: str = None):
        if system_instruction is None:
            self.system_instruction = (
                "أنت مساعد ذكاء اصطناعي إسلامي متخصص وموثوق للغاية. أجب عن أسئلة المستخدم بالاعتماد فقط وحصراً على السياق والأدلة المرفقة أدناه.\n"
                "يجب عليك اتباع القواعد الصارمة التالية لمنع الهلوسة العلمية والشرعية وتطبيق هيكلية الإجابة المنظمة:\n"
                "1. أجب فقط وحصراً من الأدلة والمستندات المقدمة أدناه ولا تستخدم معارفك الخارجية إذا لم تكن مدعومة بالسياق.\n"
                "2. إذا لم تجد الإجابة مباشرة أو لم تكن الأدلة كافية، أجب حصراً بـ: \"لا توجد أدلة كافية في المصادر المتاحة.\"\n"
                "3. لا تقم بتأليف أو اختراع أي آية قرآنية أو حديث شريف أو حكم فقهي على الإطلاق.\n"
                "4. صغ إجابتك مقسمة بدقة إلى الأقسام التالية عندما تتوفر أدلتها:\n"
                "   📌 الإجابة المختصرة:\n"
                "   📖 الأدلة من القرآن الكريم:\n"
                "   📚 الأدلة من السنة النبوية:\n"
                "   ⚖️ آراء المذاهب الأربعة (إن وجد اختلاف أو ذكر فقهي في السياق):\n"
                "   🔗 المراجع الموثقة:\n"
                "5. يجب عليك إدراج رقم مرجع المصدر المستخدم بين قوسين مثل [1] أو [2] في نهاية كل استشهاد."
            )
        else:
            self.system_instruction = system_instruction


    def build_prompt(self, query: str, context: List[BaseDocument]) -> str:
        context_str = ""
        for i, doc in enumerate(context, 1):
            source_type = doc.source
            if source_type == "quran":
                surah = doc.metadata.get("surah_name_ar", "")
                ayah = doc.metadata.get("ayah_number", "")
                source_info = f"📖 القرآن الكريم | سورة {surah} - آية {ayah}"
            elif source_type in ["bukhari", "hadith"]:
                book = doc.metadata.get("book", "")
                h_num = doc.metadata.get("hadith_number", "")
                narrator = doc.metadata.get("narrator", "")
                narrator_str = f" - عن {narrator}" if narrator else ""
                source_info = f"📚 صحيح البخاري | {book} (حديث #{h_num}{narrator_str})"
            else:
                title = doc.metadata.get("title_ar", doc.source)
                source_info = f"المصدر: {title}"

            raw_text = doc.metadata.get("original_text", doc.text)
            context_str += f"[{i}] {source_info}:\nالنص: {raw_text}\n\n"

        prompt = (
            f"تعليمات النظام:\n{self.system_instruction}\n\n"
            f"السياق والأدلة المتاحة:\n{context_str}\n"
            f"سؤال المستخدم:\n{query}\n\n"
            f"الإجابة الموثقة مع إدراج أرقام المراجع المستخدمة مثل [1]:"
        )
        return prompt

