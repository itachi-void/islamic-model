# -*- coding: utf-8 -*-
"""
Tafsir Ibn Kathir Dataset & Evaluation Benchmark Generator
"""
import os
import json

TAFSIR_DIR = r"d:\model\data\tafsir"
os.makedirs(TAFSIR_DIR, exist_ok=True)

tafsir_records = [
    {
        "id": "tafsir_1_1",
        "type": "tafsir",
        "source": "tafsir",
        "surah_number": 1,
        "ayah_number": 1,
        "surah_name_ar": "الفاتحة",
        "mufassir": "ibnkathir",
        "mufassir_name_ar": "ابن كثير",
        "title_ar": "تفسير البسملة في سورة الفاتحة",
        "text": "افتتح بها الصحابة كتاب الله تعالى، واتفق العلماء على أنها بعض آية من سورة النمل، ثم اختلفوا هل هي آية مستقلة في أول كل سورة أو في أول الفاتحة فقط أو ليست بآية. ومعنى (الله) العلم على الرب تبارك وتعالى، وقيل هو الاسم الأعظم، و(الرحمن الرحيم) اسمان مشتقان من الرحمة على وجه المبالغة، والرحمن أشد مبالغة من الرحيم.",
        "topics": ["البسملة", "أسماء الله الحسنى", "سورة الفاتحة"]
    },
    {
        "id": "tafsir_2_255",
        "type": "tafsir",
        "source": "tafsir",
        "surah_number": 2,
        "ayah_number": 255,
        "surah_name_ar": "البقرة",
        "mufassir": "ibnkathir",
        "mufassir_name_ar": "ابن كثير",
        "title_ar": "تفسير آية الكرسي",
        "text": "هذه آية الكرسي ولها شأن عظيم، قد ورد الحديث عن رسول الله صلى الله عليه وسلم بأنها أفضل آية في كتاب الله. قوله (الله لا إله إلا هو الحي القيوم) إخبار بأنه هو الأحد الصمد الذي لا إله غيره، الحي في نفسه القيوم لغيره، لا تأخذه سنة ولا نوم، أي لا يغلبه نعاس ولا نوم.",
        "topics": ["آية الكرسي", "أعظم آية في القرآن", "القيوم", "صفات الله"]
    },
    {
        "id": "tafsir_2_275",
        "type": "tafsir",
        "source": "tafsir",
        "surah_number": 2,
        "ayah_number": 275,
        "surah_name_ar": "البقرة",
        "mufassir": "ibnkathir",
        "mufassir_name_ar": "ابن كثير",
        "title_ar": "تفسير آية تحريم الربا",
        "text": "لما ذكر تعالى الأبرار المتصدقين، شرع في ذكر آكلي الربا وأكلهم أموال الناس بالباطل والشهوات الباطلة. قوله (الذين يأكلون الربا لا يقومون إلا كما يقوم الذي يتخبطه الشيطان من المس) أي لا يقومون من قبورهم يوم القيامة إلا كما يقوم المصروع حال صرعه وتخبط الشيطان له.",
        "topics": ["تحريم الربا", "عقوبة آكل الربا", "أكل أموال الناس بالباطل"]
    },
    {
        "id": "tafsir_112_1",
        "type": "tafsir",
        "source": "tafsir",
        "surah_number": 112,
        "ayah_number": 1,
        "surah_name_ar": "الإخلاص",
        "mufassir": "ibnkathir",
        "mufassir_name_ar": "ابن كثير",
        "title_ar": "تفسير سورة الإخلاص",
        "text": "قُلْ هُوَ اللَّهُ أَحَدٌ أي هو الواحد الأحد الذي لا نظير له ولا وزير ولا شريك له ولا شبيه ولا عديل. (اللَّهُ الصَّمَدُ) قال ابن عباس: الصمد الذي تصمد إليه الخلائق في حوائجهم ومسائلهم. وهي تعدل ثلث القرآن كما صح في الأحاديث.",
        "topics": ["سورة الإخلاص", "التوحيد", "معنى الصمد", "ثلث القرآن"]
    }
]

with open(os.path.join(TAFSIR_DIR, "ibnkathir.json"), "w", encoding="utf-8") as f:
    json.dump(tafsir_records, f, ensure_ascii=False, indent=2)

tafsir_eval = [
    {
        "id": "tf001",
        "query": "تفسير البسملة والفرق بين الرحمن والرحيم ابن كثير",
        "category": "Tafsir",
        "expected_ids": ["tafsir_1_1"]
    },
    {
        "id": "tf002",
        "query": "تفسير آية الكرسي ومعنى القيوم عند ابن كثير",
        "category": "Tafsir",
        "expected_ids": ["tafsir_2_255"]
    },
    {
        "id": "tf003",
        "query": "تفسير الذين يأكلون الربا لا يقومون إلا كما يقوم الذي يتخبطه الشيطان",
        "category": "Tafsir",
        "expected_ids": ["tafsir_2_275"]
    },
    {
        "id": "tf004",
        "query": "تفسير قل هو الله احد ومعنى الصمد في سورة الإخلاص",
        "category": "Tafsir",
        "expected_ids": ["tafsir_112_1"]
    }
]

with open(r"d:\model\data\evaluation_tafsir.json", "w", encoding="utf-8") as f:
    json.dump(tafsir_eval, f, ensure_ascii=False, indent=2)

print("Generated Tafsir Ibn Kathir dataset & evaluation benchmark.")
