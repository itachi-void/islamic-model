# دليل رفع مشروع Islamic AI Engine على استضافة مجانية (Hugging Face Spaces)

هذا الدليل يشرح كيفية رفع مشروع **Islamic AI Engine** مجاناً 100% باستخدام **Hugging Face Spaces** والتي توفر **16GB RAM و 2vCPU مجاناً مدى الحياة**.

---

## الخطوة 1: الحصول على مفتاح Groq API المجاني (للسرعة العالية للـ LLM)

1. اذهب إلى موقع [Groq Console](https://console.groq.com/).
2. سجل حساب مجاني.
3. اضغط على **API Keys** ثم **Create API Key**.
4. انسخ المفتاح المولد (يبدأ بـ `gsk_...`).

---

## الخطوة 2: إنشاء Space على Hugging Face

1. قم بزيارة موقع [Hugging Face Spaces](https://huggingface.co/spaces).
2. اضغط على **Create new Space**.
3. ادخل البيانات التالية:
   - **Space name**: اختر اسماً (مثل `islamic-ai-engine`).
   - **License**: `mit` أو `apache-2.0`.
   - **Select the Space SDK**: اختر **Docker**.
   - **Choose a Docker template**: اختر **Blank**.
   - **Space hardware**: اختر الباقة المجانية (`CPU basic - 2 vCPU - 16 GB RAM`).
   - **Public / Private**: اختر كما تحب (Public ليراه الجميع).
4. اضغط على **Create Space**.

---

## الخطوة 3: إضافة مفتاح الـ API في إعدادات الـ Space

1. داخل الـ Space الخاص بك على Hugging Face، اذهب إلى تبويب **Settings**.
2. انزل إلى قسم **Variables and secrets**.
3. اضغط على **New secret**:
   - **Key**: `GROQ_API_KEY`
   - **Value**: ضع مفتاح Groq الذي حصلت عليه في الخطوة الأولى (gsk_...).
4. اضغط **Save**.

---

## الخطوة 4: رفع الكود باستخدام Git

افتح التيرمينال في مجلد المشروع المحلي (`d:\model`) ونفذ الأوامر التالية:

```bash
# 1. التأكد من حفظ التغيرات المحلية
git add .
git commit -m "Prepare app for Hugging Face deployment with Groq API and Docker"

# 2. ربط الـ Space بمستودع Git المحلي (استبدل YOUR_USERNAME باسم حسابك في Hugging Face)
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/islamic-ai-engine

# 3. رفع الكود
git push space main -f
```

---

## الخطوة 5: تشغيل التطبيق واختباره

1. بعد الانتهاء من الـ `git push`، سيبدأ Hugging Face في بناء الـ Docker Image وتفعيل التطبيق تلقائياً (Build & Deploy).
2. سيتغير الحالة إلى **Running**.
3. ستشاهد واجهة الموقع تعمل بشكل كامل وتجيب عن الأسئلة القرآنية والأحاديث الشريفة بأقصى سرعة مجاناً!

---

## خيار بديل: الرفع على Render (Free Tier)

إذا أردت الرفع على Render:
1. قم بإنشاء حساب على [Render.com](https://render.com/).
2. اضغط **New +** ثم **Web Service**.
3. اربط مستودع GitHub الخاص بك.
4. اختر Runtime: **Docker**.
5. أضف Environment Variable:
   - `GROQ_API_KEY` = `مفتاح_groq`
6. اضغط **Create Web Service**.
