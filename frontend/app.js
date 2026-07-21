let currentDomain = "hybrid";
let showSanad = false;

function switchTab(tabName) {
    document.getElementById("tab-chat").classList.toggle("active", tabName === "chat");
    document.getElementById("tab-search").classList.toggle("active", tabName === "search");

    document.getElementById("view-chat").classList.toggle("hidden", tabName !== "chat");
    document.getElementById("view-search").classList.toggle("hidden", tabName !== "search");
}

function selectDomain(btn, domain) {
    document.querySelectorAll(".pill-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentDomain = domain;
}

function toggleSanadDisplay(checked) {
    showSanad = checked;
    document.querySelectorAll(".citation-card").forEach(card => {
        card.classList.toggle("show-sanad", showSanad);
    });
}

function sendSampleQuery(queryText) {
    document.getElementById("chat-input").value = queryText;
    handleChatSubmit(new Event('submit'));
}

async function handleChatSubmit(event) {
    if (event) event.preventDefault();
    const input = document.getElementById("chat-input");
    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    appendUserMessage(text);

    const { msgId, textElem, citationsElem } = createAssistantStreamingMessage();
    fetchStandardChat(text, textElem, citationsElem);
}

async function fetchStandardChat(text, textElem, citationsElem) {
    try {
        let url = `/chat?q=${encodeURIComponent(text)}&limit=5`;
        if (currentDomain !== "hybrid") {
            url += `&domain=${currentDomain}`;
        }
        const res = await fetch(url);
        if (!res.ok) throw new Error("HTTP error");
        const data = await res.json();
        
        if (data.citations) {
            renderCitations(data.citations, citationsElem);
        }
        if (data.answer) {
            textElem.innerHTML = formatParagraphs(escapeHtml(data.answer));
        } else {
            textElem.innerText = "لم يتم الحصول على إجابة من السيرفر.";
        }
        scrollToBottom();
    } catch (err) {
        textElem.innerText = "⚠️ تعذر الحصول على الإجابة من السيرفر.";
    }
}

function createAssistantStreamingMessage() {
    const history = document.getElementById("chat-history");
    const msgId = "assistant-msg-" + Date.now();
    const msg = document.createElement("div");
    msg.className = "message assistant-message";
    msg.id = msgId;

    msg.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="msg-body">
            <button class="copy-btn" onclick="copyMessageText('${msgId}')">📋 نسخ الإجابة</button>
            <div class="text-content"><p>جاري الربط والمراجعة مع الأدلة... ⏳</p></div>
            <div class="citations-box-container"></div>
        </div>
    `;

    history.appendChild(msg);
    scrollToBottom();

    return {
        msgId,
        textElem: msg.querySelector(".text-content"),
        citationsElem: msg.querySelector(".citations-box-container")
    };
}

function renderCitations(citations, container) {
    if (!citations || citations.length === 0) return;

    let html = `
        <div class="citations-box">
            <div class="citations-title">📌 الأدلة والمراجع المستشهد بها:</div>
    `;

    citations.forEach(cit => {
        const isQuran = cit.source === "quran";
        const badgeClass = isQuran ? "quran" : "bukhari";
        const cardClass = isQuran ? "quran-card" : "bukhari-card" + (showSanad ? " show-sanad" : "");
        const badgeText = isQuran ? "📖 القرآن الكريم" : "📚 صحيح البخاري";
        const sanadText = cit.details && cit.details.sanad ? `الراوي/السند: ${cit.details.sanad}` : "";

        html += `
            <div class="citation-card ${cardClass}">
                <div class="cit-header">
                    <span class="cit-badge ${badgeClass}">${badgeText}</span>
                    <span class="cit-ref">${cit.title} (${cit.reference})</span>
                </div>
                ${sanadText ? `<div class="cit-sanad">${sanadText}</div>` : ""}
                <div class="cit-text">${cit.text}</div>
            </div>
        `;
    });

    html += `</div>`;
    container.innerHTML = html;
}

async function handleSearchSubmit(event) {
    if (event) event.preventDefault();
    const input = document.getElementById("search-input");
    const text = input.value.trim();
    if (!text) return;

    const resultsContainer = document.getElementById("search-results");
    resultsContainer.innerHTML = '<div class="placeholder-text">جاري البحث في المصادر...</div>';

    try {
        let endpoint = currentDomain === "hadith" ? `/hadith/search?q=${encodeURIComponent(text)}` : `/search?q=${encodeURIComponent(text)}`;
        const res = await fetch(endpoint);
        const data = await res.json();

        resultsContainer.innerHTML = "";
        const docs = data.documents || [];

        if (docs.length === 0) {
            resultsContainer.innerHTML = '<div class="placeholder-text">لم يتم العثور على نتائج تطابق البحث.</div>';
            return;
        }

        docs.forEach(doc => {
            const card = document.createElement("div");
            const isQuran = doc.source === "quran";
            card.className = `citation-card ${isQuran ? "quran-card" : "bukhari-card"}${showSanad ? " show-sanad" : ""}`;

            const badgeText = isQuran ? "📖 القرآن الكريم" : "📚 صحيح البخاري";
            const badgeClass = isQuran ? "quran" : "bukhari";

            let refText = "";
            if (isQuran) {
                refText = `سورة ${doc.metadata.surah_name_ar || ""} (آية ${doc.metadata.ayah_number || ""})`;
            } else {
                refText = `كتاب ${doc.metadata.book || ""} (حديث #${doc.metadata.hadith_number || ""})`;
            }

            const sanadText = doc.metadata && doc.metadata.sanad ? `الراوي/السند: ${doc.metadata.sanad}` : "";

            card.innerHTML = `
                <div class="cit-header">
                    <span class="cit-badge ${badgeClass}">${badgeText}</span>
                    <span class="cit-ref">${refText}</span>
                </div>
                ${sanadText ? `<div class="cit-sanad">${sanadText}</div>` : ""}
                <div class="cit-text">${doc.text}</div>
            `;
            resultsContainer.appendChild(card);
        });
    } catch (err) {
        resultsContainer.innerHTML = '<div class="placeholder-text">⚠️ حدث خطأ في الاتصال بالخادم.</div>';
    }
}

function appendUserMessage(text) {
    const history = document.getElementById("chat-history");
    const msg = document.createElement("div");
    msg.className = "message user-message";
    msg.innerHTML = `
        <div class="msg-avatar">👤</div>
        <div class="msg-body"><p>${escapeHtml(text)}</p></div>
    `;
    history.appendChild(msg);
    scrollToBottom();
}

function copyMessageText(msgId) {
    const msgElem = document.getElementById(msgId);
    if (!msgElem) return;
    const textContent = msgElem.querySelector(".text-content").innerText;
    navigator.clipboard.writeText(textContent).then(() => {
        const btn = msgElem.querySelector(".copy-btn");
        btn.innerText = "✅ تم النسخ!";
        setTimeout(() => btn.innerText = "📋 نسخ الإجابة", 2000);
    });
}

function scrollToBottom() {
    const history = document.getElementById("chat-history");
    history.scrollTop = history.scrollHeight;
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatParagraphs(text) {
    return text.replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br>");
}
