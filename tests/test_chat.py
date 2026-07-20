# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.domain.document import BaseDocument
from backend.prompts.prompt_builder import IslamicPromptBuilder
from backend.services.chat_service import ChatService
from backend.services.pipeline import (
    ExactRetriever,
    SemanticRetriever,
    HybridRetriever,
    CoverageRanker,
    MetadataFilter,
    ResponseBuilder,
    RetrievalPipeline
)
from backend.rag.search import BaseSearchEngine

client = TestClient(app)

class MockSearchEngine(BaseSearchEngine):
    def __init__(self, docs):
        self.docs = docs

    def search(self, query: str, limit: int = 5):
        return [d for d in self.docs if query in d.text][:limit]


def test_anti_hallucination_prompt_building():
    """Verify prompt builder embeds required anti-hallucination instructions."""
    builder = IslamicPromptBuilder()
    doc = BaseDocument(
        id="2:255",
        type="verse",
        source="quran",
        text="اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
        metadata={"surah_name_ar": "البقرة", "ayah_number": 255}
    )
    prompt = builder.build_prompt("ما هي آية الكرسي؟", [doc])
    
    assert "لا تقم بتأليف أو اختراع أي آية قرآنية أو حديث شريف" in prompt
    assert "لا توجد أدلة كافية في المصادر المتاحة." in prompt
    assert "سورة البقرة - آية 255" in prompt


def test_reindex_endpoint():
    """Verify POST /reindex returns 200 and triggers background ingestion."""
    response = client.post("/reindex")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "background" in data["message"]
