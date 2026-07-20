# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.rag.router import QueryRouter

client = TestClient(app)


def test_query_router():
    router = QueryRouter()

    # Quran queries
    assert router.route("ما معنى هذه الآية من سورة البقرة") == "quran"
    assert router.route("قال الله تعالى في التنزيل") == "quran"

    # Hadith queries
    assert router.route("قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات") == "hadith"
    assert router.route("ما الحديث الذي يرويه البخاري عن أبي هريرة") == "hadith"

    # Hybrid queries
    assert router.route("ما حكم الصبر في الإسلام") == "hybrid"


def test_hadith_search_endpoint_get():
    response = client.get("/hadith/search?q=النيات&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "documents" in data
    assert data["count"] > 0
    assert data["documents"][0]["source"] == "bukhari"


def test_hadith_search_endpoint_post():
    response = client.post("/hadith/search", json={"q": "النيات", "limit": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    assert data["documents"][0]["source"] == "bukhari"


def test_chat_stream_endpoint():
    response = client.get("/chat/stream?q=النيات&limit=2")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

