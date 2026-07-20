# -*- coding: utf-8 -*-
import json
import re
from typing import Dict, Any, List, Optional
from backend.core.settings import settings
from backend.domain.query import SearchQuery
from backend.domain.citation import KnowledgeCitation
from backend.services.pipeline import RetrievalPipeline
from backend.rag.router import QueryRouter
from backend.rag.hadith_search import HadithSearchService
from backend.llm.answer_generator import AnswerGenerator


class ChatService:
    def __init__(self, quran_pipeline: RetrievalPipeline, hadith_service: Optional[HadithSearchService] = None):
        self.quran_pipeline = quran_pipeline
        self.hadith_service = hadith_service or HadithSearchService()
        self.router = QueryRouter()
        self.answer_generator = AnswerGenerator()

    def _prepare_context(self, query_text: str, limit: int = 5, filters: Dict[str, Any] = None, domain: Optional[str] = None):
        target_domain = domain if domain in ["quran", "hadith", "hybrid"] else self.router.route(query_text)
        search_query = SearchQuery(text=query_text, limit=limit, filters=filters)

        if target_domain == "quran":
            quran_resp = self.quran_pipeline.retrieve(search_query)
            retrieved_docs = quran_resp.documents
        elif target_domain == "hadith":
            hadith_resp = self.hadith_service.search(query_text, limit=limit)
            retrieved_docs = hadith_resp.documents
        else:  # hybrid search
            quran_resp = self.quran_pipeline.retrieve(search_query)
            hadith_resp = self.hadith_service.search(query_text, limit=limit)
            all_docs = quran_resp.documents + hadith_resp.documents
            all_docs.sort(key=lambda d: d.score or 0.0, reverse=True)
            retrieved_docs = all_docs[:limit]

        citations = []
        for doc in retrieved_docs:
            source_type = doc.source
            surah_val = doc.metadata.get("surah_name_ar")
            ayah_val = doc.metadata.get("ayah_number")

            if source_type == "quran":
                title = f"سورة {surah_val}"
                ref = f"سورة {surah_val}: {ayah_val}"
            elif source_type in ["bukhari", "hadith"]:
                book = doc.metadata.get("book", "")
                h_num = doc.metadata.get("hadith_number", "")
                narrator = doc.metadata.get("narrator", "")
                title = f"صحيح البخاري - {book}"
                ref = f"حديث رقم {h_num} (الراوي: {narrator})" if narrator else f"حديث رقم {h_num}"
            else:
                title = doc.metadata.get("title_ar", doc.source)
                ref = doc.id

            cit = KnowledgeCitation(
                id=doc.id,
                source=doc.source,
                source_type=source_type,
                title=title,
                reference=ref,
                text=doc.text,
                surah=surah_val,
                ayah=ayah_val,
                confidence=round(doc.score or 1.0, 4),
                details=doc.metadata
            )
            citations.append(cit.model_dump())

        return target_domain, retrieved_docs, citations

    def chat(self, query_text: str, limit: int = 5, filters: Dict[str, Any] = None, domain: Optional[str] = None) -> Dict[str, Any]:
        target_domain, retrieved_docs, citations = self._prepare_context(query_text, limit, filters, domain)
        answer = self.answer_generator.generate_answer(query_text, retrieved_docs)

        if "لا توجد أدلة كافية" in answer or not retrieved_docs:
            return {
                "answer": "لا توجد أدلة كافية في المصادر المتاحة.",
                "route": target_domain,
                "citations": [],
                "retrieved_documents": [doc.model_dump() for doc in retrieved_docs],
                "model": settings.MODEL_NAME,
                "search_type": target_domain
            }

        return {
            "answer": answer,
            "route": target_domain,
            "citations": citations,
            "retrieved_documents": [doc.model_dump() for doc in retrieved_docs],
            "model": settings.MODEL_NAME,
            "search_type": target_domain
        }

    def chat_stream(self, query_text: str, limit: int = 5, filters: Dict[str, Any] = None, domain: Optional[str] = None):
        target_domain, retrieved_docs, citations = self._prepare_context(query_text, limit, filters, domain)

        meta_payload = {
            "type": "meta",
            "route": target_domain,
            "citations": citations,
            "model": settings.MODEL_NAME
        }
        yield f"data: {json.dumps(meta_payload, ensure_ascii=False)}\n\n"

        if not retrieved_docs:
            no_evidence = {"type": "token", "content": "لا توجد أدلة كافية في المصادر المتاحة."}
            yield f"data: {json.dumps(no_evidence, ensure_ascii=False)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
            return

        for token in self.answer_generator.generate_answer_stream(query_text, retrieved_docs):
            chunk = {"type": "token", "content": token}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"


