# -*- coding: utf-8 -*-
import json
import os
import chromadb
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.settings import settings
from backend.data.loader import load_search_documents
from backend.rag.search import ExactSearchEngine
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore
from backend.domain.query import SearchQuery
from backend.services.pipeline import (
    ExactRetriever,
    SemanticRetriever,
    HybridRetriever,
    CoverageRanker,
    MetadataFilter,
    ResponseBuilder,
    RetrievalPipeline
)
from backend.services.chat_service import ChatService

client = TestClient(app)

def run_verification():
    report = {}

    # 1. Collection count verification
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    quran_col = chroma_client.get_collection("quran")
    vector_count = quran_col.count()
    report["vector_count"] = vector_count
    print(f"[1] Verified 'quran' collection vector count: {vector_count}")

    # 2. Wire production RAG pipeline
    docs = load_search_documents()
    exact_engine = ExactSearchEngine(docs)
    exact_retriever = ExactRetriever(exact_engine)
    
    embedding_provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
    chroma_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PATH,
        embedding_provider=embedding_provider,
        collection_name="quran"
    )
    semantic_retriever = SemanticRetriever(chroma_store)
    hybrid_retriever = HybridRetriever(exact_retriever, semantic_retriever)

    pipeline = RetrievalPipeline(
        retriever=hybrid_retriever,
        ranker=CoverageRanker(),
        filter_step=MetadataFilter(),
        response_builder=ResponseBuilder()
    )

    chat_service = ChatService(pipeline)

    # 3. Evaluation Benchmark (Hits@1, Hits@5, Hits@10, MRR)
    eval_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluation.json")
    with open(eval_file, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    total_queries = len(benchmarks)
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr_sum = 0.0

    eval_details = []

    for item in benchmarks:
        q = item["question"]
        expected = set(item["expected_ids"])

        # Fetch top 10 candidates
        search_query = SearchQuery(text=q, limit=10)
        res = pipeline.retrieve(search_query)
        retrieved_ids = [doc.id for doc in res.documents]

        h1 = any(doc_id in expected for doc_id in retrieved_ids[:1])
        h5 = any(doc_id in expected for doc_id in retrieved_ids[:5])
        h10 = any(doc_id in expected for doc_id in retrieved_ids[:10])

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected:
                rr = 1.0 / rank
                break

        if h1: hits_at_1 += 1
        if h5: hits_at_5 += 1
        if h10: hits_at_10 += 1
        mrr_sum += rr

        eval_details.append({
            "question": q,
            "expected": list(expected),
            "retrieved_top_5": retrieved_ids[:5],
            "hits_at_5": h5,
            "rr": round(rr, 4)
        })

    report["evaluation_metrics"] = {
        "total_queries": total_queries,
        "hits_at_1": round(hits_at_1 / total_queries, 4),
        "hits_at_5": round(hits_at_5 / total_queries, 4),
        "hits_at_10": round(hits_at_10 / total_queries, 4),
        "mrr": round(mrr_sum / total_queries, 4),
        "details": eval_details
    }

    print(f"[2] Evaluation Benchmark Completed: Hits@1={report['evaluation_metrics']['hits_at_1']}, Hits@5={report['evaluation_metrics']['hits_at_5']}, MRR={report['evaluation_metrics']['mrr']}")

    # 4. End-to-End Chat Endpoint Testing (4 test queries)
    test_queries = [
        "ما هي آية الكرسي؟",
        "الله لا إله إلا هو الحي القيوم",
        "ما حكم الصبر في القرآن؟",
        "كم عدد كواكب المجرة؟"
    ]

    chat_results = []
    for tq in test_queries:
        print(f"Executing Chat query: '{tq}'...")
        res = chat_service.chat(tq)
        chat_results.append({
            "query": tq,
            "response": res
        })

    report["chat_test_results"] = chat_results

    # Output JSON report
    report_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "verification_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[3] Full verification report written to {report_file}")
    return report

if __name__ == "__main__":
    run_verification()
