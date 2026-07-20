# -*- coding: utf-8 -*-
import json
import os
import time
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

DEV_EVAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dev_eval.json")
FAILED_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eval_failed_queries.json")

# Failure Enums
FAILURE_REASONS = {
    "hard_famous": "ALIAS_MISS",
    "incomplete_quote": "EXACT_MISS",
    "synonym_topic": "TOPIC_MISS",
    "multi_prophet": "PROPHET_MISS",
    "multi_surah_story": "STORY_MISS",
    "typo_dialect": "TYPO_FAILURE",
    "out_of_domain": "OOD_FAILURE",
    "non_famous_verse": "RANKING_ERROR",
    "fiqh_mirath": "TOPIC_MISS",
    "fiqh_riba": "TOPIC_MISS",
    "creation_topic": "TOPIC_MISS",
    "tawbah_topic": "TOPIC_MISS",
    "hard_quote": "EXACT_MISS",
    "parent_birr": "TOPIC_MISS",
    "akhlaq_topic": "TOPIC_MISS"
}

def run_dev_eval():
    start_time = time.time()
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

    with open(DEV_EVAL_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    category_stats = {}
    failure_counts = {}
    failed_queries = []

    total = len(queries)
    in_domain = 0
    hits_1 = 0
    hits_5 = 0
    mrr_sum = 0.0
    score_gaps = []

    for item in queries:
        q_id = item["id"]
        q_type = item.get("type", "General")
        q_text = item["question"]
        expected = set(item["expected_ids"])

        if q_type not in category_stats:
            category_stats[q_type] = {"total": 0, "hits_1": 0, "hits_5": 0, "mrr_sum": 0.0}
        category_stats[q_type]["total"] += 1

        if not expected:
            category_stats[q_type]["hits_1"] += 1
            category_stats[q_type]["hits_5"] += 1
            continue

        in_domain += 1
        res = pipeline.retrieve(SearchQuery(text=q_text, limit=5))
        retrieved_docs = res.documents
        retrieved_ids = [d.id for d in retrieved_docs]

        # Calculate Top 1 - Top 2 Score Gap
        if len(retrieved_docs) >= 2 and retrieved_docs[0].score is not None and retrieved_docs[1].score is not None:
            gap = round(retrieved_docs[0].score - retrieved_docs[1].score, 4)
            score_gaps.append(gap)

        h1 = any(doc_id in expected for doc_id in retrieved_ids[:1])
        h5 = any(doc_id in expected for doc_id in retrieved_ids[:5])

        rank_found = None
        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected:
                rr = 1.0 / rank
                rank_found = rank
                break

        if h1:
            hits_1 += 1
            category_stats[q_type]["hits_1"] += 1
        if h5:
            hits_5 += 1
            category_stats[q_type]["hits_5"] += 1

        mrr_sum += rr
        category_stats[q_type]["mrr_sum"] += rr

        if not h1:
            f_reason = "RANKING_ERROR" if h5 else FAILURE_REASONS.get(q_type, "RANKING_ERROR")
            failure_counts[f_reason] = failure_counts.get(f_reason, 0) + 1
            failed_queries.append({
                "id": q_id,
                "type": q_type,
                "query": q_text,
                "expected": list(expected),
                "retrieved_top5": retrieved_ids[:5],
                "rank": rank_found if rank_found else "not_in_top5",
                "failure_enum": f_reason
            })

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    h1_rate = round(hits_1 / in_domain, 4) if in_domain > 0 else 0.0
    h5_rate = round(hits_5 / in_domain, 4) if in_domain > 0 else 0.0
    mrr_rate = round(mrr_sum / in_domain, 4) if in_domain > 0 else 0.0
    avg_score_gap = round(sum(score_gaps) / len(score_gaps), 4) if score_gaps else 0.0

    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(failed_queries, f, ensure_ascii=False, indent=2)

    print("\n================ FAST DEV EVALUATION DASHBOARD ================")
    print(f"Total Queries: {total} | In-Domain: {in_domain} | Latency: {elapsed_ms}ms")
    print(f"Hits@1: {h1_rate * 100:.1f}%  | Hits@5: {h5_rate * 100:.1f}%  | MRR: {mrr_rate:.4f}")
    print(f"Average Top1-Top2 Score Gap (Confidence Metric): {avg_score_gap:.4f}")
    print("---------------- Category Breakdown ----------------")
    for cat, s in category_stats.items():
        cat_tot = s["total"]
        c1 = round(s["hits_1"] / cat_tot, 4) * 100 if cat_tot > 0 else 0.0
        c5 = round(s["hits_5"] / cat_tot, 4) * 100 if cat_tot > 0 else 0.0
        print(f" - {cat:<20}: Hits@1={c1:5.1f}% | Hits@5={c5:5.1f}% ({cat_tot} queries)")

    if failure_counts:
        print("---------------- Standardized Failure Reasons ----------------")
        for f_enum, count in failure_counts.items():
            print(f" - {f_enum:<18}: {count} queries")
    print(f"Failed Queries Log: {FAILED_LOG_PATH} ({len(failed_queries)} items)")
    print("===============================================================\n")

    return {
        "hits_1": h1_rate,
        "hits_5": h5_rate,
        "mrr": mrr_rate,
        "avg_score_gap": avg_score_gap,
        "failure_counts": failure_counts,
        "failed_count": len(failed_queries)
    }

if __name__ == "__main__":
    run_dev_eval()
