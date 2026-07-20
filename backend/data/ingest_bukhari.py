# -*- coding: utf-8 -*-
"""
Phase 2: Sahih Al-Bukhari Ingestion Pipeline
============================================
Downloads, validates, normalizes, and indexes all Sahih Al-Bukhari hadiths
into an independent ChromaDB collection named 'bukhari'.

Schema per Hadith:
  id           : "bukhari_{hadith_number}"
  source       : "bukhari"
  book         : Arabic book name (كتاب الإيمان)
  book_number  : integer
  chapter      : Arabic chapter name (باب ...)
  chapter_number: integer
  hadith_number: integer
  narrator     : narrator chain (أبو هريرة, etc.)
  matn         : main hadith text (Arabic)
  grade        : "Sahih"
  topics       : list of thematic topics
  keywords     : key term list
"""
import sys
import os
import json
import re
import time
import argparse

sys.path.insert(0, r"d:\model")

from backend.core.settings import settings
from backend.domain.document import BaseDocument
from backend.rag.embeddings import BGEEmbeddingProvider
from backend.rag.vector_store import ChromaVectorStore

BUKHARI_RAW_PATH = r"d:\model\data\bukhari\bukhari_raw.json"
BUKHARI_PROCESSED_PATH = r"d:\model\data\bukhari\bukhari_processed.json"
COLLECTION_NAME = "bukhari"


def load_raw_bukhari() -> list:
    """Load raw Bukhari JSON file."""
    if not os.path.exists(BUKHARI_RAW_PATH):
        raise FileNotFoundError(
            f"Bukhari raw data not found at {BUKHARI_RAW_PATH}\n"
            "Please run: python scratch/download_bukhari.py first."
        )
    with open(BUKHARI_RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_arabic_text(text: str) -> str:
    """Light normalization for storage (preserve original diacritics for display)."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def validate_hadith(record: dict, idx: int) -> bool:
    """Validates a single hadith record has required fields."""
    required = ["hadith_number", "matn"]
    for field in required:
        if not record.get(field):
            print(f"[SKIP] Record #{idx} missing field: {field}")
            return False
    if len(str(record.get("matn", ""))) < 10:
        print(f"[SKIP] Record #{idx} matn too short.")
        return False
    return True


def enrich_topics(record: dict) -> list:
    """
    Derives thematic topics from book/chapter metadata.
    These are used by ExactSearchEngine for topic-based retrieval.
    """
    topics = []
    book = record.get("book", "")
    chapter = record.get("chapter", "")
    if book:
        topics.append(book)
    if chapter:
        topics.append(chapter)
    # Add narrator as a topic alias for person-based queries
    narrator = record.get("narrator", "")
    if narrator:
        topics.append(f"حديث {narrator}")
    return topics


def build_doc_id(record: dict) -> str:
    """Builds a unique stable document ID for a Bukhari hadith."""
    return f"bukhari_{record['hadith_number']}"


def process_bukhari(raw_records: list) -> list:
    """Validates, normalizes, and enriches all Bukhari hadith records."""
    processed = []
    skipped = 0

    for idx, record in enumerate(raw_records):
        if not validate_hadith(record, idx):
            skipped += 1
            continue

        doc_id = build_doc_id(record)
        topics = enrich_topics(record)

        processed_record = {
            "id": doc_id,
            "source": "bukhari",
            "book": normalize_arabic_text(record.get("book", "")),
            "book_number": record.get("book_number"),
            "chapter": normalize_arabic_text(record.get("chapter", "")),
            "chapter_number": record.get("chapter_number"),
            "hadith_number": int(record["hadith_number"]),
            "narrator": normalize_arabic_text(record.get("narrator", "")),
            "matn": normalize_arabic_text(record["matn"]),
            "sanad": normalize_arabic_text(record.get("sanad", "")),
            "grade": record.get("grade", "صحيح"),
            "topics": topics,
            "keywords": record.get("keywords", []),
            "aliases": record.get("aliases", []),
        }
        processed.append(processed_record)

    print(f"\n[Validation] Total: {len(raw_records)} | Processed: {len(processed)} | Skipped: {skipped}")
    return processed


def build_chroma_documents(processed_records: list) -> list:
    """Converts processed hadith records to BaseDocument objects for ChromaDB.
    
    ChromaDB metadata rules:
    - Values must be str, int, float, or bool.
    - Empty lists are NOT allowed.
    - Lists are serialized as pipe-separated strings.
    """
    def serialize_list(val) -> str:
        """Convert a list to a pipe-separated string; return '' if empty."""
        if isinstance(val, list):
            return " | ".join(str(v) for v in val if v)
        return str(val) if val else ""

    docs = []
    seen_ids = {}
    for record in processed_records:
        raw_meta = {
            "source": record["source"],
            "book": record.get("book", ""),
            "book_number": record.get("book_number", 0),
            "chapter": record.get("chapter", ""),
            "chapter_number": record.get("chapter_number", 0),
            "hadith_number": record["hadith_number"],
            "narrator": record.get("narrator", ""),
            "grade": record.get("grade", "صحيح"),
            "topics": serialize_list(record.get("topics", [])),
            "keywords": serialize_list(record.get("keywords", [])),
            "aliases": serialize_list(record.get("aliases", [])),
            "sanad": record.get("sanad", ""),
        }
        # Keep only non-empty, non-None values
        meta = {k: v for k, v in raw_meta.items() if v is not None and v != "" and v != 0}
        # Always keep required fields
        meta["source"] = raw_meta["source"]
        meta["hadith_number"] = raw_meta["hadith_number"]
        meta["book_number"] = raw_meta.get("book_number", 0)
        meta["grade"] = raw_meta.get("grade", "صحيح")

        # Guarantee unique ID even if hadith_number repeats across narrations
        base_id = record["id"]
        if base_id in seen_ids:
            seen_ids[base_id] += 1
            unique_id = f"{base_id}_{seen_ids[base_id]}"
        else:
            seen_ids[base_id] = 0
            unique_id = base_id

        docs.append(BaseDocument(
            id=unique_id,
            type="hadith",
            source="bukhari",
            text=record["matn"],
            metadata=meta
        ))
    return docs


def ingest_to_chroma(docs: list, reset: bool = False):
    """Embeds and indexes all documents into the 'bukhari' ChromaDB collection."""
    embedding_provider = BGEEmbeddingProvider(settings.EMBEDDING_MODEL)
    chroma_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PATH,
        embedding_provider=embedding_provider,
        collection_name=COLLECTION_NAME
    )

    if reset:
        print(f"[Ingestion] Resetting existing '{COLLECTION_NAME}' collection...")
        chroma_store.reset_collection()

    try:
        existing_ids = set(chroma_store.collection.get(include=[])["ids"])
    except Exception:
        existing_ids = set()

    docs_to_index = [d for d in docs if d.id not in existing_ids]

    print(f"[Ingestion] Total documents: {len(docs)} | Already indexed: {len(existing_ids)} | Remaining to index: {len(docs_to_index)}")

    if not docs_to_index:
        print(f"[Ingestion] ✅ All {len(docs)} hadiths are already indexed in collection '{COLLECTION_NAME}'. Nothing to do!")
        return

    BATCH_SIZE = 50
    total = len(docs_to_index)
    print(f"[Ingestion] Starting batch indexing for {total} remaining hadiths → ChromaDB collection '{COLLECTION_NAME}'")

    for i in range(0, total, BATCH_SIZE):
        batch = docs_to_index[i:i + BATCH_SIZE]
        chroma_store.add_documents(batch)
        processed_count = len(existing_ids) + i + len(batch)
        pct = min(100.0, round(processed_count / len(docs) * 100, 1))
        print(f"  [{pct}%] Indexed {processed_count}/{len(docs)} hadiths...", end="\r")
        time.sleep(0.05)  # avoid hammering Ollama

    print(f"\n[Ingestion] ✅ Done. {len(docs)} hadiths indexed in collection '{COLLECTION_NAME}'.")


def main():
    parser = argparse.ArgumentParser(description="Sahih Al-Bukhari Ingestion Pipeline")
    parser.add_argument("--reset", action="store_true", help="Reset collection before indexing")
    args, _ = parser.parse_known_args()

    os.makedirs(os.path.dirname(BUKHARI_RAW_PATH), exist_ok=True)

    print("=" * 70)
    print("PHASE 2: SAHIH AL-BUKHARI INGESTION PIPELINE")
    print("=" * 70)

    # Step 1: Load raw data
    print("\n[Step 1] Loading raw Bukhari data...")
    raw_records = load_raw_bukhari()
    print(f"  Loaded {len(raw_records)} raw records.")

    # Step 2: Validate + normalize + enrich
    print("\n[Step 2] Validating, normalizing, enriching metadata...")
    processed = process_bukhari(raw_records)

    # Step 3: Save processed JSON for reference
    print(f"\n[Step 3] Saving processed data → {BUKHARI_PROCESSED_PATH}")
    with open(BUKHARI_PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(processed)} processed hadiths.")

    # Step 4: Build BaseDocument objects
    print("\n[Step 4] Building ChromaDB document objects...")
    docs = build_chroma_documents(processed)
    print(f"  Built {len(docs)} document objects.")

    # Step 5: Ingest into ChromaDB
    print("\n[Step 5] Embedding and indexing into ChromaDB...")
    ingest_to_chroma(docs, reset=args.reset)

    print("\n" + "=" * 70)
    print("PHASE 2 INGESTION COMPLETE")
    print(f"  Collection : '{COLLECTION_NAME}'")
    print(f"  Total Docs : {len(docs)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

