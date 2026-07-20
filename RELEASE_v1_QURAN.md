# 📜 RELEASE v1.1 — Quran Engine Final Frozen Baseline

**Project**: Merath (تراث) — Enterprise Islamic RAG Engine  
**Module**: Quran Search Engine (`quran`)  
**Release Version**: `v1.1-frozen-baseline`  
**Date**: 2026-07-19  
**Previous Release**: v1.0-official → v1.1 upgrades listed below

---

## 🏛️ System Architecture & Routing Pipeline

```text
User Question
    │
    ├──► MetadataLookupStep (Explicit Surah/Ayah regex parsing → Direct Metadata Lookup)
    │        └─► Hits@1 = 100% (Instant 1.0 Confidence)
    │
    └──► Hybrid Retriever (Exact Keyword [IDF-Weighted] + BGE-M3 Vector Search)
            │
            ▼
        CoverageRanker (RRF Fusion + Metadata Boost + Prefix Bonus)
            │
            ▼
        MetadataFilter
            │
            ▼
        ResponseBuilder (Polymorphic Quran Envelope)
```

---

## 📦 Dataset & Collection Metadata

- **Corpus**: Complete Quranic Text (6,236 Verses).
- **ChromaDB Collection**: `quran`
- **Total Indexed Chunks**: `6,237` (Verse 2:282 split into 2 chunks due to 1000 char threshold).
- **Embedding Model**: `bge-m3` via Ollama.
- **Enriched Metadata**: `title_ar`, `aliases`, `topics`, `surah_number`, `surah_name_ar`, `ayah_number`, `revelation_type`.

---

## 📊 Official Standardized Benchmarks (500-Query Suite) — v1.1 Final

### 1. Fast Benchmark (100-Query Golden Suite - `data/evaluation_golden.json`)
- **Queries Evaluated**: 95 In-Domain Queries (5 Out-of-Domain)
- **Hits@1**: `85.26%`
- **Hits@5**: `88.42%`
- **MRR**: `0.8684`
- **Avg Latency**: `514.75 ms` per query

### 2. Official Release Benchmark (500-Query Suite - `data/evaluation_quran_500.json`)
- **Queries Evaluated**: 450 In-Domain Queries (50 Out-of-Domain)
- **Hits@1**: `71.78%`
- **Hits@5**: `86.22%`
- **MRR**: `0.7750`
- **Avg Latency**: `501.66 ms` per query


### 2. Metadata Lookup Benchmark (Explicit Surah & Ayah References)
- **Queries Evaluated**: 50 Surah/Ayah Queries (e.g. سورة البقرة آية 94)
- **Hits@1**: `100.00%`
- **Hits@5**: `100.00%`
- **MRR**: `1.0000`

### 3. Full Islamic System Benchmark (All 450 In-Domain End-to-End Queries)
- **Queries Evaluated**: 450 In-Domain Queries
- **Hits@1**: `71.78%` _(was 62.89% in v1.0 → +8.89%)_
- **Hits@5**: `86.22%` _(was 76.22% in v1.0 → +10.00%)_
- **MRR**: `0.7750` _(was 0.6819 in v1.0 → +0.0931)_
- **Average Latency**: `501.66 ms` per query

### 4. Category Breakdown (Hits@1) — v1.1
| Category            | Hits@1   | Hits@5   |
|---------------------|----------|----------|
| Exact Quote         | 94.00%   | 100.00%  |
| Surah Reference     | 100.00%  | 100.00%  |
| Colloquial          | 50.00%   | 80.00%   |
| Prophets & Persons  | 80.00%   | 100.00%  |
| Rulings & Topics    | 52.00%   | 70.67%   |
| Asbab al-Nuzul      | 60.00%   | 60.00%   |
| Out-of-domain       | 0.00%    | 0.00%    |

---

## 🔧 Optimization Iterations Applied (v1.0 → v1.1)

| Iteration | Fix Applied | Delta Hits@1 | Status |
|---|---|:---:|:---:|
| 1 | MetadataLookupStep (Surah/Ayah regex routing) | +7.55% | ✅ Kept |
| 2 | Dialectal Query Normalization (`strip_dialectal_phrases`) | +0.23% (System) | ✅ Kept |
| 3 | Exact Verse Prefix Match Bonus in CoverageRanker | +0.23% | ✅ Kept |
| 4 | Pattern A: Contiguous N-gram Phrase Matching | 0.00% | ❌ Rejected |
| 5 | Pattern C: IDF-Weighted Root/Stem Coverage Scoring | +1.11% | ✅ Kept |

---

## 🔍 Remaining Failure Analysis (127 Total Failures)

| Category | Count | % of Failures | Treatability |
|---|:---:|:---:|:---:|
| Missing Knowledge (Tafsir/Fiqh/Asbab) | 56 | 44.09% | Phase 3+ |
| Ranking Failure (target in Top-5, not Top-1) | 51 | 40.16% | Plateau reached |
| Query Normalization (unseen colloquial forms) | 20 | 15.75% | Partial (Phase 1 exhausted) |

---

## 🔒 Frozen Baseline Status — v1.1

> **⚠️ FROZEN**: This baseline is officially frozen. No further modifications to the Quran engine  
> retrieval logic, ranking, or metadata will be made unless a critical bug is discovered.

- **Code Pipeline**: Frozen in `backend/rag/search.py` (IDF-weighted ExactSearchEngine) and `backend/services/pipeline.py` (CoverageRanker + MetadataLookupStep).
- **ChromaDB Index**: Persistent snapshot at `./data/chroma/` (`quran` collection).
- **Metadata Reference**: Fixed at `./data/quran/famous_verses.json`.
- **Evaluation Benchmark**: Fixed at `./data/evaluation_quran_500.json` (500-query suite).
- **Metrics Snapshot**: `./data/metrics_quran_v1_1.json`.

---

## 🚀 Next Phase

**Phase 2**: Sahih Al-Bukhari (`bukhari` collection) — Independent retrieval engine.  
Architecture: Mirror of Phase 1 Quran engine with Hadith-specific Schema, Metadata, and Benchmark.  
**No mixing with Quran collection until Intent Router is built (Phase 7+).**
