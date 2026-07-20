# Islamic AI Data Architecture (Retrieval-Focused)

Directory structure dedicated to Knowledge Ingestion, Retrieval Benchmarks, Diagnostic Failure Logging, and Regression Prevention:

```
data/
├── knowledge/          # Core text corpora (Quran, Bukhari, Tafsir, Fiqh)
│   ├── quran/
│   └── bukhari/
├── benchmarks/         # Standard benchmark evaluation files (Golden 100, Quran 500, Bukhari 500)
├── retrieval_queries/  # Curated test queries, phrasing variants, and domain synonyms
├── metadata/           # Surah metadata, Hadith book schemas, narrator lists, and topic tags
├── failures/           # Diagnostic logs of retrieval misses and failure analyses
└── regression/         # Permanent regression test suite (regression_suite.json)
```

## Running Regression Tests

To verify that no retrieval improvements cause regressions:

```bash
python tests/run_regression.py
```
