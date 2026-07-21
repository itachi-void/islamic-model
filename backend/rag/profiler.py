# -*- coding: utf-8 -*-
"""
Stage-by-Stage Retrieval Pipeline Latency Profiler
===================================================
Measures millisecond execution time across all 7 stages of retrieval:
1. Normalize Query & Dialect Stripping
2. Knowledge Graph Entity Expansion
3. Exact Search & BM25 Scoring
4. Chroma Vector Store Embedding & Similarity Search
5. Reciprocal Rank Fusion (RRF)
6. Neural Cross-Encoder Re-ranking
7. Candidate Filtering & Sorting
"""
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PipelineProfiler:
    """Stage-by-stage execution time tracker."""
    def __init__(self):
        self.stage_times_ms: Dict[str, float] = {}

    def measure(self, stage_name: str, func, *args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        elapsed = (time.time() - t0) * 1000.0
        self.stage_times_ms[stage_name] = round(elapsed, 2)
        return res

    def get_summary(self) -> Dict[str, float]:
        total = sum(self.stage_times_ms.values())
        summary = dict(self.stage_times_ms)
        summary["total_pipeline_ms"] = round(total, 2)
        return summary

    def log_breakdown(self):
        summary = self.get_summary()
        logger.info(f"=== PIPELINE LATENCY PROFILING (Total: {summary['total_pipeline_ms']} ms) ===")
        for stage, ms in self.stage_times_ms.items():
            pct = (ms / summary['total_pipeline_ms'] * 100) if summary['total_pipeline_ms'] > 0 else 0
            logger.info(f"  - {stage:<35}: {ms:>7.2f} ms ({pct:>5.1f}%)")


if __name__ == "__main__":
    profiler = PipelineProfiler()
    profiler.measure("dummy_stage", time.sleep, 0.05)
    profiler.log_breakdown()
