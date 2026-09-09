"""Memory Profiler for UAQE inference and candidate evaluation.

Profiles process RSS, memory deltas across optimization phases, and ensures
systematic garbage collection between candidate runs to avoid accumulation of
temporary NumPy arrays and PyTorch tensors.
"""

from __future__ import annotations

import os
import gc
import psutil
from typing import Dict, Any, Optional


class MemoryProfiler:
    """Monitors process memory footprint and temporary buffer utilization."""

    def __init__(self):
        self._process = psutil.Process(os.getpid())
        self._peak_rss_mb = 0.0
        self._baseline_rss_mb = self.get_current_rss_mb()

    def get_current_rss_mb(self) -> float:
        """Return current resident set size in megabytes."""
        rss = self._process.memory_info().rss / (1024.0 * 1024.0)
        if rss > self._peak_rss_mb:
            self._peak_rss_mb = rss
        return round(rss, 2)

    def get_peak_rss_mb(self) -> float:
        return round(self._peak_rss_mb, 2)

    def get_delta_mb(self) -> float:
        return round(self.get_current_rss_mb() - self._baseline_rss_mb, 2)

    def cleanup(self) -> Dict[str, float]:
        """Perform deterministic garbage collection and return before/after memory."""
        before = self.get_current_rss_mb()
        gc.collect()
        after = self.get_current_rss_mb()
        return {
            "before_cleanup_mb": before,
            "after_cleanup_mb": after,
            "freed_mb": round(before - after, 2)
        }

    def get_memory_breakdown(self) -> Dict[str, Any]:
        """Return structured breakdown of process memory."""
        mem_info = self._process.memory_info()
        return {
            "baseline_rss_mb": self._baseline_rss_mb,
            "current_rss_mb": round(mem_info.rss / (1024.0 * 1024.0), 2),
            "peak_rss_mb": self.get_peak_rss_mb(),
            "vms_mb": round(mem_info.vms / (1024.0 * 1024.0), 2)
        }
