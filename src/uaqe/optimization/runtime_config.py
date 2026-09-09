"""Runtime configuration and execution mode governance for UAQE.

Maintains strict separation between:
- CORRECTNESS mode: The baseline validated execution path.
- PERFORMANCE mode: Isolated optimizations including optimal CPU thread count,
  reusable contiguous buffer allocation, and pure inference timing.

Provides automated fallback: if performance mode encounters an exception,
it safely falls back to correctness mode without interrupting the pipeline.
"""

from __future__ import annotations

import enum
import os
import dataclasses
from typing import Dict, Any, Optional, Tuple


class RuntimeMode(str, enum.Enum):
    """Runtime execution mode."""
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"


@dataclasses.dataclass
class RuntimeConfig:
    """Configuration options for model evaluation runtime."""
    mode: RuntimeMode = RuntimeMode.PERFORMANCE
    performance_enabled: bool = True
    num_threads: int = 2
    reuse_buffers: bool = True
    separate_pure_latency: bool = True
    gc_collect_per_candidate: bool = True
    fallback_on_error: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "performance_enabled": self.performance_enabled,
            "num_threads": self.num_threads,
            "reuse_buffers": self.reuse_buffers,
            "separate_pure_latency": self.separate_pure_latency,
            "gc_collect_per_candidate": self.gc_collect_per_candidate,
            "fallback_on_error": self.fallback_on_error
        }

    @classmethod
    def create_correctness_mode(cls) -> RuntimeConfig:
        """Create standard correctness configuration preserving exact original behavior."""
        return cls(
            mode=RuntimeMode.CORRECTNESS,
            performance_enabled=False,
            num_threads=1,
            reuse_buffers=False,
            separate_pure_latency=False,
            gc_collect_per_candidate=False,
            fallback_on_error=True
        )

    @classmethod
    def create_performance_mode(cls, num_threads: int = 2) -> RuntimeConfig:
        """Create performance configuration with safe CPU optimizations."""
        return cls(
            mode=RuntimeMode.PERFORMANCE,
            performance_enabled=True,
            num_threads=num_threads,
            reuse_buffers=True,
            separate_pure_latency=True,
            gc_collect_per_candidate=True,
            fallback_on_error=True
        )
