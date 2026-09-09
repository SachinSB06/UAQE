"""``uaqe.optimizer`` — multi-objective structural optimization and
final memory-arena planning for a quantized/compressed ``IMR``, for the
Universal AI Quantization Engine.

This package covers the same responsibilities as the locked
``uaqe.domain.optimization`` module described in
``10_Module_Development_Guide.md`` §8 (search -> plan memory), decomposed
into ten single-responsibility files rather than the doc's two —
following the same precedent ``uaqe.compression`` and
``uaqe.quantization`` each already set for their own locked
counterparts (see those packages' ``__init__`` docstrings).

Pipeline (run in this order; each is its own ``PipelineStage`` except
where noted):

1. :class:`~uaqe.optimizer.optimizer.Optimizer` — registered under
   context key ``"optimizer"``. Runs the multi-objective structural
   search: builds and evaluates a bounded set of candidate structural
   configurations, scores each on every objective in the run's
   ``OptimizationConfig.objectives``, and selects the highest-scoring
   candidate (reporting the full Pareto front alongside it).
2. :class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer` —
   registered under context key ``"memory_optimizer"``. Plans and
   validates final activation-arena memory usage for the winning
   candidate against ``HardwareProfile.tensor_memory_bytes``, the last
   IMR-adjacent checkpoint before ``Exporter``.

Structural passes (plain collaborators, applied by
:class:`~uaqe.optimizer.optimization_planner.OptimizationPlanner` —
never invoked directly outside of this package):

- :class:`~uaqe.optimizer.graph_optimizer.GraphOptimizer` — removes
  pass-through (``Identity``/``Dropout``) and bit-identical duplicate
  layers, rewiring their consumers.
- :class:`~uaqe.optimizer.fusion_optimizer.FusionOptimizer` — collapses
  linear chains of known-fusible layers (e.g.
  ``Conv2D -> BatchNorm -> ReLU``) into single fused layers.

Objective scorers (plain collaborators, invoked once per candidate by
:class:`~uaqe.optimizer.optimizer.Optimizer`):

- :class:`~uaqe.optimizer.latency_optimizer.LatencyOptimizer` — scores
  the ``"latency"`` objective via
  :class:`~uaqe.hardware.latency_estimator.LatencyEstimator` and flags
  top latency contributors.
- :class:`~uaqe.optimizer.power_optimizer.PowerOptimizer` — scores the
  ``"power"`` objective via
  :class:`~uaqe.hardware.power_estimator.PowerEstimator`.

The ``"memory"`` objective is scored in-search via a fast static-weight
proxy (see ``Optimizer._estimate_memory_proxy``); the exact
activation-arena plan is computed once, for the winning candidate only,
by ``MemoryOptimizer``.

Planning and orchestration:

- :class:`~uaqe.optimizer.optimization_planner.OptimizationPlanner` —
  decides which candidate configurations to try (bounded by
  ``OptimizationConfig.max_search_iterations``) and materializes each
  one via the structural passes above.
- :class:`~uaqe.optimizer.optimization_result.OptimizationResult` /
  :class:`~uaqe.optimizer.optimization_result.CandidateConfiguration` —
  the search's result dataclasses.

Reporting:

- :class:`~uaqe.optimizer.optimization_report.
  OptimizationReportRenderer` summarizes a completed
  ``OptimizationResult`` (and, optionally, the final ``MemoryPlan``)
  into a Markdown/dict
  :class:`~uaqe.optimizer.optimization_report.
  OptimizationReportDocument`.

Per ``09_Architecture_Lock.md`` §8, this package depends only on
``uaqe.common``, ``uaqe.domain`` (for ``PipelineStage``,
``PipelineContext``, ``HardwareProfile``), and ``uaqe.hardware`` (for
the latency/power/memory estimation models it composes over) — never on
``uaqe.infrastructure`` or ``uaqe.interface``.
"""

from uaqe.optimizer.fusion_optimizer import FusionOptimizer, FusionResult
from uaqe.optimizer.graph_optimizer import GraphOptimizer, GraphOptimizationResult
from uaqe.optimizer.latency_optimizer import LatencyObjectiveScore, LatencyOptimizer
from uaqe.optimizer.memory_optimizer import MemoryOptimizer
from uaqe.optimizer.optimization_planner import (
    BASELINE,
    GRAPH_CLEANUP,
    GRAPH_CLEANUP_FUSION,
    OptimizationPlan,
    OptimizationPlanner,
)
from uaqe.optimizer.optimization_report import (
    OptimizationReportDocument,
    OptimizationReportRenderer,
)
from uaqe.optimizer.optimization_result import CandidateConfiguration, OptimizationResult
from uaqe.optimizer.optimizer import Optimizer
from uaqe.optimizer.power_optimizer import PowerObjectiveScore, PowerOptimizer

__all__ = [
    # graph_optimizer.py
    "GraphOptimizer",
    "GraphOptimizationResult",
    # fusion_optimizer.py
    "FusionOptimizer",
    "FusionResult",
    # latency_optimizer.py
    "LatencyOptimizer",
    "LatencyObjectiveScore",
    # power_optimizer.py
    "PowerOptimizer",
    "PowerObjectiveScore",
    # optimization_planner.py
    "OptimizationPlanner",
    "OptimizationPlan",
    "BASELINE",
    "GRAPH_CLEANUP",
    "GRAPH_CLEANUP_FUSION",
    # optimization_result.py
    "OptimizationResult",
    "CandidateConfiguration",
    # optimizer.py
    "Optimizer",
    # memory_optimizer.py
    "MemoryOptimizer",
    # optimization_report.py
    "OptimizationReportRenderer",
    "OptimizationReportDocument",
]
