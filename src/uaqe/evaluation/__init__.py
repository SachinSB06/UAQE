"""``uaqe.evaluation`` — post-export accuracy, resource, and
compatibility evaluation for a quantized/compressed/optimized ``IMR``,
for the Universal AI Quantization Engine.

This package covers the same responsibility as the locked
``uaqe.domain.evaluation`` module described in
``03_API_Specification.md`` §10 (compare baseline vs. optimized model
outcomes), decomposed into ten single-responsibility files rather than
the doc's one — following the same precedent
``uaqe.compression``, ``uaqe.quantization``, and ``uaqe.optimizer``
each already set for their own locked counterparts (see those
packages' ``__init__`` docstrings).

Pipeline (run within a single ``execute()`` call; only
:class:`~uaqe.evaluation.evaluator.Evaluator` itself is a
``PipelineStage``, registered under context key ``"evaluator"``, per
``09_Architecture_Lock.md`` §12's ``{evaluator, benchmarker}`` join
point after ``exporter``):

1. :class:`~uaqe.evaluation.accuracy_evaluator.AccuracyEvaluator` —
   always runs. Scores precision-retention between the baseline and
   final ``IMR`` from static per-layer precision/parameter analysis
   (``uaqe`` has no forward-pass execution engine — see that module's
   docstring), optionally informed by a representative evaluation
   dataset's sample count.
2. :class:`~uaqe.evaluation.size_evaluator.SizeEvaluator` — always
   runs. Compares estimated (or actual, post-export) serialized model
   size.
3. :class:`~uaqe.evaluation.latency_evaluator.LatencyEvaluator`,
   :class:`~uaqe.evaluation.memory_evaluator.MemoryEvaluator`,
   :class:`~uaqe.evaluation.power_evaluator.PowerEvaluator` — run when a
   resolved ``HardwareProfile`` is available this run. Each wraps the
   matching estimator/planner in :mod:`uaqe.hardware`, run once for the
   baseline and once for the final ``IMR``.
4. :class:`~uaqe.evaluation.compatibility_evaluator.
   CompatibilityEvaluator` — runs alongside the three above. Re-checks
   the final ``IMR`` against the target profile, since several
   transformation stages have run since ``HardwareManager``'s own
   check.
5. :class:`~uaqe.evaluation.regression_evaluator.RegressionEvaluator` —
   runs when ``EvaluationConfig.baseline_metrics`` is supplied.
   Compares this run's metrics against a prior run's.

Planning and orchestration:

- :class:`~uaqe.evaluation.evaluation_planner.EvaluationPlanner` /
  :class:`~uaqe.evaluation.evaluation_planner.EvaluationConfig` /
  :class:`~uaqe.evaluation.evaluation_planner.EvaluationPlan` — decides
  which of the dimensions above a given run can attempt, based on which
  optional inputs are available.
- :class:`~uaqe.evaluation.evaluation_result.EvaluationResult` and its
  per-dimension sub-score dataclasses (:class:`~uaqe.evaluation.
  evaluation_result.AccuracyScore`, :class:`~uaqe.evaluation.
  evaluation_result.LatencyScore`, :class:`~uaqe.evaluation.
  evaluation_result.MemoryScore`, :class:`~uaqe.evaluation.
  evaluation_result.PowerScore`, :class:`~uaqe.evaluation.
  evaluation_result.SizeScore`, :class:`~uaqe.evaluation.
  evaluation_result.CompatibilityScore`, :class:`~uaqe.evaluation.
  evaluation_result.RegressionFinding`) — this package's result
  dataclasses.
- :class:`~uaqe.evaluation.evaluator.Evaluator` — the orchestrating
  ``PipelineStage``; also exposes the locked
  ``evaluate(baseline_imr, optimized_imr, dataset_path) ->
  EvaluationResult`` method directly for callers outside the pipeline.

Reporting:

- :class:`~uaqe.evaluation.evaluation_report.EvaluationReportRenderer`
  summarizes a completed ``EvaluationResult`` into a Markdown/dict
  :class:`~uaqe.evaluation.evaluation_report.EvaluationReportDocument`.

Per ``09_Architecture_Lock.md`` §8, this package depends only on
``uaqe.common``, ``uaqe.domain`` (for ``PipelineStage``,
``PipelineContext``, ``HardwareProfile``), ``uaqe.hardware`` (for the
latency/power/memory/compatibility models it composes over), and
``uaqe.exporter`` (for ``DeploymentArtifact``, an optional, already-
produced input) — never on ``uaqe.infrastructure`` or ``uaqe.interface``.
"""

from uaqe.evaluation.accuracy_evaluator import (
    ACCURACY_RETENTION_BY_PRECISION,
    AccuracyEvaluator,
)
from uaqe.evaluation.compatibility_evaluator import CompatibilityEvaluator
from uaqe.evaluation.evaluation_planner import (
    ACCURACY,
    COMPATIBILITY,
    LATENCY,
    MEMORY,
    POWER,
    REGRESSION,
    SIZE,
    EvaluationConfig,
    EvaluationPlan,
    EvaluationPlanner,
)
from uaqe.evaluation.evaluation_report import (
    EvaluationReportDocument,
    EvaluationReportRenderer,
)
from uaqe.evaluation.evaluation_result import (
    AccuracyScore,
    CompatibilityScore,
    EvaluationResult,
    LatencyScore,
    MemoryScore,
    PowerScore,
    RegressionFinding,
    SizeScore,
)
from uaqe.evaluation.evaluator import Evaluator
from uaqe.evaluation.latency_evaluator import LatencyEvaluator
from uaqe.evaluation.memory_evaluator import MemoryEvaluator
from uaqe.evaluation.power_evaluator import PowerEvaluator
from uaqe.evaluation.regression_evaluator import DEFAULT_THRESHOLDS, RegressionEvaluator
from uaqe.evaluation.size_evaluator import SizeEvaluator

__all__ = [
    # accuracy_evaluator.py
    "AccuracyEvaluator",
    "ACCURACY_RETENTION_BY_PRECISION",
    # latency_evaluator.py
    "LatencyEvaluator",
    # memory_evaluator.py
    "MemoryEvaluator",
    # power_evaluator.py
    "PowerEvaluator",
    # size_evaluator.py
    "SizeEvaluator",
    # compatibility_evaluator.py
    "CompatibilityEvaluator",
    # regression_evaluator.py
    "RegressionEvaluator",
    "DEFAULT_THRESHOLDS",
    # evaluation_planner.py
    "EvaluationPlanner",
    "EvaluationPlan",
    "EvaluationConfig",
    "ACCURACY",
    "LATENCY",
    "MEMORY",
    "POWER",
    "SIZE",
    "COMPATIBILITY",
    "REGRESSION",
    # evaluation_result.py
    "EvaluationResult",
    "AccuracyScore",
    "LatencyScore",
    "MemoryScore",
    "PowerScore",
    "SizeScore",
    "CompatibilityScore",
    "RegressionFinding",
    # evaluator.py
    "Evaluator",
    # evaluation_report.py
    "EvaluationReportRenderer",
    "EvaluationReportDocument",
]
