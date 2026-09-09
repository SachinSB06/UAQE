"""``uaqe.benchmark`` — measured (or faithfully simulated) hardware
performance for a completed export, for the Universal AI Quantization
Engine.

This package covers the same responsibility as the locked
``uaqe.domain.benchmark`` module described in ``03_API_Specification.md``
§11 and ``10_Module_Development_Guide.md`` §10 (measure latency /
throughput / memory, real hardware if available, else analytical
simulation, else ``BenchmarkError``), decomposed into twelve
single-responsibility files rather than the doc's one — following the
same precedent ``uaqe.evaluation``, ``uaqe.optimizer``, and
``uaqe.exporter`` each already set for their own locked counterparts
(see those packages' ``__init__`` docstrings).

Pipeline (run within a single
:meth:`~uaqe.benchmark.benchmark.Benchmarker.execute` call; only
:class:`~uaqe.benchmark.benchmark.Benchmarker` itself is a
``PipelineStage``, registered under context key ``"benchmarker"``, per
``09_Architecture_Lock.md`` §12's ``{evaluator, benchmarker}`` join
point after ``exporter``):

1. :class:`~uaqe.benchmark.benchmark_planner.BenchmarkPlanner` /
   :class:`~uaqe.benchmark.benchmark_planner.BenchmarkConfig` /
   :class:`~uaqe.benchmark.benchmark_planner.BenchmarkPlan` — resolves
   ``trials``/``warmup_trials``/``timeout_seconds_per_trial`` and
   decides which optional dimensions (memory, power) this run attempts,
   before any sub-benchmark runs.
2. :class:`~uaqe.benchmark.runtime_benchmark.RuntimeBenchmark` (with
   its :class:`~uaqe.benchmark.runtime_benchmark.TrialRunner` strategy
   interface — :class:`~uaqe.benchmark.runtime_benchmark.
   RealHardwareTrialRunner` and :class:`~uaqe.benchmark.
   runtime_benchmark.SimulationTrialRunner`) — drives the warmup/trial
   loop against real hardware if attached and
   ``use_real_hardware_if_available=true``, else the
   ``HardwareProfile``-derived analytical simulation path if
   ``simulation_fallback_enabled=true``, else raises ``BenchmarkError``
   (``06_Config_Spec.md`` §6).
3. :class:`~uaqe.benchmark.latency_benchmark.LatencyBenchmark` —
   reduces the raw per-trial samples (with warmup trials already
   excluded) to ``latency_ms_p50``/``latency_ms_p99`` plus mean/min/max.
4. :class:`~uaqe.benchmark.throughput_benchmark.ThroughputBenchmark` —
   derives single-stream throughput from ``latency_ms_p50``.
5. :class:`~uaqe.benchmark.memory_benchmark.MemoryBenchmark` — always
   runs. Plans peak activation-arena memory via
   :class:`~uaqe.hardware.memory_planner.MemoryPlanner`, degrading to a
   zero-filled, warned score when no ``IMR`` is available.
6. :class:`~uaqe.benchmark.power_benchmark.PowerBenchmark` — runs when
   an ``IMR`` is available. Estimates power/energy via
   :class:`~uaqe.hardware.power_estimator.PowerEstimator`, from this
   run's own measured latency rather than a re-derived estimate.
7. :class:`~uaqe.benchmark.size_benchmark.SizeBenchmark` — always
   runs. Reduces the benchmarked ``DeploymentArtifact``'s own size
   figures.

Cross-target comparison (a plain collaborator, never invoked by
``Benchmarker`` itself — see its own module docstring for why):

- :class:`~uaqe.benchmark.comparison_benchmark.ComparisonBenchmark` —
  ranks ``N`` already-produced ``BenchmarkResult`` instances across
  hardware targets for the same model in a single ``O(N)`` pass
  (``01_Project_Architecture.md`` §18), producing a
  :class:`~uaqe.benchmark.benchmark_result.ComparisonBenchmarkResult`.

Result dataclasses:

- :class:`~uaqe.benchmark.benchmark_result.BenchmarkResult` and its
  per-dimension sub-score dataclasses
  (:class:`~uaqe.benchmark.benchmark_result.LatencyBenchmarkScore`,
  :class:`~uaqe.benchmark.benchmark_result.ThroughputBenchmarkScore`,
  :class:`~uaqe.benchmark.benchmark_result.MemoryBenchmarkScore`,
  :class:`~uaqe.benchmark.benchmark_result.PowerBenchmarkScore`,
  :class:`~uaqe.benchmark.benchmark_result.SizeBenchmarkScore`,
  :class:`~uaqe.benchmark.benchmark_result.RuntimeBenchmarkScore`) —
  this package's result dataclasses.
- :class:`~uaqe.benchmark.benchmark_result.ComparisonEntry` /
  :class:`~uaqe.benchmark.benchmark_result.ComparisonBenchmarkResult` —
  the result shape produced by
  :class:`~uaqe.benchmark.comparison_benchmark.ComparisonBenchmark`.

Orchestration:

- :class:`~uaqe.benchmark.benchmark.Benchmarker` — the orchestrating
  ``PipelineStage``; also exposes the locked ``run_benchmark(artifact,
  profile, trials) -> BenchmarkResult`` method directly for callers
  outside the pipeline.

Reporting:

- :class:`~uaqe.benchmark.benchmark_report.BenchmarkReportRenderer`
  summarizes a completed ``BenchmarkResult`` (and, optionally, a
  ``ComparisonBenchmarkResult``) into a Markdown/dict
  :class:`~uaqe.benchmark.benchmark_report.BenchmarkReportDocument`.

Per ``09_Architecture_Lock.md`` §8, this package depends only on
``uaqe.common``, ``uaqe.domain`` (for ``PipelineStage``,
``PipelineContext``, ``HardwareProfile``), ``uaqe.hardware`` (for the
latency/power/memory estimation models it composes over), and
``uaqe.exporter`` (for ``DeploymentArtifact``, a required input) —
never on ``uaqe.infrastructure`` or ``uaqe.interface``.
"""

from uaqe.benchmark.benchmark import Benchmarker
from uaqe.benchmark.benchmark_planner import (
    DEFAULT_TIMEOUT_SECONDS_PER_TRIAL,
    DEFAULT_TRIALS,
    DEFAULT_WARMUP_TRIALS,
    BenchmarkConfig,
    BenchmarkPlan,
    BenchmarkPlanner,
)
from uaqe.benchmark.benchmark_report import (
    BenchmarkReportDocument,
    BenchmarkReportRenderer,
)
from uaqe.benchmark.benchmark_result import (
    BenchmarkResult,
    ComparisonBenchmarkResult,
    ComparisonEntry,
    LatencyBenchmarkScore,
    MemoryBenchmarkScore,
    PowerBenchmarkScore,
    RuntimeBenchmarkScore,
    SizeBenchmarkScore,
    ThroughputBenchmarkScore,
)
from uaqe.benchmark.comparison_benchmark import ComparisonBenchmark
from uaqe.benchmark.latency_benchmark import LatencyBenchmark
from uaqe.benchmark.memory_benchmark import MemoryBenchmark
from uaqe.benchmark.power_benchmark import PowerBenchmark
from uaqe.benchmark.runtime_benchmark import (
    RealHardwareTrialRunner,
    RuntimeBenchmark,
    SimulationTrialRunner,
    TrialRunner,
)
from uaqe.benchmark.size_benchmark import SizeBenchmark
from uaqe.benchmark.throughput_benchmark import ThroughputBenchmark

__all__ = [
    # runtime_benchmark.py
    "RuntimeBenchmark",
    "TrialRunner",
    "RealHardwareTrialRunner",
    "SimulationTrialRunner",
    # latency_benchmark.py
    "LatencyBenchmark",
    # throughput_benchmark.py
    "ThroughputBenchmark",
    # memory_benchmark.py
    "MemoryBenchmark",
    # power_benchmark.py
    "PowerBenchmark",
    # size_benchmark.py
    "SizeBenchmark",
    # comparison_benchmark.py
    "ComparisonBenchmark",
    # benchmark_planner.py
    "BenchmarkPlanner",
    "BenchmarkPlan",
    "BenchmarkConfig",
    "DEFAULT_TRIALS",
    "DEFAULT_WARMUP_TRIALS",
    "DEFAULT_TIMEOUT_SECONDS_PER_TRIAL",
    # benchmark_result.py
    "BenchmarkResult",
    "LatencyBenchmarkScore",
    "ThroughputBenchmarkScore",
    "MemoryBenchmarkScore",
    "PowerBenchmarkScore",
    "SizeBenchmarkScore",
    "RuntimeBenchmarkScore",
    "ComparisonEntry",
    "ComparisonBenchmarkResult",
    # benchmark.py
    "Benchmarker",
    # benchmark_report.py
    "BenchmarkReportRenderer",
    "BenchmarkReportDocument",
]
