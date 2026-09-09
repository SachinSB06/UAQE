"""Hardware benchmarking — the locked ``uaqe.domain.benchmark.
benchmarker`` contract (``03_API_Specification.md`` §11.1), decomposed
into this package.

``Benchmarker`` is the sole orchestrating ``PipelineStage`` for this
package (registered under context key ``"benchmarker"``). Per
``09_Architecture_Lock.md`` §12, it runs after ``Exporter`` alongside
``Evaluator`` — the one point in the pipeline where two stages may
execute independently before the next join point
(``deployment_readiness_scorer``). Per ``01_Project_Architecture.md``
§6's strict responsibility split (and ``10_Module_Development_Guide.md``
§10's Integration Checklist), ``Benchmarker`` never computes accuracy —
that is ``Evaluator``'s job — and measures real, on-device (or
faithfully simulated) timing, never a static estimate.

``run_benchmark`` (locked minimal signature ``run_benchmark(artifact,
profile, trials) -> BenchmarkResult``, extended here with an optional
``imr`` keyword argument the same way
:meth:`~uaqe.evaluation.evaluator.Evaluator.evaluate` and
:meth:`~uaqe.optimizer.optimizer.Optimizer.search` each extend their
own locked signatures):

1. :class:`~uaqe.benchmark.benchmark_planner.BenchmarkPlanner` resolves
   ``trials``/``warmup_trials``/``timeout_seconds_per_trial`` and
   decides which optional dimensions (memory, power) this run attempts,
   before any sub-benchmark runs.
2. :class:`~uaqe.benchmark.runtime_benchmark.RuntimeBenchmark` drives
   the warmup/trial loop against the real-hardware or simulation
   execution path and returns the raw per-trial
   :class:`~uaqe.benchmark.benchmark_result.RuntimeBenchmarkScore`.
3. :class:`~uaqe.benchmark.latency_benchmark.LatencyBenchmark` reduces
   the raw per-trial samples to percentile/summary statistics.
4. :class:`~uaqe.benchmark.throughput_benchmark.ThroughputBenchmark`
   derives single-stream throughput from that latency score.
5. :class:`~uaqe.benchmark.memory_benchmark.MemoryBenchmark` always
   runs (degrading to a zero-filled, warned score without an ``IMR``).
6. :class:`~uaqe.benchmark.power_benchmark.PowerBenchmark` runs when an
   ``IMR`` is available, from the measured latency score.
7. :class:`~uaqe.benchmark.size_benchmark.SizeBenchmark` always runs,
   reducing the benchmarked artifact's own size figures.

:class:`~uaqe.benchmark.comparison_benchmark.ComparisonBenchmark` is a
separate, non-``PipelineStage`` collaborator for ranking ``N``
already-produced ``BenchmarkResult`` instances across hardware targets
in one pass (``01_Project_Architecture.md`` §18) — called by a
``BatchRunner``-level caller, never by ``Benchmarker`` itself, since a
single pipeline run benchmarks exactly one resolved ``HardwareProfile``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from uaqe.benchmark.benchmark_planner import BenchmarkConfig, BenchmarkPlanner
from uaqe.benchmark.benchmark_result import BenchmarkResult
from uaqe.benchmark.latency_benchmark import LatencyBenchmark
from uaqe.benchmark.memory_benchmark import MemoryBenchmark
from uaqe.benchmark.power_benchmark import PowerBenchmark
from uaqe.benchmark.runtime_benchmark import RuntimeBenchmark
from uaqe.benchmark.size_benchmark import SizeBenchmark
from uaqe.benchmark.throughput_benchmark import ThroughputBenchmark
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage

if TYPE_CHECKING:
    # Deferred: this stage only needs DeploymentArtifact as a type
    # hint for a required input already produced elsewhere; importing
    # uaqe.exporter at module scope would otherwise pull in that
    # package's full __init__ for no runtime benefit here, mirroring
    # the same TYPE_CHECKING-guarded cross-package type hint pattern
    # uaqe.evaluation already uses for the same type.
    from uaqe.exporter.exporter import DeploymentArtifact

#: The context key holding the completed export this stage benchmarks.
_EXPORTER_STAGE_NAME = "exporter"

#: The ``HardwareProfile``-producing stage, read at most once per
#: ``09_Architecture_Lock.md`` §13 rule 3.
_HARDWARE_STAGE_NAME = "hardware_manager"

#: The context key this stage reads the final ``IMR`` from when
#: present, per ``uaqe.optimizer.memory_optimizer.MemoryOptimizer``'s
#: own registered stage name — mirroring ``Evaluator``'s and
#: ``Exporter``'s identical resolution chain, since this stage runs
#: alongside ``Evaluator``, immediately after ``Exporter``.
_MEMORY_OPTIMIZER_STAGE_NAME = "memory_optimizer"

#: Fallback context keys, in resolution order, when
#: ``"memory_optimizer"`` has not (yet) run.
_FALLBACK_STAGE_NAMES = (
    "optimizer",
    "compression_planner",
    "quantization_planner",
    "model_loader",
)


class Benchmarker(PipelineStage):
    """Measures latency, throughput, memory, power, and size for a
    completed export against its resolved deployment target.

    Attributes:
        _logger: Structured logging sink.
        _config: The run's ``BenchmarkConfig``, supplied via
            constructor injection.
        _planner: Decides trials/warmup/timeout and which optional
            dimensions this run attempts.
        _runtime_benchmark: Drives the warmup/trial execution loop.
        _latency_benchmark: Reduces raw per-trial samples to
            percentile/summary statistics.
        _throughput_benchmark: Derives throughput from the latency
            score.
        _memory_benchmark: Measures peak activation-arena memory.
        _power_benchmark: Estimates power/energy from measured latency.
        _size_benchmark: Reduces the benchmarked artifact's size
            figures.
    """

    def __init__(
        self,
        logger: ILogger,
        config: Optional[BenchmarkConfig] = None,
        planner: Optional[BenchmarkPlanner] = None,
        runtime_benchmark: Optional[RuntimeBenchmark] = None,
        latency_benchmark: Optional[LatencyBenchmark] = None,
        throughput_benchmark: Optional[ThroughputBenchmark] = None,
        memory_benchmark: Optional[MemoryBenchmark] = None,
        power_benchmark: Optional[PowerBenchmark] = None,
        size_benchmark: Optional[SizeBenchmark] = None,
    ) -> None:
        """Initialize the ``Benchmarker``.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            config: The run's ``BenchmarkConfig``, sourced from
                ``IConfigRepository`` by the ``CompositionRoot``. A
                default (``benchmark.json``-matching) config is used if
                omitted.
            planner: Resolves trials/warmup/timeout and dimension
                gating; a fresh
                :class:`~uaqe.benchmark.benchmark_planner.
                BenchmarkPlanner` is constructed if omitted.
            runtime_benchmark: Drives the warmup/trial execution loop; a
                fresh :class:`~uaqe.benchmark.runtime_benchmark.
                RuntimeBenchmark` is constructed if omitted.
            latency_benchmark: Reduces raw per-trial samples to
                percentile/summary statistics; a fresh
                :class:`~uaqe.benchmark.latency_benchmark.
                LatencyBenchmark` is constructed if omitted.
            throughput_benchmark: Derives throughput from the latency
                score; a fresh
                :class:`~uaqe.benchmark.throughput_benchmark.
                ThroughputBenchmark` is constructed if omitted.
            memory_benchmark: Measures peak activation-arena memory; a
                fresh :class:`~uaqe.benchmark.memory_benchmark.
                MemoryBenchmark` is constructed if omitted.
            power_benchmark: Estimates power/energy from measured
                latency; a fresh
                :class:`~uaqe.benchmark.power_benchmark.PowerBenchmark`
                is constructed if omitted.
            size_benchmark: Reduces the benchmarked artifact's size
                figures; a fresh
                :class:`~uaqe.benchmark.size_benchmark.SizeBenchmark`
                is constructed if omitted.
        """
        self._logger = logger
        self._config = config or BenchmarkConfig()
        self._planner = planner or BenchmarkPlanner(logger)
        self._runtime_benchmark = runtime_benchmark or RuntimeBenchmark(logger)
        self._latency_benchmark = latency_benchmark or LatencyBenchmark(logger)
        self._throughput_benchmark = (
            throughput_benchmark or ThroughputBenchmark(logger)
        )
        self._memory_benchmark = memory_benchmark or MemoryBenchmark(logger)
        self._power_benchmark = power_benchmark or PowerBenchmark(logger)
        self._size_benchmark = size_benchmark or SizeBenchmark(logger)

    def execute(self, context: PipelineContext) -> StageResult:
        """Run this run's benchmark.

        Reads the ``DeploymentArtifact`` from ``"exporter"``, the
        resolved ``HardwareProfile`` from ``"hardware_manager"``, and
        the final ``IMR`` per the resolution order documented in the
        module docstring (``None`` if none has been recorded, which the
        memory/power dimensions each degrade gracefully for).

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is the finalized
            ``BenchmarkResult``.
        """
        start = time.monotonic()
        exporter_payload = context.get(_EXPORTER_STAGE_NAME).payload
        artifact: "DeploymentArtifact" = (
            exporter_payload[0]
            if isinstance(exporter_payload, tuple)
            else exporter_payload
        )
        profile: HardwareProfile = context.get(_HARDWARE_STAGE_NAME).payload
        imr = self._resolve_current_imr(context)

        result = self.run_benchmark(
            artifact, profile, self._config.trials, imr=imr
        )

        duration_ms = (time.monotonic() - start) * 1000.0
        self._logger.info(
            "Benchmark complete.",
            stage_name=self.name(),
            hardware_profile_id=result.hardware_profile_id,
            trials=result.trials,
            warmup_trials=result.warmup_trials,
            latency_ms_p50=result.latency_ms_p50,
            peak_memory_bytes=result.peak_memory_bytes,
            duration_ms=duration_ms,
            log_type="performance",
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=result,
            warnings=list(result.warnings),
            duration_ms=duration_ms,
        )

    def run_benchmark(
        self,
        artifact: "DeploymentArtifact",
        profile: HardwareProfile,
        trials: int,
        *,
        imr: Optional[IMR] = None,
    ) -> BenchmarkResult:
        """Benchmark ``artifact`` against ``profile``.

        Args:
            artifact: The ``DeploymentArtifact`` under benchmark — the
                locked ``run_benchmark`` parameter of the same name.
            profile: The resolved deployment target — the locked
                ``run_benchmark`` parameter of the same name.
            trials: The number of non-warmup trials to run — the locked
                ``run_benchmark`` parameter of the same name; always
                wins over ``BenchmarkConfig.trials`` per
                :meth:`~uaqe.benchmark.benchmark_planner.
                BenchmarkPlanner.build_plan`'s documented precedence.
            imr: The model ``artifact`` was exported from, if
                available. Required for the analytical simulation
                execution path (raises ``BenchmarkError`` if that path
                is selected without one); optional for the memory/power
                dimensions, which each degrade gracefully without one.

        Returns:
            The finalized :class:`~uaqe.benchmark.benchmark_result.
            BenchmarkResult`.

        Raises:
            BenchmarkError: If no execution path is available/allowed
                for this run, or if any trial exceeds
                ``timeout_seconds_per_trial``.
        """
        plan = self._planner.build_plan(
            self._config, trials_override=trials, has_imr=imr is not None
        )

        runtime_score = self._runtime_benchmark.run(
            artifact,
            profile,
            imr,
            trials=plan.trials,
            warmup_trials=plan.warmup_trials,
            timeout_seconds_per_trial=plan.timeout_seconds_per_trial,
            attempt_real_hardware=plan.attempt_real_hardware,
            allow_simulation_fallback=plan.allow_simulation_fallback,
        )

        latency_score = self._latency_benchmark.measure(
            runtime_score.per_trial_latency_ms
        )
        throughput_score = self._throughput_benchmark.measure(latency_score)
        memory_score = self._memory_benchmark.measure(imr, profile)
        size_score = self._size_benchmark.measure(artifact)

        power_score = None
        if plan.run_power and imr is not None:
            power_score = self._power_benchmark.measure(imr, profile, latency_score)

        warnings = (
            [] if plan.rationale == "All dimensions enabled." else [plan.rationale]
        )

        result = BenchmarkResult(
            latency_ms_p50=latency_score.latency_ms_p50,
            latency_ms_p99=latency_score.latency_ms_p99,
            throughput_inferences_per_sec=(
                throughput_score.throughput_inferences_per_sec
            ),
            peak_memory_bytes=memory_score.peak_memory_bytes,
            hardware_profile_id=profile.profile_id,
            trials=plan.trials,
            warmup_trials=plan.warmup_trials,
            used_real_hardware=runtime_score.used_real_hardware,
            latency=latency_score,
            throughput=throughput_score,
            memory=memory_score,
            power=power_score,
            size=size_score,
            runtime=runtime_score,
            warnings=warnings,
        )

        self._logger.info(
            "Benchmark aggregate result.",
            hardware_profile_id=result.hardware_profile_id,
            trials=result.trials,
            warmup_trials=result.warmup_trials,
            latency_ms_p50=result.latency_ms_p50,
            latency_ms_p99=result.latency_ms_p99,
            throughput_inferences_per_sec=result.throughput_inferences_per_sec,
            peak_memory_bytes=result.peak_memory_bytes,
            used_real_hardware=result.used_real_hardware,
        )
        return result

    def _resolve_current_imr(self, context: PipelineContext) -> Optional[IMR]:
        """Resolve the final ``IMR`` from ``context``, if any has been
        recorded.

        Args:
            context: The current run's pipeline context.

        Returns:
            The ``IMR`` produced by ``"memory_optimizer"`` if that
            stage has run, else the first of
            :data:`_FALLBACK_STAGE_NAMES` found in ``context``, else
            ``None`` — unlike
            :meth:`~uaqe.evaluation.evaluator.Evaluator.
            _resolve_current_imr`, this never raises: an ``IMR`` is not
            part of the locked ``run_benchmark(artifact, profile,
            trials)`` signature's required inputs, and the memory/power
            dimensions each degrade gracefully without one.
        """
        if context.has(_MEMORY_OPTIMIZER_STAGE_NAME):
            payload = context.get(_MEMORY_OPTIMIZER_STAGE_NAME).payload
            return payload[1] if isinstance(payload, tuple) else payload

        for stage_name in _FALLBACK_STAGE_NAMES:
            if context.has(stage_name):
                payload = context.get(stage_name).payload
                return payload[1] if isinstance(payload, tuple) else payload

        return None
