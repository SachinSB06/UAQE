"""Decides how a benchmark run executes, and under which configuration.

``BenchmarkPlanner`` is an internal collaborator of
:class:`~uaqe.benchmark.benchmark.Benchmarker` — the same role
:class:`~uaqe.evaluation.evaluation_planner.EvaluationPlanner` plays
for :class:`~uaqe.evaluation.evaluator.Evaluator`: it does not itself
read/write ``PipelineContext``, run any trial, or return a
``StageResult``; ``Benchmarker`` owns that. Its one job is
:meth:`build_plan`, which resolves ``trials``/``warmup_trials``/
``timeout_seconds_per_trial`` (config defaults, overridable per-call
exactly like the locked ``run_benchmark(artifact, profile, trials)``
signature already allows for ``trials``) and decides which execution
path(s) and which optional dimensions (memory, power) this run
attempts, before any sub-benchmark runs.

``BenchmarkConfig`` mirrors ``benchmark.json`` (``06_Config_Spec.md``
§6) field-for-field — unlike ``EvaluationConfig``
(:mod:`uaqe.evaluation.evaluation_planner`), a ``BenchmarkConfig`` *is*
named in the locked configuration surface, so this dataclass's five
fields are the locked schema's five fields exactly, with no additions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from uaqe.common.interfaces.i_logger import ILogger

#: ``benchmark.json``'s locked default ``trials`` (``06_Config_Spec.md``
#: §6).
DEFAULT_TRIALS: int = 20

#: ``benchmark.json``'s locked default ``warmup_trials``.
DEFAULT_WARMUP_TRIALS: int = 3

#: ``benchmark.json``'s locked default ``timeout_seconds_per_trial``.
DEFAULT_TIMEOUT_SECONDS_PER_TRIAL: int = 30


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration governing how :class:`~uaqe.benchmark.benchmark.
    Benchmarker` executes a run, mirroring ``benchmark.json``
    (``06_Config_Spec.md`` §6) field-for-field.

    Attributes:
        trials: Default value for ``run_benchmark(..., trials)`` when
            not overridden by the caller or ``RunRequest.config_overrides``.
        warmup_trials: Excluded from every ``BenchmarkResult`` aggregate
            statistic.
        timeout_seconds_per_trial: Raises ``BenchmarkError`` on a
            simulated per-trial timeout.
        use_real_hardware_if_available: If ``True`` and a physical
            device connection is detected (mechanism is
            infrastructure-specific, defined per backend — see
            :mod:`uaqe.benchmark.runtime_benchmark`), benchmarks run
            on-device.
        simulation_fallback_enabled: If ``True``, falls back to
            ``HardwareProfile``-derived analytical estimates when no
            device is attached; if ``False``, raises ``BenchmarkError``
            instead.
    """

    trials: int = DEFAULT_TRIALS
    warmup_trials: int = DEFAULT_WARMUP_TRIALS
    timeout_seconds_per_trial: int = DEFAULT_TIMEOUT_SECONDS_PER_TRIAL
    use_real_hardware_if_available: bool = True
    simulation_fallback_enabled: bool = True


@dataclass(frozen=True)
class BenchmarkPlan:
    """How a single benchmark run should execute.

    Attributes:
        trials: The resolved number of non-warmup trials to run.
        warmup_trials: The resolved number of warmup trials to run.
        timeout_seconds_per_trial: The resolved per-trial timeout.
        attempt_real_hardware: Whether
            :class:`~uaqe.benchmark.runtime_benchmark.RuntimeBenchmark`
            should first try the real-hardware execution path.
        allow_simulation_fallback: Whether the analytical simulation
            path may be used when the real-hardware path is unavailable
            or was not attempted.
        run_memory: Whether
            :class:`~uaqe.benchmark.memory_benchmark.MemoryBenchmark`
            can produce a non-degraded score this run (an ``IMR`` is
            available for analytical activation-arena planning).
        run_power: Whether
            :class:`~uaqe.benchmark.power_benchmark.PowerBenchmark`
            can produce a score this run.
        rationale: A human-readable explanation of any degraded
            dimension.
    """

    trials: int
    warmup_trials: int
    timeout_seconds_per_trial: int
    attempt_real_hardware: bool
    allow_simulation_fallback: bool
    run_memory: bool = True
    run_power: bool = True
    rationale: str = ""


class BenchmarkPlanner:
    """Builds a :class:`BenchmarkPlan` from a run's ``BenchmarkConfig``
    and the inputs available for that run.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            planner operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``BenchmarkPlanner``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def build_plan(
        self,
        config: BenchmarkConfig,
        *,
        trials_override: Optional[int] = None,
        has_imr: bool,
    ) -> BenchmarkPlan:
        """Resolve a :class:`BenchmarkPlan` for one run.

        Args:
            config: The run's ``BenchmarkConfig``.
            trials_override: The ``trials`` value explicitly passed to
                ``run_benchmark(artifact, profile, trials)`` this call,
                if any — the locked signature's explicit parameter
                always wins over ``config.trials``, mirroring
                ``config_overrides``' precedence rule in
                ``06_Config_Spec.md`` §8.
            has_imr: Whether a baseline/final ``IMR`` is available for
                this run — gates :attr:`BenchmarkPlan.run_memory` (the
                analytical activation-arena plan needs one) and informs
                :attr:`BenchmarkPlan.allow_simulation_fallback` (the
                analytical latency/power simulation path also needs
                one).

        Returns:
            The finalized :class:`BenchmarkPlan`.
        """
        trials = trials_override if trials_override is not None else config.trials
        skipped_notes = []

        allow_simulation_fallback = config.simulation_fallback_enabled and has_imr
        if config.simulation_fallback_enabled and not has_imr:
            skipped_notes.append(
                "simulation_fallback_enabled is True but no IMR is "
                "available this run; the simulation execution path "
                "cannot run without one."
            )

        run_memory = has_imr
        if not has_imr:
            skipped_notes.append(
                "no IMR available; MemoryBenchmark will report a "
                "zero-filled, warned score instead of an analytical plan."
            )

        rationale = (
            "All dimensions enabled." if not skipped_notes else " ".join(skipped_notes)
        )

        plan = BenchmarkPlan(
            trials=trials,
            warmup_trials=config.warmup_trials,
            timeout_seconds_per_trial=config.timeout_seconds_per_trial,
            attempt_real_hardware=config.use_real_hardware_if_available,
            allow_simulation_fallback=allow_simulation_fallback,
            run_memory=run_memory,
            run_power=True,
            rationale=rationale,
        )

        if self.logger is not None:
            self.logger.info(
                "Benchmark plan built.",
                trials=plan.trials,
                warmup_trials=plan.warmup_trials,
                attempt_real_hardware=plan.attempt_real_hardware,
                allow_simulation_fallback=plan.allow_simulation_fallback,
            )

        return plan
