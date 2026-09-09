"""Result dataclasses produced by a completed benchmark run.

Per ``03_API_Specification.md`` §11.1, the locked ``BenchmarkResult``
shape is exactly ``latency_ms_p50: float``, ``latency_ms_p99: float``,
``throughput_inferences_per_sec: float``, ``peak_memory_bytes: int``.
This module keeps those four fields and adds one sub-score dataclass
per benchmark dimension (latency, throughput, memory, power, size,
runtime execution) as a strict superset — the same locked-resolution
precedent :class:`~uaqe.evaluation.evaluation_result.EvaluationResult`
and :class:`~uaqe.optimizer.optimization_result.OptimizationResult`
already establish for their own locked counterparts.

Per ``10_Module_Development_Guide.md`` §10, "additional metrics (e.g.,
energy consumption per inference) are additive fields on
``BenchmarkResult``, never a breaking rename" — :class:`PowerBenchmarkScore`
is exactly that additive field: it is not part of the locked minimal
shape and is always ``Optional``, defaulting to ``None`` when power
could not be estimated.

This module also carries :class:`ComparisonEntry` and
:class:`ComparisonBenchmarkResult`, the result shape produced by
:class:`~uaqe.benchmark.comparison_benchmark.ComparisonBenchmark` when
comparing :class:`BenchmarkResult` instances across multiple
``HardwareProfile`` targets for the same model — the capability
``01_Project_Architecture.md`` §"Report/Benchmark scaling" describes:
"``Benchmarker`` [...] designed to consume N hardware-target results
per model in a single pass, avoiding O(N²) recomputation when
comparing multiple targets for the same model."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LatencyBenchmarkScore:
    """The outcome of :class:`~uaqe.benchmark.latency_benchmark.
    LatencyBenchmark.measure`.

    Attributes:
        latency_ms_p50: The locked minimal field
            (``03_API_Specification.md`` §11.1): the median measured
            (or simulated) per-trial inference latency, across every
            trial excluding ``warmup_trials``.
        latency_ms_p99: The locked minimal field: the 99th-percentile
            per-trial latency.
        latency_ms_mean: The arithmetic mean per-trial latency, for
            context alongside the two locked percentile figures.
        latency_ms_min: The fastest observed trial.
        latency_ms_max: The slowest observed trial.
        sample_count: The number of trials this score was computed
            over (``trials``, excluding ``warmup_trials``).
    """

    latency_ms_p50: float = 0.0
    latency_ms_p99: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_min: float = 0.0
    latency_ms_max: float = 0.0
    sample_count: int = 0


@dataclass
class ThroughputBenchmarkScore:
    """The outcome of :class:`~uaqe.benchmark.throughput_benchmark.
    ThroughputBenchmark.measure`.

    Attributes:
        throughput_inferences_per_sec: The locked minimal field
            (``03_API_Specification.md`` §11.1): the steady-state
            single-stream inference throughput implied by
            ``LatencyBenchmarkScore.latency_ms_p50`` — see the module
            docstring of :mod:`uaqe.benchmark.throughput_benchmark` for
            why the median, rather than the mean, is the figure this
            derivation uses.
        derived_from: The name of the latency figure
            ``throughput_inferences_per_sec`` was derived from, always
            ``"latency_ms_p50"`` today; recorded so a downstream reader
            never has to guess.
    """

    throughput_inferences_per_sec: float = 0.0
    derived_from: str = "latency_ms_p50"


@dataclass
class MemoryBenchmarkScore:
    """The outcome of :class:`~uaqe.benchmark.memory_benchmark.
    MemoryBenchmark.measure`.

    Attributes:
        peak_memory_bytes: The locked minimal field
            (``03_API_Specification.md`` §11.1): the highest
            simultaneous activation-arena usage observed for this run.
        static_weight_bytes: The total size of every parameter tensor,
            reported for context, mirroring
            :attr:`~uaqe.evaluation.evaluation_result.MemoryScore.
            optimized_static_weight_bytes`.
        used_real_hardware: Whether ``peak_memory_bytes`` came from an
            on-device memory probe (``True``) rather than
            :class:`~uaqe.hardware.memory_planner.MemoryPlanner`'s
            analytical activation-arena plan (``False``) — always
            ``False`` today, since no on-device memory probe is wired
            into this codebase yet (see the module docstring of
            :mod:`uaqe.benchmark.memory_benchmark`).
        assumptions: Human-readable notes on the estimation model used
            and its known limitations.
    """

    peak_memory_bytes: int = 0
    static_weight_bytes: int = 0
    used_real_hardware: bool = False
    assumptions: List[str] = field(default_factory=list)


@dataclass
class PowerBenchmarkScore:
    """The outcome of :class:`~uaqe.benchmark.power_benchmark.
    PowerBenchmark.measure`.

    Not part of the locked minimal ``BenchmarkResult`` shape — an
    additive field per ``10_Module_Development_Guide.md`` §10.
    ``BenchmarkResult.power`` is ``Optional`` and only populated when a
    dominant precision could be resolved for the benchmarked model
    (see :meth:`~uaqe.benchmark.power_benchmark.PowerBenchmark.
    measure`).

    Attributes:
        average_power_mw: The estimated average active-mode power
            draw, in milliwatts, for this run's dominant precision.
        peak_power_mw: The estimated peak power draw, in milliwatts.
        energy_per_inference_mj: The estimated energy consumed by one
            inference, in millijoules, derived from
            ``average_power_mw`` and the measured
            ``LatencyBenchmarkScore.latency_ms_p50``.
        assumptions: Human-readable notes on the estimation model used
            and its known limitations.
    """

    average_power_mw: float = 0.0
    peak_power_mw: float = 0.0
    energy_per_inference_mj: float = 0.0
    assumptions: List[str] = field(default_factory=list)


@dataclass
class SizeBenchmarkScore:
    """The outcome of :class:`~uaqe.benchmark.size_benchmark.
    SizeBenchmark.measure`.

    Attributes:
        artifact_size_bytes: The benchmarked
            ``DeploymentArtifact.size_bytes`` — the actual, already-
            exported on-disk size, never a static estimate (unlike
            :class:`~uaqe.evaluation.evaluation_result.SizeScore`,
            which falls back to one when no artifact is available; a
            benchmark run always has an artifact, per the locked
            ``run_benchmark(artifact, profile, trials)`` signature).
        export_format: The benchmarked artifact's ``ExportFormat``
            value, e.g. ``"TFLITE"``.
        target_profile_id: The ``HardwareProfile.profile_id`` this
            artifact was produced for and benchmarked against.
    """

    artifact_size_bytes: int = 0
    export_format: str = ""
    target_profile_id: str = ""


@dataclass
class RuntimeBenchmarkScore:
    """The outcome of :class:`~uaqe.benchmark.runtime_benchmark.
    RuntimeBenchmark.run` — the raw per-trial execution record every
    other sub-benchmark in this package is derived from.

    Attributes:
        used_real_hardware: Whether trials were executed on a physical
            device (``True``) or via the ``HardwareProfile``-derived
            analytical simulation path (``False``).
        trials_run: The number of non-warmup trials actually recorded.
        warmup_trials: The number of warmup trials actually run and
            excluded from every aggregate statistic, per
            ``06_Config_Spec.md`` §6.
        per_trial_latency_ms: Each non-warmup trial's measured (or
            simulated) latency, in milliseconds, in execution order.
        assumptions: Human-readable notes on which execution path ran
            and, for the simulation path, its known limitations.
    """

    used_real_hardware: bool = False
    trials_run: int = 0
    warmup_trials: int = 0
    per_trial_latency_ms: List[float] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """The complete outcome of one benchmark run.

    Attributes:
        latency_ms_p50: The locked minimal field
            (``03_API_Specification.md`` §11.1), mirrored from
            ``latency.latency_ms_p50``.
        latency_ms_p99: The locked minimal field, mirrored from
            ``latency.latency_ms_p99``.
        throughput_inferences_per_sec: The locked minimal field,
            mirrored from ``throughput.throughput_inferences_per_sec``.
        peak_memory_bytes: The locked minimal field, mirrored from
            ``memory.peak_memory_bytes``.
        hardware_profile_id: The ``HardwareProfile.profile_id`` this
            run benchmarked against, always populated, per
            ``11_Implementation_Rules.md`` §10.5.
        trials: The ``trials`` count actually used this run, per
            ``11_Implementation_Rules.md`` §10.5.
        warmup_trials: The ``warmup_trials`` count actually used this
            run, per ``11_Implementation_Rules.md`` §10.5.
        used_real_hardware: Whether this run executed on a physical
            device, mirrored from ``runtime.used_real_hardware``.
        latency: The full :class:`LatencyBenchmarkScore`, always
            populated.
        throughput: The full :class:`ThroughputBenchmarkScore`, always
            populated.
        memory: The full :class:`MemoryBenchmarkScore`, always
            populated (degrading to a zero-filled, warned score when no
            ``IMR`` was available for analytical planning — see
            :mod:`uaqe.benchmark.memory_benchmark`).
        power: The full :class:`PowerBenchmarkScore`, or ``None`` when
            no dominant precision could be resolved — an additive field
            per the module docstring.
        size: The full :class:`SizeBenchmarkScore`, always populated.
        runtime: The full :class:`RuntimeBenchmarkScore`, always
            populated; the raw record every other dimension derives
            from.
        warnings: Non-fatal warnings accumulated while building this
            result.
    """

    latency_ms_p50: float = 0.0
    latency_ms_p99: float = 0.0
    throughput_inferences_per_sec: float = 0.0
    peak_memory_bytes: int = 0
    hardware_profile_id: str = ""
    trials: int = 0
    warmup_trials: int = 0
    used_real_hardware: bool = False
    latency: LatencyBenchmarkScore = field(default_factory=LatencyBenchmarkScore)
    throughput: ThroughputBenchmarkScore = field(
        default_factory=ThroughputBenchmarkScore
    )
    memory: MemoryBenchmarkScore = field(default_factory=MemoryBenchmarkScore)
    power: Optional[PowerBenchmarkScore] = None
    size: SizeBenchmarkScore = field(default_factory=SizeBenchmarkScore)
    runtime: RuntimeBenchmarkScore = field(default_factory=RuntimeBenchmarkScore)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this result as a plain, JSON-serializable ``dict``."""
        return {
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p99": self.latency_ms_p99,
            "throughput_inferences_per_sec": self.throughput_inferences_per_sec,
            "peak_memory_bytes": self.peak_memory_bytes,
            "hardware_profile_id": self.hardware_profile_id,
            "trials": self.trials,
            "warmup_trials": self.warmup_trials,
            "used_real_hardware": self.used_real_hardware,
            "latency": {
                "latency_ms_p50": self.latency.latency_ms_p50,
                "latency_ms_p99": self.latency.latency_ms_p99,
                "latency_ms_mean": self.latency.latency_ms_mean,
                "latency_ms_min": self.latency.latency_ms_min,
                "latency_ms_max": self.latency.latency_ms_max,
                "sample_count": self.latency.sample_count,
            },
            "throughput": {
                "throughput_inferences_per_sec": (
                    self.throughput.throughput_inferences_per_sec
                ),
                "derived_from": self.throughput.derived_from,
            },
            "memory": {
                "peak_memory_bytes": self.memory.peak_memory_bytes,
                "static_weight_bytes": self.memory.static_weight_bytes,
                "used_real_hardware": self.memory.used_real_hardware,
            },
            "power": (
                {
                    "average_power_mw": self.power.average_power_mw,
                    "peak_power_mw": self.power.peak_power_mw,
                    "energy_per_inference_mj": self.power.energy_per_inference_mj,
                }
                if self.power is not None
                else None
            ),
            "size": {
                "artifact_size_bytes": self.size.artifact_size_bytes,
                "export_format": self.size.export_format,
                "target_profile_id": self.size.target_profile_id,
            },
            "runtime": {
                "used_real_hardware": self.runtime.used_real_hardware,
                "trials_run": self.runtime.trials_run,
                "warmup_trials": self.runtime.warmup_trials,
            },
            "warnings": list(self.warnings),
        }


@dataclass
class ComparisonEntry:
    """One ``HardwareProfile`` target's standing within a
    :class:`ComparisonBenchmarkResult`.

    Attributes:
        profile_id: The ``HardwareProfile.profile_id`` this entry
            summarizes.
        result: The full :class:`BenchmarkResult` this entry was built
            from, kept for a caller that needs the underlying detail.
        normalized_score: This target's aggregate score, normalized so
            that the best-performing target in the compared set scores
            ``1.0`` and every other target scores proportionally lower
            — see :mod:`uaqe.benchmark.comparison_benchmark` for the
            exact normalization/aggregation model.
        rank: This target's 1-indexed rank within the compared set,
            ``1`` being the best ``normalized_score``.
    """

    profile_id: str
    result: BenchmarkResult
    normalized_score: float = 0.0
    rank: int = 0


@dataclass
class ComparisonBenchmarkResult:
    """The outcome of :class:`~uaqe.benchmark.comparison_benchmark.
    ComparisonBenchmark.compare` across two or more ``HardwareProfile``
    targets benchmarked for the same model.

    Attributes:
        entries: One :class:`ComparisonEntry` per compared target,
            sorted best (``rank=1``) first.
        fastest_profile_id: The ``profile_id`` with the lowest
            ``latency_ms_p50``.
        lowest_memory_profile_id: The ``profile_id`` with the lowest
            ``peak_memory_bytes``.
        smallest_size_profile_id: The ``profile_id`` with the lowest
            ``size.artifact_size_bytes``.
        best_overall_profile_id: The ``profile_id`` of ``entries[0]``
            — the highest ``normalized_score``.
        rationale: A human-readable explanation of the comparison.
    """

    entries: List[ComparisonEntry] = field(default_factory=list)
    fastest_profile_id: str = ""
    lowest_memory_profile_id: str = ""
    smallest_size_profile_id: str = ""
    best_overall_profile_id: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return this result as a plain, JSON-serializable ``dict``."""
        return {
            "entries": [
                {
                    "profile_id": entry.profile_id,
                    "normalized_score": entry.normalized_score,
                    "rank": entry.rank,
                    "result": entry.result.to_dict(),
                }
                for entry in self.entries
            ],
            "fastest_profile_id": self.fastest_profile_id,
            "lowest_memory_profile_id": self.lowest_memory_profile_id,
            "smallest_size_profile_id": self.smallest_size_profile_id,
            "best_overall_profile_id": self.best_overall_profile_id,
            "rationale": self.rationale,
        }
