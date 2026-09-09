"""Peak activation-arena memory measurement for a benchmark run.

``MemoryBenchmark`` wraps :class:`~uaqe.hardware.memory_planner.
MemoryPlanner` (see that module's docstring for the ping-pong
buffer-arena planning model), the same collaborator
:class:`~uaqe.evaluation.memory_evaluator.MemoryEvaluator` already
wraps for its own baseline-vs-optimized comparison. Unlike that
evaluator, this benchmark plans for exactly one ``IMR`` — the model the
run's :class:`~uaqe.exporter.exporter.DeploymentArtifact` was exported
from — since a benchmark run reports one target's absolute figures,
never a before/after delta.

Per ``10_Module_Development_Guide.md`` §10 ("`Benchmarker` device-
detection mechanism is infrastructure-specific per backend [...] the
domain-level `Benchmarker` calls an injected capability, it does not
itself probe hardware"), no on-device memory probe is wired into this
codebase yet — ``MemoryBenchmarkScore.used_real_hardware`` is always
``False`` today, mirroring the same scope note
:mod:`uaqe.hardware.latency_estimator` and
:mod:`uaqe.hardware.power_estimator` already carry, and the analytical
:class:`~uaqe.hardware.memory_planner.MemoryPlanner` plan is the only
figure this benchmark can produce.

Per :class:`~uaqe.benchmark.benchmark_planner.BenchmarkPlanner`'s
``run_memory`` gate (populated only when an ``IMR`` is available this
run — the locked ``run_benchmark(artifact, profile, trials)`` signature
does not itself carry one, so a caller invoking it without the extended
``imr`` keyword has none to plan from), this benchmark degrades to a
zero-filled, warned :class:`~uaqe.benchmark.benchmark_result.
MemoryBenchmarkScore` rather than raising, so a run missing an ``IMR``
still produces every other dimension's figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from uaqe.benchmark.benchmark_result import MemoryBenchmarkScore
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.hardware.memory_planner import MemoryPlanner

if TYPE_CHECKING:
    from uaqe.domain.hardware_manager import HardwareProfile

#: The note recorded (and, when a logger is injected, logged at
#: ``warning``) when :meth:`MemoryBenchmark.measure` is called with no
#: ``IMR`` available.
_NO_IMR_WARNING = (
    "No IMR available this benchmark run; MemoryBenchmark cannot build "
    "an analytical activation-arena plan. Reporting a zero-filled "
    "MemoryBenchmarkScore instead of raising, so the run's other "
    "dimensions can still complete."
)


class MemoryBenchmark:
    """Measures peak activation-arena memory usage for one ``IMR``/
    ``HardwareProfile`` pair.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            benchmark operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        planner: Optional[MemoryPlanner] = None,
    ) -> None:
        """Initialize a ``MemoryBenchmark``.

        Args:
            logger: Optional structured logging sink.
            planner: The underlying activation-arena planner to
                delegate to; a fresh
                :class:`~uaqe.hardware.memory_planner.MemoryPlanner` is
                constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._planner = planner or MemoryPlanner(logger)

    def measure(
        self, imr: Optional[IMR], profile: "HardwareProfile"
    ) -> MemoryBenchmarkScore:
        """Measure peak activation-arena memory for ``imr`` on
        ``profile``.

        Args:
            imr: The model the benchmarked artifact was exported from,
                if available.
            profile: The resolved deployment target.

        Returns:
            The resulting :class:`~uaqe.benchmark.benchmark_result.
            MemoryBenchmarkScore`; a zero-filled, warned score if
            ``imr`` is ``None``.
        """
        if imr is None:
            if self.logger is not None:
                self.logger.warning(_NO_IMR_WARNING)
            return MemoryBenchmarkScore(assumptions=[_NO_IMR_WARNING])

        # strict=False: a benchmark run reports whatever the plan
        # measures rather than gating on the tensor_memory_bytes
        # ceiling — that gate already ran, strictly, in
        # uaqe.optimizer.memory_optimizer.MemoryOptimizer, the last
        # checkpoint before Exporter; re-raising here would only
        # duplicate a check this run's artifact has already passed.
        plan = self._planner.plan(imr, profile, strict=False)

        assumptions = list(plan.assumptions)
        assumptions.append(
            "peak_memory_bytes is an analytical activation-arena plan, "
            "not an on-device memory probe reading; no such probe is "
            "wired into this codebase yet."
        )

        if self.logger is not None:
            self.logger.info(
                "Memory benchmarked.",
                profile_id=profile.profile_id,
                peak_memory_bytes=plan.peak_memory_bytes,
                static_weight_bytes=plan.static_weight_bytes,
            )

        return MemoryBenchmarkScore(
            peak_memory_bytes=plan.peak_memory_bytes,
            static_weight_bytes=plan.static_weight_bytes,
            used_real_hardware=False,
            assumptions=assumptions,
        )
