"""Final activation-arena memory planning stage — the locked
``uaqe.domain.optimization.memory_optimizer`` contract
(``03_API_Specification.md`` §8.2).

``MemoryOptimizer`` is the last IMR-mutation-adjacent stage in the
pipeline (``09_Architecture_Lock.md`` §12:
``... -> optimization_engine -> memory_optimizer -> exporter -> ...``,
where this package's ``optimization_engine`` analog is
:class:`~uaqe.optimizer.optimizer.Optimizer`, registered under context
key ``"optimizer"``). Per ``04_Data_Flow.md`` §7, downstream stages
(``Exporter``, ``Evaluator``, ``Benchmarker``) read the final ``IMR``
from this stage's context key, ``"memory_optimizer"`` — no stage after
this one may alter tensor data or layer graph structure.

This stage does not reimplement activation-arena planning: it wraps
:class:`~uaqe.hardware.memory_planner.MemoryPlanner`, the same planner
this codebase's ``hardware`` package documents as the "reference
implementation" for this exact locked responsibility (see the scope
note in :mod:`uaqe.hardware.hardware_manager`). Composing over that
planner here — rather than duplicating its ping-pong buffer-arena
model — keeps the ``tensor_memory_bytes`` hard-ceiling logic
(``05_Hardware_Profile_Spec.md`` §6 rule 4) defined in exactly one
place.

Locked-resolution note on this stage's ``StageResult.payload``: per
``09_Architecture_Lock.md`` §9's footnote, the richer domain dataclass
(``MemoryPlan``) is the payload, not a bare ``IMR`` — but ``04_Data_Flow
.md`` also documents this stage as producing "the final IMR" for
downstream stages to read. Both are satisfied, following the same
precedent :class:`~uaqe.compression.compression_planner.
CompressionPlanner` and
:class:`~uaqe.quantization.quantization_planner.QuantizationPlanner`
already set: the payload is a ``(MemoryPlan, IMR)`` tuple. Memory
planning does not itself transform the graph, so the returned ``IMR``
is the same object passed in — the "final IMR" is simply the winning
candidate :class:`~uaqe.optimizer.optimizer.Optimizer` already selected.
"""

from __future__ import annotations

import time
from typing import Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.hardware.memory_planner import MemoryPlan, MemoryPlanner

#: The context key this stage reads the optimized ``IMR`` from when
#: present, per :class:`~uaqe.optimizer.optimizer.Optimizer`'s own
#: registered stage name.
_OPTIMIZER_STAGE_NAME = "optimizer"

#: Fallback context keys, in resolution order, when the ``"optimizer"``
#: stage has not (yet) run — mirroring
#: ``Optimizer._resolve_current_imr``'s own fallback chain.
_FALLBACK_STAGE_NAMES = ("compression_planner", "quantization_planner", "model_loader")

#: The ``HardwareProfile``-producing stage this stage plans against.
_HARDWARE_STAGE_NAME = "hardware_manager"


class MemoryOptimizer(PipelineStage):
    """Plans and validates activation-arena memory usage for the final
    optimized ``IMR``.

    Attributes:
        _logger: Structured logging sink.
        _planner: The underlying activation-arena planner delegated to.
    """

    def __init__(
        self, logger: ILogger, planner: Optional[MemoryPlanner] = None
    ) -> None:
        """Initialize the ``MemoryOptimizer``.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            planner: The underlying activation-arena planner; a fresh
                :class:`~uaqe.hardware.memory_planner.MemoryPlanner` is
                constructed if omitted.
        """
        self._logger = logger
        self._planner = planner or MemoryPlanner(logger)

    def execute(self, context: PipelineContext) -> StageResult:
        """Plan and validate this run's final activation-arena memory
        usage.

        Reads the current ``IMR`` per the resolution order documented
        in the module docstring, and the target ``HardwareProfile``
        from ``"hardware_manager"``.

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is a tuple of
            ``(MemoryPlan, IMR)`` — the finalized plan and the (already
            final, unmodified) ``IMR`` it was planned against.

        Raises:
            HardwareIncompatibilityError: If the planned peak activation
                memory exceeds ``profile.tensor_memory_bytes`` (see
                :meth:`plan`).
        """
        start = time.monotonic()
        imr = self._resolve_current_imr(context)
        profile = context.get(_HARDWARE_STAGE_NAME).payload

        memory_plan = self.plan(imr, profile)

        duration_ms = (time.monotonic() - start) * 1000.0
        self._logger.info(
            "Final memory plan validated.",
            stage_name=self.name(),
            peak_memory_bytes=memory_plan.peak_memory_bytes,
            tensor_memory_bytes=profile.tensor_memory_bytes,
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=(memory_plan, imr),
            duration_ms=duration_ms,
        )

    def plan(self, imr: IMR, profile: HardwareProfile) -> MemoryPlan:
        """Plan activation-arena memory usage for ``imr`` on ``profile``.

        Args:
            imr: The final, post-optimization model.
            profile: The resolved deployment target.

        Returns:
            The resulting ``MemoryPlan``.

        Raises:
            HardwareIncompatibilityError: If the planned peak exceeds
                ``profile.tensor_memory_bytes`` — enforced strictly,
                since this is the last checkpoint before ``Exporter``
                (``01_Project_Architecture.md`` §17's peak memory
                budget check).
        """
        return self._planner.plan(imr, profile, strict=True)

    def _resolve_current_imr(self, context: PipelineContext) -> IMR:
        """Resolve the final ``IMR`` from ``context``.

        Args:
            context: The current run's pipeline context.

        Returns:
            The ``IMR`` produced by ``"optimizer"`` if that stage has
            run, else the first of :data:`_FALLBACK_STAGE_NAMES` found
            in ``context``.

        Raises:
            KeyError: If none of ``"optimizer"`` or
                :data:`_FALLBACK_STAGE_NAMES` has been recorded in
                ``context``.
        """
        if context.has(_OPTIMIZER_STAGE_NAME):
            payload = context.get(_OPTIMIZER_STAGE_NAME).payload
            return payload[1] if isinstance(payload, tuple) else payload

        for stage_name in _FALLBACK_STAGE_NAMES:
            if context.has(stage_name):
                payload = context.get(stage_name).payload
                return payload[1] if isinstance(payload, tuple) else payload

        raise KeyError(
            "No IMR-producing stage recorded in context; expected one "
            f"of {(_OPTIMIZER_STAGE_NAME,) + _FALLBACK_STAGE_NAMES!r}."
        )
