"""Orchestration and application of the final compression plan.

``CompressionPlanner`` is the single ``PipelineStage`` entry point for
this package, mirroring the role
:class:`~uaqe.quantization.quantization_planner.QuantizationPlanner`
plays for ``uaqe.quantization``: it decides which
``CompressionType`` values (from the run's ``CompressionConfig``) to
apply and in what order, resolves each to a concrete registered
``ICompressionStrategy``, and applies them in sequence to the current
``IMR``.

Per ``04_Data_Flow.md`` §6 and ``09_Architecture_Lock.md`` §12, the
compression stage runs after quantization and reads the *current* IMR
from the latest stage in the model-transforming chain rather than a
fixed context key. Since ``uaqe.quantization`` (this codebase's actual
decomposition of the locked ``uaqe.domain.quantization`` module, see
its own package docstring) writes its output under the context key
``"quantization_planner"`` as a ``(QuantizationPlan, IMR)`` tuple, this
planner reads from that key when present, falling back to
``"model_loader"`` for a pipeline run with quantization disabled or
not yet executed — never assuming a fixed key name is always "the IMR"
(``04_Data_Flow.md`` §7 rule 2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from uaqe.common.exceptions import CompressionError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_compression_strategy import ICompressionStrategy
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.common.types import CompressionType
from uaqe.common.value_objects import CompressionConfig
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage

#: The order in which enabled ``CompressionType`` values are applied
#: when more than one is selected: lossy size-reducing transforms
#: (pruning, then clustering) before lossless entropy coding (Huffman,
#: then RLE), since each stage's output tends to make the next stage's
#: job easier (more zeros, more repeated byte values) rather than the
#: reverse.
_CANONICAL_TYPE_ORDER: List[CompressionType] = [
    CompressionType.PRUNING,
    CompressionType.WEIGHT_CLUSTERING,
    CompressionType.HUFFMAN,
    CompressionType.RLE,
]

#: The registered ``ICompressionStrategy.name()`` selected for every
#: ``CompressionType`` other than ``PRUNING``, which instead resolves
#: via :data:`_PRUNING_STRATEGY_NAME_BY_MODE` (more than one strategy
#: implements ``PRUNING``, unlike the other three types).
_STRATEGY_NAME_BY_TYPE: Dict[CompressionType, str] = {
    CompressionType.WEIGHT_CLUSTERING: "weight_clustering",
    CompressionType.HUFFMAN: "huffman",
    CompressionType.RLE: "rle",
}

#: The registered ``ICompressionStrategy.name()`` selected for
#: ``CompressionType.PRUNING`` depending on whether the resolved
#: pruning mode is "magnitude" (finer-grained, unstructured) or
#: "structured" (coarser, hardware-friendly whole-group removal).
_PRUNING_STRATEGY_NAME_BY_MODE: Dict[str, str] = {
    "magnitude": "magnitude_pruning",
    "structured": "structured_pruning",
}

#: The ``CompressionConfig.pruning_sparsity`` threshold at or above
#: which structured pruning is preferred over magnitude pruning: a
#: sufficiently aggressive sparsity target is assumed to be motivated
#: by a hardware/latency constraint that benefits from structured
#: (dense-execution-preserving) removal rather than the scattered
#: unstructured sparsity magnitude pruning produces.
_STRUCTURED_PRUNING_SPARSITY_THRESHOLD = 0.6

#: The context key this planner reads the pre-compression ``IMR`` from
#: when present, per ``uaqe.quantization``'s actual (not
#: ``04_Data_Flow.md``'s aspirational ``"quantization_engine"``)
#: output key — see the module docstring.
_QUANTIZATION_STAGE_NAME = "quantization_planner"

#: The context key this planner falls back to reading the IMR from when
#: ``_QUANTIZATION_STAGE_NAME`` has not (yet) run.
_MODEL_LOADER_STAGE_NAME = "model_loader"


@dataclass(frozen=True)
class CompressionPlan:
    """The complete, finalized compression decision for one run.

    Superset of the locked minimal shape (``selected_types``,
    ``target_ratio``, ``rationale`` — ``03_API_Specification.md``
    §7.1) with the additional per-type strategy resolution this
    package's decomposition needs to actually apply the plan, following
    the same precedent as ``uaqe.quantization.quantization_planner.
    QuantizationPlan`` (``09_Architecture_Lock.md``'s locked-resolution
    note: the richer domain dataclass, not the bare config, is this
    stage's ``StageResult.payload``).

    Attributes:
        selected_types: The ``CompressionType`` values selected for
            this run, in application order.
        target_ratio: The desired overall compression ratio this plan
            was built toward, carried through from ``CompressionConfig``.
        rationale: A human-readable explanation of why each selected
            type resolved to the strategy it did.
        strategy_names: The registered ``ICompressionStrategy.name()``
            actually invoked for each entry of ``selected_types``.
    """

    selected_types: List[CompressionType] = field(default_factory=list)
    target_ratio: float = 1.0
    rationale: str = ""
    strategy_names: Dict[CompressionType, str] = field(default_factory=dict)


class CompressionPlanner(PipelineStage):
    """Builds a :class:`CompressionPlan` and applies it to the run's
    ``IMR``.

    Attributes:
        _logger: Structured logging sink.
        _config: The run's ``CompressionConfig``, supplied via
            constructor injection.
        _strategies: Every available ``ICompressionStrategy``, keyed by
            its ``name()``, from which this planner selects the ones
            appropriate for each selected ``CompressionType``.
    """

    def __init__(
        self,
        logger: ILogger,
        config: CompressionConfig,
        strategies: Dict[str, ICompressionStrategy],
        calibration_dataset_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the planner.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            config: The run's ``CompressionConfig``, sourced from
                ``IConfigRepository`` by the ``CompositionRoot``.
            strategies: Every registered ``ICompressionStrategy``
                (first-party or plugin), keyed by ``name()``. Which
                entries are required depends on ``config.enabled_types``
                and is validated lazily in :meth:`apply`, not here,
                since a run may enable only a subset of
                ``CompressionType`` values needing only a subset of
                strategies.
            calibration_dataset_path: Optional path for calibration.
            model_path: Optional path for the model.
        """
        self._logger = logger
        self._config = config
        self._strategies = strategies
        self._calibration_dataset_path = calibration_dataset_path
        self._model_path = model_path

    def execute(self, context: PipelineContext) -> StageResult:
        """Build and apply this run's compression plan.

        Reads the current ``IMR`` per the resolution order documented
        in the module docstring.

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is a tuple of
            ``(CompressionPlan, IMR)`` — the finalized plan and the
            new, compressed ``IMR`` produced from it.

        Raises:
            CompressionError: If plan application fails (see
                :meth:`apply`).
        """
        start = time.monotonic()
        self._logger.warning(f"DEBUG PLANNER EXECUTE: self_cal_path={self._calibration_dataset_path}, self_model_path={self._model_path}")
        if not context.has("workflow_config") and (self._calibration_dataset_path or self._model_path):
            from uaqe.application.workflow_config import WorkflowConfig
            w_config = WorkflowConfig(
                model_path=self._model_path or "",
                hardware_profile_id="",
                calibration_dataset_path=self._calibration_dataset_path
            )
            context.append("workflow_config", StageResult(stage_name="workflow_config", success=True, payload=w_config))
        imr = self._resolve_current_imr(context)

        plan = self.build_plan(imr)
        
        before_bytes = sum(len(t.data) for l in imr.layers for t in l.parameters.values())
        compressed_imr = self.apply(imr, plan, context=context)
        after_bytes = sum(len(t.data) for l in compressed_imr.layers for t in l.parameters.values())

        achieved_reduction = before_bytes - after_bytes
        compression_ratio = before_bytes / max(after_bytes, 1)

        duration_ms = (time.monotonic() - start) * 1000.0
        self._logger.info(
            "Compression plan applied.",
            stage_name=self.name(),
            selected_types=[t.value for t in plan.selected_types],
            before_bytes=before_bytes,
            after_bytes=after_bytes,
            achieved_reduction=achieved_reduction,
            compression_ratio=compression_ratio,
            duration_ms=duration_ms,
        )

        warnings = []
        if achieved_reduction <= 0:
            warning_msg = "Compression did not reduce parameter storage size."
            self._logger.warning(warning_msg, stage_name=self.name())
            warnings.append(warning_msg)

        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=(plan, compressed_imr),
            warnings=warnings,
            duration_ms=duration_ms,
        )

    def build_plan(self, imr: IMR) -> CompressionPlan:
        """Build this run's :class:`CompressionPlan` from
        ``self._config``.

        Args:
            imr: The model being compressed. Not currently consulted
                for selection (selection is config-driven), accepted so
                that a future model-aware heuristic can be added
                without a signature change.

        Returns:
            The finalized :class:`CompressionPlan`, including the
            ``ICompressionStrategy`` name resolved for every selected
            type.
        """
        del imr  # Reserved for a future model-aware selection heuristic.
        selected_types = [
            compression_type
            for compression_type in _CANONICAL_TYPE_ORDER
            if compression_type in self._config.enabled_types
        ]

        strategy_names: Dict[CompressionType, str] = {}
        rationale_parts: List[str] = []
        for compression_type in selected_types:
            if compression_type == CompressionType.PRUNING:
                exp_strategy = getattr(self._config, "experimental_pruning_strategy", "magnitude_pruning")
                if exp_strategy != "magnitude_pruning" and exp_strategy in self._strategies:
                    strategy_name = exp_strategy
                    rationale_parts.append(
                        f"PRUNING -> {strategy_name} (experimental override)"
                    )
                else:
                    mode = (
                        "structured"
                        if self._config.pruning_sparsity
                        >= _STRUCTURED_PRUNING_SPARSITY_THRESHOLD
                        else "magnitude"
                    )
                    strategy_name = _PRUNING_STRATEGY_NAME_BY_MODE[mode]
                    rationale_parts.append(
                        f"PRUNING -> {strategy_name} "
                        f"(pruning_sparsity={self._config.pruning_sparsity} "
                        f"{'>=' if mode == 'structured' else '<'} "
                        f"{_STRUCTURED_PRUNING_SPARSITY_THRESHOLD} threshold)"
                    )
            else:
                strategy_name = _STRATEGY_NAME_BY_TYPE[compression_type]
                rationale_parts.append(f"{compression_type.value} -> {strategy_name}")
            strategy_names[compression_type] = strategy_name

        rationale = (
            "; ".join(rationale_parts)
            if rationale_parts
            else "No CompressionType enabled; IMR passed through unchanged."
        )
        return CompressionPlan(
            selected_types=selected_types,
            target_ratio=self._config.target_ratio,
            rationale=rationale,
            strategy_names=strategy_names,
        )



    def apply(self, imr: IMR, plan: CompressionPlan, context: Optional[PipelineContext] = None) -> IMR:
        """Apply ``plan`` to ``imr`` by running each selected type's
        resolved strategy in sequence.

        Args:
            imr: The model to compress.
            plan: A plan previously produced by :meth:`build_plan`.
            context: The pipeline execution context, passed to strategy.

        Returns:
            A new, compressed ``IMR`` reflecting every selected
            compression type applied in :data:`_CANONICAL_TYPE_ORDER`.
            The input ``imr`` is not modified. Returns ``imr`` itself
            unchanged if ``plan.selected_types`` is empty.

        Raises:
            CompressionError: If any entry of ``plan.strategy_names`` is
                not a registered strategy, or if a strategy itself
                fails to apply the plan.
        """
        current_imr = imr
        for compression_type in plan.selected_types:
            strategy_name = plan.strategy_names[compression_type]
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                raise CompressionError(
                    f"No registered ICompressionStrategy named "
                    f"{strategy_name!r} for {compression_type.value}.",
                    code="COMPRESS_MISSING_STRATEGY",
                )
            try:
                # Call apply with context if accepted by signature
                import inspect
                sig = inspect.signature(strategy.apply)
                if "context" in sig.parameters:
                    current_imr = strategy.apply(current_imr, self._config, context=context)
                else:
                    current_imr = strategy.apply(current_imr, self._config)
            except Exception as e:
                if isinstance(e, CompressionError):
                    raise
                raise CompressionError(
                    f"Compression strategy '{strategy_name}' failed: {e}",
                    code="COMPRESSION_FAILED"
                ) from e
        return current_imr

    def _resolve_current_imr(self, context: PipelineContext) -> IMR:
        """Resolve the current, pre-compression ``IMR`` from
        ``context``.

        Args:
            context: The current run's pipeline context.

        Returns:
            The ``IMR`` produced by ``"quantization_planner"`` if that
            stage has run, otherwise the ``IMR`` produced by
            ``"model_loader"``.

        Raises:
            KeyError: If neither stage has been recorded in ``context``
                (propagated from ``PipelineContext.get``).
        """
        if context.has(_QUANTIZATION_STAGE_NAME):
            _, quantized_imr = context.get(_QUANTIZATION_STAGE_NAME).payload
            return quantized_imr
        return context.get(_MODEL_LOADER_STAGE_NAME).payload
