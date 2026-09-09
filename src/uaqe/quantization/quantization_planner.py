"""Orchestration and application of the final per-layer quantization plan.

``QuantizationPlanner`` is the single ``PipelineStage`` entry point for
this package: it runs after
:class:`~uaqe.quantization.calibrator.Calibrator` and
:class:`~uaqe.quantization.sensitivity_analyzer.SensitivityAnalyzer`
(each a separate stage in its own right, so their intermediate results
are independently inspectable in ``PipelineContext``/reports), then
uses :class:`~uaqe.quantization.precision_recommender.
PrecisionRecommender` to decide a per-layer precision mapping and an
injected ``IQuantizationStrategy`` to actually transform the ``IMR``.

Per ``01_Project_Architecture.md`` §5 and this package's overall
contract (see ``uaqe.quantization.__init__``), the ``IMR`` produced here
is always a new object — the ``IMR`` read from ``"model_loader"``
remains valid and unmodified in ``PipelineContext`` history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from uaqe.common.exceptions import QuantizationError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.interfaces.i_quantization_strategy import IQuantizationStrategy
from uaqe.common.result_types import StageResult
from uaqe.common.types import Precision
from uaqe.common.value_objects import QuantizationConfig
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.quantization.calibrator import CalibrationStatistics
from uaqe.quantization.precision_recommender import PrecisionRecommender
from uaqe.quantization.sensitivity_analyzer import SensitivityReport

#: The ``IQuantizationStrategy.name()`` selected whenever the resolved
#: per-layer precision mapping is uniform (every layer maps to the same
#: precision), and that precision has a dedicated single-precision
#: strategy registered.
_SINGLE_PRECISION_STRATEGY_NAMES = {
    Precision.INT8: "int8",
    Precision.INT4: "int4",
}

#: The ``IQuantizationStrategy.name()`` selected whenever the resolved
#: per-layer precision mapping is not uniform.
_MIXED_PRECISION_STRATEGY_NAME = "mixed_precision"


@dataclass(frozen=True)
class QuantizationPlan:
    """The complete, finalized quantization decision for one run.

    Attributes:
        per_layer_precision: The precision every layer will be quantized
            to, keyed by ``IMRLayer.name`` — the output of
            :class:`~uaqe.quantization.precision_recommender.
            PrecisionRecommender.recommend`.
        rationale: A human-readable explanation of why each layer
            received its precision, keyed by the same layer names.
        calibration: The :class:`~uaqe.quantization.calibrator.
            CalibrationStatistics` this plan was recommended from.
        sensitivity: The :class:`~uaqe.quantization.
            sensitivity_analyzer.SensitivityReport` this plan was
            recommended from.
        selected_strategy_name: The ``IQuantizationStrategy.name()``
            actually invoked to apply this plan.
        overridden_layers: Names of layers whose requested precision
            was not supported by the target hardware and was therefore
            overridden, as computed by
            :meth:`~uaqe.quantization.precision_recommender.
            PrecisionRecommender.recommend`.
        stepped_up_layers: Names of layers flagged as
            quantization-sensitive whose resolved precision was stepped
            up to a more permissive, supported precision, as computed by
            :meth:`~uaqe.quantization.precision_recommender.
            PrecisionRecommender.recommend`.
    """

    per_layer_precision: Dict[str, Precision] = field(default_factory=dict)
    rationale: Dict[str, str] = field(default_factory=dict)
    calibration: Optional[CalibrationStatistics] = None
    sensitivity: Optional[SensitivityReport] = None
    selected_strategy_name: str = ""
    overridden_layers: List[str] = field(default_factory=list)
    stepped_up_layers: List[str] = field(default_factory=list)


class QuantizationPlanner(PipelineStage):
    """Builds a :class:`QuantizationPlan` and applies it to the run's
    ``IMR``.

    Attributes:
        _logger: Structured logging sink.
        _precision_recommender: Per-layer precision recommendation
            logic, hard-filtered against hardware support.
        _config: The run's ``QuantizationConfig``, supplied via
            constructor injection.
        _strategies: Every available ``IQuantizationStrategy``, keyed
            by its ``name()``, from which this planner selects the one
            appropriate for the resolved plan's precision distribution.
    """

    def __init__(
        self,
        logger: ILogger,
        precision_recommender: PrecisionRecommender,
        config: QuantizationConfig,
        strategies: Dict[str, IQuantizationStrategy],
    ) -> None:
        """Initialize the planner.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            precision_recommender: Recommends the per-layer precision
                mapping this planner turns into a
                :class:`QuantizationPlan`.
            config: The run's ``QuantizationConfig``, sourced from
                ``IConfigRepository`` by the ``CompositionRoot``.
            strategies: Every registered ``IQuantizationStrategy``
                (first-party or plugin), keyed by ``name()``. Must
                include entries for every precision this run might
                resolve to using — at minimum ``"mixed_precision"``,
                since it can apply any precision distribution.

        Raises:
            QuantizationError: If ``strategies`` does not include a
                ``"mixed_precision"`` entry.
        """
        if _MIXED_PRECISION_STRATEGY_NAME not in strategies:
            raise QuantizationError(
                "QuantizationPlanner requires a registered "
                f"{_MIXED_PRECISION_STRATEGY_NAME!r} strategy as a "
                f"fallback for non-uniform precision plans.",
                code="QUANT_MISSING_STRATEGY",
            )
        self._logger = logger
        self._precision_recommender = precision_recommender
        self._config = config
        self._strategies = strategies

    def execute(self, context: PipelineContext) -> StageResult:
        """Build and apply this run's quantization plan.

        Reads the ``IMR`` from ``"model_loader"``, the target
        ``HardwareProfile`` from ``"hardware_manager"``, the
        :class:`~uaqe.quantization.calibrator.CalibrationStatistics`
        from ``"calibrator"``, and the
        :class:`~uaqe.quantization.sensitivity_analyzer.
        SensitivityReport` from ``"sensitivity_analyzer"``.

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is a tuple of
            ``(QuantizationPlan, IMR)`` — the finalized plan and the
            new, quantized ``IMR`` produced from it.

        Raises:
            QuantizationError: If plan application fails (see
                :meth:`apply`).
        """
        start = time.monotonic()
        imr = context.get("model_loader").payload
        profile = context.get("hardware_manager").payload
        calibration = context.get("calibrator").payload
        sensitivity = context.get("sensitivity_analyzer").payload

        plan = self.build_plan(imr, profile, calibration, sensitivity)
        quantized_imr = self.apply(imr, plan)

        duration_ms = (time.monotonic() - start) * 1000.0
        self._logger.info(
            "Quantization plan applied.",
            stage_name=self.name(),
            selected_strategy=plan.selected_strategy_name,
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=(plan, quantized_imr),
            duration_ms=duration_ms,
        )

    def build_plan(
        self,
        imr: IMR,
        profile: HardwareProfile,
        calibration: CalibrationStatistics,
        sensitivity: SensitivityReport,
    ) -> QuantizationPlan:
        """Build this run's :class:`QuantizationPlan`.

        Args:
            imr: The model being quantized.
            profile: The resolved deployment target.
            calibration: The gathered calibration statistics.
            sensitivity: The estimated per-layer sensitivity report.

        Returns:
            The finalized :class:`QuantizationPlan`, including the
            ``IQuantizationStrategy`` name selected to apply it.

        Raises:
            QuantizationError: If no registered strategy can apply the
                recommended precision distribution (see
                :meth:`_select_strategy_name`).
        """
        recommendation = self._precision_recommender.recommend(
            imr, profile, self._config, sensitivity
        )
        strategy_name = self._select_strategy_name(recommendation.per_layer_precision)
        return QuantizationPlan(
            per_layer_precision=recommendation.per_layer_precision,
            rationale=recommendation.rationale,
            calibration=calibration,
            sensitivity=sensitivity,
            selected_strategy_name=strategy_name,
            overridden_layers=recommendation.overridden_layers,
            stepped_up_layers=recommendation.stepped_up_layers,
        )

    def apply(self, imr: IMR, plan: QuantizationPlan) -> IMR:
        """Apply ``plan`` to ``imr`` via its selected strategy.

        Args:
            imr: The model to quantize.
            plan: A plan previously produced by :meth:`build_plan`.

        Returns:
            A new, quantized ``IMR``. The input ``imr`` is not
            modified.

        Raises:
            QuantizationError: If ``plan.selected_strategy_name`` is not
                a registered strategy, or if the strategy itself fails
                to apply the plan.
        """
        strategy = self._strategies.get(plan.selected_strategy_name)
        if strategy is None:
            raise QuantizationError(
                f"No registered IQuantizationStrategy named "
                f"{plan.selected_strategy_name!r}.",
                code="QUANT_MISSING_STRATEGY",
            )
        strategy_config = QuantizationConfig(
            default_precision=self._config.default_precision,
            per_layer_overrides=plan.per_layer_precision,
            calibration_batch_size=self._config.calibration_batch_size,
            sensitivity_threshold=self._config.sensitivity_threshold,
        )
        return strategy.apply(imr, strategy_config)

    def _select_strategy_name(
        self, per_layer_precision: Dict[str, Precision]
    ) -> str:
        """Select the registered strategy name appropriate for a
        resolved per-layer precision mapping.

        A uniform mapping (every layer resolves to the same precision)
        prefers that precision's dedicated single-precision strategy
        when one is registered, falling back to
        ``"mixed_precision"`` otherwise; a non-uniform mapping always
        uses ``"mixed_precision"``.

        Args:
            per_layer_precision: The per-layer precision mapping to
                select a strategy for.

        Returns:
            The selected strategy's registration name.
        """
        distinct_precisions = set(per_layer_precision.values())
        if len(distinct_precisions) == 1:
            (only_precision,) = distinct_precisions
            preferred_name = _SINGLE_PRECISION_STRATEGY_NAMES.get(only_precision)
            if preferred_name is not None and preferred_name in self._strategies:
                return preferred_name
        return _MIXED_PRECISION_STRATEGY_NAME
