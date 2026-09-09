"""Per-layer quantization sensitivity estimation.

``SensitivityAnalyzer`` runs after :class:`~uaqe.quantization.calibrator.
Calibrator` and before :class:`~uaqe.quantization.precision_recommender.
PrecisionRecommender` in this package's pipeline. It has no ground-truth
evaluation dataset or execution runtime available to it (see the module
docstring of ``uaqe.quantization.calibrator`` for why), so it estimates
each layer's *susceptibility* to quantization error from the
distributional shape of the ``CalibrationStatistics`` gathered for that
layer, rather than measuring an actual accuracy delta end to end. A
layer whose observed value range is stretched far beyond where the bulk
of its mass actually lives — a handful of true outliers dominating an
otherwise well-behaved distribution — wastes much of a fixed-step
quantizer's representable levels on those rare extremes, coarsening
resolution for everything else; that "how far past the bulk do the
extremes reach" ratio, adjusted for how far extremes are *expected* to
reach by chance alone as sample count grows (see ``_estimate_sensitivity``),
is this analyzer's proxy for per-layer accuracy impact. A merely
unimodal/peaked-but-ordinary distribution (which describes almost every
real weight or activation tensor) is not by itself evidence of
sensitivity and is scored at or near ``0.0``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.quantization.calibrator import CalibrationStatistics

#: Safety margin (in standard deviations) added on top of the
#: extreme-value-theory expected spread in :meth:`SensitivityAnalyzer.
#: _estimate_sensitivity`, so that ordinary sampling variation in a
#: well-conditioned, roughly-unimodal distribution never produces a
#: spurious positive score. Only spread genuinely in excess of this
#: margin is attributed to real outliers.
_SPREAD_SAFETY_MARGIN = 1.0


@dataclass(frozen=True)
class SensitivityReport:
    """Per-layer quantization sensitivity, estimated from calibration
    statistics.

    Attributes:
        per_layer_accuracy_delta: An estimated, unitless accuracy-impact
            score for each layer, keyed by ``IMRLayer.name``. Higher
            values indicate a layer whose value distribution is
            expected to lose more effective precision under a fixed-step
            symmetric quantizer; ``0.0`` for a degenerate (empty or
            all-zero) layer.
        flagged_layers: Names of layers whose
            ``per_layer_accuracy_delta`` exceeded the analysis
            ``threshold``, in descending order of severity.
    """

    per_layer_accuracy_delta: Dict[str, float] = field(default_factory=dict)
    flagged_layers: List[str] = field(default_factory=list)


class SensitivityAnalyzer(PipelineStage):
    """Estimates each layer's susceptibility to quantization error.

    Attributes:
        _logger: Structured logging sink.
        _default_threshold: The sensitivity threshold ``execute()``
            passes to :meth:`analyze`, sourced from the run's
            ``QuantizationConfig.sensitivity_threshold`` by the
            ``CompositionRoot`` at construction time
            (``10_Module_Development_Guide.md`` §0 rule 7 — this stage
            never reads config out of ``PipelineContext`` or a config
            file itself).
    """

    def __init__(self, logger: ILogger, default_threshold: float = 0.0) -> None:
        """Initialize the analyzer.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            default_threshold: The ``QuantizationConfig.
                sensitivity_threshold`` value ``execute()`` will use;
                supplied via constructor injection, consistent with
                every config value object in this codebase.
        """
        self._logger = logger
        self._default_threshold = default_threshold

    def execute(self, context: PipelineContext) -> StageResult:
        """Analyze the current run's calibration statistics.

        Reads the ``IMR`` from ``"model_loader"`` and the
        :class:`~uaqe.quantization.calibrator.CalibrationStatistics`
        from ``"calibrator"``'s ``StageResult.payload``, then analyzes
        them against ``self._default_threshold``.

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is the computed
            :class:`SensitivityReport`. Every flagged layer name is also
            surfaced as a ``StageResult.warnings`` entry.
        """
        start = time.monotonic()
        imr = context.get("model_loader").payload
        stats = context.get("calibrator").payload
        threshold = self._default_threshold

        report = self.analyze(imr, stats, threshold)
        duration_ms = (time.monotonic() - start) * 1000.0
        warnings = [
            f"Layer {name!r} flagged as quantization-sensitive "
            f"(score={report.per_layer_accuracy_delta[name]:.4f} > "
            f"threshold={threshold:.4f})."
            for name in report.flagged_layers
        ]
        for warning in warnings:
            self._logger.warning(warning, stage_name=self.name())

        self._logger.info(
            "Sensitivity analysis complete.",
            stage_name=self.name(),
            flagged_layer_count=len(report.flagged_layers),
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=report,
            warnings=warnings,
            duration_ms=duration_ms,
        )

    def analyze(
        self, imr: IMR, stats: CalibrationStatistics, threshold: float
    ) -> SensitivityReport:
        """Estimate per-layer quantization sensitivity from ``stats``.

        Args:
            imr: The calibrated model.
            stats: The calibration statistics gathered by
                :class:`~uaqe.quantization.calibrator.Calibrator`.
            threshold: The minimum ``per_layer_accuracy_delta`` score at
                or above which a layer is added to
                ``SensitivityReport.flagged_layers``.

        Returns:
            The computed :class:`SensitivityReport`.
        """
        per_layer_score: Dict[str, float] = {}
        for layer in imr.layers:
            value_range = stats.per_layer_activation_range.get(layer.name, (0.0, 0.0))
            histogram = stats.per_layer_histogram.get(layer.name, [])
            per_layer_score[layer.name] = self._estimate_sensitivity(
                value_range, histogram
            )

        flagged = sorted(
            (name for name, score in per_layer_score.items() if score > threshold),
            key=lambda name: per_layer_score[name],
            reverse=True,
        )
        return SensitivityReport(
            per_layer_accuracy_delta=per_layer_score, flagged_layers=flagged
        )

    @staticmethod
    def _estimate_sensitivity(value_range, histogram: List[int]) -> float:
        """Estimate one layer's sensitivity score from its calibration
        range and histogram.

        The score measures how far the layer's observed extremes reach
        beyond where the bulk of its values actually live, in units of
        the layer's own (histogram-reconstructed) standard deviation,
        in excess of how far those extremes are *expected* to reach by
        chance alone given how many samples were observed.

        This two-part design deliberately avoids two failure modes a
        naive peakedness/spread measure falls into:

        - Measuring "spread" as ``span / magnitude`` degenerates to a
          near-constant ``~2.0`` for any symmetric mixed-sign range
          regardless of the tensor's actual shape, contributing no
          real signal.
        - Treating a histogram's peak-bin fraction, on its own, as a
          sensitivity signal misidentifies *any* ordinary unimodal
          distribution (which is what almost every real weight/
          activation tensor looks like) as "sensitive", since a
          concentrated central bin is the norm for such distributions,
          not an anomaly.

        Instead, the layer's mean and standard deviation are
        reconstructed from the histogram (bin midpoints weighted by
        bin counts), and the observed extent from the mean (in units
        of that standard deviation) is compared against the expected
        extreme value of ``N`` i.i.d. samples from a roughly unimodal
        distribution, which by extreme-value theory grows on the order
        of ``sqrt(2 * ln(N))`` standard deviations purely from sample
        count — with no relation to quantization risk. Only spread in
        genuine excess of that expectation (plus
        ``_SPREAD_SAFETY_MARGIN``) reflects true outliers stretching
        the layer's quantization range far past where its bulk of
        values live, and is scored above ``0.0``.

        Args:
            value_range: The layer's observed ``(min, max)`` value
                range.
            histogram: The layer's equal-width value histogram bin
                counts.

        Returns:
            A non-negative sensitivity score; ``0.0`` for a degenerate
            (empty, all-zero-range, or ordinary/well-conditioned)
            layer, growing above ``0.0`` only as genuine outliers pull
            the observed range past what sample count alone explains.
        """
        value_min, value_max = value_range
        total_samples = sum(histogram)
        if total_samples == 0 or value_max == value_min:
            return 0.0

        num_bins = len(histogram)
        bin_width = (value_max - value_min) / num_bins

        def _bin_center(index: int) -> float:
            return value_min + (index + 0.5) * bin_width

        mean = (
            sum(count * _bin_center(i) for i, count in enumerate(histogram) if count)
            / total_samples
        )
        variance = (
            sum(
                count * (_bin_center(i) - mean) ** 2
                for i, count in enumerate(histogram)
                if count
            )
            / total_samples
        )
        # A histogram with (nearly) all of its mass in a single bin
        # still carries irreducible within-bin uncertainty: a uniform
        # distribution spanning one bin has variance ``bin_width**2 /
        # 12``. Flooring the reconstructed standard deviation at this
        # value keeps the ratio below finite for such cases instead of
        # exploding purely from histogram binning resolution, while
        # still yielding a very high (but finite, and correctly
        # positive) score when genuine outliers are present.
        std = math.sqrt(max(variance, (bin_width ** 2) / 12.0))

        extent = max(value_max - mean, mean - value_min)
        observed_spread = extent / std

        expected_spread = (
            math.sqrt(2.0 * math.log(max(total_samples, 2))) + _SPREAD_SAFETY_MARGIN
        )

        return max(0.0, (observed_spread - expected_spread) / expected_spread)
