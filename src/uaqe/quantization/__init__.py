"""``uaqe.quantization`` — decides and applies numeric-precision
transforms to a calibrated ``IMR`` for the Universal AI Quantization
Engine.

This package covers the same responsibilities as the locked
``uaqe.domain.quantization`` module described in
``10_Module_Development_Guide.md`` §6 (advise -> calibrate -> analyze
sensitivity -> apply), decomposed into eleven single-responsibility
files rather than the doc's four:

Pipeline (run in this order; each is its own ``PipelineStage`` except
where noted):

1. :class:`~uaqe.quantization.calibrator.Calibrator` — gathers
   per-layer numeric-range statistics
   (:class:`~uaqe.quantization.calibrator.CalibrationStatistics`) from
   layer weights and an optional
   :class:`~uaqe.quantization.calibration_dataset.CalibrationDataset`.
2. :class:`~uaqe.quantization.sensitivity_analyzer.SensitivityAnalyzer`
   — estimates each layer's susceptibility to quantization error
   (:class:`~uaqe.quantization.sensitivity_analyzer.SensitivityReport`)
   from those statistics.
3. :class:`~uaqe.quantization.quantization_planner.QuantizationPlanner`
   — the orchestrating stage. Internally uses
   :class:`~uaqe.quantization.precision_recommender.
   PrecisionRecommender` (not a stage itself) to build a
   :class:`~uaqe.quantization.quantization_planner.QuantizationPlan`,
   then applies it via the appropriate injected
   ``IQuantizationStrategy`` (one of
   :class:`~uaqe.quantization.int8_quantizer.Int8Quantizer`,
   :class:`~uaqe.quantization.int4_quantizer.Int4Quantizer`, or
   :class:`~uaqe.quantization.mixed_precision_quantizer.
   MixedPrecisionQuantizer`), each of which shares one implementation
   of the actual quantize/dequantize arithmetic via
   :class:`~uaqe.quantization.layer_quantizer.LayerQuantizer`.

Reporting:

- :class:`~uaqe.quantization.quantization_report.
  QuantizationReportRenderer` summarizes a completed
  ``QuantizationPlan`` into a Markdown/dict
  :class:`~uaqe.quantization.quantization_report.
  QuantizationReportDocument`.

Per ``09_Architecture_Lock.md`` §8, this package depends only on
``uaqe.common`` and ``uaqe.domain`` (for ``PipelineStage``,
``PipelineContext``, and ``HardwareProfile``) — never on
``uaqe.infrastructure`` or ``uaqe.interface``. The three
``IQuantizationStrategy`` implementations are registered with
``uaqe.infrastructure.plugins.PluginRegistry`` the same way a
third-party plugin would be
(``10_Module_Development_Guide.md`` §18), so
``QuantizationPlanner`` depends on them only through the
``IQuantizationStrategy`` port passed to its constructor, never by
importing a concrete strategy class directly.
"""

from uaqe.quantization.calibration_dataset import CalibrationBatch, CalibrationDataset
from uaqe.quantization.calibrator import Calibrator, CalibrationStatistics
from uaqe.quantization.int4_quantizer import Int4Quantizer
from uaqe.quantization.int8_quantizer import Int8Quantizer
from uaqe.quantization.layer_quantizer import LayerQuantizer, QuantizationParams
from uaqe.quantization.mixed_precision_quantizer import MixedPrecisionQuantizer
from uaqe.quantization.precision_recommender import (
    PrecisionRecommendation,
    PrecisionRecommender,
)
from uaqe.quantization.quantization_planner import QuantizationPlan, QuantizationPlanner
from uaqe.quantization.quantization_report import (
    QuantizationReportDocument,
    QuantizationReportRenderer,
)
from uaqe.quantization.sensitivity_analyzer import (
    SensitivityAnalyzer,
    SensitivityReport,
)

__all__ = [
    # calibration_dataset.py
    "CalibrationDataset",
    "CalibrationBatch",
    # calibrator.py
    "Calibrator",
    "CalibrationStatistics",
    # sensitivity_analyzer.py
    "SensitivityAnalyzer",
    "SensitivityReport",
    # precision_recommender.py
    "PrecisionRecommender",
    "PrecisionRecommendation",
    # layer_quantizer.py
    "LayerQuantizer",
    "QuantizationParams",
    # int8_quantizer.py
    "Int8Quantizer",
    # int4_quantizer.py
    "Int4Quantizer",
    # mixed_precision_quantizer.py
    "MixedPrecisionQuantizer",
    # quantization_planner.py
    "QuantizationPlanner",
    "QuantizationPlan",
    # quantization_report.py
    "QuantizationReportRenderer",
    "QuantizationReportDocument",
]
