"""Composition facade over every ``uaqe.analyzer`` sub-analyzer.

``Analyzer`` is the single entry point ``ModelAnalyzer.analyze()``
(``model_analyzer.py``) delegates to; it owns no analysis logic of its
own beyond wiring ``LayerAnalyzer``, ``ParameterCounter``,
``FlopsEstimator``, ``MemoryAnalyzer``, ``ComplexityAnalyzer``, and
``UnsupportedLayerDetector`` together and assembling their outputs
into one ``AnalysisReport``. Internal helper for ``uaqe.analyzer``; not
part of the locked class inventory (``09_Architecture_Lock.md`` §11).
"""

from __future__ import annotations

from typing import Optional

from uaqe.analyzer.analysis_report import AnalysisReport
from uaqe.analyzer.complexity_analyzer import ComplexityAnalyzer
from uaqe.analyzer.flops_estimator import FlopsEstimator
from uaqe.analyzer.layer_analyzer import LayerAnalyzer
from uaqe.analyzer.memory_analyzer import MemoryAnalyzer
from uaqe.analyzer.parameter_counter import ParameterCounter
from uaqe.analyzer.unsupported_layer_detector import UnsupportedLayerDetector
from uaqe.common.imr import IMR


class Analyzer:
    """Runs every sub-analyzer against an ``IMR`` and assembles an ``AnalysisReport``.

    Attributes:
        _layer_analyzer: Computes graph structure summaries.
        _parameter_counter: Computes parameter-element counts.
        _flops_estimator: Computes FLOPs estimates.
        _memory_analyzer: Computes memory-footprint estimates.
        _complexity_analyzer: Aggregates metrics into a complexity
            classification.
        _unsupported_layer_detector: Flags layers with an unrecognized
            ``op_type``.
    """

    def __init__(
        self,
        layer_analyzer: Optional[LayerAnalyzer] = None,
        parameter_counter: Optional[ParameterCounter] = None,
        flops_estimator: Optional[FlopsEstimator] = None,
        memory_analyzer: Optional[MemoryAnalyzer] = None,
        complexity_analyzer: Optional[ComplexityAnalyzer] = None,
        unsupported_layer_detector: Optional[UnsupportedLayerDetector] = None,
    ) -> None:
        """Initialize the facade, defaulting every sub-analyzer to its
        stateless first-party implementation.

        Every sub-analyzer is stateless (no constructor parameters, no
        mutable instance state beyond its fixed lookup tables), so
        constructing default instances here — rather than requiring
        every caller to inject all six — keeps ``Analyzer()`` usable
        with zero arguments while still allowing test doubles to be
        substituted (``01_Project_Architecture.md`` §15).

        Args:
            layer_analyzer: Overrides the default ``LayerAnalyzer``.
            parameter_counter: Overrides the default ``ParameterCounter``.
            flops_estimator: Overrides the default ``FlopsEstimator``.
            memory_analyzer: Overrides the default ``MemoryAnalyzer``.
            complexity_analyzer: Overrides the default
                ``ComplexityAnalyzer``.
            unsupported_layer_detector: Overrides the default
                ``UnsupportedLayerDetector``.
        """
        self._layer_analyzer: LayerAnalyzer = layer_analyzer or LayerAnalyzer()
        self._parameter_counter: ParameterCounter = (
            parameter_counter or ParameterCounter()
        )
        self._flops_estimator: FlopsEstimator = flops_estimator or FlopsEstimator()
        self._memory_analyzer: MemoryAnalyzer = memory_analyzer or MemoryAnalyzer()
        self._complexity_analyzer: ComplexityAnalyzer = (
            complexity_analyzer or ComplexityAnalyzer()
        )
        self._unsupported_layer_detector: UnsupportedLayerDetector = (
            unsupported_layer_detector or UnsupportedLayerDetector()
        )

    def analyze(self, imr: IMR) -> AnalysisReport:
        """Run every sub-analyzer against ``imr`` and assemble the result.

        Args:
            imr: The model representation to analyze.

        Returns:
            The full ``AnalysisReport`` for ``imr``.
        """
        layer_graph_summary = self._layer_analyzer.build_summary(imr)
        parameter_count = self._parameter_counter.count_total(imr)
        estimated_flops = self._flops_estimator.estimate_total(imr)
        estimated_memory_bytes = self._memory_analyzer.estimate_total(imr)

        complexity = self._complexity_analyzer.build_profile(
            parameter_count=parameter_count,
            estimated_flops=estimated_flops,
            estimated_memory_bytes=estimated_memory_bytes,
            graph_depth=layer_graph_summary["graph_depth"],
        )

        warnings = list(self._flops_estimator.lower_bound_warnings(imr))
        warnings.extend(self._unsupported_layer_detector.build_warnings(imr))

        return AnalysisReport(
            layer_graph_summary=layer_graph_summary,
            parameter_count=parameter_count,
            estimated_flops=estimated_flops,
            estimated_memory_bytes=estimated_memory_bytes,
            op_type_histogram=self._layer_analyzer.op_type_histogram(imr),
            per_layer_parameter_count=self._parameter_counter.count_per_layer(imr),
            per_layer_flops=self._flops_estimator.estimate_per_layer(imr),
            per_layer_memory_bytes=self._memory_analyzer.per_layer_weight_bytes(imr),
            complexity=complexity,
            unsupported_layers=self._unsupported_layer_detector.find_unsupported(imr),
            warnings=warnings,
        )
