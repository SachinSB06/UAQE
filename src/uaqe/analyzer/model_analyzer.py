"""Computes the layer graph summary, parameter counts, FLOPs, memory
footprint, and op-type histogram for a loaded ``IMR``.

Per ``01_Project_Architecture.md`` §6, ``ModelAnalyzer``'s
responsibility is strictly to *compute* these metrics — it must never
modify the ``IMR`` it analyzes.

Locked contract: ``03_API_Specification.md`` §3.3.
Locked payload mapping: ``09_Architecture_Lock.md`` §9
(``model_analyzer`` -> ``AnalysisResult``).
Locked pipeline position: ``09_Architecture_Lock.md`` §12 — runs
immediately after ``model_validator`` and before ``hardware_manager``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from uaqe.analyzer.analysis_report import AnalysisReport
from uaqe.analyzer.analyzer import Analyzer
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage


@dataclass(frozen=True)
class AnalysisResult:
    """The locked, minimal analysis payload consumed by downstream stages.

    Per ``04_Data_Flow.md`` §4, ``LayerCompatibilityChecker`` and every
    later stage that reads ``"model_analyzer"``'s ``StageResult.payload``
    depends on exactly this shape. The richer per-analyzer breakdown
    (``ComplexityProfile``, unsupported-layer names, per-layer maps) is
    available on the fuller
    ``uaqe.analyzer.analysis_report.AnalysisReport`` that
    :meth:`ModelAnalyzer.analyze` builds internally but does not
    return here, to keep this locked dataclass exactly as specified.

    Attributes:
        layer_graph_summary: Structural summary of the layer graph
            (layer count, op-type histogram, graph depth, per-layer
            descriptors).
        parameter_count: Total learnable-parameter element count.
        estimated_flops: Total estimated FLOPs (see
            ``uaqe.analyzer.flops_estimator`` for exactness caveats).
        estimated_memory_bytes: Total estimated memory footprint, in
            bytes (weights + estimated activation working memory).
        op_type_histogram: Per-``op_type`` layer counts.
    """

    layer_graph_summary: Dict[str, Any]
    parameter_count: int
    estimated_flops: int
    estimated_memory_bytes: int
    op_type_histogram: Dict[str, int]


class ModelAnalyzer(PipelineStage):
    """Computes structural and numerical properties of a loaded ``IMR``.

    Attributes:
        _logger: The logger this stage reports analysis start/end and
            accumulated warnings to.
        _analyzer: The composition facade over every
            ``uaqe.analyzer`` sub-analyzer that performs the actual
            computation.
    """

    def __init__(self, logger: ILogger) -> None:
        """Initialize the analyzer stage.

        Args:
            logger: The logger to report analysis events to.
        """
        self._logger: ILogger = logger
        self._analyzer: Analyzer = Analyzer()

    def execute(self, context: PipelineContext) -> StageResult:
        """Analyze the ``IMR`` produced by the ``model_loader`` stage.

        Args:
            context: The run's pipeline context; per ``04_Data_Flow.md``
                §4, the ``IMR`` to analyze is read from
                ``context.get("model_loader").payload`` (not
                ``model_validator``, whose payload is always ``None``).

        Returns:
            A ``StageResult`` whose ``payload`` is this run's
            ``AnalysisResult``. Any FLOPs lower-bound caveats or
            unrecognized-``op_type`` notices are surfaced in
            ``warnings`` rather than raised, since neither condition
            prevents analysis from completing.
        """
        started_at = time.monotonic()
        self._logger.info("model_analyzer: starting analysis", stage="model_analyzer")

        imr: IMR = context.get("model_loader").payload
        report: AnalysisReport = self._analyzer.analyze(imr)
        result = self._narrow(report)

        duration_ms = (time.monotonic() - started_at) * 1000.0
        self._logger.info(
            "model_analyzer: analysis complete",
            stage="model_analyzer",
            parameter_count=result.parameter_count,
            estimated_flops=result.estimated_flops,
            estimated_memory_bytes=result.estimated_memory_bytes,
            unsupported_layer_count=len(report.unsupported_layers),
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=result,
            warnings=report.warnings,
            duration_ms=duration_ms,
        )

    def analyze(self, imr: IMR) -> AnalysisResult:
        """Compute the locked ``AnalysisResult`` for ``imr``.

        Args:
            imr: The model representation to analyze.

        Returns:
            The locked-shape ``AnalysisResult``, narrowed down from
            the fuller ``AnalysisReport`` the internal ``Analyzer``
            facade computes.
        """
        return self._narrow(self._analyzer.analyze(imr))

    @staticmethod
    def _narrow(report: AnalysisReport) -> AnalysisResult:
        """Project a full ``AnalysisReport`` down to the locked ``AnalysisResult`` shape.

        Args:
            report: The full analysis output to narrow.

        Returns:
            An ``AnalysisResult`` carrying exactly the fields locked by
            ``03_API_Specification.md`` §3.3.
        """
        return AnalysisResult(
            layer_graph_summary=report.layer_graph_summary,
            parameter_count=report.parameter_count,
            estimated_flops=report.estimated_flops,
            estimated_memory_bytes=report.estimated_memory_bytes,
            op_type_histogram=report.op_type_histogram,
        )
