"""Rich, report-oriented analysis output for one analyzed ``IMR``.

``AnalysisReport`` is a strict superset of the locked
``AnalysisResult`` dataclass (``model_analyzer.py``,
``03_API_Specification.md`` §3.3): it carries every field
``ModelAnalyzer.analyze()`` must return, plus the finer-grained
per-analyzer breakdowns (``ComplexityProfile``, unsupported-layer
names, per-layer parameter/FLOPs/memory maps, and warnings) consumed
by ``ReportGenerator``/``OptimizationAdvisor`` once those modules are
implemented. Internal helper for ``uaqe.analyzer``; not part of the
locked class inventory — ``ModelAnalyzer.analyze()`` narrows an
``AnalysisReport`` down to the locked ``AnalysisResult`` shape before
returning (``09_Architecture_Lock.md`` §9, §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from uaqe.analyzer.complexity_analyzer import ComplexityProfile


@dataclass(frozen=True)
class AnalysisReport:
    """The full output of ``Analyzer.analyze()`` for one ``IMR``.

    Attributes:
        layer_graph_summary: Structural graph summary, as produced by
            ``LayerAnalyzer.build_summary``.
        parameter_count: Total parameter element count, as produced by
            ``ParameterCounter.count_total``.
        estimated_flops: Total estimated FLOPs, as produced by
            ``FlopsEstimator.estimate_total``.
        estimated_memory_bytes: Total estimated memory footprint, in
            bytes, as produced by ``MemoryAnalyzer.estimate_total``.
        op_type_histogram: Per-``op_type`` layer counts, as produced by
            ``LayerAnalyzer.op_type_histogram``.
        per_layer_parameter_count: Per-layer parameter-count
            breakdown, as produced by
            ``ParameterCounter.count_per_layer``.
        per_layer_flops: Per-layer FLOPs breakdown, as produced by
            ``FlopsEstimator.estimate_per_layer``.
        per_layer_memory_bytes: Per-layer weight-memory breakdown, as
            produced by ``MemoryAnalyzer.per_layer_weight_bytes``.
        complexity: The aggregate complexity classification, as
            produced by ``ComplexityAnalyzer.build_profile``.
        unsupported_layers: Names of layers with an unrecognized
            ``op_type``, as produced by
            ``UnsupportedLayerDetector.find_unsupported``.
        warnings: Every non-fatal warning accumulated across the
            sub-analyzers (FLOPs lower-bound caveats, unsupported-layer
            notices), suitable for ``StageResult.warnings``.
    """

    layer_graph_summary: Dict[str, Any]
    parameter_count: int
    estimated_flops: int
    estimated_memory_bytes: int
    op_type_histogram: Dict[str, int]
    per_layer_parameter_count: Dict[str, int]
    per_layer_flops: Dict[str, int]
    per_layer_memory_bytes: Dict[str, int]
    complexity: ComplexityProfile
    unsupported_layers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
