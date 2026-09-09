"""Model analysis for the Universal AI Quantization Engine: computes
structural and numerical properties (parameter counts, FLOPs, memory
footprint, op-type histogram, complexity classification, and
unsupported-operator detection) for a loaded ``IMR``.

Locked public class inventory (``09_Architecture_Lock.md`` §3,
``uaqe.domain.model``): ``AnalysisResult`` and ``ModelAnalyzer``
(``model_analyzer.py``) — these are the only two classes other
packages should depend on.

Every other class in this package (``Analyzer``, ``LayerAnalyzer``,
``ParameterCounter``, ``FlopsEstimator``, ``MemoryAnalyzer``,
``ComplexityAnalyzer``, ``ComplexityProfile``,
``UnsupportedLayerDetector``, ``AnalysisReport``) is an internal helper
that backs ``ModelAnalyzer``'s locked behavior without being part of
the locked class inventory itself (``09_Architecture_Lock.md`` §11,
"Internal API") — mirroring how ``uaqe.domain.model`` already exposes
``FrameworkDetector``/``LoaderFactory`` alongside ``ModelLoader``. They
are exported here for direct unit testing but are not a stable public
contract: ``ModelAnalyzer`` is the only class other packages should
depend on.

Per ``01_Project_Architecture.md`` §7's prohibited edges, nothing in
this package imports ``uaqe.infrastructure`` or ``uaqe.application``.
"""

from __future__ import annotations

from uaqe.analyzer.analysis_report import AnalysisReport
from uaqe.analyzer.analyzer import Analyzer
from uaqe.analyzer.complexity_analyzer import ComplexityAnalyzer, ComplexityProfile
from uaqe.analyzer.flops_estimator import FlopsEstimator
from uaqe.analyzer.layer_analyzer import LayerAnalyzer
from uaqe.analyzer.memory_analyzer import MemoryAnalyzer
from uaqe.analyzer.model_analyzer import AnalysisResult, ModelAnalyzer
from uaqe.analyzer.parameter_counter import ParameterCounter
from uaqe.analyzer.unsupported_layer_detector import UnsupportedLayerDetector

__all__ = [
    "AnalysisResult",
    "ModelAnalyzer",
    "Analyzer",
    "AnalysisReport",
    "LayerAnalyzer",
    "ParameterCounter",
    "FlopsEstimator",
    "MemoryAnalyzer",
    "ComplexityAnalyzer",
    "ComplexityProfile",
    "UnsupportedLayerDetector",
]
