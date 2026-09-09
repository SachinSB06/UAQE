"""Model ingestion, validation, and loader-resolution for the Universal
AI Quantization Engine.

Per ``09_Architecture_Lock.md`` §2, this package is
``uaqe.domain.model``. Its locked public class inventory
(``09_Architecture_Lock.md`` §3) is exactly ``ModelLoader``,
``ModelValidator``, ``AnalysisResult``, and ``ModelAnalyzer`` —
``AnalysisResult``/``ModelAnalyzer`` (``model_analyzer.py``) are out of
scope for this file set and are not re-exported here.

``FrameworkDetector`` and ``LoaderFactory`` are internal helper classes
that back ``ModelLoader``'s locked behavior (framework detection and
adapter resolution, respectively — see ``01_Project_Architecture.md``
§5 steps 2–3) without being part of the locked class inventory
themselves (``09_Architecture_Lock.md`` §11, "Internal API"). They are
exported for direct unit testing but are not a stable public contract:
``ModelLoader`` is the only class other packages should depend on.

Per ``10_Module_Development_Guide.md`` §3 Dependencies and
``01_Project_Architecture.md`` §7's prohibited edges, nothing in this
package imports ``uaqe.infrastructure`` or any concrete
``IFrameworkAdapter`` implementation.
"""

from __future__ import annotations

from uaqe.model_loader.framework_detector import FrameworkDetector
from uaqe.model_loader.loader_factory import LoaderFactory
from uaqe.model_loader.model_loader import ModelLoader
from uaqe.model_loader.model_validator import ModelValidator

__all__ = [
    "ModelLoader",
    "ModelValidator",
    "FrameworkDetector",
    "LoaderFactory",
]
