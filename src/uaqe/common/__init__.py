"""``uaqe.common`` — framework-agnostic primitives shared across the
Universal AI Quantization Engine.

This package defines no business logic and performs no I/O
(``10_Module_Development_Guide.md`` §1 "Purpose"). It contains:

- The Internal Model Representation (``imr.py``).
- Shared enumerations (``types.py``).
- Immutable configuration value objects (``value_objects.py``).
- The ``UAQEError`` exception hierarchy (``exceptions.py``).
- Result/report dataclasses (``result_types.py``).

Per ``09_Architecture_Lock.md`` §8, ``uaqe.common`` has no dependency on
any other ``uaqe`` package, so this ``__init__.py`` only imports from its
own sibling modules — never from ``uaqe.domain``, ``uaqe.application``,
``uaqe.infrastructure``, or ``uaqe.interface``.

The re-exports below are exactly the ``uaqe.common`` class inventory
locked in ``09_Architecture_Lock.md`` §3. No name here may be renamed,
and no additional name may be added, without an RFC per §14 of that
document.

Note: the ``interfaces/`` sub-package (``ILogger``, ``IFrameworkAdapter``,
``IQuantizationStrategy``, ``ICompressionStrategy``, ``IExporterBackend``,
``IReportRenderer``, ``IHardwareProfileRepository``, ``IConfigRepository``)
is also part of the locked ``uaqe.common`` inventory but is out of scope
for this file, as its source modules were not provided; it is exported
via its own ``interfaces/__init__.py`` and is not re-imported here to
avoid depending on modules that do not yet exist on disk.
"""

from uaqe.common.exceptions import (
    BenchmarkError,
    CompressionError,
    ConfigurationError,
    EvaluationError,
    ExportError,
    HardwareIncompatibilityError,
    ModelLoadError,
    ModelValidationError,
    OptimizationError,
    PluginLoadError,
    QuantizationError,
    UAQEError,
    UnsupportedLayerError,
)
from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.result_types import CompatibilityReport, RunResult, StageResult
from uaqe.common.types import (
    CompressionType,
    ExportFormat,
    HardwareClass,
    PipelineErrorPolicy,
    Precision,
)
from uaqe.common.value_objects import (
    CompressionConfig,
    ExecutionConfig,
    HardwareConfig,
    OptimizationConfig,
    QuantizationConfig,
)

__all__ = [
    # imr.py
    "IMR",
    "IMRLayer",
    "IMRMetadata",
    "IMRTensor",
    # types.py
    "Precision",
    "HardwareClass",
    "CompressionType",
    "ExportFormat",
    "PipelineErrorPolicy",
    # value_objects.py
    "QuantizationConfig",
    "CompressionConfig",
    "HardwareConfig",
    "OptimizationConfig",
    "ExecutionConfig",
    # exceptions.py
    "UAQEError",
    "ModelLoadError",
    "ModelValidationError",
    "UnsupportedLayerError",
    "HardwareIncompatibilityError",
    "QuantizationError",
    "CompressionError",
    "OptimizationError",
    "ExportError",
    "EvaluationError",
    "BenchmarkError",
    "ConfigurationError",
    "PluginLoadError",
    # result_types.py
    "StageResult",
    "RunResult",
    "CompatibilityReport",
]
