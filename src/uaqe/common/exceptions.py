"""Exception hierarchy for the Universal AI Quantization Engine.

Every exception raised across a public ``uaqe`` boundary is a subclass of
``UAQEError`` (``10_Module_Development_Guide.md`` §0 rule 6). This module
has no dependencies on any other ``uaqe`` package, per the layering rules
in ``09_Architecture_Lock.md`` §8.

Locked contract: ``03_API_Specification.md`` §1.4.
Locked class inventory: ``09_Architecture_Lock.md`` §3 (twelve named
subclasses of ``UAQEError``).
"""

from typing import Any, Dict, Optional


class UAQEError(Exception):
    """Base exception for every error raised within ``uaqe``.

    Carries structured context so that callers at any layer (CLI, REST,
    batch) can render a consistent, non-leaking error surface instead of
    a raw traceback, and so that ``ILogger`` implementations can log
    structured fields rather than a bare message.

    Attributes:
        code: A short, stable, machine-readable error code (e.g.
            ``"MODEL_LOAD_FAILED"``) suitable for programmatic handling
            and for correlating log lines with user-facing messages.
        message: A human-readable description of what went wrong.
        remediation_hint: An optional suggestion for how the caller might
            resolve or work around the error.
        stage: The name of the pipeline stage that raised the error, if
            the error originated from within ``PipelineStage.execute()``;
            ``None`` for errors raised outside the pipeline (e.g. during
            composition-root wiring).

    Supports standard Python exception chaining via ``raise ... from
    original_exception``; ``__cause__``/``__context__`` are populated by
    the interpreter as usual and are surfaced through
    :meth:`to_log_context`.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        remediation_hint: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Initialize a ``UAQEError``.

        Args:
            message: Human-readable description of the failure.
            code: Short, stable, machine-readable error code.
            remediation_hint: Optional suggestion for resolving the error.
            stage: Optional name of the pipeline stage that raised this
                error.
        """
        super().__init__(message)
        self.message: str = message
        self.code: str = code
        self.remediation_hint: Optional[str] = remediation_hint
        self.stage: Optional[str] = stage

    def to_log_context(self) -> Dict[str, Any]:
        """Return a structured field mapping suitable for ``ILogger``.

        This is an internal convenience method (not part of the locked
        public API surface) that lets callers pass this error directly
        as ``**fields`` to an ``ILogger`` call without manually
        re-extracting each attribute.

        Returns:
            A mapping of field name to value, including the exception's
            chained cause, if any.
        """
        context: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "remediation_hint": self.remediation_hint,
            "stage": self.stage,
        }
        if self.__cause__ is not None:
            context["caused_by"] = repr(self.__cause__)
        return context

    def __repr__(self) -> str:
        """Return an unambiguous representation including error code."""
        return (
            f"{self.__class__.__name__}(code={self.code!r}, "
            f"message={self.message!r}, stage={self.stage!r})"
        )


class ModelLoadError(UAQEError):
    """Raised when a model file cannot be loaded into the IMR.

    Typically raised by ``IFrameworkAdapter.load()`` or
    ``ModelLoader.detect_framework()`` when the source file is missing,
    corrupt, or in an unsupported framework format.
    """


class ModelValidationError(UAQEError):
    """Raised when a loaded ``IMR`` fails structural validation.

    Typically raised by ``ModelValidator.validate()`` for issues such as
    dangling edges, cycles, or invalid tensor shapes.
    """


class UnsupportedLayerError(UAQEError):
    """Raised when an ``IMR`` contains a layer type that cannot be
    processed by a downstream stage (e.g. an unsupported ``op_type``
    encountered during compatibility checking or quantization).
    """


class HardwareIncompatibilityError(UAQEError):
    """Raised when a model or plan is incompatible with the resolved
    ``HardwareProfile`` (e.g. exceeds memory, unsupported op on target).
    """


class QuantizationError(UAQEError):
    """Raised when quantization cannot be applied or completed.

    Typically raised by ``IQuantizationStrategy.apply()`` or
    ``QuantizationEngine`` during plan application.
    """


class CompressionError(UAQEError):
    """Raised when compression cannot be applied or completed.

    Typically raised by ``ICompressionStrategy.apply()`` or
    ``CompressionEngine`` during plan application.
    """


class OptimizationError(UAQEError):
    """Raised when an optimization pass (e.g. memory optimization,
    multi-objective search) cannot be completed successfully.
    """


class ExportError(UAQEError):
    """Raised when export to a deployment artifact fails.

    Typically raised by ``IExporterBackend.export()``.
    """


class EvaluationError(UAQEError):
    """Raised when model evaluation (e.g. accuracy scoring) cannot be
    completed successfully.
    """


class BenchmarkError(UAQEError):
    """Raised when hardware benchmarking cannot be completed
    successfully.
    """


class ConfigurationError(UAQEError):
    """Raised when configuration cannot be loaded, is missing required
    fields, or otherwise fails validation.

    Typically raised by ``IConfigRepository`` implementations and
    ``IHardwareProfileRepository.get()``.
    """


class PluginLoadError(UAQEError):
    """Raised when a plugin cannot be discovered, loaded, or registered.

    Typically raised by ``PluginRegistry.discover()`` and
    ``PluginRegistry.get_quantization_strategy()``.
    """
