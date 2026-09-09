"""Logging port consumed by every module in the Universal AI
Quantization Engine.

Per ``10_Module_Development_Guide.md`` §0 rule 5, every module logs
through ``ILogger``; no module calls ``print()`` or the stdlib
``logging`` module directly. ``uaqe.infrastructure.logging.StructuredLogger``
is the sole first-party implementation, injected via ``CompositionRoot``.

Locked contract: ``03_API_Specification.md`` §1.6.
"""

from abc import ABC, abstractmethod


class ILogger(ABC):
    """Abstract structured-logging port.

    Every method accepts a free-text message plus arbitrary structured
    fields, so implementations can emit machine-parseable log lines
    (e.g. JSON) rather than pre-formatted strings.
    """

    @abstractmethod
    def debug(self, msg: str, **fields) -> None:
        """Log a diagnostic message useful only during development."""
        raise NotImplementedError

    @abstractmethod
    def info(self, msg: str, **fields) -> None:
        """Log a routine, expected event."""
        raise NotImplementedError

    @abstractmethod
    def warning(self, msg: str, **fields) -> None:
        """Log a non-fatal condition that may warrant attention."""
        raise NotImplementedError

    @abstractmethod
    def error(self, msg: str, **fields) -> None:
        """Log a failure that halted the current operation."""
        raise NotImplementedError

    @abstractmethod
    def critical(self, msg: str, **fields) -> None:
        """Log a failure severe enough to halt the entire pipeline run."""
        raise NotImplementedError
