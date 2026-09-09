"""``LoggerFactory`` — resolves and validates ``settings.yaml``'s
``logging`` section into the exact constructor arguments
``StructuredLogger`` takes, without ever instantiating it.

.. important::
    **This class is not part of the locked architecture.**
    ``09_Architecture_Lock.md`` §8's module ownership table for
    ``uaqe.infrastructure.logging`` names exactly ``StructuredLogger``.
    This file was generated on explicit request despite that gap; it
    has not been reconciled via the RFC process in §14, and — like
    ``PluginRepository``/``ModelRepository`` — should be formalized or
    removed before being wired into a real ``CompositionRoot``.

    An earlier version of this class also called
    ``StructuredLogger(...)`` directly, which conflicted with
    ``11_Implementation_Rules.md`` §3.4: "No class instantiates
    ``StructuredLogger`` directly except ``CompositionRoot``." That
    conflict is why this class exists in its current, narrower form:
    it now does config resolution and validation only —
    :meth:`build_logger_params` returns a plain, inert
    :class:`StructuredLoggerParams` value, never a live ``ILogger``.
    ``CompositionRoot`` remains the sole call site that writes
    ``StructuredLogger(**dataclasses.asdict(params))`` (or the
    field-by-field equivalent), so §3.4 holds without any reinterpretation
    of "instantiates ... directly" being required. Neither
    ``StructuredLogger`` nor ``ILogger`` is imported or modified here.

Per ``06_Config_Spec.md`` §2, ``settings.yaml``'s ``logging`` section
(``min_level``, ``sink``) and ``config.json``'s ``logs_path`` are the
two config sources this factory reads. Per ``02_Folder_Structure.md``
§13, a file sink is written under ``logs/<run_id>/``.

Severity-level validation reuses
:data:`uaqe.infrastructure.logging.log_formatter.LEVEL_ORDER` — the
same locked five-level set ``StructuredLogger`` itself validates
against — rather than duplicating that list, since ``LogFormatter``
is a plain data/formatting module with no instantiation restriction
of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from uaqe.infrastructure.logging.log_formatter import LEVEL_ORDER

#: Recognized values of ``settings.yaml``'s ``logging.sink`` field
#: (``06_Config_Spec.md`` §2).
_SINK_FILE = "file"
_SINK_STDOUT = "stdout"
_SINK_BOTH = "both"
_VALID_SINKS = frozenset({_SINK_FILE, _SINK_STDOUT, _SINK_BOTH})

#: The literal ``StructuredLogger`` recognizes as "write to the
#: console" (see that module's ``_STDOUT_SINK``).
_STDOUT_SINK_PATH = "stdout"

#: The log file name written under ``logs/<run_id>/``
#: (``02_Folder_Structure.md`` §13).
_LOG_FILE_NAME = "run.log"


@dataclass(frozen=True)
class StructuredLoggerParams:
    """The exact, validated keyword arguments ``StructuredLogger.__init__`` takes.

    Deliberately field-for-field identical to ``StructuredLogger``'s
    locked constructor (``03_API_Specification.md`` §18.1), so
    ``CompositionRoot`` can construct one with
    ``StructuredLogger(**dataclasses.asdict(params))``. This dataclass
    holds no behavior and creates no logger itself.

    Attributes:
        run_id: The run identifier to attach to every emitted log line.
        min_level: The validated minimum severity level name.
        sink_paths: The validated, resolved list of sink destinations.
    """

    run_id: str
    min_level: str
    sink_paths: List[str] = field(default_factory=list)


class LoggerFactory:
    """Resolves and validates logging config; never constructs a logger.

    Attributes:
        logs_path: The root directory under which one ``<run_id>/``
            subdirectory is addressed per run that logs to a file
            (``config.json``'s ``logs_path``).
    """

    def __init__(self, logs_path: str) -> None:
        """Initialize a ``LoggerFactory`` rooted at ``logs_path``.

        Args:
            logs_path: The root log directory (``config.json``'s
                ``logs_path``), under which this factory addresses one
                ``<run_id>/`` subdirectory per run that logs to a
                file. This factory never creates the directory itself
                — ``StructuredLogger`` owns file-sink creation
                (``uaqe.infrastructure.logging.structured_logger``).
        """
        self.logs_path: str = logs_path

    def validate_min_level(self, min_level: str) -> str:
        """Validate a ``logging.min_level`` value and normalize its case.

        Pure — reads no instance state, mutates nothing. Safe to call
        from any number of threads concurrently.

        Args:
            min_level: The minimum severity level, by name (one of
                ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
                ``CRITICAL``, case-insensitive) — ``settings.yaml``'s
                ``logging.min_level``.

        Returns:
            ``min_level`` upper-cased, once confirmed to be one of the
            five recognized severity level names.

        Raises:
            ValueError: If ``min_level`` is not one of the five
                recognized severity level names.
        """
        normalized_min_level = min_level.upper()
        if normalized_min_level not in LEVEL_ORDER:
            raise ValueError(
                f"Unrecognized min_level {min_level!r}; expected one of "
                f"{sorted(LEVEL_ORDER, key=lambda name: LEVEL_ORDER[name])}."
            )
        return normalized_min_level

    def validate_sink(self, sink: str) -> str:
        """Validate a ``logging.sink`` value.

        Pure — reads no instance state, mutates nothing. Safe to call
        from any number of threads concurrently.

        Args:
            sink: One of ``"file"``, ``"stdout"``, or ``"both"`` —
                ``settings.yaml``'s ``logging.sink``.

        Returns:
            ``sink`` unchanged, once confirmed to be one of the three
            recognized values.

        Raises:
            ValueError: If ``sink`` is not one of the three recognized
                values.
        """
        if sink not in _VALID_SINKS:
            raise ValueError(
                f"Unrecognized logging sink {sink!r}; expected one of "
                f"{sorted(_VALID_SINKS)}."
            )
        return sink

    def resolve_sink_paths(self, run_id: str, sink: str) -> List[str]:
        """Resolve ``logging.sink`` into the ``sink_paths`` list ``StructuredLogger`` expects.

        Depends only on :attr:`logs_path` (read, never written after
        ``__init__``) and its arguments — reads no other instance
        state and mutates nothing. Safe to call from any number of
        threads concurrently.

        Args:
            run_id: The run identifier, used to build the per-run log
                file path when ``sink`` writes to a file.
            sink: One of ``"file"``, ``"stdout"``, or ``"both"``.

        Returns:
            ``["stdout"]`` for ``"stdout"``; ``["<logs_path>/<run_id>/run.log"]``
            for ``"file"``; both, in that order, for ``"both"``.

        Raises:
            ValueError: If ``sink`` is not one of the three recognized
                values.
        """
        self.validate_sink(sink)

        file_path = str(Path(self.logs_path) / run_id / _LOG_FILE_NAME)

        if sink == _SINK_STDOUT:
            return [_STDOUT_SINK_PATH]
        if sink == _SINK_FILE:
            return [file_path]
        return [_STDOUT_SINK_PATH, file_path]

    def build_logger_params(
        self, run_id: str, min_level: str, sink: str
    ) -> StructuredLoggerParams:
        """Validate and normalize logging config into ``StructuredLogger`` kwargs.

        This is the one method ``CompositionRoot`` is expected to call.
        It never constructs a logger and never imports
        ``StructuredLogger`` — it returns an inert
        ``StructuredLoggerParams``, which ``CompositionRoot`` then
        unpacks into ``StructuredLogger(...)`` itself. Composes
        :meth:`validate_min_level` and :meth:`resolve_sink_paths`
        (which itself calls :meth:`validate_sink`) — reads no instance
        state beyond :attr:`logs_path` and mutates nothing, so it is
        safe to call from any number of threads concurrently, e.g. to
        prepare params for several concurrent runs.

        Args:
            run_id: The run identifier to attach to every emitted log
                line.
            min_level: The minimum severity level, by name — see
                :meth:`validate_min_level`.
            sink: One of ``"file"``, ``"stdout"``, or ``"both"`` — see
                :meth:`validate_sink`.

        Returns:
            A ``StructuredLoggerParams`` ready to be unpacked directly
            into ``StructuredLogger(...)`` by ``CompositionRoot``.

        Raises:
            ValueError: If ``sink`` is not one of the three recognized
                values, or ``min_level`` is not one of the five
                recognized severity level names.
        """
        normalized_min_level = self.validate_min_level(min_level)
        sink_paths = self.resolve_sink_paths(run_id, sink)

        return StructuredLoggerParams(
            run_id=run_id,
            min_level=normalized_min_level,
            sink_paths=sink_paths,
        )

    def prepare_constructor_args(
        self, run_id: str, min_level: str, sink: str
    ) -> StructuredLoggerParams:
        """Alias for :meth:`build_logger_params`, kept for call-site compatibility.

        Args:
            run_id: See :meth:`build_logger_params`.
            min_level: See :meth:`build_logger_params`.
            sink: See :meth:`build_logger_params`.

        Returns:
            See :meth:`build_logger_params`.

        Raises:
            ValueError: See :meth:`build_logger_params`.
        """
        return self.build_logger_params(run_id, min_level, sink)
