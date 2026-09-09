"""Structured, JSON-lines implementation of ``ILogger``.

This module provides the single concrete logging sink for the Universal
AI Quantization Engine. Per ``01_Project_Architecture.md`` §13, every
``PipelineStage`` (and every other injected consumer of ``ILogger``)
logs exclusively through this interface — never via bare ``print()`` or
the stdlib ``logging`` module directly (``07_Coding_Standards.md`` §6).

Only ``uaqe.interface.composition_root.CompositionRoot`` is permitted to
instantiate this class directly (``11_Implementation_Rules.md`` §3.4);
every other consumer receives an ``ILogger`` via constructor injection.

JSON-lines record construction (severity ordering, timestamping,
serialization) is delegated to
:class:`~uaqe.infrastructure.logging.log_formatter.LogFormatter`, an
internal collaborator colocated in this package and never used outside
this class; see its module docstring for why it exists as a separate
module. This class retains sole responsibility for sink lifecycle
(opening/closing file handles), severity gating against ``min_level``,
and write concurrency.

Locked contract: ``03_API_Specification.md`` §18.1.
"""

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, TextIO

from uaqe.common.interfaces.i_logger import ILogger
from uaqe.infrastructure.logging.log_formatter import LEVEL_ORDER, LogFormatter

#: Sentinel sink path recognized as "write to the console" rather than to
#: a log file on disk, per ``06_Config_Spec.md`` §2 (``logging.sink``:
#: ``file`` | ``stdout`` | ``both``). ``CompositionRoot`` resolves that
#: enum into a concrete ``sink_paths`` list before construction, using
#: this literal to represent the ``stdout``/``both`` portion.
_STDOUT_SINK: str = "stdout"


class StructuredLogger(ILogger):
    """Structured, JSON-lines ``ILogger`` implementation.

    Writes one JSON object per log call to every configured sink
    (console and/or a per-run log file under ``logs/<run_id>/``), per
    ``02_Folder_Structure.md`` §13 and ``01_Project_Architecture.md``
    §13. Every emitted line carries ``timestamp``, ``run_id``, ``level``,
    and ``message``, plus whatever structured ``**fields`` the caller
    supplies (e.g. ``stage``, ``module``, ``metrics``), per
    ``07_Coding_Standards.md`` §6.

    File sinks are opened in line-buffered append mode and all writes
    are guarded by an internal lock, so that concurrent stage executions
    (``01_Project_Architecture.md`` §16 "Threading Strategy") sharing a
    single ``StructuredLogger`` instance never interleave partial JSON
    lines within or across sinks (``11_Implementation_Rules.md`` §11.3).

    Attributes:
        run_id: The run identifier attached to every emitted log line.
        min_level: The minimum severity level, by name, that this logger
            will emit; log calls below this level are silently dropped.
        sink_paths: The destinations this logger writes to. Each entry
            is either the literal ``"stdout"`` (console output) or a
            filesystem path to a JSON-lines log file.
    """

    def __init__(self, run_id: str, min_level: str, sink_paths: List[str]) -> None:
        """Initialize a ``StructuredLogger``.

        Args:
            run_id: The run identifier attached to every emitted log
                line.
            min_level: The minimum severity level, by name (one of
                ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
                ``CRITICAL``), that this logger will emit.
            sink_paths: The destinations to write to. Each entry is
                either the literal ``"stdout"`` or a filesystem path;
                file sink parent directories are created if they do not
                already exist.

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

        self.run_id: str = run_id
        self.min_level: str = normalized_min_level
        self.sink_paths: List[str] = list(sink_paths)

        self._formatter = LogFormatter(run_id)
        self._lock = threading.Lock()
        self._file_handles: Dict[str, TextIO] = {}
        for sink_path in self.sink_paths:
            if sink_path == _STDOUT_SINK:
                continue
            resolved_path = Path(sink_path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered (buffering=1) text-mode append: each write()
            # call below emits exactly one JSON-lines record, so
            # line-buffering flushes each record immediately without
            # requiring an explicit flush() after every call.
            self._file_handles[sink_path] = resolved_path.open(
                mode="a", encoding="utf-8", buffering=1
            )

    def debug(self, msg: str, **fields: Any) -> None:
        """Emit a ``DEBUG``-level structured log line.

        Args:
            msg: The human-readable log message.
            **fields: Arbitrary structured context (e.g. ``stage``,
                ``module``, ``metrics``) merged into the emitted record.
        """
        self._emit("DEBUG", msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        """Emit an ``INFO``-level structured log line.

        Args:
            msg: The human-readable log message.
            **fields: Arbitrary structured context (e.g. ``stage``,
                ``module``, ``metrics``) merged into the emitted record.
        """
        self._emit("INFO", msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        """Emit a ``WARNING``-level structured log line.

        Args:
            msg: The human-readable log message.
            **fields: Arbitrary structured context (e.g. ``stage``,
                ``module``, ``metrics``) merged into the emitted record.
        """
        self._emit("WARNING", msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        """Emit an ``ERROR``-level structured log line.

        Args:
            msg: The human-readable log message.
            **fields: Arbitrary structured context (e.g. ``stage``,
                ``module``, ``metrics``) merged into the emitted record.
        """
        self._emit("ERROR", msg, fields)

    def critical(self, msg: str, **fields: Any) -> None:
        """Emit a ``CRITICAL``-level structured log line.

        Args:
            msg: The human-readable log message.
            **fields: Arbitrary structured context (e.g. ``stage``,
                ``module``, ``metrics``) merged into the emitted record.
        """
        self._emit("CRITICAL", msg, fields)

    def close(self) -> None:
        """Flush and close every open file sink.

        Idempotent: safe to call more than once. Intended to be invoked
        by ``CompositionRoot`` (or a run-scoped context manager built on
        top of it) once a run has finished emitting log lines. Console
        sinks require no cleanup and are unaffected.
        """
        with self._lock:
            for file_handle in self._file_handles.values():
                if not file_handle.closed:
                    file_handle.close()

    def _emit(self, level: str, msg: str, fields: Dict[str, Any]) -> None:
        """Build and write one JSON-lines record to every configured sink.

        Args:
            level: The severity level name of this log call.
            msg: The human-readable log message.
            fields: Arbitrary structured context supplied by the caller.

        Records below ``self.min_level`` are dropped without writing to
        any sink.
        """
        if LEVEL_ORDER[level] < LEVEL_ORDER[self.min_level]:
            return

        line = self._formatter.format(level, msg, fields)

        with self._lock:
            for sink_path in self.sink_paths:
                if sink_path == _STDOUT_SINK:
                    stream: TextIO = sys.stdout
                    stream.write(line)
                    stream.flush()
                else:
                    self._file_handles[sink_path].write(line)
