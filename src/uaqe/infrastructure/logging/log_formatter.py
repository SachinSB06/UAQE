"""JSON-lines record formatting for the Logging module.

``LogFormatter`` is an internal collaborator of
:class:`uaqe.infrastructure.logging.structured_logger.StructuredLogger`.
It is **not** part of the locked ``uaqe.infrastructure.logging`` class
inventory (``09_Architecture_Lock.md`` §8, which names exactly
``StructuredLogger``) — it exists purely to separate "what a log
record looks like on the wire" (severity ordering, timestamping, JSON
serialization) from "where a log record is written and how sink
lifecycle/concurrency is managed", which remains ``StructuredLogger``'s
responsibility. It is never imported outside this package.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

#: Ordering of severity levels, lowest to highest, used to decide
#: whether a given log call is at or above a logger's configured
#: ``min_level``. Per ``01_Project_Architecture.md`` §13, these are the
#: only five levels, and their exact spelling is part of the locked
#: ``ILogger`` contract (``03_API_Specification.md`` §1.6).
LEVEL_ORDER: Dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


class LogFormatter:
    """Builds one JSON-lines record string per log call.

    Attributes:
        run_id: The run identifier attached to every formatted record.
    """

    def __init__(self, run_id: str) -> None:
        """Initialize a ``LogFormatter``.

        Args:
            run_id: The run identifier attached to every record this
                formatter builds.
        """
        self.run_id: str = run_id

    def format(self, level: str, msg: str, fields: Dict[str, Any]) -> str:
        """Build a single newline-terminated JSON-lines record.

        Args:
            level: The severity level name of this log call (one of
                ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
                ``CRITICAL``).
            msg: The human-readable log message.
            fields: Arbitrary structured context supplied by the
                caller (e.g. ``stage``, ``module``, ``metrics``),
                merged into the emitted record. Values that are not
                natively JSON-serializable are coerced via ``str()``.

        Returns:
            A single JSON object, followed by ``"\\n"``, ready to be
            written directly to a sink.
        """
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "level": level,
            "message": msg,
            **fields,
        }
        return json.dumps(record, default=str) + "\n"
