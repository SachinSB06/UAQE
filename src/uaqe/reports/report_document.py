"""The rendered-report value object every renderer in this package
produces.

Per ``03_API_Specification.md`` §13.1, the locked minimal shape is
exactly ``report_type: str``, ``content: str``, ``file_path: str`` —
the return type of ``IReportRenderer.render()``
(``uaqe.common.interfaces.i_report_renderer``). Every other package's
own ``*_report.py`` module (``evaluation_report.py``,
``benchmark_report.py``, ``compression_report.py``,
``optimization_report.py``, ``quantization_report.py``,
``export_report.py``) has so far deliberately stayed only
*structurally* compatible with ``IReportRenderer`` rather than
formally implementing it, each noting in its own module docstring that
"the locked ``ReportDocument`` type is not yet implemented anywhere in
this codebase." This module is that implementation, and
:mod:`uaqe.report.markdown_report`, :mod:`uaqe.report.html_report`,
:mod:`uaqe.report.json_report`, :mod:`uaqe.report.csv_report`, and
:mod:`uaqe.report.pdf_report` are the first renderers in the codebase
to formally subclass ``IReportRenderer``.

Note on location: ``03_API_Specification.md`` §13.1 and
``09_Architecture_Lock.md`` §2 place this type at
``uaqe.domain.reporting.report_generator.ReportDocument``. This
codebase's actual package layout already departs from that nested
``uaqe.domain.*``/``uaqe.application``/``uaqe.infrastructure`` tree
throughout (e.g. ``HardwareManager`` lives at
``uaqe.domain.hardware_manager``, not
``uaqe.domain.hardware.hardware_manager``; every engine/planner/report
package — ``evaluation``, ``benchmark``, ``compression``,
``quantization``, ``optimizer``, ``exporter``, ``analyzer`` — is a
top-level ``uaqe.*`` package rather than nested under
``uaqe.domain.*``). This module follows that same, already-established
flat layout: ``ReportDocument`` lives at
``uaqe.report.report_document.ReportDocument``. This is a location
deviation only — the field shape, semantics, and the
``IReportRenderer.render() -> ReportDocument`` contract are unchanged
from the locked specification.

``content`` is defined as ``str`` in the locked contract. Text-format
renderers (Markdown, HTML, JSON, CSV) populate it directly. The PDF
renderer produces binary output, which cannot be represented as
``str`` without a lossy encoding; per the same "additive field, zero
modification to existing names" precedent every other package in this
codebase already uses for its own locked-type extensions
(``09_Architecture_Lock.md`` §0 rule 2), this module adds
``binary_content: Optional[bytes]`` for that one case. ``content``
stays ``\"\"`` whenever ``binary_content`` is populated, and vice
versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ReportDocument:
    """One renderer's complete output for a single run.

    Attributes:
        report_type: This document's format/kind identifier, matching
            the producing renderer's ``report_type()`` return value
            (e.g. ``\"markdown\"``, ``\"html\"``, ``\"json\"``,
            ``\"csv\"``, ``\"pdf\"``, ``\"summary\"``, ``\"comparison\"``)
            — the locked field (``03_API_Specification.md`` §13.1).
        content: The rendered textual content, for every text-format
            renderer — the locked field. ``\"\"`` for the binary PDF
            renderer; see the module docstring.
        file_path: The path this document was (or should be) written
            to under ``reports/<run_id>/`` — the locked field. ``\"\"``
            until a writer (e.g. :class:`~uaqe.report.report_generator.
            ReportGenerator`) actually persists it.
        binary_content: The rendered binary content, populated only by
            the PDF renderer. ``None`` for every text-format renderer.
        mime_type: The MIME type ``content``/``binary_content``
            represents (e.g. ``\"text/markdown\"``,
            ``\"application/pdf\"``), for callers that write this
            document via an HTTP response or need to pick a file
            extension without inspecting ``report_type`` themselves.
        run_id: The run this document reports on.
        warnings: Non-fatal warnings accumulated while rendering this
            document (e.g. an upstream stage that had not run yet).
    """

    report_type: str
    content: str = ""
    file_path: str = ""
    binary_content: Optional[bytes] = None
    mime_type: str = "text/plain"
    run_id: str = ""
    warnings: List[str] = field(default_factory=list)

    def is_binary(self) -> bool:
        """Report whether this document's payload lives in
        ``binary_content`` rather than ``content``.

        Returns:
            ``True`` if ``binary_content`` is populated.
        """
        return self.binary_content is not None

    def size_bytes(self) -> int:
        """Return this document's rendered payload size, in bytes.

        Returns:
            ``len(binary_content)`` when :meth:`is_binary` is ``True``,
            else the UTF-8 encoded byte length of ``content``.
        """
        if self.binary_content is not None:
            return len(self.binary_content)
        return len(self.content.encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        """Return this document's metadata as a plain, JSON-serializable
        ``dict``.

        ``binary_content`` is never included (it is not JSON-safe and
        would bloat any manifest built from this method); use
        :attr:`binary_content` directly when the raw bytes are needed.

        Returns:
            This document's metadata, plus ``content`` only when this
            document is not binary.
        """
        payload: Dict[str, Any] = {
            "report_type": self.report_type,
            "file_path": self.file_path,
            "mime_type": self.mime_type,
            "run_id": self.run_id,
            "is_binary": self.is_binary(),
            "size_bytes": self.size_bytes(),
            "warnings": list(self.warnings),
        }
        if not self.is_binary():
            payload["content"] = self.content
        return payload
