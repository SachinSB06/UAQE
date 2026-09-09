"""PDF rendering of a run's :class:`~uaqe.report.report_result.ReportResult`.

Per ``07_Coding_Standards.md`` §14 rule 4, a new top-level dependency
requires a ``pyproject.toml`` update and stated justification, and per
``11_Implementation_Rules.md`` §2, third-party libraries are scoped to
the layer that specifically needs them. Rather than adding a PDF
library dependency for what this package needs (paginated, left-
aligned monospace text — no images, no vector graphics, no embedded
fonts beyond a standard PDF base-14 font), :class:`_MinimalPdfWriter`
below hand-assembles a minimal, spec-valid PDF 1.4 document directly:
one ``/Catalog``, one ``/Pages`` tree, one ``/Font`` (Helvetica, a
standard PDF base-14 font requiring no embedding), and one ``/Page`` +
content-stream object pair per page, with a correctly computed
cross-reference table. This keeps PDF export dependency-free while
still producing a document that opens in any standards-compliant PDF
reader.

The rendered text is the same section content
:mod:`uaqe.report.markdown_report` produces, stripped of Markdown
syntax and re-wrapped to fit the fixed-width page — see
:func:`_plain_text_lines`.
"""

from __future__ import annotations

import re
from typing import List

from uaqe.common.interfaces.i_report_renderer import IReportRenderer
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.reports.markdown_report import render_markdown
from uaqe.reports.report_document import ReportDocument
from uaqe.reports.report_result import ReportResult, build_report_result

#: This renderer's report-type identifier, returned by
#: :meth:`PdfReportRenderer.report_type`.
REPORT_TYPE = "pdf"

#: Matches a Markdown table separator row (e.g. ``|---|---|``), dropped
#: entirely from the plain-text rendering since it carries no content.
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|$")

#: Matches leading Markdown heading markers (``#``, ``##``, ...).
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

#: Matches inline Markdown emphasis/code markers this renderer strips
#: for plain-text output (bold, inline code, backtick spans).
_INLINE_MARKUP_RE = re.compile(r"(\*\*|`)")


def _plain_text_lines(markdown_text: str, *, wrap_width: int = 96) -> List[str]:
    """Convert ``markdown_text`` into plain, wrapped text lines.

    A lightweight, purpose-built conversion (heading markers become
    underlined-by-blank-line titles, table pipes become fixed-width
    columns, emphasis/code markers are stripped) rather than a general
    Markdown parser, since this function only ever needs to handle the
    fixed subset of Markdown
    :func:`~uaqe.report.markdown_report.render_markdown` itself
    produces.

    Args:
        markdown_text: The Markdown document to convert, as produced by
            :func:`~uaqe.report.markdown_report.render_markdown`.
        wrap_width: The maximum character width a non-table,
            non-heading line is wrapped to.

    Returns:
        Plain-text lines, ready for direct pagination by
        :class:`_MinimalPdfWriter`.
    """
    lines: List[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if _TABLE_SEPARATOR_RE.match(line):
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level, title = heading_match.groups()
            lines.append("")
            lines.append(title.strip())
            lines.append("-" * len(title.strip()) if len(level) <= 2 else "")
            continue

        line = _INLINE_MARKUP_RE.sub("", line)

        if line.startswith("- "):
            line = "  * " + line[2:]

        if not line:
            lines.append("")
            continue

        while len(line) > wrap_width:
            split_at = line.rfind(" ", 0, wrap_width)
            if split_at <= 0:
                split_at = wrap_width
            lines.append(line[:split_at])
            line = line[split_at:].lstrip()
        lines.append(line)

    return lines


def _pdf_escape(text: str) -> str:
    """Escape a string for use inside a PDF literal string ``(...)``.

    Args:
        text: The raw text to escape.

    Returns:
        ``text`` with backslashes and parentheses escaped, and any
        character outside the printable Latin-1 range replaced with
        ``?`` (the base-14 Helvetica font has no wider encoding, and
        this renderer never embeds a custom font per the module
        docstring).
    """
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return "".join(char if 32 <= ord(char) < 127 else "?" for char in escaped)


class _MinimalPdfWriter:
    """Assembles a minimal, dependency-free, spec-valid PDF document
    from plain left-aligned text lines.

    Attributes:
        _page_width: The page width, in PDF points (US Letter).
        _page_height: The page height, in PDF points (US Letter).
        _font_size: The Helvetica font size used for every line.
        _line_height: The vertical distance between successive lines.
        _left_margin: The left text margin, in points.
        _top_y: The baseline y-coordinate of the first line on a page.
        _bottom_margin: The minimum y-coordinate text may be placed at.
    """

    _page_width = 612.0
    _page_height = 792.0
    _font_size = 10
    _line_height = 13
    _left_margin = 48.0
    _top_y = 740.0
    _bottom_margin = 48.0

    def __init__(self) -> None:
        """Initialize an empty writer with no pages yet."""
        max_lines = int((self._top_y - self._bottom_margin) // self._line_height) + 1
        self._max_lines_per_page: int = max(max_lines, 1)
        self._pages: List[List[str]] = []

    def add_lines(self, lines: List[str]) -> None:
        """Paginate ``lines`` into one or more pages.

        Args:
            lines: The plain-text lines to add, in display order.
        """
        current_page: List[str] = []
        for line in lines:
            if len(current_page) >= self._max_lines_per_page:
                self._pages.append(current_page)
                current_page = []
            current_page.append(line)
        self._pages.append(current_page)

    def to_bytes(self) -> bytes:
        """Serialize all added pages into a complete PDF byte string.

        Returns:
            The complete PDF document, including header, body objects,
            cross-reference table, and trailer.
        """
        pages = self._pages or [[]]
        page_count = len(pages)

        # Object numbering: 1=Catalog, 2=Pages, 3=Font,
        # 4..4+N-1=Page objects, 4+N..4+2N-1=Content-stream objects.
        catalog_obj_num = 1
        pages_obj_num = 2
        font_obj_num = 3
        first_page_obj_num = 4
        first_content_obj_num = first_page_obj_num + page_count

        page_obj_nums = [first_page_obj_num + i for i in range(page_count)]
        content_obj_nums = [first_content_obj_num + i for i in range(page_count)]

        kids = " ".join(f"{num} 0 R" for num in page_obj_nums)

        objects: List[bytes] = [b""]  # index 0 unused; PDF objects are 1-indexed

        objects.append(
            f"<< /Type /Catalog /Pages {pages_obj_num} 0 R >>".encode("latin-1")
        )
        objects.append(
            (
                f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        )

        for page_obj_num, content_obj_num in zip(page_obj_nums, content_obj_nums):
            objects.append(
                (
                    f"<< /Type /Page /Parent {pages_obj_num} 0 R "
                    f"/MediaBox [0 0 {self._page_width:g} {self._page_height:g}] "
                    f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
                    f"/Contents {content_obj_num} 0 R >>"
                ).encode("latin-1")
            )

        for page_lines in pages:
            stream = self._build_content_stream(page_lines)
            objects.append(
                (
                    f"<< /Length {len(stream)} >>\nstream\n"
                ).encode("latin-1")
                + stream
                + b"\nendstream"
            )

        return self._assemble(objects, catalog_obj_num)

    def _build_content_stream(self, page_lines: List[str]) -> bytes:
        """Build one page's PDF content stream from its text lines.

        Args:
            page_lines: The lines to place on this page, top to bottom.

        Returns:
            The raw content-stream bytes (``BT ... ET`` block).
        """
        commands: List[str] = [
            "BT",
            f"/F1 {self._font_size} Tf",
            f"{self._line_height} TL",
            f"{self._left_margin:g} {self._top_y:g} Td",
        ]
        for index, line in enumerate(page_lines):
            escaped = _pdf_escape(line)
            if index == 0:
                commands.append(f"({escaped}) Tj")
            else:
                commands.append(f"T* ({escaped}) Tj")
        commands.append("ET")
        return "\n".join(commands).encode("latin-1")

    def _assemble(self, objects: List[bytes], catalog_obj_num: int) -> bytes:
        """Concatenate ``objects`` with a header, cross-reference table,
        and trailer into a complete PDF byte string.

        Args:
            objects: Object bodies, indexed by PDF object number
                (index ``0`` is an unused placeholder).
            catalog_obj_num: The object number of the document catalog,
                referenced by the trailer's ``/Root`` entry.

        Returns:
            The complete PDF document bytes.
        """
        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        buffer = bytearray(header)
        offsets = [0] * len(objects)

        for obj_num in range(1, len(objects)):
            offsets[obj_num] = len(buffer)
            buffer += f"{obj_num} 0 obj\n".encode("latin-1")
            buffer += objects[obj_num]
            buffer += b"\nendobj\n"

        xref_offset = len(buffer)
        object_count = len(objects)
        buffer += f"xref\n0 {object_count}\n".encode("latin-1")
        buffer += b"0000000000 65535 f \n"
        for obj_num in range(1, object_count):
            buffer += f"{offsets[obj_num]:010d} 00000 n \n".encode("latin-1")

        buffer += (
            f"trailer\n<< /Size {object_count} /Root {catalog_obj_num} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("latin-1")

        return bytes(buffer)


def render_pdf(result: ReportResult) -> bytes:
    """Render ``result`` into a complete PDF document.

    Args:
        result: The run summary to render.

    Returns:
        The rendered PDF, as raw bytes.
    """
    markdown_text = render_markdown(result)
    plain_lines = _plain_text_lines(markdown_text)
    writer = _MinimalPdfWriter()
    writer.add_lines(plain_lines)
    return writer.to_bytes()


class PdfReportRenderer(IReportRenderer):
    """Renders a run's ``PipelineContext`` into a PDF
    :class:`~uaqe.report.report_document.ReportDocument`.

    Formally implements ``IReportRenderer`` — see
    :mod:`uaqe.report.report_document`'s module docstring. Unlike
    every other renderer in this package, the rendered payload is
    binary: it is carried on ``ReportDocument.binary_content`` rather
    than ``ReportDocument.content`` (see that dataclass's docstring for
    why).
    """

    def render(self, context: PipelineContext) -> ReportDocument:
        """Render ``context`` into a PDF ``ReportDocument``.

        Args:
            context: The pipeline context of the run being reported on.

        Returns:
            The rendered ``ReportDocument``, with ``binary_content``
            set to the full PDF document, ``content`` left as ``\"\"``,
            and ``file_path`` set to ``reports/<run_id>/report.pdf``.
        """
        result = build_report_result(context)
        pdf_bytes = render_pdf(result)
        return ReportDocument(
            report_type=REPORT_TYPE,
            content="",
            binary_content=pdf_bytes,
            file_path=f"{result.run_id}/report.pdf",
            mime_type="application/pdf",
            run_id=result.run_id,
            warnings=list(result.warnings),
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
