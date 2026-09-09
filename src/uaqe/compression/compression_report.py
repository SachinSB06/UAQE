"""Human- and machine-readable reporting for a completed compression
run.

``CompressionReportRenderer`` summarizes a
:class:`~uaqe.compression.compression_planner.CompressionPlan` (plus
the ``IMR`` it was applied to and the ``IMR`` it produced) into a
:class:`CompressionReportDocument` — a Markdown- and dict-renderable
summary suitable for ``uaqe.domain.reporting.report_generator.
ReportGenerator`` to fold into a run's overall report bundle, mirroring
:class:`~uaqe.quantization.quantization_report.
QuantizationReportRenderer`'s role for the quantization package.

Per-layer size comparison (rather than a technique-specific metric
like "sparsity", which is meaningless for e.g. Huffman-only runs) is
used throughout so this renderer stays correct regardless of which
``CompressionType`` combination a plan selected: every
``ICompressionStrategy`` in this package returns an ``IMR`` whose
parameter tensors' ``data`` reflects that technique's actual storage
footprint, so comparing raw byte lengths before and after is a
technique-agnostic way to report the achieved reduction.

This renderer is deliberately *structurally* compatible with
``IReportRenderer`` (``render(context) -> ReportDocument``,
``report_type() -> str``) rather than formally inheriting from it, for
the same reason ``QuantizationReportRenderer`` does not: the locked
``ReportDocument`` type is not yet implemented anywhere in this
codebase, and this package should not take a hard import dependency on
a module that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from uaqe.common.imr import IMR
from uaqe.compression.compression_planner import CompressionPlan

#: This renderer's report-type identifier, returned by
#: :meth:`CompressionReportRenderer.report_type`.
REPORT_TYPE = "compression"


@dataclass(frozen=True)
class CompressionReportDocument:
    """A rendered summary of one run's compression outcome.

    Attributes:
        selected_types: The ``CompressionType.value`` strings applied,
            in application order.
        strategy_names: The ``ICompressionStrategy.name()`` resolved
            for every selected type, keyed by that type's ``.value``.
        rationale: The human-readable rationale for every type's
            resolved strategy, carried through from ``CompressionPlan``.
        target_ratio: The compression ratio this plan was built toward.
        original_size_bytes: The total parameter-tensor byte size of
            the ``IMR`` before compression.
        compressed_size_bytes: The total parameter-tensor byte size of
            the ``IMR`` after compression.
        achieved_ratio: ``compressed_size_bytes / original_size_bytes``
            (``1.0`` if ``original_size_bytes`` is ``0``, to avoid a
            division by zero for a parameter-free model).
        per_layer_size_bytes: The ``(original, compressed)`` byte-size
            pair for every layer, keyed by layer name.
        warnings: Every non-fatal warning accumulated while building
            this report.
    """

    selected_types: List[str] = field(default_factory=list)
    strategy_names: Dict[str, str] = field(default_factory=dict)
    rationale: str = ""
    target_ratio: float = 1.0
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    achieved_ratio: float = 1.0
    per_layer_size_bytes: Dict[str, "tuple[int, int]"] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this document as a plain, JSON-serializable ``dict``."""
        return {
            "selected_types": list(self.selected_types),
            "strategy_names": dict(self.strategy_names),
            "rationale": self.rationale,
            "target_ratio": self.target_ratio,
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "achieved_ratio": self.achieved_ratio,
            "per_layer_size_bytes": {
                name: list(sizes) for name, sizes in self.per_layer_size_bytes.items()
            },
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """Render this document as a Markdown summary.

        Returns:
            A Markdown string: an overall-ratio summary followed by a
            per-layer size table and any warnings.
        """
        lines: List[str] = [
            "# Compression Report",
            "",
            f"- **Applied types:** {', '.join(self.selected_types) or 'none'}",
            f"- **Target ratio:** {self.target_ratio:.4f}",
            f"- **Achieved ratio:** {self.achieved_ratio:.4f}",
            f"- **Original size:** {self.original_size_bytes:,} bytes",
            f"- **Compressed size:** {self.compressed_size_bytes:,} bytes",
            f"- **Rationale:** {self.rationale}",
            "",
            "## Per-Layer Size",
            "",
            "| Layer | Original bytes | Compressed bytes | Ratio |",
            "|---|---|---|---|",
        ]
        for layer_name in sorted(self.per_layer_size_bytes):
            original, compressed = self.per_layer_size_bytes[layer_name]
            ratio = (compressed / original) if original else 1.0
            lines.append(
                f"| {layer_name} | {original:,} | {compressed:,} | {ratio:.4f} |"
            )

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines)


class CompressionReportRenderer:
    """Builds a :class:`CompressionReportDocument` from a completed
    compression run's artifacts.

    Structurally compatible with ``IReportRenderer`` — see the module
    docstring for why this class does not formally inherit it yet.
    """

    def render(
        self, plan: CompressionPlan, original_imr: IMR, compressed_imr: IMR
    ) -> CompressionReportDocument:
        """Render a completed compression plan into a report document.

        Args:
            plan: The finalized
                :class:`~uaqe.compression.compression_planner.
                CompressionPlan` to summarize.
            original_imr: The ``IMR`` as it stood immediately before
                :meth:`~uaqe.compression.compression_planner.
                CompressionPlanner.apply` ran.
            compressed_imr: The ``IMR`` produced by that same call.

        Returns:
            The rendered :class:`CompressionReportDocument`.
        """
        original_by_layer = _parameter_bytes_by_layer(original_imr)
        compressed_by_layer = _parameter_bytes_by_layer(compressed_imr)

        per_layer_size_bytes: Dict[str, "tuple[int, int]"] = {}
        warnings: List[str] = []
        for layer_name, original_size in original_by_layer.items():
            compressed_size = compressed_by_layer.get(layer_name)
            if compressed_size is None:
                warnings.append(
                    f"Layer {layer_name!r} present before compression but "
                    f"missing afterward; excluded from size totals."
                )
                continue
            per_layer_size_bytes[layer_name] = (original_size, compressed_size)

        original_total = sum(size for size, _ in per_layer_size_bytes.values())
        compressed_total = sum(size for _, size in per_layer_size_bytes.values())
        achieved_ratio = (
            (compressed_total / original_total) if original_total else 1.0
        )

        return CompressionReportDocument(
            selected_types=[t.value for t in plan.selected_types],
            strategy_names={
                t.value: name for t, name in plan.strategy_names.items()
            },
            rationale=plan.rationale,
            target_ratio=plan.target_ratio,
            original_size_bytes=original_total,
            compressed_size_bytes=compressed_total,
            achieved_ratio=achieved_ratio,
            per_layer_size_bytes=per_layer_size_bytes,
            warnings=warnings,
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE


def _parameter_bytes_by_layer(imr: IMR) -> Dict[str, int]:
    """Sum every parameter tensor's raw byte length for each layer.

    Args:
        imr: The ``IMR`` to measure.

    Returns:
        The total ``len(tensor.data)`` across all parameters, keyed by
        layer name.
    """
    return {
        layer.name: sum(len(tensor.data) for tensor in layer.parameters.values())
        for layer in imr.layers
    }
