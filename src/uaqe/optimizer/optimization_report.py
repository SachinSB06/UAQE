"""Human- and machine-readable reporting for a completed optimization
run.

``OptimizationReportRenderer`` summarizes an
:class:`~uaqe.optimizer.optimization_result.OptimizationResult` (plus
the optional final
:class:`~uaqe.hardware.memory_planner.MemoryPlan` from
:class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer`) into an
:class:`OptimizationReportDocument` — a Markdown- and dict-renderable
summary suitable for ``uaqe.domain.reporting.report_generator.
ReportGenerator`` to fold into a run's overall report bundle, mirroring
:class:`~uaqe.compression.compression_report.CompressionReportRenderer`'s
and
:class:`~uaqe.quantization.quantization_report.
QuantizationReportRenderer`'s role for their own packages.

Like those two renderers, this class is deliberately *structurally*
compatible with ``IReportRenderer`` (``render(...) -> ReportDocument``,
``report_type() -> str``) rather than formally inheriting from it: the
locked ``ReportDocument`` type is not yet implemented anywhere in this
codebase, and this package should not take a hard import dependency on
a module that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from uaqe.hardware.memory_planner import MemoryPlan
from uaqe.optimizer.optimization_result import OptimizationResult

#: This renderer's report-type identifier, returned by
#: :meth:`OptimizationReportRenderer.report_type`.
REPORT_TYPE = "optimization"


@dataclass(frozen=True)
class OptimizationReportDocument:
    """A rendered summary of one run's optimization outcome.

    Attributes:
        selected_name: The winning candidate's label (e.g.
            ``"graph_cleanup+fusion"``).
        applied_passes: Human-readable notes on every structural pass
            applied to reach the winning candidate.
        objective_scores: The winning candidate's per-objective scores.
        aggregate_score: The winning candidate's combined ranking
            score.
        candidate_summaries: Every evaluated candidate's
            :meth:`~uaqe.optimizer.optimization_result.
            CandidateConfiguration.to_dict` summary, in evaluation
            order.
        pareto_front: The non-dominated candidates' summaries.
        rationale: The human-readable rationale for the winning
            candidate's selection, carried through from
            ``OptimizationResult``.
        peak_memory_bytes: The final activation-arena peak usage, from
            ``MemoryOptimizer``'s ``MemoryPlan``, if supplied to
            :meth:`OptimizationReportRenderer.render`. ``None`` if no
            ``MemoryPlan`` was supplied (e.g. reporting on a search
            result before ``MemoryOptimizer`` has run).
        tensor_reuse_slot_count: The number of distinct arena buffer
            slots ``MemoryPlan.tensor_reuse_map`` assigns tensors to,
            for the same conditional reason as
            ``peak_memory_bytes``.
        warnings: Every non-fatal warning accumulated while building
            this report.
    """

    selected_name: str = ""
    applied_passes: List[str] = field(default_factory=list)
    objective_scores: Dict[str, float] = field(default_factory=dict)
    aggregate_score: float = 0.0
    candidate_summaries: List[Dict[str, Any]] = field(default_factory=list)
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    peak_memory_bytes: Optional[int] = None
    tensor_reuse_slot_count: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this document as a plain, JSON-serializable ``dict``."""
        return {
            "selected_name": self.selected_name,
            "applied_passes": list(self.applied_passes),
            "objective_scores": dict(self.objective_scores),
            "aggregate_score": self.aggregate_score,
            "candidate_summaries": [dict(item) for item in self.candidate_summaries],
            "pareto_front": [dict(item) for item in self.pareto_front],
            "rationale": self.rationale,
            "peak_memory_bytes": self.peak_memory_bytes,
            "tensor_reuse_slot_count": self.tensor_reuse_slot_count,
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """Render this document as a Markdown summary.

        Returns:
            A Markdown string: a selection summary, a per-objective
            score table, the Pareto front, and any warnings.
        """
        lines: List[str] = [
            "# Optimization Report",
            "",
            f"- **Selected configuration:** {self.selected_name or 'none'}",
            f"- **Aggregate score:** {self.aggregate_score:.4f}",
            f"- **Candidates evaluated:** {len(self.candidate_summaries)}",
            f"- **Rationale:** {self.rationale}",
        ]
        if self.peak_memory_bytes is not None:
            lines.append(f"- **Peak activation memory:** {self.peak_memory_bytes:,} bytes")
        if self.tensor_reuse_slot_count is not None:
            lines.append(f"- **Arena buffer slots:** {self.tensor_reuse_slot_count}")

        lines.extend(["", "## Applied Passes", ""])
        if self.applied_passes:
            lines.extend(f"- {note}" for note in self.applied_passes)
        else:
            lines.append("- None (baseline configuration selected).")

        lines.extend(["", "## Objective Scores", "", "| Objective | Score |", "|---|---|"])
        for objective_name in sorted(self.objective_scores):
            lines.append(
                f"| {objective_name} | {self.objective_scores[objective_name]:.4f} |"
            )

        lines.extend(["", "## Pareto Front", ""])
        if self.pareto_front:
            lines.extend(
                f"- {candidate.get('name', '?')} "
                f"(aggregate_score={candidate.get('aggregate_score', 0.0):.4f})"
                for candidate in self.pareto_front
            )
        else:
            lines.append("- No candidates recorded.")

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines)


class OptimizationReportRenderer:
    """Builds an :class:`OptimizationReportDocument` from a completed
    optimization run's artifacts.

    Structurally compatible with ``IReportRenderer`` — see the module
    docstring for why this class does not formally inherit it yet.
    """

    def render(
        self,
        result: OptimizationResult,
        memory_plan: Optional[MemoryPlan] = None,
    ) -> OptimizationReportDocument:
        """Render a completed optimization result into a report
        document.

        Args:
            result: The finalized
                :class:`~uaqe.optimizer.optimization_result.
                OptimizationResult` to summarize.
            memory_plan: The final
                :class:`~uaqe.hardware.memory_planner.MemoryPlan`
                produced by
                :class:`~uaqe.optimizer.memory_optimizer.
                MemoryOptimizer`, if that stage has already run this
                run. Omit to render a report for the search result
                alone.

        Returns:
            The rendered :class:`OptimizationReportDocument`.
        """
        warnings: List[str] = []
        selected = result.selected_candidate
        if selected is None:
            warnings.append(
                "OptimizationResult has no selected_candidate; "
                "selected_name/applied_passes left empty."
            )

        return OptimizationReportDocument(
            selected_name=selected.name if selected is not None else "",
            applied_passes=list(selected.applied_passes) if selected is not None else [],
            objective_scores=dict(result.objective_scores),
            aggregate_score=selected.aggregate_score if selected is not None else 0.0,
            candidate_summaries=[candidate.to_dict() for candidate in result.candidates],
            pareto_front=list(result.pareto_front),
            rationale=result.rationale,
            peak_memory_bytes=(
                memory_plan.peak_memory_bytes if memory_plan is not None else None
            ),
            tensor_reuse_slot_count=(
                len(set(memory_plan.tensor_reuse_map.values()))
                if memory_plan is not None
                else None
            ),
            warnings=warnings,
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
