"""Human- and machine-readable reporting for a completed benchmark run.

``BenchmarkReportRenderer`` summarizes a
:class:`~uaqe.benchmark.benchmark_result.BenchmarkResult` (and,
optionally, a :class:`~uaqe.benchmark.benchmark_result.
ComparisonBenchmarkResult` when multiple hardware targets were
benchmarked for the same model) into a :class:`BenchmarkReportDocument`
— a Markdown- and dict-renderable summary suitable for
``uaqe.domain.reporting.report_generator.ReportGenerator`` to fold into
a run's overall report bundle, mirroring
:class:`~uaqe.evaluation.evaluation_report.EvaluationReportRenderer`'s
and :class:`~uaqe.optimizer.optimization_report.
OptimizationReportRenderer`'s role for their own packages.

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

from uaqe.benchmark.benchmark_result import (
    BenchmarkResult,
    ComparisonBenchmarkResult,
)

#: This renderer's report-type identifier, returned by
#: :meth:`BenchmarkReportRenderer.report_type`.
REPORT_TYPE = "benchmark"


@dataclass(frozen=True)
class BenchmarkReportDocument:
    """A rendered summary of one run's benchmark outcome.

    Attributes:
        hardware_profile_id: The benchmarked ``HardwareProfile.
            profile_id``, mirrored from ``BenchmarkResult``.
        used_real_hardware: Whether the run executed on a physical
            device rather than the analytical simulation path.
        trials: The ``trials`` count actually used this run.
        latency_ms_p50: The locked minimal figure, mirrored from
            ``BenchmarkResult``.
        latency_ms_p99: The locked minimal figure, mirrored from
            ``BenchmarkResult``.
        throughput_inferences_per_sec: The locked minimal figure,
            mirrored from ``BenchmarkResult``.
        peak_memory_bytes: The locked minimal figure, mirrored from
            ``BenchmarkResult``.
        artifact_size_bytes: The benchmarked artifact's size, from
            ``BenchmarkResult.size``.
        average_power_mw: The power dimension's headline figure, or
            ``None`` if power was not resolvable this run.
        energy_per_inference_mj: The power dimension's per-inference
            energy figure, or ``None`` for the same reason.
        comparison_ranking: When this report also summarizes a
            :class:`~uaqe.benchmark.benchmark_result.
            ComparisonBenchmarkResult`, every compared target's
            ``profile_id``/``normalized_score``/``rank``, best
            (``rank=1``) first; empty when no comparison was supplied
            to :meth:`BenchmarkReportRenderer.render`.
        best_overall_profile_id: The comparison's overall winner, or
            ``None`` if no comparison was supplied.
        warnings: Every non-fatal warning accumulated while building
            this report.
    """

    hardware_profile_id: str = ""
    used_real_hardware: bool = False
    trials: int = 0
    latency_ms_p50: float = 0.0
    latency_ms_p99: float = 0.0
    throughput_inferences_per_sec: float = 0.0
    peak_memory_bytes: int = 0
    artifact_size_bytes: int = 0
    average_power_mw: Optional[float] = None
    energy_per_inference_mj: Optional[float] = None
    comparison_ranking: List[Dict[str, Any]] = field(default_factory=list)
    best_overall_profile_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this document as a plain, JSON-serializable ``dict``."""
        return {
            "hardware_profile_id": self.hardware_profile_id,
            "used_real_hardware": self.used_real_hardware,
            "trials": self.trials,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p99": self.latency_ms_p99,
            "throughput_inferences_per_sec": self.throughput_inferences_per_sec,
            "peak_memory_bytes": self.peak_memory_bytes,
            "artifact_size_bytes": self.artifact_size_bytes,
            "average_power_mw": self.average_power_mw,
            "energy_per_inference_mj": self.energy_per_inference_mj,
            "comparison_ranking": [dict(item) for item in self.comparison_ranking],
            "best_overall_profile_id": self.best_overall_profile_id,
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """Render this document as a Markdown summary.

        Returns:
            A Markdown string: a headline-figure summary, the power
            dimension if present, the comparison ranking table if
            present, and any warnings.
        """
        lines: List[str] = [
            "# Benchmark Report",
            "",
            f"- **Hardware profile:** {self.hardware_profile_id or 'unknown'}",
            f"- **Execution path:** "
            f"{'real hardware' if self.used_real_hardware else 'simulation'}",
            f"- **Trials:** {self.trials}",
            f"- **Latency p50:** {self.latency_ms_p50:.3f} ms",
            f"- **Latency p99:** {self.latency_ms_p99:.3f} ms",
            f"- **Throughput:** {self.throughput_inferences_per_sec:.2f} inferences/sec",
            f"- **Peak memory:** {self.peak_memory_bytes:,} bytes",
            f"- **Artifact size:** {self.artifact_size_bytes:,} bytes",
        ]
        if self.average_power_mw is not None:
            lines.append(f"- **Average power:** {self.average_power_mw:.2f} mW")
        if self.energy_per_inference_mj is not None:
            lines.append(
                f"- **Energy per inference:** {self.energy_per_inference_mj:.4f} mJ"
            )

        if self.comparison_ranking:
            lines.extend(
                [
                    "",
                    "## Hardware Target Comparison",
                    "",
                    "| Rank | Profile | Normalized Score |",
                    "|---|---|---|",
                ]
            )
            for entry in self.comparison_ranking:
                lines.append(
                    f"| {entry.get('rank', '?')} | {entry.get('profile_id', '?')} "
                    f"| {entry.get('normalized_score', 0.0):.4f} |"
                )
            lines.append(
                f"\n- **Best overall:** {self.best_overall_profile_id or 'unknown'}"
            )

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines)


class BenchmarkReportRenderer:
    """Builds a :class:`BenchmarkReportDocument` from a completed
    benchmark run's result.

    Structurally compatible with ``IReportRenderer`` — see the module
    docstring for why this class does not formally inherit it yet.
    """

    def render(
        self,
        result: BenchmarkResult,
        comparison: Optional[ComparisonBenchmarkResult] = None,
    ) -> BenchmarkReportDocument:
        """Render a completed benchmark result into a report document.

        Args:
            result: The finalized :class:`~uaqe.benchmark.
                benchmark_result.BenchmarkResult` to summarize.
            comparison: The
                :class:`~uaqe.benchmark.benchmark_result.
                ComparisonBenchmarkResult` produced by
                :class:`~uaqe.benchmark.comparison_benchmark.
                ComparisonBenchmark`, if multiple hardware targets were
                benchmarked for this model this run. Omit to render a
                single-target report.

        Returns:
            The rendered :class:`BenchmarkReportDocument`.
        """
        comparison_ranking: List[Dict[str, Any]] = []
        best_overall_profile_id: Optional[str] = None
        if comparison is not None:
            comparison_ranking = [
                {
                    "profile_id": entry.profile_id,
                    "normalized_score": entry.normalized_score,
                    "rank": entry.rank,
                }
                for entry in comparison.entries
            ]
            best_overall_profile_id = comparison.best_overall_profile_id

        return BenchmarkReportDocument(
            hardware_profile_id=result.hardware_profile_id,
            used_real_hardware=result.used_real_hardware,
            trials=result.trials,
            latency_ms_p50=result.latency_ms_p50,
            latency_ms_p99=result.latency_ms_p99,
            throughput_inferences_per_sec=result.throughput_inferences_per_sec,
            peak_memory_bytes=result.peak_memory_bytes,
            artifact_size_bytes=result.size.artifact_size_bytes,
            average_power_mw=(
                result.power.average_power_mw if result.power is not None else None
            ),
            energy_per_inference_mj=(
                result.power.energy_per_inference_mj
                if result.power is not None
                else None
            ),
            comparison_ranking=comparison_ranking,
            best_overall_profile_id=best_overall_profile_id,
            warnings=list(result.warnings),
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
