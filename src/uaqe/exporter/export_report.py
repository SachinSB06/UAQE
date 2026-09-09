"""Human- and machine-readable reporting for a completed export run.

``ExportReportRenderer`` summarizes an
:class:`~uaqe.exporter.export_planner.ExportPlan`, its resulting
:class:`~uaqe.exporter.exporter.DeploymentArtifact`, and its
:class:`~uaqe.exporter.artifact_manifest.ArtifactManifest` into an
:class:`ExportReportDocument`, mirroring
:class:`~uaqe.compression.compression_report.CompressionReportRenderer`'s
role for its own package. As with that renderer, this one is not
invoked by :class:`~uaqe.exporter.exporter.Exporter` itself — it is a
separate collaborator a reporting stage (``uaqe.domain.reporting`` /
this codebase's eventual ``reports`` package) calls once, after export
has completed, the same way ``CompressionReportRenderer`` is never
called from inside ``CompressionPlanner.execute``.

This renderer is deliberately *structurally* compatible with
``IReportRenderer`` (``render(...) -> ReportDocument``,
``report_type() -> str``) rather than formally inheriting from it, for
the same reason ``CompressionReportRenderer`` does not: the locked
``ReportDocument`` type is not yet implemented anywhere in this
codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from uaqe.exporter.artifact_manifest import ArtifactManifest
from uaqe.exporter.export_planner import ExportPlan
from uaqe.exporter.exporter import DeploymentArtifact

#: This renderer's report-type identifier, returned by
#: :meth:`ExportReportRenderer.report_type`.
REPORT_TYPE = "export"


@dataclass(frozen=True)
class ExportReportDocument:
    """A rendered summary of one run's export outcome.

    Attributes:
        target_profile_id: The ``HardwareProfile.profile_id`` exported
            for.
        export_format: The ``ExportFormat.value`` produced.
        rationale: The human-readable rationale for why
            ``export_format`` was selected, carried through from
            ``ExportPlan``.
        file_paths: Every artifact file path produced.
        size_bytes: The total size, in bytes, of ``file_paths``.
        manifest_path: The path of the written ``manifest.json``.
        package_path: The path of the assembled deployment package, if
            one was built; ``None`` otherwise.
        checksums: Each artifact file's checksum, keyed by its manifest
            ``relative_path``.
        warnings: Every non-fatal warning accumulated while building
            this report.
    """

    target_profile_id: str
    export_format: str
    rationale: str
    file_paths: List[str] = field(default_factory=list)
    size_bytes: int = 0
    manifest_path: str = ""
    package_path: "str | None" = None
    checksums: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this document as a plain, JSON-serializable ``dict``."""
        return {
            "target_profile_id": self.target_profile_id,
            "export_format": self.export_format,
            "rationale": self.rationale,
            "file_paths": list(self.file_paths),
            "size_bytes": self.size_bytes,
            "manifest_path": self.manifest_path,
            "package_path": self.package_path,
            "checksums": dict(self.checksums),
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """Render this document as a Markdown summary.

        Returns:
            A Markdown string: a summary of the export outcome followed
            by a per-file checksum table and any warnings.
        """
        lines: List[str] = [
            "# Export Report",
            "",
            f"- **Target profile:** {self.target_profile_id}",
            f"- **Export format:** {self.export_format}",
            f"- **Total size:** {self.size_bytes:,} bytes",
            f"- **Manifest:** {self.manifest_path or 'n/a'}",
            f"- **Package:** {self.package_path or 'n/a'}",
            f"- **Rationale:** {self.rationale}",
            "",
            "## Files",
            "",
            "| Relative path | Checksum (sha256) |",
            "|---|---|",
        ]
        for relative_path in sorted(self.checksums):
            lines.append(f"| {relative_path} | {self.checksums[relative_path]} |")

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines)


class ExportReportRenderer:
    """Builds an :class:`ExportReportDocument` from a completed export
    run's artifacts.

    Structurally compatible with ``IReportRenderer`` — see the module
    docstring for why this class does not formally inherit it yet.
    """

    def render(
        self,
        plan: ExportPlan,
        artifact: DeploymentArtifact,
        manifest: ArtifactManifest,
    ) -> ExportReportDocument:
        """Render a completed export into a report document.

        Args:
            plan: The finalized
                :class:`~uaqe.exporter.export_planner.ExportPlan` that
                was executed.
            artifact: The ``DeploymentArtifact`` produced by the
                selected backend, with ``manifest_path``/``package_path``
                already populated by
                :class:`~uaqe.exporter.exporter.Exporter`.
            manifest: The ``ArtifactManifest`` built for ``artifact``.

        Returns:
            The rendered :class:`ExportReportDocument`.
        """
        warnings: List[str] = []
        if len(manifest.files) != len(artifact.file_paths):
            warnings.append(
                f"Manifest lists {len(manifest.files)} file(s) but the "
                f"artifact reports {len(artifact.file_paths)}; the report's "
                f"checksum table reflects the manifest only."
            )

        return ExportReportDocument(
            target_profile_id=artifact.target_profile_id,
            export_format=artifact.export_format.value,
            rationale=plan.rationale,
            file_paths=list(artifact.file_paths),
            size_bytes=artifact.size_bytes,
            manifest_path=artifact.manifest_path,
            package_path=artifact.package_path,
            checksums={
                entry.relative_path: entry.checksum for entry in manifest.files
            },
            warnings=warnings,
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
