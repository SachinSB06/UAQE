"""Human- and machine-readable reporting for a completed quantization run.

``QuantizationReportRenderer`` summarizes a
:class:`~uaqe.quantization.quantization_planner.QuantizationPlan` (plus
its supporting ``CalibrationStatistics`` and ``SensitivityReport``) into
a :class:`QuantizationReportDocument` — a Markdown- and dict-renderable
summary suitable for ``uaqe.domain.reporting.report_generator.
ReportGenerator`` to fold into a run's overall report bundle.

This renderer is deliberately *structurally* compatible with
``IReportRenderer`` (``render(context) -> ReportDocument``,
``report_type() -> str``) rather than formally inheriting from it: the
locked ``ReportDocument`` type
(``uaqe.domain.reporting.report_generator.ReportDocument``, per
``03_API_Specification.md`` §1.11) is not yet implemented anywhere in
this codebase, and this package should not take a hard import
dependency on a module that does not exist. Once
``uaqe.domain.reporting`` is implemented, wrapping
``QuantizationReportDocument`` in that locked ``ReportDocument`` shape
(or having this class inherit ``IReportRenderer`` directly) is a pure
adaptation, not a rewrite, of the logic below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from uaqe.quantization.calibrator import CalibrationStatistics
from uaqe.quantization.quantization_planner import QuantizationPlan
from uaqe.quantization.sensitivity_analyzer import SensitivityReport

#: This renderer's report-type identifier, returned by
#: :meth:`QuantizationReportRenderer.report_type`.
REPORT_TYPE = "quantization"


@dataclass(frozen=True)
class QuantizationReportDocument:
    """A rendered summary of one run's quantization outcome.

    Attributes:
        hardware_profile_id: The ``HardwareProfile.profile_id`` this
            plan was recommended against.
        selected_strategy_name: The ``IQuantizationStrategy.name()``
            used to apply the plan.
        precision_distribution: The number of layers assigned to each
            ``Precision``, keyed by that precision's ``.value``.
        per_layer_precision: The precision assigned to every layer,
            keyed by layer name, as ``Precision.value`` strings.
        rationale: The human-readable rationale for every layer's
            assigned precision, keyed by layer name.
        flagged_layers: Names of layers the
            :class:`~uaqe.quantization.sensitivity_analyzer.
            SensitivityAnalyzer` flagged as quantization-sensitive.
        overridden_layers: Names of layers whose requested precision
            was unsupported by the target hardware and was therefore
            overridden, as computed by ``PrecisionRecommender`` and
            carried through the ``QuantizationPlan``.
        calibration_sample_count: The number of calibration-dataset
            samples the plan was calibrated from (``0`` if calibration
            ran from layer weights alone).
        warnings: Every non-fatal warning accumulated while building
            this report (e.g. hardware-forced precision overrides,
            sensitivity-based precision promotions).
    """

    hardware_profile_id: str = ""
    selected_strategy_name: str = ""
    precision_distribution: Dict[str, int] = field(default_factory=dict)
    per_layer_precision: Dict[str, str] = field(default_factory=dict)
    rationale: Dict[str, str] = field(default_factory=dict)
    flagged_layers: List[str] = field(default_factory=list)
    overridden_layers: List[str] = field(default_factory=list)
    calibration_sample_count: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return this document as a plain, JSON-serializable ``dict``."""
        return {
            "hardware_profile_id": self.hardware_profile_id,
            "selected_strategy_name": self.selected_strategy_name,
            "precision_distribution": dict(self.precision_distribution),
            "per_layer_precision": dict(self.per_layer_precision),
            "rationale": dict(self.rationale),
            "flagged_layers": list(self.flagged_layers),
            "overridden_layers": list(self.overridden_layers),
            "calibration_sample_count": self.calibration_sample_count,
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """Render this document as a Markdown summary.

        Returns:
            A Markdown string: a precision-distribution table followed
            by a per-layer rationale table and any warnings.
        """
        lines: List[str] = [
            "# Quantization Report",
            "",
            f"- **Hardware profile:** `{self.hardware_profile_id}`",
            f"- **Applied strategy:** `{self.selected_strategy_name}`",
            f"- **Calibration samples:** {self.calibration_sample_count}",
            "",
            "## Precision Distribution",
            "",
            "| Precision | Layer count |",
            "|---|---|",
        ]
        for precision_value, count in sorted(self.precision_distribution.items()):
            lines.append(f"| {precision_value} | {count} |")

        lines.extend(["", "## Per-Layer Detail", "", "| Layer | Precision | Rationale |", "|---|---|---|"])
        for layer_name in sorted(self.per_layer_precision):
            precision_value = self.per_layer_precision[layer_name]
            reason = self.rationale.get(layer_name, "")
            lines.append(f"| {layer_name} | {precision_value} | {reason} |")

        if self.flagged_layers:
            lines.extend(["", "## Flagged (Quantization-Sensitive) Layers", ""])
            lines.extend(f"- {name}" for name in self.flagged_layers)

        if self.overridden_layers:
            lines.extend(["", "## Overridden Layers (Hardware Constraint)", ""])
            lines.extend(f"- {name}" for name in self.overridden_layers)

        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)

        return "\n".join(lines)


class QuantizationReportRenderer:
    """Builds a :class:`QuantizationReportDocument` from a completed
    quantization run's artifacts.

    Structurally compatible with ``IReportRenderer`` — see the module
    docstring for why this class does not formally inherit it yet.
    """

    def render(
        self,
        plan: QuantizationPlan,
        hardware_profile_id: str,
        calibration: Optional[CalibrationStatistics] = None,
        sensitivity: Optional[SensitivityReport] = None,
    ) -> QuantizationReportDocument:
        """Render a completed quantization plan into a report document.

        Args:
            plan: The finalized
                :class:`~uaqe.quantization.quantization_planner.
                QuantizationPlan` to summarize.
            hardware_profile_id: The ``HardwareProfile.profile_id`` the
                plan was recommended against.
            calibration: The calibration statistics the plan was built
                from, if available; defaults to ``plan.calibration``
                when omitted.
            sensitivity: The sensitivity report the plan was built
                from, if available; defaults to ``plan.sensitivity``
                when omitted.

        Returns:
            The rendered :class:`QuantizationReportDocument`.
        """
        calibration = calibration if calibration is not None else plan.calibration
        sensitivity = sensitivity if sensitivity is not None else plan.sensitivity

        distribution: Dict[str, int] = {}
        for precision in plan.per_layer_precision.values():
            distribution[precision.value] = distribution.get(precision.value, 0) + 1

        warnings: List[str] = []
        flagged_layers: List[str] = []
        if sensitivity is not None:
            flagged_layers = list(sensitivity.flagged_layers)

        # Hardware-forced precision overrides: PrecisionRecommender
        # always overrides an unsupported requested precision and logs
        # a warning when it does so (05_Hardware_Profile_Spec.md §6
        # rule 2); surface that same signal here via
        # QuantizationPlan.overridden_layers, which is populated with
        # the identical set of layer names.
        for name in plan.overridden_layers:
            warnings.append(
                f"Layer {name!r} precision overridden due to hardware "
                f"constraints ({hardware_profile_id!r}): "
                f"{plan.rationale.get(name, '')}"
            )

        # Sensitivity-based precision promotions: layers flagged by the
        # SensitivityAnalyzer whose resolved precision was stepped up to
        # a more permissive, supported precision.
        for name in plan.stepped_up_layers:
            warnings.append(
                f"Layer {name!r} precision promoted due to flagged "
                f"quantization sensitivity: {plan.rationale.get(name, '')}"
            )

        return QuantizationReportDocument(
            hardware_profile_id=hardware_profile_id,
            selected_strategy_name=plan.selected_strategy_name,
            precision_distribution=distribution,
            per_layer_precision={
                name: precision.value
                for name, precision in plan.per_layer_precision.items()
            },
            rationale=dict(plan.rationale),
            flagged_layers=flagged_layers,
            overridden_layers=list(plan.overridden_layers),
            calibration_sample_count=(
                calibration.sample_count if calibration is not None else 0
            ),
            warnings=warnings,
        )

    def report_type(self) -> str:
        """Return this renderer's unique registration name."""
        return REPORT_TYPE
