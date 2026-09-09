"""Decides which single ``ExportFormat`` :class:`~uaqe.exporter.exporter.
Exporter` targets for a resolved ``HardwareProfile``.

``ExportPlanner`` is an internal collaborator of
:class:`~uaqe.exporter.exporter.Exporter` — the same "decides, does not
itself act" role :class:`~uaqe.compression.compression_planner.
CompressionPlanner`'s own advisory half and
:class:`~uaqe.benchmark.benchmark_planner.BenchmarkPlanner` already
play for their packages: :meth:`ExportPlanner.build_plan` never reads
``PipelineContext``, never invokes a backend, and never writes a file;
``Exporter.execute`` owns all three of those. Its one job is resolving
*which* ``ExportFormat`` this run targets, before
:class:`~uaqe.exporter.export_validator.ExportValidator` re-checks the
final ``IMR`` against it and :meth:`~uaqe.exporter.exporter.Exporter.
select_backend` resolves the concrete backend for it.

Selection order, first match wins:

1. An explicit ``requested_format`` argument to :meth:`build_plan` —
   e.g. a CLI ``--export-format`` override or a per-run
   ``RunRequest.config_overrides`` entry threaded through by the
   caller; this planner does not itself read either, it only accepts
   the already-resolved value so callers stay in control of precedence.
2. The first entry of ``HardwareProfile.preferred_export_formats``
   (``05_Hardware_Profile_Spec.md`` §1: "the ``ExportFormat`` values,
   in preference order, this target's exporter backends produce") —
   the profile's own declared preference, honored in the order it
   lists them.
3. :data:`DEFAULT_EXPORT_FORMAT` (``ExportFormat.BIN``), when a profile
   declares no preference at all. Raw concatenated parameter bytes are
   the one format every runtime in this codebase can consume at a
   minimum (``BinaryExporter``'s own module docstring), so it is the
   only safe universal fallback — every other backend layers
   additional structure (text encoding, container framing) on top of
   exactly that same buffer (see
   :mod:`uaqe.exporter.binary_exporter`'s module docstring for why the
   other four backends are built on it directly).

This planner never raises: an unresolvable profile simply falls
through to step 3 above. Whether the ``ExportFormat`` this planner
selects actually has a registered backend is checked downstream, by
:meth:`~uaqe.exporter.exporter.Exporter.select_backend` — a planning
decision and a registration/wiring concern are deliberately kept
separate, the same separation ``ExportValidator``'s own docstring
draws between itself and earlier compatibility-checking stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import ExportFormat
from uaqe.domain.hardware_manager import HardwareProfile

#: The universal fallback ``ExportFormat`` used when a
#: ``HardwareProfile`` declares no ``preferred_export_formats`` at all
#: — see the module docstring's selection-order step 3.
DEFAULT_EXPORT_FORMAT: ExportFormat = ExportFormat.BIN


@dataclass(frozen=True)
class ExportPlan:
    """Which single ``ExportFormat`` a run targets, and why.

    Superset of the locked minimal shape implied by
    ``03_API_Specification.md`` §9.1's ``ExportFormat``-selecting role,
    following the same "richer plan dataclass than the bare
    ``*Config``" precedent
    :class:`~uaqe.compression.compression_planner.CompressionPlan` and
    :class:`~uaqe.optimizer.optimization_result.OptimizationResult`
    already set for their own packages.

    Attributes:
        selected_format: The single ``ExportFormat`` this run targets —
            the value :class:`~uaqe.exporter.exporter.Exporter` passes
            to ``ExportValidator.validate`` and resolves a backend for.
        target_profile_id: The ``HardwareProfile.profile_id`` this plan
            was built for.
        considered_formats: ``target``'s
            ``preferred_export_formats`` at the time this plan was
            built, in the profile's own declared order, kept for
            transparency/debugging even when ``selected_format`` came
            from an explicit override or the default fallback instead.
        rationale: A short, human-readable explanation of why
            ``selected_format`` was chosen.
    """

    selected_format: ExportFormat = DEFAULT_EXPORT_FORMAT
    target_profile_id: str = ""
    considered_formats: List[ExportFormat] = field(default_factory=list)
    rationale: str = ""


class ExportPlanner:
    """Resolves an :class:`ExportPlan` for a resolved ``HardwareProfile``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            planner operates silently.
        default_format: The fallback ``ExportFormat`` used when a
            profile declares no ``preferred_export_formats`` and no
            ``requested_format`` override is supplied.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        default_format: ExportFormat = DEFAULT_EXPORT_FORMAT,
    ) -> None:
        """Initialize an ``ExportPlanner``.

        Args:
            logger: Optional structured logging sink.
            default_format: The fallback ``ExportFormat``; overridable
                mainly for tests that want to exercise the fallback
                path with a format other than ``ExportFormat.BIN``.
        """
        self.logger: Optional[ILogger] = logger
        self.default_format: ExportFormat = default_format

    def build_plan(
        self,
        profile: HardwareProfile,
        requested_format: Optional[ExportFormat] = None,
    ) -> ExportPlan:
        """Resolve which ``ExportFormat`` to target for ``profile``.

        Args:
            profile: The resolved deployment target.
            requested_format: An explicit caller-supplied override,
                honored ahead of ``profile.preferred_export_formats``
                when supplied.

        Returns:
            The resolved :class:`ExportPlan`. Never raises — see the
            module docstring's selection order for the fallback chain
            this always bottoms out in.
        """
        considered = list(profile.preferred_export_formats)

        if requested_format is not None:
            selected = requested_format
            rationale = (
                f"Using explicitly requested ExportFormat.{requested_format.name}, "
                f"overriding target {profile.profile_id!r}'s own preference order "
                f"({[fmt.value for fmt in considered] or 'none declared'})."
            )
        elif considered:
            selected = considered[0]
            rationale = (
                f"Selected ExportFormat.{selected.name}, the first entry of "
                f"target {profile.profile_id!r}'s preferred_export_formats "
                f"({[fmt.value for fmt in considered]})."
            )
        else:
            selected = self.default_format
            rationale = (
                f"Target {profile.profile_id!r} declares no preferred_export_formats; "
                f"falling back to the universal default, "
                f"ExportFormat.{selected.name}."
            )

        if self.logger is not None:
            self.logger.info(
                "Export plan built.",
                target_profile_id=profile.profile_id,
                selected_format=selected.value,
                considered_formats=[fmt.value for fmt in considered],
            )

        return ExportPlan(
            selected_format=selected,
            target_profile_id=profile.profile_id,
            considered_formats=considered,
            rationale=rationale,
        )
