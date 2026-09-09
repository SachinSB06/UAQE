"""Ranks and selects the best-fit ``HardwareProfile`` for a model's
deployment requirements.

Where ``uaqe.hardware.hardware_profile_loader.HardwareProfileLoader``
answers "which profiles exist", ``HardwareSelector`` answers "which of
these should I deploy to" — it never touches the filesystem, operating
only on an already-resolved ``List[HardwareProfile]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from uaqe.common.exceptions import ConfigurationError
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import ExportFormat, HardwareClass, Precision

if TYPE_CHECKING:
    # Deferred to break the hardware_manager <-> hardware_selector
    # import cycle; see the matching note in compatibility_checker.py.
    from uaqe.hardware.hardware_manager import HardwareProfile


@dataclass(frozen=True)
class SelectionCriteria:
    """A model's deployment requirements, used to filter and rank
    candidate ``HardwareProfile`` instances.

    Attributes:
        model_size_bytes: The model's total serialized size. A
            candidate whose ``max_model_size_bytes`` is smaller is
            ineligible. ``None`` skips this check.
        peak_memory_bytes: The model's estimated peak activation-arena
            usage (typically ``MemoryPlan.peak_memory_bytes`` from
            :mod:`uaqe.hardware.memory_planner`). A candidate whose
            ``tensor_memory_bytes`` is smaller is ineligible. ``None``
            skips this check.
        required_precisions: ``Precision`` values the model must be able
            to execute in. A candidate missing any of these from its
            ``supported_precisions`` is ineligible.
        required_export_formats: ``ExportFormat`` values at least one of
            which the candidate must offer. Empty means no constraint.
        preferred_hardware_classes: ``HardwareClass`` values that should
            be scored higher than equally-eligible candidates outside
            this list. Empty means no class preference.
    """

    model_size_bytes: Optional[int] = None
    peak_memory_bytes: Optional[int] = None
    required_precisions: List[Precision] = field(default_factory=list)
    required_export_formats: List[ExportFormat] = field(default_factory=list)
    preferred_hardware_classes: List[HardwareClass] = field(default_factory=list)


@dataclass(frozen=True)
class SelectionResult:
    """One candidate's ranking outcome.

    Attributes:
        profile: The candidate this result describes.
        eligible: Whether ``profile`` satisfies every hard requirement
            in the ``SelectionCriteria`` it was ranked against.
        score: A higher-is-better ranking score, meaningful only among
            other ``eligible`` results from the same call; ``0.0`` for
            ineligible candidates.
        reasons: Human-readable explanations — either why a candidate
            was disqualified, or what made it score as it did.
    """

    profile: HardwareProfile
    eligible: bool
    score: float
    reasons: List[str] = field(default_factory=list)


class HardwareSelector:
    """Ranks candidate ``HardwareProfile`` instances against a model's
    ``SelectionCriteria``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            selector operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``HardwareSelector``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def rank(
        self, candidates: List[HardwareProfile], criteria: SelectionCriteria
    ) -> List[SelectionResult]:
        """Rank every candidate against ``criteria``.

        Args:
            candidates: The ``HardwareProfile`` pool to rank.
            criteria: The model's deployment requirements.

        Returns:
            One :class:`SelectionResult` per candidate, sorted with
            every eligible result first (highest ``score`` first),
            followed by ineligible results in input order.
        """
        results = [self._evaluate(profile, criteria) for profile in candidates]
        eligible = sorted(
            (r for r in results if r.eligible), key=lambda r: r.score, reverse=True
        )
        ineligible = [r for r in results if not r.eligible]
        ranked = eligible + ineligible
        if self.logger is not None:
            self.logger.info(
                "Ranked hardware candidates.",
                candidate_count=len(candidates),
                eligible_count=len(eligible),
            )
        return ranked

    def select_best(
        self, candidates: List[HardwareProfile], criteria: SelectionCriteria
    ) -> HardwareProfile:
        """Return the single best-fit eligible profile.

        Args:
            candidates: The ``HardwareProfile`` pool to choose from.
            criteria: The model's deployment requirements.

        Returns:
            The highest-scoring eligible ``HardwareProfile``.

        Raises:
            ConfigurationError: If no candidate satisfies ``criteria``.
        """
        ranked = self.rank(candidates, criteria)
        if not ranked or not ranked[0].eligible:
            raise ConfigurationError(
                "No hardware profile satisfies the given selection criteria.",
                code="NO_ELIGIBLE_HARDWARE_PROFILE",
                remediation_hint=(
                    "Relax model_size_bytes, peak_memory_bytes, or "
                    "required_precisions, or add a suitable hardware profile."
                ),
            )
        return ranked[0].profile

    def _evaluate(
        self, profile: HardwareProfile, criteria: SelectionCriteria
    ) -> SelectionResult:
        """Evaluate one candidate's eligibility and score.

        Args:
            profile: The candidate to evaluate.
            criteria: The model's deployment requirements.

        Returns:
            The candidate's :class:`SelectionResult`.
        """
        reasons: List[str] = []

        if (
            criteria.model_size_bytes is not None
            and criteria.model_size_bytes > profile.max_model_size_bytes
        ):
            reasons.append(
                f"model_size_bytes ({criteria.model_size_bytes}) exceeds "
                f"max_model_size_bytes ({profile.max_model_size_bytes})."
            )

        if (
            criteria.peak_memory_bytes is not None
            and criteria.peak_memory_bytes > profile.tensor_memory_bytes
        ):
            reasons.append(
                f"peak_memory_bytes ({criteria.peak_memory_bytes}) exceeds "
                f"tensor_memory_bytes ({profile.tensor_memory_bytes})."
            )

        missing_precisions = [
            p for p in criteria.required_precisions if p not in profile.supported_precisions
        ]
        if missing_precisions:
            reasons.append(
                "Unsupported required precisions: "
                f"{[p.value for p in missing_precisions]}."
            )

        if criteria.required_export_formats and not any(
            fmt in profile.preferred_export_formats
            for fmt in criteria.required_export_formats
        ):
            reasons.append(
                "None of the required export formats "
                f"{[f.value for f in criteria.required_export_formats]} are "
                "offered."
            )

        if reasons:
            return SelectionResult(
                profile=profile, eligible=False, score=0.0, reasons=reasons
            )

        score = self._score(profile, criteria, reasons)
        return SelectionResult(profile=profile, eligible=True, score=score, reasons=reasons)

    def _score(
        self,
        profile: HardwareProfile,
        criteria: SelectionCriteria,
        reasons: List[str],
    ) -> float:
        """Compute a higher-is-better score for an eligible candidate.

        Scoring favors the smallest (cheapest) profile that still fits
        the model — the same "smallest sufficient target" heuristic a
        human deploying to constrained hardware would apply — with a
        bonus for a preferred ``hardware_class`` and a small tiebreaker
        for higher ``clock_speed_hz``.

        Args:
            profile: The candidate to score.
            criteria: The model's deployment requirements.
            reasons: Mutated in place with a human-readable note on how
                the score was composed.

        Returns:
            The candidate's score.
        """
        # Smaller tensor_memory_bytes -> higher score: an eligible
        # profile with less headroom is a tighter (cheaper) fit.
        headroom_score = 1.0 / (1.0 + profile.tensor_memory_bytes)

        class_bonus = 0.0
        if profile.hardware_class in criteria.preferred_hardware_classes:
            class_bonus = 1.0
            reasons.append(
                f"Preferred hardware_class match ({profile.hardware_class.value})."
            )

        clock_tiebreaker = (profile.clock_speed_hz or 0) * 1e-12

        score = class_bonus + headroom_score + clock_tiebreaker
        reasons.append(f"Computed selection score: {score:.6f}.")
        return score
