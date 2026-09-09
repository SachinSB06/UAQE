"""``HardwareProfile`` value object and the ``HardwareManager`` facade.

Per ``05_Hardware_Profile_Spec.md`` §1, ``HardwareProfile`` is the
in-memory representation of one deployment target (FPGA, embedded MCU,
or Raspberry Pi), resolved from the on-disk JSON database under
``hardware_profiles/**/*.json`` (``05_Hardware_Profile_Spec.md`` §2-§5).

``HardwareManager`` is the single entry point for this package: it
resolves a run's target ``HardwareProfile`` exactly once and offers
every other capability in this package (loading, selection,
compatibility checking, memory planning, and latency/power estimation)
as one coherent facade, so a caller only needs one import to reason
about "can this model run here, and how well".

Note on scope vs. ``09_Architecture_Lock.md`` §8: the locked
``uaqe.domain.hardware`` module ownership table names exactly
``HardwareProfile`` and ``HardwareManager`` for this package, with
profile loading owned by ``uaqe.infrastructure.repositories``,
compatibility checking by ``uaqe.domain.compatibility``, and memory
planning by ``uaqe.domain.optimization``. This package intentionally
bundles those concerns (plus two capabilities — latency and power
estimation — that are not in the locked spec at all) into a single
``hardware/`` module for callers who want one cohesive package rather
than four. If strict Architecture Lock conformance is required, treat
:mod:`uaqe.hardware.hardware_profile_loader`,
:mod:`uaqe.hardware.compatibility_checker`, and
:mod:`uaqe.hardware.memory_planner` as reference implementations to
relocate into their locked packages; ``HardwareManager.check_compatibility``
below still matches the locked signature exactly, so relocation does not
require changing this class's public surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from uaqe.common.exceptions import HardwareIncompatibilityError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_hardware_profile_repository import (
    IHardwareProfileRepository,
)
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import CompatibilityReport, StageResult
from uaqe.common.types import ExportFormat, HardwareClass, Precision
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.hardware.compatibility_checker import CompatibilityChecker
from uaqe.hardware.hardware_selector import (
    HardwareSelector,
    SelectionCriteria,
    SelectionResult,
)
from uaqe.hardware.latency_estimator import LatencyEstimate, LatencyEstimator
from uaqe.hardware.memory_planner import MemoryPlan, MemoryPlanner
from uaqe.hardware.power_estimator import PowerEstimate, PowerEstimator


from uaqe.domain.hardware_manager import FpgaResourceProfile, HardwareProfile


class HardwareManager(PipelineStage):
    """Resolves a run's target ``HardwareProfile`` once and provides
    every other hardware-facing capability as a single facade.

    Attributes:
        repo: The profile source this manager resolves
            ``target_profile_id`` against.
        logger: Structured logging sink for every operation performed
            through this facade.
        target_profile_id: The ``HardwareProfile.profile_id`` this
            manager resolves and memoizes on first access
            (``09_Architecture_Lock.md`` §13 rule 3: "``HardwareProfile``
            is resolved exactly once by ``HardwareManager`` and read
            [...] thereafter").
    """

    def __init__(
        self,
        repo: IHardwareProfileRepository,
        logger: ILogger,
        target_profile_id: str,
        *,
        selector: Optional[HardwareSelector] = None,
        checker: Optional[CompatibilityChecker] = None,
        memory_planner: Optional[MemoryPlanner] = None,
        latency_estimator: Optional[LatencyEstimator] = None,
        power_estimator: Optional[PowerEstimator] = None,
    ) -> None:
        """Initialize a ``HardwareManager``.

        Args:
            repo: The ``IHardwareProfileRepository`` this manager
                resolves ``target_profile_id`` against. Typically a
                :class:`~uaqe.hardware.hardware_profile_loader.
                HardwareProfileLoader`.
            logger: Structured logging sink.
            target_profile_id: The ``profile_id`` of this run's
                deployment target, resolved lazily on first use.
            selector: Optional :class:`HardwareSelector`; a default
                instance is constructed if omitted.
            checker: Optional :class:`CompatibilityChecker`; a default
                instance is constructed if omitted.
            memory_planner: Optional :class:`MemoryPlanner`; a default
                instance is constructed if omitted.
            latency_estimator: Optional :class:`LatencyEstimator`; a
                default instance is constructed if omitted.
            power_estimator: Optional :class:`PowerEstimator`; a
                default instance is constructed if omitted.
        """
        self.repo: IHardwareProfileRepository = repo
        self.logger: ILogger = logger
        self.target_profile_id: str = target_profile_id
        self._resolved_profile: Optional[HardwareProfile] = None
        self._selector: HardwareSelector = selector or HardwareSelector(logger=logger)
        self._checker: CompatibilityChecker = checker or CompatibilityChecker(logger=logger)
        self._memory_planner: MemoryPlanner = memory_planner or MemoryPlanner(logger=logger)
        self._latency_estimator: LatencyEstimator = (
            latency_estimator or LatencyEstimator(logger=logger)
        )
        self._power_estimator: PowerEstimator = power_estimator or PowerEstimator(logger=logger)

    def resolve(self) -> HardwareProfile:
        """Resolve and memoize this run's target ``HardwareProfile``.

        Returns:
            The resolved ``HardwareProfile``. Subsequent calls return
            the same cached instance without calling ``repo.get()``
            again.

        Raises:
            ConfigurationError: If ``target_profile_id`` cannot be
                resolved by ``repo``.
        """
        if self._resolved_profile is None:
            self.logger.info(
                "Resolving target hardware profile.",
                profile_id=self.target_profile_id,
            )
            self._resolved_profile = self.repo.get(self.target_profile_id)
        return self._resolved_profile

    def execute(self, context: PipelineContext) -> StageResult:
        """Resolve this run's ``HardwareProfile`` and record it in the
        pipeline context.

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is the resolved
            ``HardwareProfile``.
        """
        profile = self.resolve()
        return StageResult(stage_name=self.name(), success=True, payload=profile)

    def check_compatibility(self, imr: IMR, profile: HardwareProfile) -> CompatibilityReport:
        """Check whether ``imr`` is fully compatible with ``profile``.

        Args:
            imr: The model to check.
            profile: The resolved deployment target.

        Returns:
            The resulting ``CompatibilityReport``.

        Raises:
            HardwareIncompatibilityError: If ``imr`` is not fully
                compatible with ``profile``.
        """
        return self._checker.check(imr, profile, strict=True)

    def select_profile(
        self, candidates: List[HardwareProfile], criteria: SelectionCriteria
    ) -> HardwareProfile:
        """Select the best-fit profile from ``candidates`` for
        ``criteria``.

        Args:
            candidates: The ``HardwareProfile`` pool to rank.
            criteria: The model's deployment requirements.

        Returns:
            The highest-ranked eligible ``HardwareProfile``.

        Raises:
            ConfigurationError: If no candidate satisfies ``criteria``.
        """
        return self._selector.select_best(candidates, criteria)

    def rank_profiles(
        self, candidates: List[HardwareProfile], criteria: SelectionCriteria
    ) -> List[SelectionResult]:
        """Rank every candidate profile against ``criteria``.

        Args:
            candidates: The ``HardwareProfile`` pool to rank.
            criteria: The model's deployment requirements.

        Returns:
            Every candidate's :class:`SelectionResult`, best first.
        """
        return self._selector.rank(candidates, criteria)

    def plan_memory(self, imr: IMR, profile: HardwareProfile) -> MemoryPlan:
        """Plan activation-arena memory usage for ``imr`` on ``profile``.

        Args:
            imr: The model to plan for.
            profile: The resolved deployment target.

        Returns:
            The resulting ``MemoryPlan``.

        Raises:
            HardwareIncompatibilityError: If the planned peak memory
                exceeds ``profile.tensor_memory_bytes``.
        """
        return self._memory_planner.plan(imr, profile, strict=True)

    def estimate_latency(self, imr: IMR, profile: HardwareProfile) -> LatencyEstimate:
        """Estimate per-layer and total inference latency for ``imr``
        on ``profile``.

        Args:
            imr: The model to estimate.
            profile: The resolved deployment target.

        Returns:
            The resulting ``LatencyEstimate``.
        """
        return self._latency_estimator.estimate(imr, profile)

    def estimate_power(
        self,
        profile: HardwareProfile,
        latency: LatencyEstimate,
        dominant_precision: Precision = Precision.INT8,
    ) -> PowerEstimate:
        """Estimate power draw and per-inference energy for ``profile``.

        Args:
            profile: The resolved deployment target.
            latency: A previously computed ``LatencyEstimate`` for the
                same model/profile pair, used to derive energy.
            dominant_precision: The precision most of the model executes
                in, used to scale the reference power figure.

        Returns:
            The resulting ``PowerEstimate``.
        """
        return self._power_estimator.estimate(profile, latency, dominant_precision)

    def full_assessment(
        self, imr: IMR, profile: HardwareProfile
    ) -> "HardwareAssessment":
        """Run every check/estimate this package offers for one
        ``imr``/``profile`` pair in a single call.

        Args:
            imr: The model to assess.
            profile: The resolved deployment target.

        Returns:
            A :class:`HardwareAssessment` bundling every result. Note
            that unlike :meth:`check_compatibility` and
            :meth:`plan_memory`, this method does not raise on
            incompatibility or overflow — it reports them via
            ``compatibility.compatible`` so a caller can inspect a full
            picture even for a target the model does not fit.
        """
        compatibility = self._checker.check(imr, profile, strict=False)
        memory = self._memory_planner.plan(imr, profile, strict=False)
        latency = self._latency_estimator.estimate(imr, profile)
        power = self._power_estimator.estimate(profile, latency)
        return HardwareAssessment(
            profile=profile,
            compatibility=compatibility,
            memory=memory,
            latency=latency,
            power=power,
        )


@dataclass(frozen=True)
class HardwareAssessment:
    """The combined result of every hardware-facing check for one
    model/profile pair, as produced by ``HardwareManager.full_assessment``.

    Attributes:
        profile: The deployment target this assessment was run against.
        compatibility: The layer/precision/size compatibility result.
        memory: The activation-arena memory plan.
        latency: The estimated inference latency.
        power: The estimated power draw and per-inference energy.
    """

    profile: HardwareProfile
    compatibility: CompatibilityReport
    memory: MemoryPlan
    latency: LatencyEstimate
    power: PowerEstimate
