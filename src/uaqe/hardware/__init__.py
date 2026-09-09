"""``uaqe.hardware`` — a consolidated deployment-target package for the
Universal AI Quantization Engine: profile loading, target selection,
compatibility checking, memory planning, and latency/power estimation
behind one facade, :class:`~uaqe.hardware.hardware_manager.HardwareManager`.

Relationship to ``09_Architecture_Lock.md`` §8: the locked module
ownership table assigns this package's two classes — ``HardwareProfile``
and ``HardwareManager`` — to ``uaqe.domain.hardware``, and assigns
profile loading, compatibility checking, and memory planning to three
*other* locked packages (``uaqe.infrastructure.repositories``,
``uaqe.domain.compatibility``, ``uaqe.domain.optimization``,
respectively). This package intentionally consolidates all of that,
plus two capabilities absent from the locked spec entirely (latency and
power estimation), into one importable unit. Every class below still
implements the exact locked contract it corresponds to (e.g.
:meth:`HardwareManager.check_compatibility` matches
``03_API_Specification.md`` §5.1 signature-for-signature), so moving a
given module into its architecturally "correct" package later is a pure
relocation, not a rewrite.

Public API:

- :class:`~uaqe.hardware.hardware_manager.HardwareProfile` /
  :class:`~uaqe.hardware.hardware_manager.FpgaResourceProfile` — the
  value objects (``05_Hardware_Profile_Spec.md`` §1).
- :class:`~uaqe.hardware.hardware_manager.HardwareManager` — the facade;
  most callers only need this class.
- :class:`~uaqe.hardware.hardware_profile_loader.HardwareProfileLoader`
  — loads ``HardwareProfile`` instances from
  ``hardware_profiles/**/*.json``.
- :class:`~uaqe.hardware.hardware_selector.HardwareSelector` (with
  :class:`~uaqe.hardware.hardware_selector.SelectionCriteria` /
  :class:`~uaqe.hardware.hardware_selector.SelectionResult`) — ranks
  candidate profiles against a model's requirements.
- :class:`~uaqe.hardware.compatibility_checker.CompatibilityChecker` —
  layer/precision/size compatibility checking.
- :class:`~uaqe.hardware.memory_planner.MemoryPlanner` (with
  :class:`~uaqe.hardware.memory_planner.MemoryPlan`) — activation-arena
  memory planning.
- :class:`~uaqe.hardware.latency_estimator.LatencyEstimator` (with
  :class:`~uaqe.hardware.latency_estimator.LatencyEstimate`) — inference
  latency estimation.
- :class:`~uaqe.hardware.power_estimator.PowerEstimator` (with
  :class:`~uaqe.hardware.power_estimator.PowerEstimate`) — power/energy
  estimation.
"""

from uaqe.hardware.compatibility_checker import CompatibilityChecker
from uaqe.hardware.hardware_manager import (
    FpgaResourceProfile,
    HardwareAssessment,
    HardwareManager,
    HardwareProfile,
)
from uaqe.hardware.hardware_profile_loader import HardwareProfileLoader
from uaqe.hardware.hardware_selector import (
    HardwareSelector,
    SelectionCriteria,
    SelectionResult,
)
from uaqe.hardware.latency_estimator import LatencyEstimate, LatencyEstimator
from uaqe.hardware.memory_planner import MemoryPlan, MemoryPlanner
from uaqe.hardware.power_estimator import PowerEstimate, PowerEstimator

__all__ = [
    # hardware_manager.py
    "HardwareProfile",
    "FpgaResourceProfile",
    "HardwareManager",
    "HardwareAssessment",
    # hardware_profile_loader.py
    "HardwareProfileLoader",
    # hardware_selector.py
    "HardwareSelector",
    "SelectionCriteria",
    "SelectionResult",
    # compatibility_checker.py
    "CompatibilityChecker",
    # memory_planner.py
    "MemoryPlanner",
    "MemoryPlan",
    # latency_estimator.py
    "LatencyEstimator",
    "LatencyEstimate",
    # power_estimator.py
    "PowerEstimator",
    "PowerEstimate",
]
