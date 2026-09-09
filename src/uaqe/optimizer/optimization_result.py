"""Result dataclasses produced by the multi-objective optimization
search.

Per ``03_API_Specification.md`` §8.1, the locked ``OptimizationResult``
shape is exactly ``selected_configuration: Dict[str, Any]``,
``pareto_front: List[Dict[str, Any]]``, ``objective_scores: Dict[str,
float]``. This module keeps those three fields (populated from
:meth:`CandidateConfiguration.to_dict`, a JSON-serializable summary) and
adds :attr:`OptimizationResult.candidates`,
:attr:`OptimizationResult.selected_candidate`, and
:attr:`OptimizationResult.rationale` as a superset — the same
locked-resolution precedent
:class:`~uaqe.compression.compression_planner.CompressionPlan` and
:class:`~uaqe.quantization.quantization_planner.QuantizationPlan` each
already establish for their own locked counterparts (richer domain
dataclass carrying the rationale/scores downstream stages need,
strictly containing the locked minimal shape rather than replacing it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from uaqe.common.imr import IMR
from uaqe.hardware.latency_estimator import LatencyEstimate
from uaqe.hardware.power_estimator import PowerEstimate


@dataclass
class CandidateConfiguration:
    """One candidate structural configuration evaluated during
    :meth:`~uaqe.optimizer.optimizer.Optimizer.search`.

    Attributes:
        name: The candidate's label, e.g. ``"baseline"``,
            ``"graph_cleanup"``, ``"graph_cleanup+fusion"`` — see
            :mod:`uaqe.optimizer.optimization_planner`.
        applied_passes: Human-readable notes on which structural passes
            were applied to produce this candidate's ``imr``, carried
            through from
            :class:`~uaqe.optimizer.graph_optimizer.
            GraphOptimizationResult` and
            :class:`~uaqe.optimizer.fusion_optimizer.FusionResult`.
        latency_estimate: This candidate's
            :class:`~uaqe.hardware.latency_estimator.LatencyEstimate`,
            from
            :meth:`~uaqe.optimizer.latency_optimizer.LatencyOptimizer.
            evaluate`.
        power_estimate: This candidate's
            :class:`~uaqe.hardware.power_estimator.PowerEstimate`, from
            :meth:`~uaqe.optimizer.power_optimizer.PowerOptimizer.
            evaluate`.
        memory_proxy_bytes: An order-of-magnitude static-weight-size
            proxy for the ``"memory"`` objective — not the exact
            activation-arena plan
            :class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer`
            computes for the final selected candidate; see
            :meth:`~uaqe.optimizer.optimizer.Optimizer.
            _estimate_memory_proxy` for why the exact plan is not
            computed per-candidate here.
        objective_scores: This candidate's normalized ``[0, ~)`` score
            for every objective in the run's ``OptimizationConfig.
            objectives``, higher is better, ``1.0`` meaning "as good as
            the baseline candidate" for ratio-based objectives.
        aggregate_score: The weighted (or unweighted, if
            ``OptimizationConfig.objective_weights`` is empty) combination
            of ``objective_scores`` used to rank candidates.
        imr: The candidate's materialized ``IMR``. Excluded from
            equality/``repr`` (and from :meth:`to_dict`) since it is not
            itself a report-worthy summary value — callers needing the
            actual candidate model should read this field directly, not
            round-trip it through JSON.
    """

    name: str
    applied_passes: List[str] = field(default_factory=list)
    latency_estimate: Optional[LatencyEstimate] = None
    power_estimate: Optional[PowerEstimate] = None
    memory_proxy_bytes: int = 0
    objective_scores: Dict[str, float] = field(default_factory=dict)
    aggregate_score: float = 0.0
    imr: Optional[IMR] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return this candidate as a plain, JSON-serializable
        ``dict`` (omitting :attr:`imr`).
        """
        return {
            "name": self.name,
            "applied_passes": list(self.applied_passes),
            "total_latency_ms": (
                self.latency_estimate.total_latency_ms
                if self.latency_estimate is not None
                else None
            ),
            "average_power_mw": (
                self.power_estimate.average_power_mw
                if self.power_estimate is not None
                else None
            ),
            "energy_per_inference_mj": (
                self.power_estimate.energy_per_inference_mj
                if self.power_estimate is not None
                else None
            ),
            "memory_proxy_bytes": self.memory_proxy_bytes,
            "objective_scores": dict(self.objective_scores),
            "aggregate_score": self.aggregate_score,
        }


@dataclass
class OptimizationResult:
    """The complete outcome of one
    :meth:`~uaqe.optimizer.optimizer.Optimizer.search` call.

    Attributes:
        selected_configuration: The winning candidate's
            :meth:`CandidateConfiguration.to_dict` — the locked minimal
            field (``03_API_Specification.md`` §8.1).
        pareto_front: The non-dominated candidates' ``to_dict()``
            summaries — the locked minimal field.
        objective_scores: The winning candidate's ``objective_scores``
            — the locked minimal field.
        candidates: Every candidate evaluated during the search, in
            evaluation order (a strict superset of ``pareto_front``).
        selected_candidate: The full winning
            :class:`CandidateConfiguration` (including its ``imr``),
            not just its serialized summary.
        rationale: A human-readable explanation of why
            ``selected_candidate`` was chosen over the others.
    """

    selected_configuration: Dict[str, Any] = field(default_factory=dict)
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)
    objective_scores: Dict[str, float] = field(default_factory=dict)
    candidates: List[CandidateConfiguration] = field(default_factory=list)
    selected_candidate: Optional[CandidateConfiguration] = None
    rationale: str = ""
