"""Decides which structural optimization passes to try, and applies
them to produce each candidate ``IMR``.

``OptimizationPlanner`` is an internal collaborator of
:class:`~uaqe.optimizer.optimizer.Optimizer` — the same role
:class:`~uaqe.quantization.precision_recommender.PrecisionRecommender`
plays for
:class:`~uaqe.quantization.quantization_planner.QuantizationPlanner`:
it does not itself read/write ``PipelineContext`` or return a
``StageResult``; ``Optimizer`` owns that. It has two jobs:

1. :meth:`build_plan` turns the run's ``OptimizationConfig`` into an
   :class:`OptimizationPlan` naming which candidate structural
   configurations ``Optimizer.search`` should evaluate, bounded by
   ``OptimizationConfig.max_search_iterations``
   (``10_Module_Development_Guide.md`` §8: "``OptimizationEngine``
   tested for search convergence within ``max_search_iterations``").
2. :meth:`materialize` applies the structural passes implied by one
   candidate label to a given ``IMR`` — delegating to
   :class:`~uaqe.optimizer.graph_optimizer.GraphOptimizer` and
   :class:`~uaqe.optimizer.fusion_optimizer.FusionOptimizer` — so
   ``Optimizer`` never needs to know which passes a given label
   implies.

Candidates are structured as a strictly increasing chain of
aggressiveness (:data:`BASELINE` -> :data:`GRAPH_CLEANUP` ->
:data:`GRAPH_CLEANUP_FUSION`) rather than an arbitrary combinatorial
power set: since fusion's preconditions (single-consumer intermediate
tensors) are only *more* likely to hold after graph cleanup has already
removed pass-through/duplicate layers, "fusion without prior cleanup"
is not a materially distinct configuration worth its own search
iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.value_objects import OptimizationConfig
from uaqe.optimizer.fusion_optimizer import FusionOptimizer
from uaqe.optimizer.graph_optimizer import GraphOptimizer

#: No structural passes applied; the unmodified input ``IMR``. Always
#: the first candidate evaluated, since every other candidate's
#: objective scores are normalized relative to it (see
#: ``uaqe.optimizer.optimizer.Optimizer.search``).
BASELINE = "baseline"

#: :class:`~uaqe.optimizer.graph_optimizer.GraphOptimizer` applied
#: alone.
GRAPH_CLEANUP = "graph_cleanup"

#: :class:`~uaqe.optimizer.graph_optimizer.GraphOptimizer` followed by
#: :class:`~uaqe.optimizer.fusion_optimizer.FusionOptimizer`.
GRAPH_CLEANUP_FUSION = "graph_cleanup+fusion"

#: The full candidate chain, in increasing order of aggressiveness.
#: :meth:`OptimizationPlanner.build_plan` truncates this to
#: ``OptimizationConfig.max_search_iterations`` entries.
_CANDIDATE_ORDER: Tuple[str, ...] = (BASELINE, GRAPH_CLEANUP, GRAPH_CLEANUP_FUSION)


@dataclass(frozen=True)
class OptimizationPlan:
    """Which candidate structural configurations a search should try.

    Attributes:
        candidate_labels: The candidate labels to evaluate, in
            evaluation order, always starting with :data:`BASELINE`.
        rationale: A human-readable explanation of how
            ``candidate_labels`` was bounded from
            ``OptimizationConfig.max_search_iterations``.
    """

    candidate_labels: List[str] = field(default_factory=lambda: [BASELINE])
    rationale: str = ""


class OptimizationPlanner:
    """Builds an :class:`OptimizationPlan` and materializes each of its
    candidate labels into a concrete ``IMR``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            planner operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        graph_optimizer: Optional[GraphOptimizer] = None,
        fusion_optimizer: Optional[FusionOptimizer] = None,
    ) -> None:
        """Initialize an ``OptimizationPlanner``.

        Args:
            logger: Optional structured logging sink.
            graph_optimizer: The structural cleanup pass to delegate
                to; a fresh
                :class:`~uaqe.optimizer.graph_optimizer.GraphOptimizer`
                is constructed if omitted.
            fusion_optimizer: The operator-fusion pass to delegate to;
                a fresh
                :class:`~uaqe.optimizer.fusion_optimizer.
                FusionOptimizer` is constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._graph_optimizer = graph_optimizer or GraphOptimizer(logger)
        self._fusion_optimizer = fusion_optimizer or FusionOptimizer(logger)

    def build_plan(self, config: OptimizationConfig) -> OptimizationPlan:
        """Build this run's :class:`OptimizationPlan` from ``config``.

        Args:
            config: The run's ``OptimizationConfig``.

        Returns:
            The finalized :class:`OptimizationPlan`, always including
            :data:`BASELINE` and never exceeding
            ``config.max_search_iterations`` candidates.
        """
        max_iterations = max(1, config.max_search_iterations)
        candidate_labels = list(_CANDIDATE_ORDER[:max_iterations])

        if max_iterations >= len(_CANDIDATE_ORDER):
            rationale = (
                f"max_search_iterations={config.max_search_iterations} "
                f"covers the full {len(_CANDIDATE_ORDER)}-candidate "
                "structural chain; every candidate will be evaluated."
            )
        else:
            rationale = (
                f"max_search_iterations={config.max_search_iterations} "
                f"bounds the search to the first {max_iterations} of "
                f"{len(_CANDIDATE_ORDER)} candidates: "
                f"{candidate_labels}."
            )

        if self.logger is not None:
            self.logger.info(
                "Optimization plan built.",
                candidate_labels=candidate_labels,
            )

        return OptimizationPlan(candidate_labels=candidate_labels, rationale=rationale)

    def materialize(self, imr: IMR, label: str) -> Tuple[IMR, List[str]]:
        """Apply the structural passes implied by ``label`` to ``imr``.

        Args:
            imr: The pre-optimization ``IMR`` to transform. Not
                modified.
            label: One of :data:`BASELINE`, :data:`GRAPH_CLEANUP`, or
                :data:`GRAPH_CLEANUP_FUSION` — normally an entry of a
                previously built :class:`OptimizationPlan`'s
                ``candidate_labels``.

        Returns:
            A ``(candidate_imr, applied_pass_notes)`` tuple: the
            resulting ``IMR`` for this candidate, and the human-readable
            notes describing every structural change applied (empty
            for :data:`BASELINE`).
        """
        if label == BASELINE:
            return imr, []

        graph_result = self._graph_optimizer.optimize(imr)
        candidate_imr = graph_result.optimized_imr
        notes = list(graph_result.notes)

        if label == GRAPH_CLEANUP_FUSION:
            fusion_result = self._fusion_optimizer.optimize(candidate_imr)
            candidate_imr = fusion_result.optimized_imr
            notes.extend(fusion_result.notes)

        return candidate_imr, notes
