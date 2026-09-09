"""Multi-objective optimization search — the locked
``uaqe.domain.optimization.optimization_engine`` contract
(``03_API_Specification.md`` §8.1), decomposed into this package.

``Optimizer`` is the sole orchestrating ``PipelineStage`` for this
package's *search* half (registered under context key ``"optimizer"``);
:class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer` is the
separate, second stage that runs after it
(``04_Data_Flow.md`` §7: ``OptimizationEngine`` -> ``MemoryOptimizer``),
finalizing activation-arena planning against the winning ``IMR`` this
stage selects. Per ``09_Architecture_Lock.md`` §13 rule 2, this stage
reads the *current* IMR from the latest stage in the model-transforming
chain (``compression_planner`` -> ``quantization_planner`` ->
``model_loader``, in that fallback order — mirroring
:class:`~uaqe.compression.compression_planner.CompressionPlanner`'s own
resolution), never a fixed context key.

Search algorithm (``search`` -- locked signature
``search(imr, config) -> OptimizationResult``, extended here to also
take the resolved ``HardwareProfile`` the objective scorers need):

1. :class:`~uaqe.optimizer.optimization_planner.OptimizationPlanner`
   builds an :class:`~uaqe.optimizer.optimization_planner.
   OptimizationPlan` naming which candidate structural configurations to
   try, bounded by ``OptimizationConfig.max_search_iterations``.
2. Each candidate is materialized (graph cleanup / fusion applied per
   its label) and scored on every objective in
   ``OptimizationConfig.objectives`` by
   :class:`~uaqe.optimizer.latency_optimizer.LatencyOptimizer` and
   :class:`~uaqe.optimizer.power_optimizer.PowerOptimizer`, plus an
   in-search memory proxy (see :meth:`Optimizer._estimate_memory_proxy`).
   Every ratio-based objective is normalized relative to the first
   (``"baseline"``) candidate, so a score of ``1.0`` always means "no
   better or worse than doing nothing."
3. Scores are combined into one ``aggregate_score`` per
   ``OptimizationConfig.objective_weights`` (or an unweighted average
   over ``objectives`` if no weights are given), and the highest-scoring
   candidate is selected. The non-dominated subset of all candidates is
   reported as the Pareto front.

Per ``01_Project_Architecture.md`` §16 and
``10_Module_Development_Guide.md`` §8, candidate evaluation is
"embarrassingly parallel" and may be parallelized internally (e.g. via
``concurrent.futures.ProcessPoolExecutor``) as a pure performance
optimization; this implementation evaluates candidates sequentially
since ``OptimizationConfig.max_search_iterations`` bounds the search to
at most three IMR-sized structural passes, and parallelizing three
candidates is not yet a measured bottleneck
(``11_Implementation_Rules.md`` §18's "optimize only after profiling").
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.common.value_objects import OptimizationConfig
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.optimizer.latency_optimizer import LatencyOptimizer
from uaqe.optimizer.optimization_planner import OptimizationPlanner
from uaqe.optimizer.optimization_result import CandidateConfiguration, OptimizationResult
from uaqe.optimizer.power_optimizer import PowerOptimizer

#: Objectives assumed when ``OptimizationConfig.objectives`` is empty.
_DEFAULT_OBJECTIVES: Tuple[str, ...] = ("accuracy", "latency", "memory", "power")

#: The context key this stage reads the pre-optimization ``IMR`` from
#: when present, per ``uaqe.compression``'s actual output key (see
#: ``CompressionPlanner``'s own module docstring).
_COMPRESSION_STAGE_NAME = "compression_planner"

#: Fallback context key when compression has not (yet) run, per
#: ``uaqe.quantization``'s actual output key.
_QUANTIZATION_STAGE_NAME = "quantization_planner"

#: Final fallback context key when neither compression nor quantization
#: has run.
_MODEL_LOADER_STAGE_NAME = "model_loader"

#: The ``HardwareProfile``-producing stage every objective scorer needs.
_HARDWARE_STAGE_NAME = "hardware_manager"


class Optimizer(PipelineStage):
    """Runs the multi-objective structural optimization search.

    Attributes:
        _logger: Structured logging sink.
        _config: The run's ``OptimizationConfig``, supplied via
            constructor injection.
        _planner: Builds and materializes candidate structural
            configurations.
        _latency_optimizer: Scores each candidate on the ``"latency"``
            objective.
        _power_optimizer: Scores each candidate on the ``"power"``
            objective.
    """

    def __init__(
        self,
        logger: ILogger,
        config: OptimizationConfig,
        planner: Optional[OptimizationPlanner] = None,
        latency_optimizer: Optional[LatencyOptimizer] = None,
        power_optimizer: Optional[PowerOptimizer] = None,
    ) -> None:
        """Initialize the ``Optimizer``.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            config: The run's ``OptimizationConfig``, sourced from
                ``IConfigRepository`` by the ``CompositionRoot``.
            planner: Builds/materializes candidate configurations; a
                fresh
                :class:`~uaqe.optimizer.optimization_planner.
                OptimizationPlanner` is constructed if omitted.
            latency_optimizer: Scores the ``"latency"`` objective; a
                fresh
                :class:`~uaqe.optimizer.latency_optimizer.
                LatencyOptimizer` is constructed if omitted.
            power_optimizer: Scores the ``"power"`` objective; a fresh
                :class:`~uaqe.optimizer.power_optimizer.PowerOptimizer`
                is constructed if omitted.
        """
        self._logger = logger
        self._config = config
        self._planner = planner or OptimizationPlanner(logger)
        self._latency_optimizer = latency_optimizer or LatencyOptimizer(logger)
        self._power_optimizer = power_optimizer or PowerOptimizer(logger)

    def execute(self, context: PipelineContext) -> StageResult:
        """Run this run's multi-objective optimization search.

        Reads the current ``IMR`` per the resolution order documented
        in the module docstring, and the target ``HardwareProfile``
        from ``"hardware_manager"``.

        Args:
            context: The current run's pipeline context.

        Returns:
            A ``StageResult`` whose ``payload`` is a tuple of
            ``(OptimizationResult, IMR)`` — the finalized search result
            and the winning candidate's optimized ``IMR``.
        """
        start = time.monotonic()
        imr = self._resolve_current_imr(context)
        profile = context.get(_HARDWARE_STAGE_NAME).payload

        result, optimized_imr = self.search(imr, profile, self._config)

        duration_ms = (time.monotonic() - start) * 1000.0
        self._logger.info(
            "Multi-objective optimization search complete.",
            stage_name=self.name(),
            selected=(
                result.selected_candidate.name
                if result.selected_candidate is not None
                else None
            ),
            candidate_count=len(result.candidates),
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=(result, optimized_imr),
            duration_ms=duration_ms,
        )

    def search(
        self, imr: IMR, profile: HardwareProfile, config: OptimizationConfig
    ) -> Tuple[OptimizationResult, IMR]:
        """Search over candidate structural configurations of ``imr``.

        Args:
            imr: The pre-optimization model.
            profile: The resolved deployment target.
            config: The run's ``OptimizationConfig``.

        Returns:
            A ``(result, optimized_imr)`` tuple: the finalized
            :class:`~uaqe.optimizer.optimization_result.
            OptimizationResult` and the selected candidate's ``IMR``.
        """
        plan = self._planner.build_plan(config)
        objectives = list(config.objectives) if config.objectives else list(
            _DEFAULT_OBJECTIVES
        )

        candidates: List[CandidateConfiguration] = []
        baseline_latency_ms: Optional[float] = None
        baseline_energy_mj: Optional[float] = None
        baseline_memory_bytes: Optional[int] = None

        for label in plan.candidate_labels:
            candidate_imr, applied_passes = self._planner.materialize(imr, label)

            latency_score = self._latency_optimizer.evaluate(candidate_imr, profile)
            power_score = self._power_optimizer.evaluate(
                candidate_imr, profile, latency_score.estimate
            )
            memory_proxy_bytes = self._estimate_memory_proxy(candidate_imr)

            if baseline_latency_ms is None:
                baseline_latency_ms = latency_score.estimate.total_latency_ms
                baseline_energy_mj = power_score.estimate.energy_per_inference_mj
                baseline_memory_bytes = memory_proxy_bytes

            objective_scores = self._score_objectives(
                objectives,
                latency_ms=latency_score.estimate.total_latency_ms,
                energy_mj=power_score.estimate.energy_per_inference_mj,
                memory_bytes=memory_proxy_bytes,
                baseline_latency_ms=baseline_latency_ms,
                baseline_energy_mj=baseline_energy_mj,
                baseline_memory_bytes=baseline_memory_bytes,
            )
            aggregate_score = self._aggregate(
                objectives, objective_scores, config.objective_weights
            )

            candidates.append(
                CandidateConfiguration(
                    name=label,
                    applied_passes=applied_passes,
                    latency_estimate=latency_score.estimate,
                    power_estimate=power_score.estimate,
                    memory_proxy_bytes=memory_proxy_bytes,
                    objective_scores=objective_scores,
                    aggregate_score=aggregate_score,
                    imr=candidate_imr,
                )
            )

        best = max(candidates, key=lambda candidate: candidate.aggregate_score)
        pareto_front = self._pareto_front(candidates, objectives)

        result = OptimizationResult(
            selected_configuration=best.to_dict(),
            pareto_front=[candidate.to_dict() for candidate in pareto_front],
            objective_scores=best.objective_scores,
            candidates=candidates,
            selected_candidate=best,
            rationale=(
                f"{plan.rationale} Selected {best.name!r} "
                f"(aggregate_score={best.aggregate_score:.4f}) out of "
                f"{len(candidates)} candidate(s) evaluated."
            ),
        )
        return result, best.imr

    def _estimate_memory_proxy(self, imr: IMR) -> int:
        """Estimate a fast, order-of-magnitude ``"memory"`` objective
        proxy for one candidate.

        Uses total static parameter-tensor byte size rather than the
        exact activation-arena plan
        :class:`~uaqe.optimizer.memory_optimizer.MemoryOptimizer`
        computes, since that plan's hard-ceiling check
        (``05_Hardware_Profile_Spec.md`` §6 rule 4) is reserved for the
        single final, selected candidate — running it speculatively for
        every candidate here would duplicate that check for candidates
        that are never selected.

        Args:
            imr: The candidate model to measure.

        Returns:
            The total ``len(tensor.data)`` across every layer's
            parameters.
        """
        return sum(
            len(tensor.data)
            for layer in imr.layers
            for tensor in layer.parameters.values()
        )

    def _score_objectives(
        self,
        objectives: List[str],
        *,
        latency_ms: float,
        energy_mj: float,
        memory_bytes: int,
        baseline_latency_ms: float,
        baseline_energy_mj: float,
        baseline_memory_bytes: int,
    ) -> Dict[str, float]:
        """Normalize one candidate's raw measurements into per-objective
        scores, each relative to the baseline candidate.

        Args:
            objectives: The objective names to produce a score for.
            latency_ms: This candidate's total estimated latency.
            energy_mj: This candidate's estimated per-inference energy.
            memory_bytes: This candidate's memory proxy.
            baseline_latency_ms: The ``"baseline"`` candidate's total
                estimated latency.
            baseline_energy_mj: The ``"baseline"`` candidate's estimated
                per-inference energy.
            baseline_memory_bytes: The ``"baseline"`` candidate's memory
                proxy.

        Returns:
            A mapping of objective name to score, higher is better,
            ``1.0`` meaning "same as baseline" for every ratio-based
            objective.
        """
        scores: Dict[str, float] = {}
        for objective in objectives:
            if objective == "latency":
                scores["latency"] = baseline_latency_ms / max(latency_ms, 1e-9)
            elif objective == "power":
                scores["power"] = baseline_energy_mj / max(energy_mj, 1e-9)
            elif objective == "memory":
                scores["memory"] = baseline_memory_bytes / max(memory_bytes, 1)
            elif objective == "accuracy":
                # Not evaluated by this stage: OptimizationConfig carries
                # no labeled dataset path, and uaqe.optimizer must not
                # depend on the downstream uaqe.domain.evaluation
                # package. A neutral score keeps this objective from
                # skewing aggregate_score toward or away from any
                # candidate; see uaqe.domain.evaluation.Evaluator for
                # the actual accuracy-delta measurement, run later in
                # the pipeline.
                scores["accuracy"] = 1.0
            else:
                scores[objective] = 1.0
        return scores

    def _aggregate(
        self,
        objectives: List[str],
        objective_scores: Dict[str, float],
        objective_weights: Dict[str, float],
    ) -> float:
        """Combine per-objective scores into one ranking score.

        Args:
            objectives: The objective names scored for this candidate.
            objective_scores: This candidate's per-objective scores,
                from :meth:`_score_objectives`.
            objective_weights: ``OptimizationConfig.objective_weights``;
                an empty mapping falls back to an unweighted average.

        Returns:
            The candidate's aggregate score.
        """
        if not objectives:
            return 1.0
        if objective_weights:
            total_weight = sum(
                objective_weights.get(objective, 0.0) for objective in objectives
            )
            if total_weight > 0:
                return (
                    sum(
                        objective_scores.get(objective, 1.0)
                        * objective_weights.get(objective, 0.0)
                        for objective in objectives
                    )
                    / total_weight
                )
        return sum(objective_scores.values()) / len(objectives)

    def _pareto_front(
        self, candidates: List[CandidateConfiguration], objectives: List[str]
    ) -> List[CandidateConfiguration]:
        """Compute the non-dominated subset of ``candidates``.

        Args:
            candidates: Every candidate evaluated this search.
            objectives: The objective names to compare candidates on.

        Returns:
            Every candidate not dominated by another (a candidate ``a``
            dominates ``b`` if ``a`` scores at least as well on every
            objective and strictly better on at least one).
        """
        front: List[CandidateConfiguration] = []
        for candidate in candidates:
            if not any(
                other is not candidate
                and self._dominates(other, candidate, objectives)
                for other in candidates
            ):
                front.append(candidate)
        return front

    def _dominates(
        self,
        a: CandidateConfiguration,
        b: CandidateConfiguration,
        objectives: List[str],
    ) -> bool:
        """Report whether candidate ``a`` Pareto-dominates ``b``.

        Args:
            a: The candidate to test as the dominator.
            b: The candidate to test as dominated.
            objectives: The objective names to compare on.

        Returns:
            ``True`` if ``a`` scores at least as well as ``b`` on every
            objective and strictly better on at least one.
        """
        at_least_as_good = all(
            a.objective_scores.get(objective, 0.0)
            >= b.objective_scores.get(objective, 0.0)
            for objective in objectives
        )
        strictly_better = any(
            a.objective_scores.get(objective, 0.0)
            > b.objective_scores.get(objective, 0.0)
            for objective in objectives
        )
        return at_least_as_good and strictly_better

    def _resolve_current_imr(self, context: PipelineContext) -> IMR:
        """Resolve the current, pre-optimization ``IMR`` from
        ``context``.

        Args:
            context: The current run's pipeline context.

        Returns:
            The ``IMR`` produced by ``"compression_planner"`` if that
            stage has run, else by ``"quantization_planner"``, else by
            ``"model_loader"``.

        Raises:
            KeyError: If none of the three stages has been recorded in
                ``context`` (propagated from ``PipelineContext.get``).
        """
        for stage_name in (_COMPRESSION_STAGE_NAME, _QUANTIZATION_STAGE_NAME):
            if context.has(stage_name):
                payload = context.get(stage_name).payload
                return payload[1] if isinstance(payload, tuple) else payload
        return context.get(_MODEL_LOADER_STAGE_NAME).payload
