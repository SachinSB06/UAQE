"""Multi-objective candidate ranking and Pareto frontier analysis for Phase B.

Provides transparent, explainable ranking of mixed-precision candidate models
across four key dimensions:
- Accuracy recovery (relative to FP32 and Full INT8)
- Model size compression (relative to FP32)
- Host inference latency
- INT8 compute coverage retention (%)

Supports four configurable deployment objectives:
- `balanced` (default)
- `accuracy_first`
- `size_first`
- `latency_first`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CandidateMetrics:
    """Standardized metrics container for a single candidate model."""

    experiment_id: str
    name: str
    policy_name: str
    accuracy: float
    accuracy_delta_vs_int8: float
    accuracy_delta_vs_fp32: float
    cosine_similarity: float
    mae: float
    rmse: float
    model_size_bytes: int
    model_size_mb: float
    latency_ms: float
    int8_tensor_count: int
    fp16_tensor_count: int
    fp32_tensor_count: int
    int32_tensor_count: int
    int8_coverage_percent: float
    selected_layers: List[str] = field(default_factory=list)
    stability_status: str = "PENDING"
    status: str = "COMPLETED"


@dataclass
class CandidateScore:
    """Candidate with calculated objective scores and Pareto status."""

    candidate: CandidateMetrics
    composite_score: float
    rank: int
    is_pareto_optimal: bool
    objective: str
    sub_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert score to dictionary."""
        d = asdict(self)
        return d


class CandidateRanker:
    """Ranks mixed-precision candidates transparently using multi-criteria decision analysis."""

    OBJECTIVE_WEIGHTS = {
        "balanced": {
            "accuracy": 0.40,
            "size": 0.25,
            "latency": 0.20,
            "int8_coverage": 0.15,
        },
        "accuracy_first": {
            "accuracy": 0.70,
            "size": 0.10,
            "latency": 0.10,
            "int8_coverage": 0.10,
        },
        "size_first": {
            "accuracy": 0.25,
            "size": 0.50,
            "latency": 0.15,
            "int8_coverage": 0.10,
        },
        "latency_first": {
            "accuracy": 0.25,
            "size": 0.15,
            "latency": 0.50,
            "int8_coverage": 0.10,
        },
    }

    @classmethod
    def rank_candidates(
        cls,
        candidates: List[CandidateMetrics],
        fp32_accuracy: float = 37.16,
        int8_baseline_accuracy: float = 21.96,
        fp32_size_bytes: int = 6428784,
        objective: str = "balanced",
    ) -> List[CandidateScore]:
        """Rank a list of candidates according to the specified objective.

        Args:
            candidates: List of candidate metrics.
            fp32_accuracy: Reference FP32 accuracy.
            int8_baseline_accuracy: Reference Full INT8 baseline accuracy.
            fp32_size_bytes: Reference FP32 model size in bytes.
            objective: One of 'balanced', 'accuracy_first', 'size_first', 'latency_first'.

        Returns:
            Sorted list of CandidateScore objects (rank 1 is best).
        """
        if not candidates:
            return []

        weights = cls.OBJECTIVE_WEIGHTS.get(objective, cls.OBJECTIVE_WEIGHTS["balanced"])

        # Find min/max bounds for normalization
        accuracies = [c.accuracy for c in candidates]
        sizes = [c.model_size_bytes for c in candidates]
        latencies = [c.latency_ms for c in candidates]
        int8_covs = [c.int8_coverage_percent for c in candidates]

        min_acc, max_acc = min(accuracies), max(max(accuracies), fp32_accuracy)
        min_size, max_size = min(sizes), max(max(sizes), fp32_size_bytes)
        min_lat, max_lat = min(latencies), max(latencies) if max(latencies) > min(latencies) else (min(latencies) + 1.0)
        min_cov, max_cov = 0.0, 100.0

        pareto_flags = cls.compute_pareto_frontier(candidates)

        scored_candidates: List[CandidateScore] = []
        for c, is_pareto in zip(candidates, pareto_flags):
            # Normalization (higher is better for all components)
            acc_score = (c.accuracy - min_acc) / (max_acc - min_acc) if max_acc > min_acc else 1.0
            size_score = (max_size - c.model_size_bytes) / (max_size - min_size) if max_size > min_size else 1.0
            lat_score = (max_lat - c.latency_ms) / (max_lat - min_lat) if max_lat > min_lat else 1.0
            cov_score = c.int8_coverage_percent / 100.0

            composite = (
                weights["accuracy"] * acc_score
                + weights["size"] * size_score
                + weights["latency"] * lat_score
                + weights["int8_coverage"] * cov_score
            )

            sub = {
                "acc_norm": round(acc_score, 4),
                "size_norm": round(size_score, 4),
                "lat_norm": round(lat_score, 4),
                "cov_norm": round(cov_score, 4),
            }

            scored_candidates.append(
                CandidateScore(
                    candidate=c,
                    composite_score=round(float(composite), 4),
                    rank=0,
                    is_pareto_optimal=is_pareto,
                    objective=objective,
                    sub_scores=sub,
                )
            )

        # Sort descending by composite score
        scored_candidates.sort(key=lambda s: s.composite_score, reverse=True)
        for r, s in enumerate(scored_candidates, start=1):
            s.rank = r

        return scored_candidates

    @classmethod
    def compute_pareto_frontier(cls, candidates: List[CandidateMetrics]) -> List[bool]:
        """Identify candidates that are non-dominated across (accuracy, -size, -latency, int8_coverage)."""
        is_pareto = [True] * len(candidates)

        # Dimension vector: higher is better
        # (accuracy, -size, -latency, int8_coverage)
        for i, c1 in enumerate(candidates):
            v1 = (c1.accuracy, -c1.model_size_bytes, -c1.latency_ms, c1.int8_coverage_percent)
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                v2 = (c2.accuracy, -c2.model_size_bytes, -c2.latency_ms, c2.int8_coverage_percent)
                # c2 dominates c1 if v2 >= v1 in all dims and strictly greater in at least one
                if all(x >= y for x, y in zip(v2, v1)) and any(x > y for x, y in zip(v2, v1)):
                    is_pareto[i] = False
                    break

        return is_pareto
