"""Multi-Objective Optimization Scoring for UAQE.

Provides profile-dependent objective weighting and composite score calculation
across accuracy retention, model size reduction, and latency improvement.
Enforces the hard accuracy safety priority over compression gains.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from .accuracy_safety_policy import AccuracySafetyPolicy, AccuracyClassification


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights assigned to multi-objective optimization dimensions."""
    accuracy_weight: float
    size_weight: float
    latency_weight: float
    profile_name: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ObjectiveFunction:
    """Computes profile-driven composite optimization scores."""

    DEFAULT_PROFILES: Dict[str, ObjectiveWeights] = {
        "balanced": ObjectiveWeights(
            accuracy_weight=0.50,
            size_weight=0.25,
            latency_weight=0.25,
            profile_name="balanced"
        ),
        "accuracy_first": ObjectiveWeights(
            accuracy_weight=0.75,
            size_weight=0.15,
            latency_weight=0.10,
            profile_name="accuracy_first"
        ),
        "size_first": ObjectiveWeights(
            accuracy_weight=0.35,
            size_weight=0.45,
            latency_weight=0.20,
            profile_name="size_first"
        ),
        "latency_first": ObjectiveWeights(
            accuracy_weight=0.35,
            size_weight=0.20,
            latency_weight=0.45,
            profile_name="latency_first"
        ),
    }

    def __init__(self, profile: str = "balanced", custom_weights: Optional[ObjectiveWeights] = None):
        self.profile = profile.lower().strip()
        if custom_weights:
            self.weights = custom_weights
        else:
            self.weights = self.DEFAULT_PROFILES.get(self.profile, self.DEFAULT_PROFILES["balanced"])

    def get_manifest(self) -> Dict[str, Any]:
        """Return objective configuration manifest for plan recording."""
        max_loss_pp = AccuracySafetyPolicy.get_max_allowed_loss_pp(self.profile)
        return {
            "profile": self.profile,
            "weights": self.weights.to_dict(),
            "max_acceptable_accuracy_loss_pp": max_loss_pp,
            "formula": (
                f"score = ({self.weights.accuracy_weight} * acc_retention) + "
                f"({self.weights.size_weight} * size_reduction) + "
                f"({self.weights.latency_weight} * latency_speedup) [if safe]"
            ),
            "safety_priority_rule": (
                "Hard constraint: Candidates with accuracy loss > "
                f"{max_loss_pp} pp are penalised/ineligible when safe alternatives exist."
            )
        }

    def compute_score(
        self,
        candidate_accuracy: float,
        candidate_size_bytes: int,
        candidate_latency_ms: float,
        fp32_accuracy: float,
        fp32_size_bytes: int,
        fp32_latency_ms: float,
        safety_evaluation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute normalized composite score for a candidate.
        
        Args:
            candidate_accuracy: Top-1 accuracy of candidate (0.0 to 1.0).
            candidate_size_bytes: Serialized size of candidate in bytes.
            candidate_latency_ms: Mean inference latency in milliseconds.
            fp32_accuracy: FP32 baseline top-1 accuracy (0.0 to 1.0).
            fp32_size_bytes: FP32 baseline size in bytes.
            fp32_latency_ms: FP32 baseline latency in milliseconds.
            safety_evaluation: Optional precomputed safety evaluation dict.
            
        Returns:
            Dict containing composite_score, sub-scores, safety_status, and eligibility.
        """
        if safety_evaluation is None:
            safety_evaluation = AccuracySafetyPolicy.evaluate_safety(
                fp32_accuracy, candidate_accuracy, self.profile
            )

        # 1. Accuracy Retention Ratio (capped at 1.0 for standard retention score)
        acc_retention = float(candidate_accuracy / max(fp32_accuracy, 1e-6))
        acc_retention_clamped = min(max(acc_retention, 0.0), 1.2)

        # 2. Size Reduction Ratio (0.0 = no reduction, 0.75 = 75% reduction)
        size_reduction = float(1.0 - (candidate_size_bytes / max(fp32_size_bytes, 1)))
        size_reduction_clamped = max(size_reduction, 0.0)

        # 3. Latency Speedup Ratio (0.0 = no speedup, 0.40 = 40% faster)
        latency_reduction = float(1.0 - (candidate_latency_ms / max(fp32_latency_ms, 1e-4)))
        latency_reduction_clamped = max(latency_reduction, 0.0)

        # 4. Raw Weighted Score
        w = self.weights
        raw_score = (
            (w.accuracy_weight * acc_retention_clamped) +
            (w.size_weight * size_reduction_clamped) +
            (w.latency_weight * latency_reduction_clamped)
        )

        # 5. Accuracy Safety Multiplier
        # If safety constraint is not satisfied or candidate is CRITICAL,
        # heavily scale down score so safe candidates strictly dominate.
        is_satisfied = safety_evaluation["is_satisfied"]
        is_critical = safety_evaluation["is_critical"]

        if not is_satisfied or is_critical:
            # Harsh penalty multiplier: unsafe candidates receive max 0.20 scaling
            penalty_multiplier = 0.15
            final_score = round(raw_score * penalty_multiplier, 6)
            is_eligible = False
        else:
            final_score = round(raw_score, 6)
            is_eligible = True

        return {
            "composite_score": final_score,
            "raw_score": round(raw_score, 6),
            "accuracy_retention": round(acc_retention, 4),
            "size_reduction": round(size_reduction, 4),
            "latency_reduction": round(latency_reduction, 4),
            "is_eligible": is_eligible,
            "is_satisfied": is_satisfied,
            "safety_classification": safety_evaluation["classification"],
            "accuracy_loss_pp": safety_evaluation["accuracy_loss_pp"],
            "weights": w.to_dict()
        }
