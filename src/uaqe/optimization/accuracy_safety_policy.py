"""Accuracy Safety Policy for UAQE Autonomous Optimization.

Defines hard quality thresholds for accuracy preservation across optimization profiles:
- EXCELLENT: accuracy loss <= 1.0 pp
- ACCEPTABLE: 1.0 < accuracy loss <= 4.0 pp
- CRITICAL: accuracy loss > 4.0 pp

Enforces task-aware baseline validity guards and profile-specific safety boundaries.
"""

from __future__ import annotations

import enum
from typing import Dict, Any, Tuple, Optional


class AccuracyClassification(str, enum.Enum):
    """Classification of candidate accuracy loss relative to FP32 baseline."""
    EXCELLENT = "EXCELLENT"
    ACCEPTABLE = "ACCEPTABLE"
    CRITICAL = "CRITICAL"
    INVALID_BASELINE = "INVALID_BASELINE"


class BaselineStatus(str, enum.Enum):
    """Status of reference FP32 baseline."""
    VALID = "VALID"
    INVALID_BASELINE = "INVALID_BASELINE"
    PENDING = "PENDING"
    UNAVAILABLE = "UNAVAILABLE"


class AccuracySafetyPolicy:
    """Evaluates and enforces accuracy safety constraints and baseline validity."""

    # Default task-aware baseline validity policy configuration
    DEFAULT_BASELINE_VALIDITY_POLICY: Dict[str, Any] = {
        "version": "1.0",
        "classification": {
            "min_accuracy_fraction": 0.25,  # 25.0% minimum baseline accuracy (2.5x random chance on 10 classes)
            "enabled": True
        }
    }

    # Maximum acceptable accuracy loss (in percentage points) per profile
    PROFILE_THRESHOLDS: Dict[str, float] = {
        "balanced": 4.0,
        "accuracy_first": 1.0,
        "size_first": 4.0,
        "latency_first": 4.0
    }

    @classmethod
    def validate_baseline(
        cls,
        fp32_accuracy: float,
        task_type: str = "classification",
        class_count: int = 10,
        policy: Optional[Dict[str, Any]] = None,
        adaptation_status: str = "NOT_REQUIRED",
        weight_source: str = "ORIGINAL_MODEL"
    ) -> Dict[str, Any]:
        """Validate whether an FP32 baseline is sufficiently accurate to serve as reference."""
        active_policy = policy or cls.DEFAULT_BASELINE_VALIDITY_POLICY
        policy_version = active_policy.get("version", "1.0")
        task_cfg = active_policy.get(task_type, active_policy.get("classification", {}))

        min_threshold = float(task_cfg.get("min_accuracy_fraction", 0.25))
        enabled = bool(task_cfg.get("enabled", True))

        # Normalize accuracy to 0.0-1.0
        acc_norm = fp32_accuracy / 100.0 if fp32_accuracy > 1.0 else fp32_accuracy

        # Check for untrained random head without training
        if adaptation_status == "RANDOM_HEAD" and weight_source != "TRAINED_AFTER_INITIALIZATION":
            return {
                "is_valid": False,
                "baseline_status": BaselineStatus.INVALID_BASELINE.value,
                "threshold_fraction": min_threshold,
                "threshold_percent": round(min_threshold * 100.0, 2),
                "policy_version": policy_version,
                "reason": (
                    f"Model classifier head is untrained (adaptation_status=RANDOM_HEAD, weight_source={weight_source}). "
                    f"FP32 baseline accuracy ({acc_norm*100:.2f}%) reflects untrained random initialization and is invalid."
                )
            }

        if enabled and acc_norm < min_threshold:
            return {
                "is_valid": False,
                "baseline_status": BaselineStatus.INVALID_BASELINE.value,
                "threshold_fraction": min_threshold,
                "threshold_percent": round(min_threshold * 100.0, 2),
                "policy_version": policy_version,
                "reason": (
                    f"FP32 baseline accuracy ({acc_norm*100:.2f}%) is below the configured validity threshold "
                    f"({min_threshold*100:.1f}%). The model appears untrained or incompatible with this dataset."
                )
            }

        return {
            "is_valid": True,
            "baseline_status": BaselineStatus.VALID.value,
            "threshold_fraction": min_threshold,
            "threshold_percent": round(min_threshold * 100.0, 2),
            "policy_version": policy_version,
            "reason": f"FP32 baseline accuracy ({acc_norm*100:.2f}%) satisfies validity threshold ({min_threshold*100:.1f}%)."
        }

    @staticmethod
    def calculate_loss_pp(fp32_accuracy: float, candidate_accuracy: float) -> float:
        """Calculate accuracy loss in percentage points (baseline - candidate).
        
        Example: 75.0% baseline, 74.6% candidate -> +0.40 pp loss
        """
        fp32_pct = fp32_accuracy * 100.0 if fp32_accuracy <= 1.0 else fp32_accuracy
        cand_pct = candidate_accuracy * 100.0 if candidate_accuracy <= 1.0 else candidate_accuracy
        return round(float(fp32_pct - cand_pct), 4)

    @staticmethod
    def calculate_delta_pp(fp32_accuracy: float, candidate_accuracy: float) -> float:
        """Calculate accuracy delta in percentage points (candidate - baseline).
        
        Example: 75.0% baseline, 75.5% candidate -> +0.50 pp delta (improvement)
        Example: 75.0% baseline, 74.6% candidate -> -0.40 pp delta (loss)
        """
        fp32_pct = fp32_accuracy * 100.0 if fp32_accuracy <= 1.0 else fp32_accuracy
        cand_pct = candidate_accuracy * 100.0 if candidate_accuracy <= 1.0 else candidate_accuracy
        return round(float(cand_pct - fp32_pct), 4)

    @staticmethod
    def calculate_latency_change_pct(baseline_latency_ms: float, candidate_latency_ms: float) -> float:
        """Calculate latency change percentage.
        
        Formula: ((baseline - candidate) / baseline) * 100
        Positive = speedup (e.g. 100ms -> 80ms = +20.00% speedup)
        Negative = slower (e.g. 100ms -> 125ms = -25.00% slower)
        """
        if baseline_latency_ms <= 0:
            return 0.0
        speedup = ((baseline_latency_ms - candidate_latency_ms) / baseline_latency_ms) * 100.0
        return round(float(speedup), 2)

    @classmethod
    def classify(cls, accuracy_loss_pp: float) -> AccuracyClassification:
        """Classify accuracy loss into EXCELLENT, ACCEPTABLE, or CRITICAL.
        
        Rules:
            loss_pp <= 1.0       -> EXCELLENT
            1.0 < loss_pp <= 4.0 -> ACCEPTABLE
            loss_pp > 4.0        -> CRITICAL
        """
        if accuracy_loss_pp <= 1.0:
            return AccuracyClassification.EXCELLENT
        elif accuracy_loss_pp <= 4.0:
            return AccuracyClassification.ACCEPTABLE
        else:
            return AccuracyClassification.CRITICAL

    @classmethod
    def get_max_allowed_loss_pp(cls, profile: str) -> float:
        """Get the maximum acceptable accuracy loss in percentage points for a profile."""
        norm_prof = profile.lower().strip()
        return cls.PROFILE_THRESHOLDS.get(norm_prof, 4.0)

    @classmethod
    def evaluate_safety(
        cls,
        fp32_accuracy: float,
        candidate_accuracy: float,
        profile: str = "balanced",
        baseline_status: str = "VALID"
    ) -> Dict[str, Any]:
        """Perform comprehensive safety evaluation for a candidate."""
        loss_pp = cls.calculate_loss_pp(fp32_accuracy, candidate_accuracy)
        delta_pp = cls.calculate_delta_pp(fp32_accuracy, candidate_accuracy)
        max_allowed = cls.get_max_allowed_loss_pp(profile)

        # Precedence: INVALID_BASELINE overrides all candidate classifications
        if baseline_status == BaselineStatus.INVALID_BASELINE.value:
            return {
                "accuracy_loss_pp": loss_pp,
                "accuracy_delta_pp": delta_pp,
                "classification": AccuracyClassification.INVALID_BASELINE.value,
                "safety_status": AccuracyClassification.INVALID_BASELINE.value,
                "max_allowed_loss_pp": max_allowed,
                "is_satisfied": False,
                "is_critical": True,
                "profile": profile,
                "baseline_status": baseline_status
            }

        classification = cls.classify(loss_pp)
        is_satisfied = bool(loss_pp <= max_allowed)

        return {
            "accuracy_loss_pp": loss_pp,
            "accuracy_delta_pp": delta_pp,
            "classification": classification.value,
            "safety_status": classification.value,
            "max_allowed_loss_pp": max_allowed,
            "is_satisfied": is_satisfied,
            "is_critical": classification == AccuracyClassification.CRITICAL,
            "profile": profile,
            "baseline_status": baseline_status
        }
