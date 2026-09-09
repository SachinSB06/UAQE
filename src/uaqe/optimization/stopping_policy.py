"""Stopping Policy for UAQE Autonomous Optimization.

Determines when the autonomous optimization controller should terminate search,
evaluating budget constraints, accuracy safety requirements, and convergence.
"""

from __future__ import annotations

import enum
from typing import Dict, List, Any, Optional


class StoppingReason(str, enum.Enum):
    """Reason for terminating autonomous optimization search."""
    TARGET_CONSTRAINTS_SATISFIED = "TARGET_CONSTRAINTS_SATISFIED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NO_LEGAL_CANDIDATES_REMAIN = "NO_LEGAL_CANDIDATES_REMAIN"
    PARETO_CONVERGENCE = "PARETO_CONVERGENCE"
    EARLY_STOPPING_CRITERIA_MET = "EARLY_STOPPING_CRITERIA_MET"
    UNSATISFIED_SEARCH_EXHAUSTED = "UNSATISFIED_SEARCH_EXHAUSTED"


class StoppingPolicy:
    """Evaluates search termination conditions."""

    def __init__(
        self,
        max_candidates: int = 10,
        target_accuracy_loss_pp: float = 1.0,
        target_size_reduction: float = 0.70,
        enable_early_stopping: bool = True
    ):
        self.max_candidates = max_candidates
        self.target_accuracy_loss_pp = target_accuracy_loss_pp
        self.target_size_reduction = target_size_reduction
        self.enable_early_stopping = enable_early_stopping

    def evaluate(
        self,
        candidate_count: int,
        history: List[Dict[str, Any]],
        has_more_candidates: bool,
        current_best_safe: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, StoppingReason, str]:
        """Evaluate whether the search should stop.
        
        Returns:
            Tuple of (should_stop, stopping_reason, description).
        """
        # 1. Check if no more legal candidates can be generated
        if not has_more_candidates:
            if current_best_safe is not None:
                return (
                    True,
                    StoppingReason.NO_LEGAL_CANDIDATES_REMAIN,
                    "Candidate search space fully explored. Best safe candidate selected."
                )
            else:
                return (
                    True,
                    StoppingReason.UNSATISFIED_SEARCH_EXHAUSTED,
                    "Search space exhausted without satisfying accuracy constraint."
                )

        # 2. Check early stopping if target criteria are met (e.g. EXCELLENT accuracy + high compression)
        if self.enable_early_stopping and current_best_safe is not None:
            acc_loss_pp = current_best_safe.get("accuracy_loss_pp", 99.0)
            size_red = current_best_safe.get("size_reduction", 0.0)
            classification = current_best_safe.get("safety_classification", "")

            if classification == "EXCELLENT" and size_red >= self.target_size_reduction:
                return (
                    True,
                    StoppingReason.TARGET_CONSTRAINTS_SATISFIED,
                    (
                        f"Target constraints satisfied: EXCELLENT accuracy (loss={acc_loss_pp:.2f} pp <= "
                        f"{self.target_accuracy_loss_pp} pp) and substantial size reduction ({size_red*100:.1f}%)."
                    )
                )

        # 3. Check search budget
        if candidate_count >= self.max_candidates:
            return (
                True,
                StoppingReason.BUDGET_EXHAUSTED,
                f"Search budget of {self.max_candidates} candidates reached."
            )

        return (False, StoppingReason.TARGET_CONSTRAINTS_SATISFIED, "Continuing search...")
