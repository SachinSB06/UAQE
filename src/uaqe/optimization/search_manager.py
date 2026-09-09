"""Search Manager and Pareto Frontier Tracking for UAQE Autonomous Optimization.

Maintains candidate history, deduplication, Pareto frontier over (accuracy, size, latency),
and selects the top-scoring candidate satisfying accuracy safety constraints.
"""

from __future__ import annotations

import os
import csv
import json
from typing import Dict, List, Any, Optional, Tuple

from .candidate_evaluator import CandidateEvaluationResult


class SearchManager:
    """Manages search history, Pareto frontier, and candidate selection."""

    def __init__(self, job_dir: str):
        self.job_dir = job_dir
        self.history: List[CandidateEvaluationResult] = []

    def record_candidate(self, result: CandidateEvaluationResult) -> None:
        """Record an evaluated candidate result in history."""
        self.history.append(result)

    def compute_pareto_frontier(self) -> List[CandidateEvaluationResult]:
        """Compute the Pareto frontier over (accuracy, size, latency).
        
        Objectives:
        - Accuracy: maximize
        - Size: minimize
        - Latency: minimize
        """
        frontier: List[CandidateEvaluationResult] = []

        for cand_a in self.history:
            is_dominated = False
            for cand_b in self.history:
                if cand_a.candidate_id == cand_b.candidate_id:
                    continue

                # Check if B dominates A:
                # B is >= A in accuracy, <= A in size, <= A in latency, and strictly better in >= 1
                b_acc = cand_b.top1_accuracy
                a_acc = cand_a.top1_accuracy
                b_size = cand_b.model_size_bytes
                a_size = cand_a.model_size_bytes
                b_lat = cand_b.latency_mean_ms
                a_lat = cand_a.latency_mean_ms

                if (b_acc >= a_acc and b_size <= a_size and b_lat <= a_lat) and (
                    b_acc > a_acc or b_size < a_size or b_lat < a_lat
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                frontier.append(cand_a)

        return frontier

    def select_best_candidate(self) -> Tuple[Optional[CandidateEvaluationResult], bool]:
        """Select the best candidate adhering to the accuracy safety hierarchy.
        
        Returns:
            Tuple of (best_candidate, constraint_satisfied_bool).
        """
        if not self.history:
            return None, False

        # 1. Filter safe / satisfied candidates
        safe_candidates = [c for c in self.history if c.is_satisfied and not c.is_critical]

        if safe_candidates:
            # Sort by composite score descending
            safe_sorted = sorted(safe_candidates, key=lambda c: c.composite_score, reverse=True)
            return safe_sorted[0], True

        # 2. If NO candidate satisfies safety constraint, select the best candidate with least accuracy loss
        fallback_sorted = sorted(self.history, key=lambda c: (c.accuracy_loss_pp, -c.composite_score))
        return fallback_sorted[0], False

    def export_artifacts(self) -> Dict[str, str]:
        """Export optimization_history.json, pareto_frontier.json, and candidate_results.csv."""
        os.makedirs(self.job_dir, exist_ok=True)

        history_path = os.path.join(self.job_dir, "optimization_history.json")
        pareto_path = os.path.join(self.job_dir, "pareto_frontier.json")
        csv_path = os.path.join(self.job_dir, "candidate_results.csv")

        # 1. Export History JSON
        history_data = [c.to_dict() for c in self.history]
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)

        # 2. Export Pareto Frontier JSON
        pareto_candidates = self.compute_pareto_frontier()
        pareto_data = [c.to_dict() for c in pareto_candidates]
        with open(pareto_path, "w", encoding="utf-8") as f:
            json.dump(pareto_data, f, indent=2)

        # 3. Export Candidate Results CSV
        fieldnames = [
            "candidate_id",
            "candidate_name",
            "strategy_type",
            "top1_accuracy",
            "accuracy_loss_pp",
            "safety_classification",
            "is_satisfied",
            "model_size_bytes",
            "size_reduction_pct",
            "latency_mean_ms",
            "latency_reduction_pct",
            "throughput_ips",
            "prediction_agreement_pct",
            "composite_score",
            "execution_duration_sec"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.history:
                writer.writerow({
                    "candidate_id": c.candidate_id,
                    "candidate_name": c.candidate_name,
                    "strategy_type": c.strategy_type,
                    "top1_accuracy": round(c.top1_accuracy, 4),
                    "accuracy_loss_pp": round(c.accuracy_loss_pp, 2),
                    "safety_classification": c.safety_classification,
                    "is_satisfied": c.is_satisfied,
                    "model_size_bytes": c.model_size_bytes,
                    "size_reduction_pct": round(c.size_reduction * 100.0, 2),
                    "latency_mean_ms": round(c.latency_mean_ms, 3),
                    "latency_reduction_pct": round(c.latency_reduction * 100.0, 2),
                    "throughput_ips": round(c.throughput_ips, 2),
                    "prediction_agreement_pct": round(c.prediction_agreement, 2),
                    "composite_score": round(c.composite_score, 6),
                    "execution_duration_sec": c.execution_duration_sec
                })

        return {
            "optimization_history_json": history_path,
            "pareto_frontier_json": pareto_path,
            "candidate_results_csv": csv_path
        }
