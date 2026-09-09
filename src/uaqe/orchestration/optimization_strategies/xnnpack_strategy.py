"""
src/uaqe/orchestration/optimization_strategies/xnnpack_strategy.py
XNNPACK-Compatible INT8 Orchestration Strategy for UAQE.
"""

from __future__ import annotations

import os
import json
import shutil
import hashlib
from typing import Dict, Any

from .base_strategy import BaseOptimizationStrategy
from uaqe.optimization.strategies.xnnpack_int8 import (
    XNNPACKCompatibleINT8Strategy,
    STRATEGY_NAME
)


class XNNPACKStrategy(BaseOptimizationStrategy):
    """Strategy wrapper for XNNPACK-compatible INT8 execution in UAQE Orchestration."""

    def __init__(self):
        super().__init__(STRATEGY_NAME)

    def can_handle(
        self,
        model_descriptor: Dict[str, Any],
        dataset_descriptor: Dict[str, Any],
        optimization_plan: Dict[str, Any]
    ) -> bool:
        """Check if workload is eligible for XNNPACK strategy."""
        is_eligible, _ = XNNPACKCompatibleINT8Strategy.check_eligibility(
            model_descriptor=model_descriptor
        )
        return is_eligible

    def execute(self, job_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the XNNPACK strategy, generating isolated artifact and metadata."""
        job_id = job_context["job_id"]
        job_dir = job_context["job_dir"]
        final_dir = os.path.join(job_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        candidate_artifact_dir = os.path.join(job_dir, "candidates", "cand_xnnpack")
        model_path = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(
            candidate_dir=candidate_artifact_dir,
            job_context=job_context
        )

        final_model_file = os.path.join(final_dir, "optimized_model.tflite")
        shutil.copy2(model_path, final_model_file)

        return {
            "job_id": job_id,
            "status": "COMPLETED",
            "strategy": STRATEGY_NAME,
            "final_model_path": final_model_file
        }
