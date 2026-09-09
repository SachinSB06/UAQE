"""Base Optimization Strategy Interface for UAQE.

All pluggable optimization pipelines (e.g. ResNet ONNX PTQ, MobileNetV3 Adaptive,
Transformer dynamic quantization, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseOptimizationStrategy(ABC):
    """Abstract base class for capability-driven optimization strategies."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def can_handle(
        self,
        model_descriptor: Dict[str, Any],
        dataset_descriptor: Dict[str, Any],
        optimization_plan: Dict[str, Any]
    ) -> bool:
        """Probe whether this strategy can optimize the given model and dataset under the plan.

        Args:
            model_descriptor: Model metadata and architecture descriptor.
            dataset_descriptor: Dataset metadata and splits descriptor.
            optimization_plan: The dry-run optimization plan.

        Returns:
            True if this strategy supports the workload, False otherwise.
        """
        pass

    @abstractmethod
    def execute(
        self,
        job_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the optimization strategy, evaluate results, and package artifacts.

        Args:
            job_context: Complete job context including job_id, job_dir, model_path,
                         model_desc, dataset_ingestor, dataset_desc, preprocess_config,
                         opt_plan, hw_profile, adapted_model, adaptation_record, etc.

        Returns:
            Dictionary containing metrics, manifest, and artifact paths.
        """
        pass
