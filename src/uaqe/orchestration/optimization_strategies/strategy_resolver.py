"""Optimization Strategy Resolver for UAQE.

Maintains a pluggable registry of optimization strategies and resolves the best
applicable strategy based on model capabilities, dataset properties, and target hardware.
"""

from typing import Dict, List, Optional, Any

from .base_strategy import BaseOptimizationStrategy
from .resnet_onnx_ptq_strategy import ResNetONNXPTQStrategy
from .mobilenet_adaptive_strategy import MobileNetAdaptiveStrategy
from .xnnpack_strategy import XNNPACKStrategy
from .generic_fallback_strategy import GenericFallbackStrategy


class OptimizationStrategyResolver:
    """Pluggable registry and capability-driven resolver for optimization strategies."""

    _STRATEGIES: List[BaseOptimizationStrategy] = [
        ResNetONNXPTQStrategy(),
        MobileNetAdaptiveStrategy(),
        XNNPACKStrategy(),
        GenericFallbackStrategy()
    ]

    @classmethod
    def register_strategy(cls, strategy: BaseOptimizationStrategy, insert_at_front: bool = True) -> None:
        """Register a new custom optimization strategy."""
        if insert_at_front:
            cls._STRATEGIES.insert(0, strategy)
        else:
            cls._STRATEGIES.append(strategy)

    @classmethod
    def resolve(
        cls,
        model_descriptor: Dict[str, Any],
        dataset_descriptor: Dict[str, Any],
        optimization_plan: Dict[str, Any]
    ) -> BaseOptimizationStrategy:
        """Resolve the first registered strategy that can handle the given workload."""
        for strat in cls._STRATEGIES:
            if strat.can_handle(model_descriptor, dataset_descriptor, optimization_plan):
                return strat
        return GenericFallbackStrategy()
