"""Quantization-strategy port consumed by
``uaqe.domain.quantization.QuantizationEngine``.

Concrete strategies (first-party or plugin, discovered via
``PluginRegistry``) implement this interface to apply a
``QuantizationConfig`` to an ``IMR`` without ``QuantizationEngine``
needing to know the specific numeric technique used.

Locked contract: ``03_API_Specification.md`` §1.8.
"""

from abc import ABC, abstractmethod

from uaqe.common.imr import IMR
from uaqe.common.value_objects import QuantizationConfig


class IQuantizationStrategy(ABC):
    """Abstract port for applying quantization to an ``IMR``."""

    @abstractmethod
    def apply(self, imr: IMR, plan: QuantizationConfig) -> IMR:
        """Apply this strategy's quantization technique to ``imr``.

        Args:
            imr: The model to quantize.
            plan: The configuration governing precision selection.

        Returns:
            A new ``IMR`` with the quantization applied.

        Raises:
            QuantizationError: If quantization cannot be applied or
                completed.
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Return this strategy's unique registration name."""
        raise NotImplementedError
