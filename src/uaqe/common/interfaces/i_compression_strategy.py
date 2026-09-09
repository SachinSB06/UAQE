"""Compression-strategy port consumed by
``uaqe.domain.compression.CompressionEngine``.

Concrete strategies (first-party or plugin, discovered via
``PluginRegistry``) implement this interface to apply a
``CompressionConfig`` to an ``IMR`` without ``CompressionEngine``
needing to know the specific technique used.

Locked contract: ``03_API_Specification.md`` §1.9.
"""

from abc import ABC, abstractmethod

from uaqe.common.imr import IMR
from uaqe.common.value_objects import CompressionConfig


class ICompressionStrategy(ABC):
    """Abstract port for applying compression to an ``IMR``."""

    @abstractmethod
    def apply(self, imr: IMR, plan: CompressionConfig) -> IMR:
        """Apply this strategy's compression technique to ``imr``.

        Args:
            imr: The model to compress.
            plan: The configuration governing which techniques and
                target ratio to apply.

        Returns:
            A new ``IMR`` with the compression applied.

        Raises:
            CompressionError: If compression cannot be applied or
                completed.
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Return this strategy's unique registration name."""
        raise NotImplementedError
