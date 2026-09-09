"""Config repository port consumed by ``CompositionRoot`` to source every
``*Config`` value object handed to domain stages via constructor
injection.

``uaqe.infrastructure.repositories.ConfigRepository`` is the sole
first-party implementation. Per ``10_Module_Development_Guide.md`` §0
rule 7, no module other than an ``IConfigRepository`` implementation
reads a config file directly.

Locked contract: ``03_API_Specification.md`` §1.13.
"""

from abc import ABC, abstractmethod

from uaqe.common.value_objects import (
    CompressionConfig,
    ExecutionConfig,
    OptimizationConfig,
    QuantizationConfig,
)


class IConfigRepository(ABC):
    """Abstract port for loading every top-level configuration value
    object.
    """

    @abstractmethod
    def load_quantization_config(self) -> QuantizationConfig:
        """Load the current ``QuantizationConfig``."""
        raise NotImplementedError

    @abstractmethod
    def load_compression_config(self) -> CompressionConfig:
        """Load the current ``CompressionConfig``."""
        raise NotImplementedError

    @abstractmethod
    def load_execution_config(self) -> ExecutionConfig:
        """Load the current ``ExecutionConfig``."""
        raise NotImplementedError

    @abstractmethod
    def load_optimization_config(self) -> OptimizationConfig:
        """Load the current ``OptimizationConfig``."""
        raise NotImplementedError
