"""Base Model Adapter Interface for UAQE.

Defines the abstract interface and capability contract for model format adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import os


class BaseModelAdapter(ABC):
    """Abstract base class for framework and format-specific model inspection and handling."""

    def __init__(self, model_path: str):
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        self.source_sha256 = self._compute_sha256(self.model_path)

    @staticmethod
    def _compute_sha256(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @abstractmethod
    def inspect(self) -> Dict[str, Any]:
        """Inspect model and return normalized descriptor metadata."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """Return supported optimization capabilities for this model format/architecture.

        Example keys:
            - supports_int8 (bool)
            - supports_fp16 (bool)
            - supports_pruning (bool)
            - supports_sparse (bool)
            - supports_rle (bool)
            - supports_clustering (bool)
            - supports_runtime_reconstruction (bool)
            - supports_classifier_adaptation (bool)
        """
        pass

    @abstractmethod
    def forward(self, input_tensor: Any) -> Any:
        """Execute a test forward pass through the model."""
        pass

    @abstractmethod
    def get_parameter_count(self) -> int:
        """Return total parameter count."""
        pass
