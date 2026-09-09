"""Universal Model Ingestor for UAQE.

Provides framework-agnostic model format detection, adapter resolution,
capability inspection, and normalized descriptor generation.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any, Type

from .model_adapters.base_model_adapter import BaseModelAdapter
from .model_adapters.onnx_adapter import ONNXModelAdapter
from .model_adapters.safetensors_adapter import SafeTensorsModelAdapter
from .model_adapters.pytorch_adapter import PyTorchModelAdapter
from .model_adapters.tflite_adapter import TFLiteModelAdapter


class UniversalModelIngestor:
    """Universal Model Ingestor and Adapter Resolver."""

    # File extension mapping
    FORMAT_EXTENSION_MAP: Dict[str, str] = {
        ".onnx": "onnx",
        ".safetensors": "safetensors",
        ".pt": "pytorch_checkpoint",
        ".pth": "pytorch_checkpoint",
        ".bin": "pytorch_checkpoint",
        ".tflite": "tflite"
    }

    # Adapter Registry
    ADAPTER_REGISTRY: Dict[str, Type[BaseModelAdapter]] = {
        "onnx": ONNXModelAdapter,
        "safetensors": SafeTensorsModelAdapter,
        "pytorch_checkpoint": PyTorchModelAdapter,
        "tflite": TFLiteModelAdapter
    }

    def __init__(self, model_path: str):
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file does not exist at: {self.model_path}")

        self.detected_format = self.detect_format(self.model_path)
        self.adapter: Optional[BaseModelAdapter] = None
        self._init_adapter()

    @classmethod
    def detect_format(cls, path: str) -> str:
        """Detect model format from file extension and magic headers."""
        if not os.path.exists(path):
            return "unknown"

        _, ext = os.path.splitext(path.lower())
        if ext in cls.FORMAT_EXTENSION_MAP:
            return cls.FORMAT_EXTENSION_MAP[ext]

        # Magic header checks
        try:
            with open(path, "rb") as f:
                header = f.read(16)
                if header.startswith(b"TFL3") or b"TFL" in header[:8]:
                    return "tflite"
        except Exception:
            pass

        return "unknown"

    def _init_adapter(self) -> None:
        if self.detected_format not in self.ADAPTER_REGISTRY:
            raise ValueError(
                f"Model format '{self.detected_format}' unrecognized or unsupported for file: {self.model_path}. "
                f"Supported formats: {list(self.ADAPTER_REGISTRY.keys())}"
            )

        adapter_cls = self.ADAPTER_REGISTRY[self.detected_format]
        try:
            self.adapter = adapter_cls(self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Model format recognized ('{self.detected_format}'), but this architecture is not currently supported by an available adapter. "
                f"Details: {e}"
            ) from e

    def inspect(self) -> Dict[str, Any]:
        """Return normalized model descriptor."""
        if self.adapter is None:
            raise RuntimeError("Model adapter not initialized")
        return self.adapter.inspect()

    def get_descriptor(self) -> Dict[str, Any]:
        """Return normalized model descriptor (alias for inspect)."""
        return self.inspect()

    def get_capabilities(self) -> Dict[str, bool]:
        """Return optimization capability flags."""
        if self.adapter is None:
            raise RuntimeError("Model adapter not initialized")
        return self.adapter.get_capabilities()

    def generate_descriptor(self, output_path: str) -> Dict[str, Any]:
        """Export model_descriptor.json."""
        descriptor = self.inspect()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptor, f, indent=2)
        return descriptor

    def generate_capability_report(self, output_path: str) -> Dict[str, Any]:
        """Export capability_report.json."""
        desc = self.inspect()
        caps = self.get_capabilities()
        report = {
            "model_path": self.model_path,
            "format": desc["format"],
            "framework": desc["framework"],
            "architecture": desc["architecture"],
            "supported": any(caps.values()),
            "adapter": self.adapter.__class__.__name__,
            "reason": "Model architecture and format fully supported by adapter" if any(caps.values()) else "Limited support",
            "supported_optimization_modes": [k for k, v in caps.items() if v]
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report
