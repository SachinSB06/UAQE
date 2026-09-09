"""Safetensors Model Adapter for UAQE.

Inspects SafeTensors checkpoint files and accompanying configuration files,
extracting tensor shapes, dtypes, parameter counts, and architecture configurations.
"""

import os
import json
import struct
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .base_model_adapter import BaseModelAdapter


class SafeTensorsModelAdapter(BaseModelAdapter):
    """Adapter for SafeTensors checkpoint artifacts."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.header_meta: Dict[str, Any] = {}
        self.config_meta: Dict[str, Any] = {}
        self._descriptor_cache: Optional[Dict[str, Any]] = None
        self._inspect_header()

    def _inspect_header(self) -> None:
        try:
            with open(self.model_path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                self.header_meta = json.loads(f.read(header_len).decode("utf-8"))

            # Check for adjacent config.json
            model_dir = os.path.dirname(self.model_path)
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config_meta = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse SafeTensors header at {self.model_path}: {e}") from e

    def inspect(self) -> Dict[str, Any]:
        if self._descriptor_cache is not None:
            return self._descriptor_cache

        param_count = 0
        dtypes = set()
        tensor_names = []

        for k, v in self.header_meta.items():
            if k == "__metadata__":
                continue
            tensor_names.append(k)
            shape = v.get("shape", [])
            param_count += int(np.prod(shape)) if shape else 1
            dtypes.add(v.get("dtype", "F32"))

        # Determine architecture
        architecture = "Generic SafeTensors Checkpoint"
        detected_task = "image_classification"
        input_shape = [1, 3, 224, 224]
        output_classes = 1000

        if self.config_meta:
            arch_list = self.config_meta.get("architectures", [])
            if arch_list:
                architecture = arch_list[0]
            elif "model_type" in self.config_meta:
                architecture = self.config_meta["model_type"]

            if "num_labels" in self.config_meta:
                output_classes = self.config_meta["num_labels"]
            elif "id2label" in self.config_meta:
                output_classes = len(self.config_meta["id2label"])
            elif "hidden_sizes" in self.config_meta:
                # E.g. ResNet hidden_sizes
                output_classes = 1000

            if "image_size" in self.config_meta:
                sz = self.config_meta["image_size"]
                input_shape = [1, self.config_meta.get("num_channels", 3), sz, sz]
        else:
            # Infer from tensor keys
            if any("resnet" in k.lower() for k in tensor_names):
                architecture = "ResNetForImageClassification"
                output_classes = 1000
            elif any("vit." in k.lower() or "patch_embeddings" in k.lower() or "encoder.layer" in k.lower() for k in tensor_names):
                architecture = "ViTForImageClassification"
                if "classifier.weight" in self.header_meta:
                    output_classes = self.header_meta["classifier.weight"].get("shape", [1000])[0]
                else:
                    output_classes = 1000

        descriptor = {
            "format": "safetensors",
            "framework": "PyTorch / SafeTensors",
            "architecture": architecture,
            "task": detected_task,
            "input_shape": input_shape,
            "inputs": [{"name": "input_image", "shape": input_shape, "dtype": "float32"}],
            "output_shape": [1, output_classes],
            "outputs": [{"name": "logits", "shape": [1, output_classes], "dtype": "float32"}],
            "dtype": "float32",
            "parameter_count": param_count,
            "trainable_parameters": param_count,
            "source_sha256": self.source_sha256,
            "tensor_count": len(tensor_names),
            "config_found": bool(self.config_meta)
        }

        self._descriptor_cache = descriptor
        return descriptor

    def get_capabilities(self) -> Dict[str, bool]:
        desc = self.inspect()
        is_supported_arch = desc["architecture"] in ["ResNetForImageClassification", "ResNet-50", "resnet"]
        return {
            "supports_int8": is_supported_arch,
            "supports_fp16": is_supported_arch,
            "supports_pruning": is_supported_arch,
            "supports_sparse": is_supported_arch,
            "supports_rle": is_supported_arch,
            "supports_clustering": is_supported_arch,
            "supports_runtime_reconstruction": is_supported_arch,
            "supports_classifier_adaptation": is_supported_arch,
            "supports_export_tflite": is_supported_arch,
            "supports_export_uaqe": is_supported_arch
        }

    def forward(self, input_tensor: Any) -> Any:
        # For full forward pass, PyTorch model reconstructor is invoked
        raise NotImplementedError("Direct forward pass on raw safetensors requires loading into PyTorch module")

    def get_parameter_count(self) -> int:
        desc = self.inspect()
        return desc["parameter_count"]
