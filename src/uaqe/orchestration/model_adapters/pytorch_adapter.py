"""PyTorch Model Adapter for UAQE.

Inspects PyTorch checkpoint files (.pt / .pth) safely, extracting parameter counts,
layer tensors, and metadata.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from .base_model_adapter import BaseModelAdapter


class PyTorchModelAdapter(BaseModelAdapter):
    """Adapter for PyTorch checkpoint (.pt / .pth) artifacts."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._descriptor_cache: Optional[Dict[str, Any]] = None
        self.state_dict: Optional[Dict[str, Any]] = None
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        try:
            # Safe loading
            try:
                ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
            except Exception:
                # Fallback to standard torch.load if weights_only fails on legacy format
                ckpt = torch.load(self.model_path, map_location="cpu")

            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    self.state_dict = ckpt["model_state_dict"]
                elif "state_dict" in ckpt:
                    self.state_dict = ckpt["state_dict"]
                else:
                    self.state_dict = ckpt
            elif isinstance(ckpt, torch.nn.Module):
                self.state_dict = ckpt.state_dict()
            else:
                raise ValueError(f"Unrecognized PyTorch checkpoint contents: {type(ckpt)}")
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model at {self.model_path}: {e}") from e

    def inspect(self) -> Dict[str, Any]:
        if self._descriptor_cache is not None:
            return self._descriptor_cache

        param_count = 0
        output_classes = 1000
        input_shape = [1, 3, 224, 224]
        architecture = "PyTorch State Dict"

        tensor_keys = list(self.state_dict.keys()) if self.state_dict else []
        for k, v in self.state_dict.items():
            if isinstance(v, torch.Tensor):
                param_count += v.numel()

        # Check last layer for output classes
        if tensor_keys:
            last_key = tensor_keys[-1]
            if "weight" in last_key or "bias" in last_key:
                last_tensor = self.state_dict[last_key]
                if last_tensor.dim() >= 1:
                    output_classes = last_tensor.shape[0]

        # Architecture detection heuristics
        if any("mobilenet" in k.lower() or "features." in k.lower() for k in tensor_keys):
            architecture = "MobileNetV3-Small"
            input_shape = [1, 3, 128, 128]
        elif any("resnet" in k.lower() for k in tensor_keys):
            architecture = "ResNetForImageClassification"
            input_shape = [1, 3, 224, 224]
        elif any("vit." in k.lower() or "patch_embeddings" in k.lower() or "encoder.layer" in k.lower() for k in tensor_keys):
            architecture = "ViTForImageClassification"
            input_shape = [1, 3, 224, 224]

        descriptor = {
            "format": "pytorch_checkpoint",
            "framework": "PyTorch",
            "architecture": architecture,
            "task": "image_classification",
            "input_shape": input_shape,
            "inputs": [{"name": "input", "shape": input_shape, "dtype": "float32"}],
            "output_shape": [1, output_classes],
            "outputs": [{"name": "logits", "shape": [1, output_classes], "dtype": "float32"}],
            "dtype": "float32",
            "parameter_count": param_count,
            "trainable_parameters": param_count,
            "source_sha256": self.source_sha256,
            "tensor_count": len(tensor_keys)
        }

        self._descriptor_cache = descriptor
        return descriptor

    def get_capabilities(self) -> Dict[str, bool]:
        desc = self.inspect()
        is_known = desc["architecture"] in ["MobileNetV3-Small", "ResNetForImageClassification", "ResNet-50"]
        return {
            "supports_int8": is_known,
            "supports_fp16": is_known,
            "supports_pruning": is_known,
            "supports_sparse": is_known,
            "supports_rle": is_known,
            "supports_clustering": is_known,
            "supports_runtime_reconstruction": is_known,
            "supports_classifier_adaptation": is_known,
            "supports_export_tflite": is_known,
            "supports_export_uaqe": is_known
        }

    def forward(self, input_tensor: Any) -> Any:
        raise NotImplementedError("Direct forward pass on state_dict requires module reconstruction")

    def get_parameter_count(self) -> int:
        desc = self.inspect()
        return desc["parameter_count"]
