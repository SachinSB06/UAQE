"""TFLite Model Adapter for UAQE.

Inspects TFLite FlatBuffer files, extracting tensor metadata, quantization info,
and executing validation via TFLite runtime.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf

from .base_model_adapter import BaseModelAdapter


class TFLiteModelAdapter(BaseModelAdapter):
    """Adapter for TFLite FlatBuffer model artifacts."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._descriptor_cache: Optional[Dict[str, Any]] = None
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self._load_interpreter()

    def _load_interpreter(self) -> None:
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            raise ValueError(f"Failed to load TFLite model at {self.model_path}: {e}") from e

    def inspect(self) -> Dict[str, Any]:
        if self._descriptor_cache is not None:
            return self._descriptor_cache

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        tensor_details = self.interpreter.get_tensor_details()

        inputs_meta = []
        for inp in input_details:
            inputs_meta.append({
                "name": inp["name"],
                "shape": list(inp["shape"]),
                "dtype": str(inp["dtype"])
            })

        outputs_meta = []
        for out in output_details:
            outputs_meta.append({
                "name": out["name"],
                "shape": list(out["shape"]),
                "dtype": str(out["dtype"])
            })

        # Calculate parameter count from weight tensors
        param_count = 0
        for t in tensor_details:
            # Tensors with non-empty shape and values
            shape = t.get("shape", [])
            if len(shape) > 0:
                param_count += int(np.prod(shape))

        primary_input_shape = inputs_meta[0]["shape"] if inputs_meta else [1, 224, 224, 3]
        primary_output_shape = outputs_meta[0]["shape"] if outputs_meta else [1, 1000]

        descriptor = {
            "format": "tflite",
            "framework": "TensorFlow Lite",
            "architecture": "TFLite FlatBuffer",
            "task": "image_classification",
            "input_shape": primary_input_shape,
            "inputs": inputs_meta,
            "output_shape": primary_output_shape,
            "outputs": outputs_meta,
            "dtype": str(input_details[0]["dtype"]) if input_details else "float32",
            "parameter_count": param_count,
            "trainable_parameters": 0,  # TFLite models are frozen runtime models
            "source_sha256": self.source_sha256,
            "tensor_count": len(tensor_details)
        }

        self._descriptor_cache = descriptor
        return descriptor

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "supports_int8": True,
            "supports_fp16": True,
            "supports_pruning": False,
            "supports_sparse": True,
            "supports_rle": True,
            "supports_clustering": False,
            "supports_runtime_reconstruction": True,
            "supports_classifier_adaptation": False,
            "supports_export_tflite": True,
            "supports_export_uaqe": True
        }

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        inp_idx = self.interpreter.get_input_details()[0]["index"]
        out_idx = self.interpreter.get_output_details()[0]["index"]
        self.interpreter.set_tensor(inp_idx, input_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(out_idx)

    def get_parameter_count(self) -> int:
        desc = self.inspect()
        return desc["parameter_count"]
