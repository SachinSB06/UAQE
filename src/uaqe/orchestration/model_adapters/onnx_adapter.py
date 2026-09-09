"""ONNX Model Adapter for UAQE.

Inspects ONNX graphs, extracts metadata, parameter counts, input/output tensors,
and validates execution via ONNX Runtime.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    onnx = None
    ort = None
    HAS_ONNX = False

from .base_model_adapter import BaseModelAdapter


class ONNXModelAdapter(BaseModelAdapter):
    """Adapter for ONNX model artifacts."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.onnx_model: Optional[Any] = None
        self.ort_session: Optional[Any] = None
        self._descriptor_cache: Optional[Dict[str, Any]] = None
        self._load_model()

    def _load_model(self) -> None:
        if not HAS_ONNX:
            raise ImportError(
                "ONNX and ONNX Runtime are required for ONNX model inspection. "
                "Please install them via: pip install onnx onnxruntime"
            )
        try:
            self.onnx_model = onnx.load(self.model_path)
            onnx.checker.check_model(self.onnx_model)
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # Error only
            self.ort_session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise ValueError(f"Failed to load/validate ONNX model at {self.model_path}: {e}") from e

    def inspect(self) -> Dict[str, Any]:
        if self._descriptor_cache is not None:
            return self._descriptor_cache

        graph = self.onnx_model.graph

        # Extract inputs
        inputs_meta = []
        for inp in self.ort_session.get_inputs():
            shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
            inputs_meta.append({
                "name": inp.name,
                "shape": shape,
                "dtype": inp.type
            })

        # Extract outputs
        outputs_meta = []
        for out in self.ort_session.get_outputs():
            shape = [dim if isinstance(dim, int) else -1 for dim in out.shape]
            outputs_meta.append({
                "name": out.name,
                "shape": shape,
                "dtype": out.type
            })

        # Parameter count
        param_count = 0
        trainable_count = 0
        for init in graph.initializer:
            n_elems = int(np.prod(init.dims))
            param_count += n_elems
            trainable_count += n_elems  # Weights/biases in ONNX are parameters

        # Architecture detection
        architecture = "Generic ONNX Graph"
        detected_task = "image_classification"
        op_types = {node.op_type for node in graph.node}

        # Check for characteristic MobileNetV3 / inverted residual blocks
        has_conv = "Conv" in op_types
        has_global_pool = "GlobalAveragePool" in op_types or "AveragePool" in op_types
        has_gemm = "Gemm" in op_types or "MatMul" in op_types
        
        # Check node naming or initializer names for known patterns
        node_names = [n.name for n in graph.node]
        if any("mobilenet" in n.lower() or "invertedresidual" in n.lower() or "features." in n.lower() for n in node_names):
            architecture = "MobileNetV3-Small"
        elif has_conv and has_global_pool and has_gemm:
            architecture = "Convolutional Classifier"

        primary_input_shape = inputs_meta[0]["shape"] if inputs_meta else [1, 3, 224, 224]
        primary_output_shape = outputs_meta[0]["shape"] if outputs_meta else [1, 1000]

        descriptor = {
            "format": "onnx",
            "framework": "ONNX",
            "architecture": architecture,
            "task": detected_task,
            "input_shape": primary_input_shape,
            "inputs": inputs_meta,
            "output_shape": primary_output_shape,
            "outputs": outputs_meta,
            "dtype": "float32",
            "parameter_count": param_count,
            "trainable_parameters": trainable_count,
            "source_sha256": self.source_sha256,
            "op_count": len(graph.node),
            "op_types": sorted(list(op_types))
        }

        self._descriptor_cache = descriptor
        return descriptor

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "supports_int8": True,
            "supports_fp16": True,
            "supports_pruning": True,
            "supports_sparse": True,
            "supports_rle": True,
            "supports_clustering": True,
            "supports_runtime_reconstruction": True,
            "supports_classifier_adaptation": True,
            "supports_export_tflite": True,
            "supports_export_uaqe": True
        }

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        inp_name = self.ort_session.get_inputs()[0].name
        if not isinstance(input_tensor, np.ndarray):
            input_tensor = np.array(input_tensor, dtype=np.float32)
        outputs = self.ort_session.run(None, {inp_name: input_tensor})
        return outputs[0]

    def get_parameter_count(self) -> int:
        desc = self.inspect()
        return desc["parameter_count"]
