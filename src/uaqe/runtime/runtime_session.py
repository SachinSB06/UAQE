"""
UAQE Phase D.5 Runtime Session API
High-level execution API for loading UAQE compressed packages, allocating in-memory
FlatBuffers, executing live inference, and performing deterministic benchmarks.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

import tensorflow as tf

from src.uaqe.runtime.runtime_decoder import RuntimeDecoder


CLASS_NAMES = ["bridge", "clean", "cmp", "crack", "opens", "other", "particle", "scratch", "vias"]


class RuntimeSession:
    """Provides high-level session management for UAQE model execution."""

    def __init__(
        self,
        decoder: RuntimeDecoder,
        template_tflite_path: Optional[str] = None
    ):
        self.decoder = decoder
        self.template_tflite_path = template_tflite_path
        self.model_bytes: Optional[bytearray] = None
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self._input_details: Optional[List[Dict[str, Any]]] = None
        self._output_details: Optional[List[Dict[str, Any]]] = None
        self._is_allocated = False

    @classmethod
    def from_archive(
        cls,
        archive_path: str,
        template_tflite_path: Optional[str] = "output/phase_c4/models/c4_best_int8.tflite"
    ) -> RuntimeSession:
        """Factory method to initialize a session from a compressed .bin / .uaqe file."""
        decoder = RuntimeDecoder(base_template_path=template_tflite_path or "output/phase_c4/models/c4_best_int8.tflite")
        decoder.load(archive_path)
        return cls(decoder=decoder, template_tflite_path=template_tflite_path)

    def inspect(self) -> Dict[str, Any]:
        """Returns archive inspection metadata."""
        return self.decoder.inspect()

    def load(self) -> RuntimeSession:
        """Reconstructs in-memory FlatBuffer model from decoded tensor streams."""
        self.model_bytes = self.decoder.reconstruct(self.template_tflite_path)
        return self

    def allocate(self) -> RuntimeSession:
        """Instantiates and allocates tensors on the standard TFLite interpreter."""
        if self.model_bytes is None:
            self.load()

        self.interpreter = tf.lite.Interpreter(
            model_content=bytes(self.model_bytes),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self._is_allocated = True
        return self

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Executes inference on preprocessed input array."""
        if not self._is_allocated or self.interpreter is None:
            self.allocate()

        in_idx = self._input_details[0]["index"]
        out_idx = self._output_details[0]["index"]

        # Handle batch or single sample
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        input_data = input_data.astype(np.float32)

        results = []
        for i in range(len(input_data)):
            sample = input_data[i:i+1]
            self.interpreter.set_tensor(in_idx, sample)
            self.interpreter.invoke()
            out_tensor = self.interpreter.get_tensor(out_idx)
            results.append(out_tensor[0])

        return np.array(results)

    def predict_class(self, input_data: np.ndarray) -> Union[int, List[int]]:
        """Executes inference and returns argmax class index (or list of indices)."""
        logits = self.predict(input_data)
        classes = np.argmax(logits, axis=1).tolist()
        if input_data.ndim == 3 or (input_data.ndim == 4 and input_data.shape[0] == 1):
            return int(classes[0])
        return classes

    def benchmark(
        self,
        input_data: Optional[np.ndarray] = None,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmarks warm inference latency on the current host system."""
        if not self._is_allocated or self.interpreter is None:
            self.allocate()

        if input_data is None:
            dummy_shape = self._input_details[0]["shape"]
            dummy_input = np.random.randn(*dummy_shape).astype(np.float32)
        else:
            dummy_input = input_data[:1].astype(np.float32)

        in_idx = self._input_details[0]["index"]

        # Warmup
        for _ in range(warmup_runs):
            self.interpreter.set_tensor(in_idx, dummy_input)
            self.interpreter.invoke()

        # Timed runs
        timings = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.interpreter.set_tensor(in_idx, dummy_input)
            self.interpreter.invoke()
            timings.append((time.perf_counter() - t0) * 1000.0)

        timings_np = np.array(timings)
        return {
            "mean_ms": float(np.mean(timings_np)),
            "median_ms": float(np.median(timings_np)),
            "p95_ms": float(np.percentile(timings_np, 95)),
            "min_ms": float(np.min(timings_np)),
            "max_ms": float(np.max(timings_np)),
            "std_ms": float(np.std(timings_np)),
            "fps": float(1000.0 / max(0.001, np.mean(timings_np)))
        }

    def verify(self, source_tflite_path: str) -> Dict[str, Any]:
        """Runs tensor-by-tensor mathematical verification against source model."""
        return self.decoder.verify_tensors(source_tflite_path)

    def close(self) -> None:
        """Releases interpreter resources and clears internal memory buffers."""
        self.interpreter = None
        self.model_bytes = None
        self._is_allocated = False
