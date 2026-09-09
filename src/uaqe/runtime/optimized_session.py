"""
UAQE Phase E.1 Optimized Runtime Session
Supports both 'portable' (on-the-fly fast decoding) and 'cached' (prebuilt verified TFLite) execution modes.
Provides high-level session management, deterministic benchmark routines, and memory reclamation.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

import tensorflow as tf

# Ensure uaqe imports work cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.runtime.optimized_decoder import OptimizedRuntimeDecoder
from src.uaqe.runtime.runtime_cache import RuntimeCacheManager


CLASS_NAMES = ["bridge", "clean", "cmp", "crack", "opens", "other", "particle", "scratch", "vias"]


class OptimizedRuntimeSession:
    """High-level execution session supporting 'portable' and 'cached' runtime deployment modes."""

    def __init__(
        self,
        decoder: OptimizedRuntimeDecoder,
        template_tflite_path: Optional[str] = None,
        mode: str = "portable",
        cache_manager: Optional[RuntimeCacheManager] = None
    ):
        if mode not in ["portable", "cached"]:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes: 'portable', 'cached'.")

        self.decoder = decoder
        self.template_tflite_path = template_tflite_path
        self.mode = mode
        self.cache_manager = cache_manager or RuntimeCacheManager()

        self.model_bytes: Optional[Union[bytes, bytearray]] = None
        self.interpreter: Optional[tf.lite.Interpreter] = None
        self._input_details: Optional[List[Dict[str, Any]]] = None
        self._output_details: Optional[List[Dict[str, Any]]] = None
        self._is_allocated = False
        self._cache_hit = False

    @classmethod
    def from_archive(
        cls,
        archive_path: str,
        template_tflite_path: Optional[str] = "output/phase_c4/models/c4_best_int8.tflite",
        mode: str = "portable",
        cache_dir: str = "output/phase_e1/runtime_cache"
    ) -> OptimizedRuntimeSession:
        """Initializes an optimized session directly from a compressed archive path."""
        decoder = OptimizedRuntimeDecoder(base_template_path=template_tflite_path or "output/phase_c4/models/c4_best_int8.tflite")
        decoder.load(archive_path)
        cache_mgr = RuntimeCacheManager(cache_dir=cache_dir)
        return cls(
            decoder=decoder,
            template_tflite_path=template_tflite_path,
            mode=mode,
            cache_manager=cache_mgr
        )

    def inspect(self) -> Dict[str, Any]:
        """Returns archive inspection metadata."""
        return self.decoder.inspect()

    def load(self) -> OptimizedRuntimeSession:
        """Loads or reconstructs model bytes based on execution mode."""
        archive_hash = self.decoder.archive_hash
        if not archive_hash:
            raise RuntimeError("Decoder has not loaded an archive.")

        if self.mode == "cached":
            cached_data = self.cache_manager.load_cached_model(archive_hash)
            if cached_data is not None:
                self.model_bytes = cached_data
                self._cache_hit = True
                return self

            # Cache miss: decode, reconstruct, and store in cache
            self.model_bytes = self.decoder.reconstruct(self.template_tflite_path)
            self.cache_manager.store_cached_model(
                archive_sha256=archive_hash,
                model_bytes=self.model_bytes,
                extra_metadata={"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            )
            self._cache_hit = False
            return self

        # Portable mode: reconstruct in memory
        self.model_bytes = self.decoder.reconstruct(self.template_tflite_path)
        self._cache_hit = False
        return self

    def allocate(self) -> OptimizedRuntimeSession:
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
        """Benchmarks warm inference latency on the host system."""
        if not self._is_allocated or self.interpreter is None:
            self.allocate()

        if input_data is None:
            dummy_shape = self._input_details[0]["shape"]
            dummy_input = np.random.randn(*dummy_shape).astype(np.float32)
        else:
            dummy_input = input_data[:1].astype(np.float32)

        in_idx = self._input_details[0]["index"]

        for _ in range(warmup_runs):
            self.interpreter.set_tensor(in_idx, dummy_input)
            self.interpreter.invoke()

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
        """Runs mathematical tensor verification against source model."""
        return self.decoder.verify_tensors(source_tflite_path)

    def close(self) -> None:
        """Releases interpreter resources and clears memory buffers."""
        self.interpreter = None
        self.model_bytes = None
        self._is_allocated = False
