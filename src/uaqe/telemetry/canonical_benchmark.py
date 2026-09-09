"""Single Canonical Benchmark Implementation for UAQE.

Provides mathematically rigorous, methodologically identical pure-inference
benchmarking for FP32 (ONNX Runtime) and INT8 (TensorFlow Lite) models.

PROTOCOL SPECIFICATION:
- Batch size: 1
- Thread count: 2 (configured on runtime session / interpreter)
- Warmup invokes: exactly 10 (untimed)
- Timed invokes: exactly 100 (timed strictly around runtime invocation)
- RuntimeMonitor: MUST be stopped prior to benchmark timing
- Garbage collection: gc.collect() executed prior to timing
- Zero disk I/O, PIL loading, transforms, logging, or dataset loops in timed block
- Throughput formula: throughput_img_s = 1000.0 / pure_invoke_latency_ms
- Full provenance metadata persisted with every benchmark result
"""

from __future__ import annotations

import gc
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


@dataclass
class BenchmarkProvenance:
    evaluator: str
    runtime: str
    runtime_mode: str
    performance_enabled: bool
    num_threads: int
    batch_size: int
    input_shape: List[int]
    warmup_count: int
    timed_iterations: int
    latency_type: str = "PURE_INVOKE"
    throughput_type: str = "DERIVED_FROM_PURE_INVOKE"
    preprocessing_in_timing: bool = False
    postprocessing_in_timing: bool = False
    dataset_io_in_timing: bool = False
    runtime_monitor_active_during_timing: bool = False
    gc_before_timing: bool = True
    benchmark_version: str = "1.0.0"
    artifact_sha256: Optional[str] = None
    artifact_size_bytes: Optional[int] = None
    delegate: str = "REFERENCE"
    delegate_enabled: bool = False
    delegate_supported_ops: Optional[List[str]] = None
    delegate_fallback_ops: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalBenchmarkResult:
    model_path: str
    runtime: str
    num_threads: int
    pure_invoke_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_img_s: float
    preprocessing_latency_ms: float
    postprocessing_latency_ms: float
    end_to_end_latency_ms: float
    warmup_runs: int
    timed_iterations: int
    raw_latencies_ms: List[float] = field(default_factory=list)
    provenance: BenchmarkProvenance = field(default_factory=lambda: BenchmarkProvenance(
        evaluator="unknown", runtime="unknown", runtime_mode="unknown",
        performance_enabled=False, num_threads=1, batch_size=1,
        input_shape=[1, 3, 128, 128], warmup_count=10, timed_iterations=100
    ))

    def validate(self) -> None:
        """Enforce strict metric consistency."""
        if self.pure_invoke_latency_ms <= 0:
            raise ValueError(f"Invalid pure_invoke_latency_ms: {self.pure_invoke_latency_ms}")
        if self.throughput_img_s <= 0:
            raise ValueError(f"Invalid throughput_img_s: {self.throughput_img_s}")
        if self.warmup_runs != 10:
            raise ValueError(f"warmup_runs must be 10, got {self.warmup_runs}")
        if self.timed_iterations != 100:
            raise ValueError(f"timed_iterations must be 100, got {self.timed_iterations}")
        
        expected_throughput = 1000.0 / self.pure_invoke_latency_ms
        if abs(self.throughput_img_s - expected_throughput) > 0.1:
            raise ValueError(
                f"Throughput inconsistency: {self.throughput_img_s} != 1000 / {self.pure_invoke_latency_ms} ({expected_throughput})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "runtime": self.runtime,
            "num_threads": self.num_threads,
            "pure_invoke_latency_ms": round(self.pure_invoke_latency_ms, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "min_latency_ms": round(self.min_latency_ms, 3),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "throughput_img_s": round(self.throughput_img_s, 2),
            "preprocessing_latency_ms": round(self.preprocessing_latency_ms, 3),
            "postprocessing_latency_ms": round(self.postprocessing_latency_ms, 3),
            "end_to_end_latency_ms": round(self.end_to_end_latency_ms, 3),
            "warmup_runs": self.warmup_runs,
            "timed_iterations": self.timed_iterations,
            "provenance": self.provenance.to_dict()
        }


class CanonicalBenchmark:
    """The authoritative single source of truth for pure inference benchmarking in UAQE."""

    WARMUP_COUNT: int = 10
    TIMED_ITERATIONS: int = 100
    NUM_THREADS: int = 2

    @staticmethod
    def _compute_sha256_and_size(file_path: str) -> Tuple[str, int]:
        if not os.path.exists(file_path):
            return "NOT_AVAILABLE", 0
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest(), os.path.getsize(file_path)

    @classmethod
    def benchmark_tflite(
        cls,
        model_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 128, 128),
        num_threads: int = 2,
        warmup_runs: int = 10,
        measured_runs: int = 100,
        runtime_mode: str = "performance",
        performance_enabled: bool = True,
        use_xnnpack: bool = False
    ) -> CanonicalBenchmarkResult:
        """Canonical pure-inference benchmark for TensorFlow Lite INT8/FP32 models."""
        import tensorflow as tf

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at: {model_path}")

        sha256, size_bytes = cls._compute_sha256_and_size(model_path)

        # 1. Initialize Interpreter with selected delegate resolver
        if use_xnnpack:
            try:
                interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    num_threads=num_threads
                )
                interpreter.allocate_tensors()
                runtime_str = "TensorFlow Lite (XNNPACK)"
                delegate_str = "XNNPACK"
                delegate_enabled = True
            except RuntimeError as err:
                raise RuntimeError(f"XNNPACK acceleration failed to prepare on model {model_path}: {err}") from err
        else:
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            interpreter.allocate_tensors()
            runtime_str = "TensorFlow Lite (BUILTIN_WITHOUT_DEFAULT_DELEGATES)"
            delegate_str = "REFERENCE"
            delegate_enabled = False

        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]
        in_idx = in_details["index"]
        out_idx = out_details["index"]

        # 2. Allocate pre-allocated contiguous buffer (zero allocation in loop)
        dummy_input = np.zeros(input_shape, dtype=np.float32)
        reusable_buffer = np.empty(input_shape, dtype=np.float32)
        np.copyto(reusable_buffer, dummy_input)

        # 3. Clean memory and pause background telemetry monitors before timing
        gc.collect()
        from uaqe.telemetry.runtime_monitor import pause_active_monitors, resume_active_monitors

        pause_active_monitors()
        try:
            # 4. Untimed Warmup Invokes (strictly 10 warmups)
            for _ in range(warmup_runs):
                interpreter.set_tensor(in_idx, reusable_buffer)
                interpreter.invoke()
                _ = interpreter.get_tensor(out_idx)

            # 5. Timed Measurement (strictly 100 invokes)
            pure_latencies: List[float] = []
            prep_latencies: List[float] = []
            post_latencies: List[float] = []

            for _ in range(measured_runs):
                # Preprocessing / buffer load
                t_p0 = time.perf_counter()
                interpreter.set_tensor(in_idx, reusable_buffer)
                t_p1 = time.perf_counter()
                prep_latencies.append((t_p1 - t_p0) * 1000.0)

                # Pure inference (START -> interpreter.invoke() -> STOP)
                t_i0 = time.perf_counter()
                interpreter.invoke()
                t_i1 = time.perf_counter()
                pure_latencies.append((t_i1 - t_i0) * 1000.0)

                # Postprocessing (output extraction + argmax)
                t_post0 = time.perf_counter()
                out = interpreter.get_tensor(out_idx)[0]
                _ = int(np.argmax(out))
                t_post1 = time.perf_counter()
                post_latencies.append((t_post1 - t_post0) * 1000.0)
        finally:
            resume_active_monitors()

        # 6. Statistical Aggregation
        mean_ms = float(np.mean(pure_latencies))
        p50_ms = float(np.median(pure_latencies))
        p95_ms = float(np.percentile(pure_latencies, 95))
        min_ms = float(np.min(pure_latencies))
        max_ms = float(np.max(pure_latencies))
        throughput = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

        prep_mean = float(np.mean(prep_latencies))
        post_mean = float(np.mean(post_latencies))
        e2e_mean = mean_ms + prep_mean + post_mean

        provenance = BenchmarkProvenance(
            evaluator="CanonicalBenchmark.benchmark_tflite",
            runtime=runtime_str,
            runtime_mode=runtime_mode,
            performance_enabled=performance_enabled,
            num_threads=num_threads,
            batch_size=1,
            input_shape=list(input_shape),
            warmup_count=warmup_runs,
            timed_iterations=measured_runs,
            latency_type="PURE_INVOKE",
            throughput_type="DERIVED_FROM_PURE_INVOKE",
            preprocessing_in_timing=False,
            postprocessing_in_timing=False,
            dataset_io_in_timing=False,
            runtime_monitor_active_during_timing=False,
            gc_before_timing=True,
            benchmark_version="1.0.0",
            artifact_sha256=sha256,
            artifact_size_bytes=size_bytes,
            delegate=delegate_str,
            delegate_enabled=delegate_enabled
        )

        result = CanonicalBenchmarkResult(
            model_path=model_path,
            runtime=runtime_str,
            num_threads=num_threads,
            pure_invoke_latency_ms=mean_ms,
            p50_latency_ms=p50_ms,
            p95_latency_ms=p95_ms,
            min_latency_ms=min_ms,
            max_latency_ms=max_ms,
            throughput_img_s=throughput,
            preprocessing_latency_ms=prep_mean,
            postprocessing_latency_ms=post_mean,
            end_to_end_latency_ms=e2e_mean,
            warmup_runs=warmup_runs,
            timed_iterations=measured_runs,
            raw_latencies_ms=pure_latencies,
            provenance=provenance
        )
        result.validate()
        return result

    @classmethod
    def benchmark_onnx(
        cls,
        model_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 128, 128),
        num_threads: int = 2,
        warmup_runs: int = 10,
        measured_runs: int = 100,
        runtime_mode: str = "performance",
        performance_enabled: bool = True
    ) -> CanonicalBenchmarkResult:
        """Canonical pure-inference benchmark for ONNX Runtime FP32/INT8 models."""
        import onnxruntime as ort

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at: {model_path}")

        sha256, size_bytes = cls._compute_sha256_and_size(model_path)

        # 1. Initialize ONNX Session with identical 2-thread configuration
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = num_threads
        sess_opt.inter_op_num_threads = 1
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_opt, providers=["CPUExecutionProvider"])

        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name

        # 2. Pre-allocate contiguous buffer (zero allocation in loop)
        dummy_input = np.zeros(input_shape, dtype=np.float32)
        reusable_buffer = np.empty(input_shape, dtype=np.float32)
        np.copyto(reusable_buffer, dummy_input)

        # 3. Clean memory and pause background telemetry monitors before timing
        gc.collect()
        from uaqe.telemetry.runtime_monitor import pause_active_monitors, resume_active_monitors

        pause_active_monitors()
        try:
            # 4. Untimed Warmup Invokes (strictly 10 warmups)
            for _ in range(warmup_runs):
                _ = session.run([out_name], {in_name: reusable_buffer})

            # 5. Timed Measurement (strictly 100 invokes)
            pure_latencies: List[float] = []
            prep_latencies: List[float] = []
            post_latencies: List[float] = []

            for _ in range(measured_runs):
                # Preprocessing / buffer load
                t_p0 = time.perf_counter()
                np.copyto(reusable_buffer, dummy_input)
                t_p1 = time.perf_counter()
                prep_latencies.append((t_p1 - t_p0) * 1000.0)

                # Pure inference (START -> session.run() -> STOP)
                t_i0 = time.perf_counter()
                outputs = session.run([out_name], {in_name: reusable_buffer})
                t_i1 = time.perf_counter()
                pure_latencies.append((t_i1 - t_i0) * 1000.0)

                # Postprocessing (output extraction + argmax)
                t_post0 = time.perf_counter()
                out = outputs[0][0]
                _ = int(np.argmax(out))
                t_post1 = time.perf_counter()
                post_latencies.append((t_post1 - t_post0) * 1000.0)
        finally:
            resume_active_monitors()

        # 6. Statistical Aggregation
        mean_ms = float(np.mean(pure_latencies))
        p50_ms = float(np.median(pure_latencies))
        p95_ms = float(np.percentile(pure_latencies, 95))
        min_ms = float(np.min(pure_latencies))
        max_ms = float(np.max(pure_latencies))
        throughput = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

        prep_mean = float(np.mean(prep_latencies))
        post_mean = float(np.mean(post_latencies))
        e2e_mean = mean_ms + prep_mean + post_mean

        provenance = BenchmarkProvenance(
            evaluator="CanonicalBenchmark.benchmark_onnx",
            runtime="ONNX Runtime (CPUExecutionProvider)",
            runtime_mode=runtime_mode,
            performance_enabled=performance_enabled,
            num_threads=num_threads,
            batch_size=1,
            input_shape=list(input_shape),
            warmup_count=warmup_runs,
            timed_iterations=measured_runs,
            latency_type="PURE_INVOKE",
            throughput_type="DERIVED_FROM_PURE_INVOKE",
            preprocessing_in_timing=False,
            postprocessing_in_timing=False,
            dataset_io_in_timing=False,
            runtime_monitor_active_during_timing=False,
            gc_before_timing=True,
            benchmark_version="1.0.0",
            artifact_sha256=sha256,
            artifact_size_bytes=size_bytes,
            delegate="CPUExecutionProvider",
            delegate_enabled=True
        )

        result = CanonicalBenchmarkResult(
            model_path=model_path,
            runtime="ONNX Runtime (CPUExecutionProvider)",
            num_threads=num_threads,
            pure_invoke_latency_ms=mean_ms,
            p50_latency_ms=p50_ms,
            p95_latency_ms=p95_ms,
            min_latency_ms=min_ms,
            max_latency_ms=max_ms,
            throughput_img_s=throughput,
            preprocessing_latency_ms=prep_mean,
            postprocessing_latency_ms=post_mean,
            end_to_end_latency_ms=e2e_mean,
            warmup_runs=warmup_runs,
            timed_iterations=measured_runs,
            raw_latencies_ms=pure_latencies,
            provenance=provenance
        )
        result.validate()
        return result
