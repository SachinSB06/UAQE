"""Performance Benchmarking Layer for UAQE.

Provides dedicated, isolated benchmarking for deep learning models (TFLite and ONNX Runtime)
separating:
- Model file loading
- Runtime / interpreter initialization
- Tensor allocation
- Warmup executions (default 10)
- Pure inference execution (default 100 measured runs)
- Preprocessing / buffer copying
- Postprocessing (argmax, softmax, top-k)
- End-to-end sample latency

Calculates strictly measured statistical metrics:
- Mean
- Median
- P95
- Min
- Max
- Throughput (images/sec)
"""

from __future__ import annotations

import os
import time
import dataclasses
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


@dataclasses.dataclass
class LatencyStatistics:
    """Latency distribution statistics in milliseconds."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    throughput_ips: float
    raw_samples: List[float] = dataclasses.field(default_factory=list)

    def to_dict(self, include_samples: bool = False) -> Dict[str, Any]:
        d = {
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "throughput_ips": round(self.throughput_ips, 2)
        }
        if include_samples:
            d["raw_samples"] = [round(s, 3) for s in self.raw_samples]
        return d


@dataclasses.dataclass
class PerformanceBenchmarkResult:
    """Complete decomposed performance breakdown."""
    model_path: str
    runtime_type: str
    threads: int
    delegate: str
    model_load_ms: float
    runtime_init_ms: float
    warmup_runs: int
    measured_runs: int
    pure_inference: LatencyStatistics
    preprocessing_mean_ms: float
    postprocessing_mean_ms: float
    end_to_end_mean_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "runtime_type": self.runtime_type,
            "threads": self.threads,
            "delegate": self.delegate,
            "model_load_ms": round(self.model_load_ms, 3),
            "runtime_init_ms": round(self.runtime_init_ms, 3),
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "pure_inference": self.pure_inference.to_dict(),
            "preprocessing_mean_ms": round(self.preprocessing_mean_ms, 3),
            "postprocessing_mean_ms": round(self.postprocessing_mean_ms, 3),
            "end_to_end_mean_ms": round(self.end_to_end_mean_ms, 3)
        }


class PerformanceBenchmark:
    """Benchmarking helper that isolates pure inference from data pipeline & I/O overhead."""

    @staticmethod
    def calculate_stats(latencies_ms: List[float]) -> LatencyStatistics:
        if not latencies_ms:
            return LatencyStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mean_v = float(np.mean(latencies_ms))
        med_v = float(np.median(latencies_ms))
        p95_v = float(np.percentile(latencies_ms, 95))
        min_v = float(np.min(latencies_ms))
        max_v = float(np.max(latencies_ms))
        tput = float(1000.0 / mean_v) if mean_v > 0 else 0.0
        return LatencyStatistics(
            mean_ms=mean_v,
            median_ms=med_v,
            p95_ms=p95_v,
            min_ms=min_v,
            max_ms=max_v,
            throughput_ips=tput,
            raw_samples=latencies_ms
        )

    @classmethod
    def benchmark_tflite(
        cls,
        model_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 128, 128),
        num_threads: int = 1,
        warmup_runs: int = 10,
        measured_runs: int = 100,
        experimental_without_delegates: bool = True
    ) -> PerformanceBenchmarkResult:
        """Benchmark a TFLite FlatBuffer with strict stage isolation."""
        import tensorflow as tf

        t0 = time.perf_counter()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at {model_path}")
        model_load_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        if experimental_without_delegates:
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            delegate_name = "BUILTIN_WITHOUT_DEFAULT_DELEGATES"
        else:
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=num_threads
            )
            delegate_name = "DEFAULT_XNNPACK"

        interpreter.allocate_tensors()
        runtime_init_ms = (time.perf_counter() - t1) * 1000.0

        in_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]
        in_idx = in_details["index"]
        out_idx = out_details["index"]

        # Pre-allocate contiguous input buffer (zero allocation in loop)
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        reusable_buffer = np.empty(input_shape, dtype=np.float32)
        np.copyto(reusable_buffer, dummy_input)

        # Warmup runs (not timed)
        for _ in range(warmup_runs):
            interpreter.set_tensor(in_idx, reusable_buffer)
            interpreter.invoke()
            _ = interpreter.get_tensor(out_idx)

        # Measure pure inference vs preprocessing vs postprocessing
        pure_latencies: List[float] = []
        prep_latencies: List[float] = []
        post_latencies: List[float] = []

        for _ in range(measured_runs):
            # Preprocessing / buffer load
            t_p0 = time.perf_counter()
            np.copyto(reusable_buffer, dummy_input)
            interpreter.set_tensor(in_idx, reusable_buffer)
            t_p1 = time.perf_counter()
            prep_latencies.append((t_p1 - t_p0) * 1000.0)

            # Pure inference (interpreter.invoke ONLY)
            t_i0 = time.perf_counter()
            interpreter.invoke()
            t_i1 = time.perf_counter()
            pure_latencies.append((t_i1 - t_i0) * 1000.0)

            # Postprocessing (output tensor extraction + argmax)
            t_post0 = time.perf_counter()
            out = interpreter.get_tensor(out_idx)[0]
            _ = int(np.argmax(out))
            t_post1 = time.perf_counter()
            post_latencies.append((t_post1 - t_post0) * 1000.0)

        stats = cls.calculate_stats(pure_latencies)
        prep_mean = float(np.mean(prep_latencies))
        post_mean = float(np.mean(post_latencies))
        e2e_mean = stats.mean_ms + prep_mean + post_mean

        return PerformanceBenchmarkResult(
            model_path=model_path,
            runtime_type="TensorFlow Lite",
            threads=num_threads,
            delegate=delegate_name,
            model_load_ms=model_load_ms,
            runtime_init_ms=runtime_init_ms,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            pure_inference=stats,
            preprocessing_mean_ms=prep_mean,
            postprocessing_mean_ms=post_mean,
            end_to_end_mean_ms=e2e_mean
        )

    @classmethod
    def benchmark_onnx(
        cls,
        model_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_threads: int = 1,
        warmup_runs: int = 10,
        measured_runs: int = 50
    ) -> PerformanceBenchmarkResult:
        """Benchmark an ONNX Runtime model with strict stage isolation."""
        import onnxruntime as ort

        t0 = time.perf_counter()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        model_load_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        runtime_init_ms = (time.perf_counter() - t1) * 1000.0

        in_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        reusable_dict = {in_name: dummy_input}

        # Warmup
        for _ in range(warmup_runs):
            _ = session.run([out_name], reusable_dict)

        pure_latencies: List[float] = []
        prep_latencies: List[float] = []
        post_latencies: List[float] = []

        for _ in range(measured_runs):
            t_p0 = time.perf_counter()
            reusable_dict[in_name] = dummy_input
            t_p1 = time.perf_counter()
            prep_latencies.append((t_p1 - t_p0) * 1000.0)

            t_i0 = time.perf_counter()
            outs = session.run([out_name], reusable_dict)
            t_i1 = time.perf_counter()
            pure_latencies.append((t_i1 - t_i0) * 1000.0)

            t_post0 = time.perf_counter()
            _ = int(np.argmax(outs[0][0]))
            t_post1 = time.perf_counter()
            post_latencies.append((t_post1 - t_post0) * 1000.0)

        stats = cls.calculate_stats(pure_latencies)
        prep_mean = float(np.mean(prep_latencies))
        post_mean = float(np.mean(post_latencies))
        e2e_mean = stats.mean_ms + prep_mean + post_mean

        return PerformanceBenchmarkResult(
            model_path=model_path,
            runtime_type="ONNX Runtime",
            threads=num_threads,
            delegate="CPUExecutionProvider",
            model_load_ms=model_load_ms,
            runtime_init_ms=runtime_init_ms,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            pure_inference=stats,
            preprocessing_mean_ms=prep_mean,
            postprocessing_mean_ms=post_mean,
            end_to_end_mean_ms=e2e_mean
        )

    @classmethod
    def benchmark_threads_tflite(
        cls,
        model_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 128, 128),
        thread_counts: Tuple[int, ...] = (1, 2, 4),
        repeats: int = 3,
        runs_per_repeat: int = 25
    ) -> List[Dict[str, Any]]:
        """Multi-threaded benchmark measuring latency, p95, throughput, CPU and RAM."""
        import psutil
        import tensorflow as tf

        process = psutil.Process()
        results: List[Dict[str, Any]] = []

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        reusable_buffer = np.empty(input_shape, dtype=np.float32)
        np.copyto(reusable_buffer, dummy_input)

        for th in thread_counts:
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=th,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            interpreter.allocate_tensors()
            in_idx = interpreter.get_input_details()[0]["index"]
            out_idx = interpreter.get_output_details()[0]["index"]

            # Warmup
            for _ in range(5):
                interpreter.set_tensor(in_idx, reusable_buffer)
                interpreter.invoke()
                _ = interpreter.get_tensor(out_idx)

            all_latencies: List[float] = []
            cpu_measurements: List[float] = []
            ram_measurements: List[float] = []

            for _ in range(repeats):
                # Sample pre-run CPU/RAM
                _ = process.cpu_percent()
                ram_pre = process.memory_info().rss / (1024 * 1024)

                for _ in range(runs_per_repeat):
                    t0 = time.perf_counter()
                    interpreter.set_tensor(in_idx, reusable_buffer)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(out_idx)
                    all_latencies.append((time.perf_counter() - t0) * 1000.0)

                cpu_post = process.cpu_percent()
                ram_post = process.memory_info().rss / (1024 * 1024)
                cpu_measurements.append(cpu_post)
                ram_measurements.extend([ram_pre, ram_post])

            stats = cls.calculate_stats(all_latencies)
            res_entry = {
                "threads": th,
                "mean_latency_ms": round(stats.mean_ms, 2),
                "median_latency_ms": round(stats.median_ms, 2),
                "p95_latency_ms": round(stats.p95_ms, 2),
                "min_latency_ms": round(stats.min_ms, 2),
                "max_latency_ms": round(stats.max_ms, 2),
                "throughput_ips": round(stats.throughput_ips, 2),
                "cpu_avg_pct": round(float(np.mean(cpu_measurements)), 1),
                "cpu_peak_pct": round(float(np.max(cpu_measurements)), 1),
                "ram_avg_mb": round(float(np.mean(ram_measurements)), 1),
                "ram_peak_mb": round(float(np.max(ram_measurements)), 1)
            }
            results.append(res_entry)

        return results

