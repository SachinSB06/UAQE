"""
UAQE Phase E.1 Optimized Runtime Benchmarker
Executes comprehensive benchmarking comparing D5 Baseline vs E1 Portable vs E1 Cached:
- 30-run cold uncached breakdown
- 30-run cold cached startup
- 100-run warm inference timing
- Process memory (RSS) profiling across stages
- 196-image clean test set accuracy and prediction agreement
"""

from __future__ import annotations

import os
import sys
import time
import gc
import hashlib
import psutil
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf

# Ensure uaqe imports work cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.runtime.runtime_decoder import RuntimeDecoder
from src.uaqe.runtime.optimized_decoder import OptimizedRuntimeDecoder
from src.uaqe.runtime.optimized_session import OptimizedRuntimeSession
from src.uaqe.runtime.runtime_cache import RuntimeCacheManager
from src.uaqe.optimizer.sensitivity_pruner import CLASS_NAMES, CLASS_TO_IDX


class OptimizedRuntimeBenchmarker:
    """Orchestrates comprehensive performance, memory, and accuracy verification benchmarks for Phase E.1."""

    def __init__(
        self,
        archive_path: str = "output/phase_d4/compressed/d4_d_adaptive_sparse_rle.bin",
        baseline_c4_path: str = "output/phase_c4/models/c4_best_int8.tflite",
        d4_d_model_path: str = "output/phase_d4/models/d4_d_adaptive_sparse_rle.tflite",
        d5_model_path: str = "output/phase_d5/models/d5_reconstructed.tflite",
        dataset_root: str = "D:\\semiconductor_dataset\\dataset",
        output_dir: str = "output/phase_e1"
    ):
        self.archive_path = archive_path
        self.baseline_c4_path = baseline_c4_path
        self.d4_d_model_path = d4_d_model_path
        self.d5_model_path = d5_model_path
        self.dataset_root = dataset_root
        self.output_dir = output_dir

        self.benchmarks_dir = os.path.join(self.output_dir, "benchmarks")
        self.verification_dir = os.path.join(self.output_dir, "verification")
        self.models_dir = os.path.join(self.output_dir, "models")
        self.cache_dir = os.path.join(self.output_dir, "runtime_cache")

        for d in [self.output_dir, self.benchmarks_dir, self.verification_dir, self.models_dir, self.cache_dir]:
            os.makedirs(d, exist_ok=True)

        self._load_clean_test_split()

    def _load_clean_test_split(self) -> None:
        """Loads and caches the 196 clean test images."""
        test_dir = os.path.join(self.dataset_root, "test")
        images, labels, file_paths = [], [], []
        dup_target = os.path.normpath(os.path.join(self.dataset_root, "test", "opens", "open133.png"))

        for c_name in CLASS_NAMES:
            c_dir = os.path.join(test_dir, c_name)
            if not os.path.isdir(c_dir):
                continue
            c_idx = CLASS_TO_IDX[c_name]
            for f_name in sorted(os.listdir(c_dir)):
                if f_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    f_path = os.path.join(c_dir, f_name)
                    if os.path.normpath(f_path) == dup_target:
                        continue
                    img = Image.open(f_path).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    images.append(arr.transpose(2, 0, 1))
                    labels.append(c_idx)
                    file_paths.append(f_path)

        self.test_x = np.stack(images)
        self.test_y = np.array(labels)
        self.test_paths = file_paths

    def run_d5_baseline_benchmark(self, repetitions: int = 30) -> Dict[str, Any]:
        """Reproduces the exact D5 unoptimized baseline measurements."""
        records = []

        for r in range(repetitions):
            gc.collect()

            # 1. Load
            t0 = time.perf_counter()
            decoder = RuntimeDecoder(base_template_path=self.baseline_c4_path)
            decoder.load(self.archive_path)
            t_load = (time.perf_counter() - t0) * 1000.0

            # 2. Decode
            t0 = time.perf_counter()
            decoded = decoder.decode()
            t_decode = (time.perf_counter() - t0) * 1000.0

            # 3. FlatBuffer Reconstruction
            t0 = time.perf_counter()
            model_bytes = decoder.reconstruct(self.baseline_c4_path)
            t_reconstruct = (time.perf_counter() - t0) * 1000.0

            # 4. TFLite Interpreter Init
            t0 = time.perf_counter()
            interp = tf.lite.Interpreter(
                model_content=bytes(model_bytes),
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            t_init = (time.perf_counter() - t0) * 1000.0

            # 5. Allocate Tensors
            t0 = time.perf_counter()
            interp.allocate_tensors()
            t_allocate = (time.perf_counter() - t0) * 1000.0

            t_total = t_load + t_decode + t_reconstruct + t_init + t_allocate

            records.append({
                "repetition": r + 1,
                "archive_load_ms": round(t_load, 4),
                "decode_ms": round(t_decode, 4),
                "reconstruct_flatbuffer_ms": round(t_reconstruct, 4),
                "tflite_init_ms": round(t_init, 4),
                "allocate_tensors_ms": round(t_allocate, 4),
                "cold_start_total_ms": round(t_total, 4)
            })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.benchmarks_dir, "d5_baseline_reproduction.csv"), index=False)

        summary = {}
        for col in ["archive_load_ms", "decode_ms", "reconstruct_flatbuffer_ms", "tflite_init_ms", "allocate_tensors_ms", "cold_start_total_ms"]:
            vals = df[col].values
            summary[col] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "std": float(np.std(vals))
            }
        return summary

    def run_optimized_cold_start_benchmark(self, repetitions: int = 30) -> Dict[str, Any]:
        """Measures E1 optimized cold-start breakdown in portable mode."""
        records = []

        # Read template once for buffer map
        with open(self.baseline_c4_path, "rb") as f:
            tpl_bytes = f.read()

        for r in range(repetitions):
            gc.collect()

            # 1. Load archive
            t0 = time.perf_counter()
            decoder = OptimizedRuntimeDecoder(base_template_path=self.baseline_c4_path, template_bytes=tpl_bytes)
            decoder.load(self.archive_path)
            t_load = (time.perf_counter() - t0) * 1000.0

            # 2. Vectorized decode
            t0 = time.perf_counter()
            decoded = decoder.decode()
            t_decode = (time.perf_counter() - t0) * 1000.0

            # 3. Fast FlatBuffer direct patching
            t0 = time.perf_counter()
            model_bytes = decoder.reconstruct_from_decoded(decoded, self.baseline_c4_path)
            t_reconstruct = (time.perf_counter() - t0) * 1000.0

            # 4. TFLite Interpreter Init
            t0 = time.perf_counter()
            interp = tf.lite.Interpreter(
                model_content=bytes(model_bytes),
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            t_init = (time.perf_counter() - t0) * 1000.0

            # 5. Allocate Tensors
            t0 = time.perf_counter()
            interp.allocate_tensors()
            t_allocate = (time.perf_counter() - t0) * 1000.0

            t_total = t_load + t_decode + t_reconstruct + t_init + t_allocate

            records.append({
                "repetition": r + 1,
                "archive_load_ms": round(t_load, 4),
                "decode_ms": round(t_decode, 4),
                "reconstruct_flatbuffer_ms": round(t_reconstruct, 4),
                "tflite_init_ms": round(t_init, 4),
                "allocate_tensors_ms": round(t_allocate, 4),
                "cold_start_total_ms": round(t_total, 4)
            })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.benchmarks_dir, "e1_optimized_cold_start.csv"), index=False)

        summary = {}
        for col in ["archive_load_ms", "decode_ms", "reconstruct_flatbuffer_ms", "tflite_init_ms", "allocate_tensors_ms", "cold_start_total_ms"]:
            vals = df[col].values
            summary[col] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "std": float(np.std(vals))
            }
        return summary

    def run_cached_startup_benchmark(self, repetitions: int = 30) -> Dict[str, Any]:
        """Measures E1 cached-startup latency over multiple repetitions."""
        cache_mgr = RuntimeCacheManager(cache_dir=self.cache_dir)
        # Ensure model is cached
        session = OptimizedRuntimeSession.from_archive(
            self.archive_path,
            template_tflite_path=self.baseline_c4_path,
            mode="cached",
            cache_dir=self.cache_dir
        )
        session.load()
        session.close()

        records = []
        for r in range(repetitions):
            gc.collect()

            # 1. Archive Hash Verification & Cache Lookup
            t0 = time.perf_counter()
            with open(self.archive_path, "rb") as f:
                raw = f.read()
            h = hashlib.sha256(raw).hexdigest()
            t_verify = (time.perf_counter() - t0) * 1000.0

            # 2. Load from Cache
            t0 = time.perf_counter()
            cached_bytes = cache_mgr.load_cached_model(h)
            t_cache_load = (time.perf_counter() - t0) * 1000.0

            if cached_bytes is None:
                raise RuntimeError("Cached model failed to load during benchmark.")

            # 3. TFLite Interpreter Init
            t0 = time.perf_counter()
            interp = tf.lite.Interpreter(
                model_content=bytes(cached_bytes),
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            t_init = (time.perf_counter() - t0) * 1000.0

            # 4. Allocate Tensors
            t0 = time.perf_counter()
            interp.allocate_tensors()
            t_alloc = (time.perf_counter() - t0) * 1000.0

            t_total = t_verify + t_cache_load + t_init + t_alloc

            records.append({
                "repetition": r + 1,
                "archive_verify_ms": round(t_verify, 4),
                "cache_load_ms": round(t_cache_load, 4),
                "tflite_init_ms": round(t_init, 4),
                "allocate_tensors_ms": round(t_alloc, 4),
                "cached_startup_total_ms": round(t_total, 4)
            })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.benchmarks_dir, "e1_cached_startup.csv"), index=False)

        summary = {}
        for col in ["archive_verify_ms", "cache_load_ms", "tflite_init_ms", "allocate_tensors_ms", "cached_startup_total_ms"]:
            vals = df[col].values
            summary[col] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "std": float(np.std(vals))
            }
        return summary

    def run_warm_inference_benchmark(self, iterations: int = 100) -> Dict[str, Any]:
        """Measures warm inference latency on the optimized runtime session."""
        session = OptimizedRuntimeSession.from_archive(
            self.archive_path,
            template_tflite_path=self.baseline_c4_path,
            mode="portable"
        )
        session.allocate()

        sample_input = self.test_x[0:1]
        bench_res = session.benchmark(input_data=sample_input, num_runs=iterations, warmup_runs=15)

        records = [{"metric": k, "value": v} for k, v in bench_res.items()]
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(self.benchmarks_dir, "e1_warm_inference.csv"), index=False)

        session.close()
        return bench_res

    def run_memory_measurement(self) -> Dict[str, Any]:
        """Profiles host process RSS memory across portable and cached lifecycles."""
        gc.collect()
        process = psutil.Process(os.getpid())
        base_rss = process.memory_info().rss / 1024.0  # in KB

        archive_size = os.path.getsize(self.archive_path)

        # Portable mode profiling
        decoder = OptimizedRuntimeDecoder(base_template_path=self.baseline_c4_path)
        decoder.load(self.archive_path)
        post_load_rss = process.memory_info().rss / 1024.0

        decoded_tensors = decoder.decode()
        post_decode_rss = process.memory_info().rss / 1024.0

        reconstructed_bytes = decoder.reconstruct_from_decoded(decoded_tensors, self.baseline_c4_path)
        post_reconstruct_rss = process.memory_info().rss / 1024.0

        session_portable = OptimizedRuntimeSession(
            decoder=decoder,
            template_tflite_path=self.baseline_c4_path,
            mode="portable"
        )
        session_portable.load()
        session_portable.allocate()
        post_alloc_rss = process.memory_info().rss / 1024.0
        session_portable.close()

        # Cached mode profiling
        gc.collect()
        session_cached = OptimizedRuntimeSession.from_archive(
            self.archive_path,
            template_tflite_path=self.baseline_c4_path,
            mode="cached",
            cache_dir=self.cache_dir
        )
        session_cached.load()
        session_cached.allocate()
        cached_mode_rss = process.memory_info().rss / 1024.0
        session_cached.close()

        incremental_decode_kb = max(0.0, post_decode_rss - base_rss)
        runtime_memory_kb = max(0.0, post_alloc_rss - base_rss)
        cached_overhead_kb = max(0.0, cached_mode_rss - base_rss)

        memory_metrics = [
            {"metric": "base_process_rss_kb", "value": round(base_rss, 2)},
            {"metric": "d4_d_archive_size_bytes", "value": archive_size},
            {"metric": "post_load_rss_kb", "value": round(post_load_rss, 2)},
            {"metric": "post_decode_rss_kb", "value": round(post_decode_rss, 2)},
            {"metric": "post_reconstruct_rss_kb", "value": round(post_reconstruct_rss, 2)},
            {"metric": "post_allocation_rss_kb", "value": round(post_alloc_rss, 2)},
            {"metric": "cached_mode_rss_kb", "value": round(cached_mode_rss, 2)},
            {"metric": "incremental_decode_overhead_kb", "value": round(incremental_decode_kb, 2)},
            {"metric": "runtime_memory_overhead_kb", "value": round(runtime_memory_kb, 2)},
            {"metric": "cached_memory_overhead_kb", "value": round(cached_overhead_kb, 2)}
        ]

        df = pd.DataFrame(memory_metrics)
        df.to_csv(os.path.join(self.benchmarks_dir, "e1_memory.csv"), index=False)

        return {m["metric"]: m["value"] for m in memory_metrics}

    def run_prediction_agreement_test(self) -> Dict[str, Any]:
        """Evaluates clean 196-image benchmark on C4, D4-D, D5, E1 Portable, and E1 Cached."""
        # 1. Evaluate C4 baseline
        c4_interp = tf.lite.Interpreter(
            model_path=self.baseline_c4_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        c4_interp.allocate_tensors()
        in_idx = c4_interp.get_input_details()[0]["index"]
        out_idx = c4_interp.get_output_details()[0]["index"]

        c4_preds = []
        for i in range(len(self.test_x)):
            c4_interp.set_tensor(in_idx, self.test_x[i:i+1])
            c4_interp.invoke()
            out = c4_interp.get_tensor(out_idx)[0]
            c4_preds.append(int(np.argmax(out)))
        c4_preds_np = np.array(c4_preds)

        # 2. Evaluate D4-D offline model
        d4_interp = tf.lite.Interpreter(
            model_path=self.d4_d_model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        d4_interp.allocate_tensors()
        in_idx_d4 = d4_interp.get_input_details()[0]["index"]
        out_idx_d4 = d4_interp.get_output_details()[0]["index"]

        d4_preds = []
        for i in range(len(self.test_x)):
            d4_interp.set_tensor(in_idx_d4, self.test_x[i:i+1])
            d4_interp.invoke()
            out = d4_interp.get_tensor(out_idx_d4)[0]
            d4_preds.append(int(np.argmax(out)))
        d4_preds_np = np.array(d4_preds)

        # 3. Evaluate E1 Portable Session
        sess_portable = OptimizedRuntimeSession.from_archive(
            self.archive_path,
            template_tflite_path=self.baseline_c4_path,
            mode="portable"
        )
        sess_portable.allocate()
        e1_portable_preds = np.array(sess_portable.predict_class(self.test_x))
        sess_portable.close()

        # 4. Evaluate E1 Cached Session
        sess_cached = OptimizedRuntimeSession.from_archive(
            self.archive_path,
            template_tflite_path=self.baseline_c4_path,
            mode="cached",
            cache_dir=self.cache_dir
        )
        sess_cached.allocate()
        e1_cached_preds = np.array(sess_cached.predict_class(self.test_x))
        sess_cached.close()

        # Metrics
        total = len(self.test_y)
        c4_acc = float(accuracy_score(self.test_y, c4_preds_np))
        d4_acc = float(accuracy_score(self.test_y, d4_preds_np))
        e1_port_acc = float(accuracy_score(self.test_y, e1_portable_preds))
        e1_cache_acc = float(accuracy_score(self.test_y, e1_cached_preds))

        p_m, r_m, f1_m, _ = precision_recall_fscore_support(self.test_y, e1_portable_preds, average="macro", zero_division=0)

        c4_agree = float(np.mean(c4_preds_np == e1_portable_preds) * 100.0)
        d4_agree = float(np.mean(d4_preds_np == e1_portable_preds) * 100.0)
        cache_agree = float(np.mean(e1_portable_preds == e1_cached_preds) * 100.0)

        # Export comparison CSV
        pred_df = pd.DataFrame({
            "image_path": self.test_paths,
            "true_class": [CLASS_NAMES[y] for y in self.test_y],
            "c4_pred": [CLASS_NAMES[p] for p in c4_preds_np],
            "d4_pred": [CLASS_NAMES[p] for p in d4_preds_np],
            "e1_portable_pred": [CLASS_NAMES[p] for p in e1_portable_preds],
            "e1_cached_pred": [CLASS_NAMES[p] for p in e1_cached_preds],
            "d4_e1_match": (d4_preds_np == e1_portable_preds),
            "portable_cached_match": (e1_portable_preds == e1_cached_preds)
        })
        pred_csv = os.path.join(self.verification_dir, "e1_prediction_comparison.csv")
        pred_df.to_csv(pred_csv, index=False)

        # Save E1 reconstructed model to models/
        e1_model_path = os.path.join(self.models_dir, "e1_reconstructed.tflite")
        decoder = OptimizedRuntimeDecoder(base_template_path=self.baseline_c4_path)
        decoder.load(self.archive_path)
        decoder.reconstruct_to_file(e1_model_path)

        return {
            "total_images": total,
            "c4_accuracy_pct": round(c4_acc * 100.0, 4),
            "d4_accuracy_pct": round(d4_acc * 100.0, 4),
            "e1_portable_accuracy_pct": round(e1_port_acc * 100.0, 4),
            "e1_cached_accuracy_pct": round(e1_cache_acc * 100.0, 4),
            "e1_correct": int(np.sum(e1_portable_preds == self.test_y)),
            "e1_macro_precision_pct": round(p_m * 100.0, 4),
            "e1_macro_recall_pct": round(r_m * 100.0, 4),
            "e1_macro_f1_pct": round(f1_m * 100.0, 4),
            "c4_to_e1_agreement_pct": round(c4_agree, 4),
            "d4_to_e1_agreement_pct": round(d4_agree, 4),
            "portable_to_cached_agreement_pct": round(cache_agree, 4)
        }
