"""
UAQE Phase D.5 Runtime Benchmarker
Executes deterministic cold-start breakdown benchmarking, warm inference timing,
process memory profiling, and 196-image prediction agreement verification.
"""

from __future__ import annotations

import os
import time
import gc
import psutil
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

from src.uaqe.runtime.runtime_decoder import RuntimeDecoder
from src.uaqe.runtime.runtime_session import RuntimeSession
from src.uaqe.optimizer.sensitivity_pruner import CLASS_NAMES, CLASS_TO_IDX


class RuntimeBenchmarker:
    """Orchestrates comprehensive runtime performance and verification benchmarks."""

    def __init__(
        self,
        archive_path: str = "output/phase_d4/compressed/d4_d_adaptive_sparse_rle.bin",
        baseline_c4_path: str = "output/phase_c4/models/c4_best_int8.tflite",
        d4_d_model_path: str = "output/phase_d4/models/d4_d_adaptive_sparse_rle.tflite",
        dataset_root: str = "D:\\semiconductor_dataset\\dataset",
        output_dir: str = "output/phase_d5"
    ):
        self.archive_path = archive_path
        self.baseline_c4_path = baseline_c4_path
        self.d4_d_model_path = d4_d_model_path
        self.dataset_root = dataset_root
        self.output_dir = output_dir

        self.benchmarks_dir = os.path.join(self.output_dir, "benchmarks")
        self.verification_dir = os.path.join(self.output_dir, "verification")
        self.models_dir = os.path.join(self.output_dir, "models")

        for d in [self.output_dir, self.benchmarks_dir, self.verification_dir, self.models_dir]:
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

    def run_cold_start_benchmark(self, repetitions: int = 30) -> Dict[str, Any]:
        """Measures cold-start breakdown over multiple repetitions."""
        records = []

        for r in range(repetitions):
            gc.collect()

            # 1. Load archive
            t0 = time.perf_counter()
            decoder = RuntimeDecoder(base_template_path=self.baseline_c4_path)
            decoder.load(self.archive_path)
            t_load = (time.perf_counter() - t0) * 1000.0

            # 2. Decode tensor blocks
            t0 = time.perf_counter()
            decoded = decoder.decode()
            t_decode = (time.perf_counter() - t0) * 1000.0

            # 3. FlatBuffer Reconstruction
            t0 = time.perf_counter()
            model_bytes = decoder.reconstruct(self.baseline_c4_path)
            t_reconstruct = (time.perf_counter() - t0) * 1000.0

            # 4. TFLite Interpreter Initialization
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
        csv_path = os.path.join(self.benchmarks_dir, "d5_cold_start.csv")
        df.to_csv(csv_path, index=False)

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

    def run_warm_inference_benchmark(self, iterations: int = 100) -> Dict[str, Any]:
        """Measures warm inference latency on the reconstructed runtime session."""
        session = RuntimeSession.from_archive(self.archive_path, template_tflite_path=self.baseline_c4_path)
        session.allocate()

        sample_input = self.test_x[0:1]
        bench_res = session.benchmark(input_data=sample_input, num_runs=iterations, warmup_runs=15)

        records = [{
            "metric": k,
            "value": v
        } for k, v in bench_res.items()]

        df = pd.DataFrame(records)
        csv_path = os.path.join(self.benchmarks_dir, "d5_warm_inference.csv")
        df.to_csv(csv_path, index=False)

        session.close()
        return bench_res

    def run_memory_measurement(self) -> Dict[str, Any]:
        """Measures process RSS memory during the decoding and allocation lifecycle."""
        gc.collect()
        process = psutil.Process(os.getpid())
        base_rss = process.memory_info().rss / 1024.0  # in KB

        # 1. Archive on disk
        archive_size = os.path.getsize(self.archive_path)

        # 2. Decode
        decoder = RuntimeDecoder(base_template_path=self.baseline_c4_path)
        decoder.load(self.archive_path)
        post_load_rss = process.memory_info().rss / 1024.0

        decoded_tensors = decoder.decode()
        decoded_bytes_total = sum(t["data"].nbytes for t in decoded_tensors)
        post_decode_rss = process.memory_info().rss / 1024.0

        # 3. Reconstruct FlatBuffer
        reconstructed_bytes = decoder.reconstruct(self.baseline_c4_path)
        reconstructed_size = len(reconstructed_bytes)
        post_reconstruct_rss = process.memory_info().rss / 1024.0

        # 4. Instantiate Interpreter and Allocate
        session = RuntimeSession(decoder=decoder, template_tflite_path=self.baseline_c4_path)
        session.load()
        session.allocate()
        post_alloc_rss = process.memory_info().rss / 1024.0

        incremental_decode_kb = max(0.0, post_decode_rss - base_rss)
        runtime_memory_kb = max(0.0, post_alloc_rss - base_rss)

        memory_metrics = [
            {"metric": "base_process_rss_kb", "value": round(base_rss, 2)},
            {"metric": "d4_d_archive_size_bytes", "value": archive_size},
            {"metric": "d4_d_archive_size_mb", "value": round(archive_size / (1024 * 1024), 4)},
            {"metric": "decoded_weights_memory_bytes", "value": decoded_bytes_total},
            {"metric": "reconstructed_tflite_size_bytes", "value": reconstructed_size},
            {"metric": "reconstructed_tflite_size_mb", "value": round(reconstructed_size / (1024 * 1024), 4)},
            {"metric": "post_decode_rss_kb", "value": round(post_decode_rss, 2)},
            {"metric": "post_allocation_rss_kb", "value": round(post_alloc_rss, 2)},
            {"metric": "incremental_decode_overhead_kb", "value": round(incremental_decode_kb, 2)},
            {"metric": "runtime_memory_overhead_kb", "value": round(runtime_memory_kb, 2)}
        ]

        df = pd.DataFrame(memory_metrics)
        csv_path = os.path.join(self.benchmarks_dir, "d5_memory.csv")
        df.to_csv(csv_path, index=False)

        session.close()
        return {m["metric"]: m["value"] for m in memory_metrics}

    def run_prediction_agreement_test(self) -> Dict[str, Any]:
        """Runs the clean 196-image benchmark on C4 baseline, D4-D offline, and D5 runtime."""
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

        # 3. Evaluate D5 Runtime Reconstructed Session
        session = RuntimeSession.from_archive(self.archive_path, template_tflite_path=self.baseline_c4_path)
        session.allocate()
        d5_preds = session.predict_class(self.test_x)
        d5_preds_np = np.array(d5_preds)

        # Metrics
        total = len(self.test_y)
        c4_acc = float(accuracy_score(self.test_y, c4_preds_np))
        d4_acc = float(accuracy_score(self.test_y, d4_preds_np))
        d5_acc = float(accuracy_score(self.test_y, d5_preds_np))

        p_m, r_m, f1_m, _ = precision_recall_fscore_support(self.test_y, d5_preds_np, average="macro", zero_division=0)

        # Observed Prediction Agreements
        c4_d5_agree = float(np.mean(c4_preds_np == d5_preds_np) * 100.0)
        d4_d5_agree = float(np.mean(d4_preds_np == d5_preds_np) * 100.0)

        # Export prediction comparison CSV
        pred_df = pd.DataFrame({
            "image_path": self.test_paths,
            "true_class": [CLASS_NAMES[y] for y in self.test_y],
            "c4_pred": [CLASS_NAMES[p] for p in c4_preds_np],
            "d4_pred": [CLASS_NAMES[p] for p in d4_preds_np],
            "d5_pred": [CLASS_NAMES[p] for p in d5_preds_np],
            "d4_d5_match": (d4_preds_np == d5_preds_np),
            "c4_d5_match": (c4_preds_np == d5_preds_np)
        })
        pred_csv = os.path.join(self.verification_dir, "d5_prediction_comparison.csv")
        pred_df.to_csv(pred_csv, index=False)

        # Save reconstructed runtime model to models/
        d5_model_path = os.path.join(self.models_dir, "d5_reconstructed.tflite")
        decoder = RuntimeDecoder(base_template_path=self.baseline_c4_path)
        decoder.load(self.archive_path)
        decoder.reconstruct_to_file(d5_model_path)

        session.close()

        return {
            "total_images": total,
            "c4_accuracy_pct": round(c4_acc * 100.0, 4),
            "d4_accuracy_pct": round(d4_acc * 100.0, 4),
            "d5_accuracy_pct": round(d5_acc * 100.0, 4),
            "d5_correct": int(np.sum(d5_preds_np == self.test_y)),
            "d5_macro_precision_pct": round(p_m * 100.0, 4),
            "d5_macro_recall_pct": round(r_m * 100.0, 4),
            "d5_macro_f1_pct": round(f1_m * 100.0, 4),
            "c4_to_d5_agreement_pct": round(c4_d5_agree, 4),
            "d4_to_d5_agreement_pct": round(d4_d5_agree, 4)
        }
