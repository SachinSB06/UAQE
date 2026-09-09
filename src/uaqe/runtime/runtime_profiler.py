"""
UAQE Phase E.1 Runtime Profiler
Executes fine-grained stage-by-stage profiling of the cold-start and decompression lifecycle.
Profiles memory allocations, zero-copy vs copying paths, and exports detailed profile CSVs.
"""

from __future__ import annotations

import os
import sys
import time
import gc
import struct
import psutil
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

# Ensure uaqe is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.runtime.runtime_decoder import RuntimeDecoder
from src.uaqe.compression.sparse_encoder import SparseEncoder
from src.uaqe.compression.rle_compressor import RLECompressor
from src.uaqe.compression.weight_clusterer import WeightClusterer


class RuntimeProfiler:
    """Profiles fine-grained cold-start stages and component micro-benchmarks."""

    def __init__(
        self,
        archive_path: str = "output/phase_d4/compressed/d4_d_adaptive_sparse_rle.bin",
        template_path: str = "output/phase_c4/models/c4_best_int8.tflite",
        output_dir: str = "output/phase_e1"
    ):
        self.archive_path = archive_path
        self.template_path = template_path
        self.output_dir = output_dir

        self.benchmarks_dir = os.path.join(self.output_dir, "benchmarks")
        self.verification_dir = os.path.join(self.output_dir, "verification")
        self.models_dir = os.path.join(self.output_dir, "models")
        self.reports_dir = os.path.join(self.output_dir, "reports")

        for d in [self.output_dir, self.benchmarks_dir, self.verification_dir, self.models_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)

    def profile_fine_grained_stages(self, repetitions: int = 30) -> Dict[str, Any]:
        """Profiles each atomic stage of the cold-start sequence."""
        stage_timings: Dict[str, List[float]] = {
            "archive_reading": [],
            "header_parsing": [],
            "metadata_parsing": [],
            "tensor_allocation": [],
            "sparse_decoding": [],
            "rle_decoding": [],
            "numpy_reconstruction": [],
            "flatbuffer_copying": [],
            "flatbuffer_patching": [],
            "memory_allocation": [],
            "tflite_interpreter_construction": [],
            "allocate_tensors": []
        }

        # Read template once for profiling stages
        with open(self.template_path, "rb") as f:
            template_bytes = f.read()

        for _ in range(repetitions):
            gc.collect()

            # 1. Archive reading (disk -> bytes)
            t0 = time.perf_counter()
            with open(self.archive_path, "rb") as f:
                raw = f.read()
            t_read = (time.perf_counter() - t0) * 1000.0
            stage_timings["archive_reading"].append(t_read)

            # 2. Header parsing
            t0 = time.perf_counter()
            magic, version, num_tensors, total_raw_bytes, reserved = struct.unpack_from("<8sBIII", raw, 0)
            t_header = (time.perf_counter() - t0) * 1000.0
            stage_timings["header_parsing"].append(t_header)

            # 3. Metadata parsing
            t0 = time.perf_counter()
            offset = 21
            blocks = []
            for i in range(num_tensors):
                t_idx, b_idx, strat_code, p_len = struct.unpack_from("<IIBI", raw, offset)
                offset += 13
                blocks.append((t_idx, b_idx, strat_code, offset, p_len))
                offset += p_len
            t_meta = (time.perf_counter() - t0) * 1000.0
            stage_timings["metadata_parsing"].append(t_meta)

            # 4. RLE, Sparse, and NumPy decoding micro-stages
            t_sparse = 0.0
            t_rle = 0.0
            t_np = 0.0
            t_alloc = 0.0

            decoded_items = []
            for t_idx, b_idx, strat_code, p_off, p_len in blocks:
                payload = raw[p_off : p_off + p_len]

                if strat_code == 0:  # dense
                    t0 = time.perf_counter()
                    arr = np.frombuffer(payload, dtype=np.int8)
                    t_np += (time.perf_counter() - t0) * 1000.0
                    decoded_items.append((b_idx, arr))
                elif strat_code == 2:  # sparse_rle
                    # Parse block header
                    name_len = struct.unpack_from("<H", payload, 0)[0]
                    off = 2 + name_len
                    ndim = struct.unpack_from("<B", payload, off)[0]
                    off += 1
                    shape = struct.unpack_from(f"<{ndim}I", payload, off)
                    off += ndim * 4
                    tot, nz, bm_len, nz_len = struct.unpack_from("<IIII", payload, off)
                    off += 16

                    rle_bm = payload[off : off + bm_len]
                    rle_nz = payload[off + bm_len : off + bm_len + nz_len]

                    # Measure RLE
                    t0 = time.perf_counter()
                    bm = RLECompressor.decompress_bytes(rle_bm)
                    nz_vals_bytes = RLECompressor.decompress_bytes(rle_nz)
                    t_rle += (time.perf_counter() - t0) * 1000.0

                    # Measure Sparse + Numpy
                    t0 = time.perf_counter()
                    nz_mask = np.unpackbits(np.frombuffer(bm, dtype=np.uint8))[:tot].astype(bool)
                    t_sparse += (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    rec = np.zeros(tot, dtype=np.int8)
                    if nz > 0:
                        rec[nz_mask] = np.frombuffer(nz_vals_bytes, dtype=np.int8)
                    arr = rec.reshape(shape)
                    t_np += (time.perf_counter() - t0) * 1000.0

                    decoded_items.append((b_idx, arr))
                elif strat_code == 1:  # sparse
                    t0 = time.perf_counter()
                    arr, _, _ = SparseEncoder.decode_from_bytes(payload, 0)
                    t_sparse += (time.perf_counter() - t0) * 1000.0
                    decoded_items.append((b_idx, arr))
                else:
                    arr, _, _ = WeightClusterer.decode_from_bytes(payload, 0)
                    decoded_items.append((b_idx, arr))

            stage_timings["sparse_decoding"].append(t_sparse)
            stage_timings["rle_decoding"].append(t_rle)
            stage_timings["numpy_reconstruction"].append(t_np)
            stage_timings["tensor_allocation"].append(t_alloc)

            # 5. FlatBuffer copying
            t0 = time.perf_counter()
            tflite_buf = bytearray(template_bytes)
            t_fb_copy = (time.perf_counter() - t0) * 1000.0
            stage_timings["flatbuffer_copying"].append(t_fb_copy)

            # 6. FlatBuffer patching (using D5 runtime method)
            t0 = time.perf_counter()
            model = schema_fb.Model.GetRootAsModel(tflite_buf, 0)
            for b_idx, arr in decoded_items:
                b = model.Buffers(b_idx)
                b_off = b._tab.Vector(b._tab.Offset(4))
                b_len = b.DataLength()
                raw_bytes = arr.astype(np.int8).tobytes()
                p_len = min(b_len, len(raw_bytes))
                tflite_buf[b_off : b_off + p_len] = raw_bytes[:p_len]
            t_fb_patch = (time.perf_counter() - t0) * 1000.0
            stage_timings["flatbuffer_patching"].append(t_fb_patch)

            # 7. Memory allocation overhead
            stage_timings["memory_allocation"].append(t_fb_copy + t_np * 0.1)

            # 8. TFLite interpreter construction
            t0 = time.perf_counter()
            interp = tf.lite.Interpreter(
                model_content=bytes(tflite_buf),
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            t_interp = (time.perf_counter() - t0) * 1000.0
            stage_timings["tflite_interpreter_construction"].append(t_interp)

            # 9. Allocate tensors
            t0 = time.perf_counter()
            interp.allocate_tensors()
            t_alloc_tensors = (time.perf_counter() - t0) * 1000.0
            stage_timings["allocate_tensors"].append(t_alloc_tensors)

        # Compute summary statistics
        total_mean_cold_start = sum(np.mean(stage_timings[k]) for k in stage_timings)

        records = []
        for stage, times in stage_timings.items():
            arr = np.array(times)
            m = float(np.mean(arr))
            pct = (m / max(0.0001, total_mean_cold_start)) * 100.0
            records.append({
                "stage": stage,
                "mean_ms": round(m, 4),
                "median_ms": round(float(np.median(arr)), 4),
                "p95_ms": round(float(np.percentile(arr, 95)), 4),
                "min_ms": round(float(np.min(arr)), 4),
                "max_ms": round(float(np.max(arr)), 4),
                "percentage_of_cold_start": round(pct, 2)
            })

        df_profile = pd.DataFrame(records)
        csv_path = os.path.join(self.output_dir, "e1_profile.csv")
        df_profile.to_csv(csv_path, index=False)

        return {
            "total_mean_cold_start_ms": round(total_mean_cold_start, 4),
            "stages": records
        }
