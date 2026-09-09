"""
UAQE Phase D.4 Adaptive Hybrid Model Packager and Decompression Engine.
Packages pruned and quantized INT8 models using per-layer adaptive strategies
(Dense, Lossless Sparse, Sparse+RLE, Selective Clustering) into unified .bin archives,
and reconstructs executable TFLite FlatBuffers for live inference.
"""

from __future__ import annotations

import os
import json
import struct
import copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.spatial.distance import cosine

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

try:
    from src.uaqe.compression.sparse_encoder import SparseEncoder
    from src.uaqe.compression.rle_compressor import RLECompressor
    from src.uaqe.compression.weight_clusterer import WeightClusterer
except ImportError:
    from uaqe.compression.sparse_encoder import SparseEncoder
    from uaqe.compression.rle_compressor import RLECompressor
    from uaqe.compression.weight_clusterer import WeightClusterer


class AdaptiveModelPackager:
    """Manages per-layer adaptive packaging and exact FlatBuffer reconstruction."""

    MAGIC_HEADER = b"UAQE_D4\x01"

    STRAT_CODES = {
        "dense": 0,
        "KEEP_INT8": 0,
        "sparse": 1,
        "INT8_SPARSE": 1,
        "sparse_rle": 2,
        "INT8_SPARSE_RLE": 2,
        "cluster32": 3,
        "INT8_CLUSTER_32": 3,
        "cluster64": 4,
        "INT8_CLUSTER_64": 4,
        "cluster16": 5,
        "INT8_CLUSTER_16": 5,
        "cluster8": 6,
        "INT8_CLUSTER_8": 6
    }

    REV_STRAT_CODES = {
        0: "dense",
        1: "sparse",
        2: "sparse_rle",
        3: "cluster32",
        4: "cluster64",
        5: "cluster16",
        6: "cluster8"
    }

    def __init__(self, base_tflite_path: str = "output/phase_c4/models/c4_best_int8.tflite"):
        self.base_tflite_path = base_tflite_path
        if os.path.exists(base_tflite_path):
            with open(base_tflite_path, "rb") as f:
                self.template_bytes = bytearray(f.read())
        else:
            self.template_bytes = bytearray()

    def extract_weight_tensors(self, tflite_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extracts all INT8 weight tensors and FlatBuffer buffer offsets from a TFLite model."""
        target_path = tflite_path or self.base_tflite_path
        with open(target_path, "rb") as f:
            raw_buf = bytearray(f.read())

        model = schema_fb.Model.GetRootAsModel(raw_buf, 0)
        subgraph = model.Subgraphs(0)
        
        weight_tensors = []
        for i in range(subgraph.TensorsLength()):
            t = subgraph.Tensors(i)
            b_idx = t.Buffer()
            b = model.Buffers(b_idx)
            
            if b and b.DataLength() > 0 and t.Type() == 9:  # INT8
                t_name = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
                shape = [t.Shape(j) for j in range(t.ShapeLength())]
                data_np = b.DataAsNumpy().astype(np.int8)
                offset = b._tab.Vector(b._tab.Offset(4))
                
                weight_tensors.append({
                    "tensor_index": i,
                    "buffer_index": b_idx,
                    "name": t_name,
                    "shape": shape,
                    "data": data_np,
                    "byte_length": len(data_np),
                    "buffer_offset": offset
                })
                
        return weight_tensors

    @staticmethod
    def apply_layer_pruning(weight_arr: np.ndarray, prune_ratio: float) -> np.ndarray:
        """Applies deterministic magnitude pruning to a single tensor."""
        if prune_ratio <= 0.0 or weight_arr.size == 0:
            return weight_arr.copy()
        
        pruned = weight_arr.copy()
        flat_abs = np.abs(pruned.flatten())
        k = int(math_floor := np.floor(prune_ratio * flat_abs.size))
        if k > 0:
            threshold = np.partition(flat_abs, k)[k]
            # Zero out weights strictly below or at threshold to match count
            mask = flat_abs <= threshold
            # If mask has more than k elements due to ties at 0 or small values, zero exactly smallest
            sorted_indices = np.argsort(flat_abs)
            pruned_flat = pruned.flatten()
            pruned_flat[sorted_indices[:k]] = 0
            pruned = pruned_flat.reshape(weight_arr.shape)
        return pruned

    def package_adaptive_model(
        self,
        tflite_path: str,
        output_bin_path: str,
        plan: List[Dict[str, Any]],
        tflite_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Packages a model using layer-specific adaptive optimization and compression rules."""
        weight_tensors = self.extract_weight_tensors(tflite_path)
        os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)

        plan_map = {}
        buffer_idx_map = {}
        tensor_idx_map = {}
        for item in plan:
            name = item.get("layer_name", "")
            clean_name = name.replace("/", ".").strip(".")
            plan_map[clean_name] = item
            if "buffer_index" in item and item["buffer_index"] != -1:
                buffer_idx_map[item["buffer_index"]] = item
            if "tensor_index" in item and item["tensor_index"] != -1:
                tensor_idx_map[item["tensor_index"]] = item

        packed_records: List[bytes] = []
        tensor_reports: List[Dict[str, Any]] = []
        
        total_raw_weight_bytes = 0
        total_nonzero_weights = 0
        total_weights = 0

        # Also prepare modified tensors for executable TFLite generation
        reconstructed_weights = {}

        for t_info in weight_tensors:
            name = t_info["name"]
            clean_name = name.replace("/", ".").strip(".")
            t_idx = t_info["tensor_index"]
            b_idx = t_info["buffer_index"]
            arr = t_info["data"]
            shape = t_info["shape"]
            raw_len = t_info["byte_length"]

            # Lookup plan: 1) buffer_index, 2) tensor_index, 3) name match
            layer_plan = buffer_idx_map.get(b_idx) or tensor_idx_map.get(t_idx)
            if layer_plan is None:
                for k, v in plan_map.items():
                    if k in clean_name or clean_name in k:
                        layer_plan = v
                        break

            prune_ratio = layer_plan.get("pruning_ratio", 0.0) if layer_plan else 0.0
            strat_name = layer_plan.get("compression_strategy", "dense") if layer_plan else "dense"
            strat_code = self.STRAT_CODES.get(strat_name, 0)

            # 1. Apply layer pruning
            pruned_arr = self.apply_layer_pruning(arr, prune_ratio)
            nz_count = int(np.sum(pruned_arr != 0))

            total_raw_weight_bytes += raw_len
            total_weights += pruned_arr.size
            total_nonzero_weights += nz_count

            # 2. Apply per-layer compression
            is_lossless = True
            mae = 0.0
            rmse = 0.0
            cos_sim = 1.0
            max_err = 0.0

            if strat_code == 0:  # Dense
                payload = pruned_arr.tobytes()
                reconstructed_weights[name] = pruned_arr
            elif strat_code == 1:  # Sparse
                res = SparseEncoder.encode_tensor(pruned_arr, tensor_name=name)
                payload = res["serialized_payload"]
                reconstructed_weights[name] = pruned_arr
            elif strat_code == 2:  # Sparse + RLE
                res = RLECompressor.encode_sparse_rle_tensor(pruned_arr, tensor_name=name)
                payload = res["serialized_payload"]
                reconstructed_weights[name] = pruned_arr
            elif strat_code in [3, 4, 5, 6]:  # Clustering
                k_val = 32 if strat_code == 3 else (64 if strat_code == 4 else (16 if strat_code == 5 else 8))
                res = WeightClusterer.cluster_tensor(pruned_arr, num_clusters=k_val, tensor_name=name)
                payload = res["serialized_payload"]
                mae = res["mae"]
                max_err = res["max_error"]
                rmse = res["rmse"]
                cos_sim = res["cosine_similarity"]
                is_lossless = res["is_lossless"]
                # Decompress clustered array for reconstructed FlatBuffer
                dec_res = WeightClusterer.decompress_tensor(payload, shape=shape)
                reconstructed_weights[name] = dec_res["reconstructed_array"]
            else:
                payload = pruned_arr.tobytes()
                reconstructed_weights[name] = pruned_arr

            # Per-Tensor Block Header:
            # [tensor_index: 4B (I)] [buffer_index: 4B (I)] [strategy_code: 1B (B)] [payload_len: 4B (I)] [payload]
            block_hdr = struct.pack("<IIBI", t_idx, b_idx, strat_code, len(payload))
            packed_records.append(block_hdr + payload)

            tensor_reports.append({
                "tensor_index": t_idx,
                "buffer_index": b_idx,
                "tensor_name": name,
                "shape": str(shape),
                "total_elements": raw_len,
                "nonzero_elements": nz_count,
                "pruning_ratio": prune_ratio,
                "strategy": strat_name,
                "raw_bytes": raw_len,
                "compressed_bytes": len(payload),
                "compression_ratio": float(raw_len / max(1, len(payload))),
                "storage_reduction_pct": float((1.0 - len(payload) / max(1, raw_len)) * 100.0),
                "mae": mae,
                "max_error": max_err,
                "rmse": rmse,
                "cosine_similarity": cos_sim,
                "is_lossless": is_lossless
            })

        # Master Archive Header:
        # [MAGIC_HEADER: 8B] [version: 1B (B)] [num_tensors: 4B (I)] [total_raw_bytes: 4B (I)] [reserved: 4B (I)]
        num_tensors = len(weight_tensors)
        archive_header = struct.pack(
            f"<8sBIII",
            self.MAGIC_HEADER,
            1,  # Version 1
            num_tensors,
            total_raw_weight_bytes,
            0   # Reserved
        )

        all_blocks = b"".join(packed_records)
        full_archive = archive_header + all_blocks

        with open(output_bin_path, "wb") as f:
            f.write(full_archive)

        actual_bin_size = os.path.getsize(output_bin_path)
        baseline_size = 1856832  # Phase C4 baseline

        overall_sparsity = (total_weights - total_nonzero_weights) / max(1, total_weights)
        overall_reduction = float((baseline_size - actual_bin_size) / baseline_size * 100.0)

        # Reconstruct runnable FlatBuffer if output path requested
        if tflite_output_path:
            self.reconstruct_to_tflite_file(output_bin_path, tflite_output_path)

        summary = {
            "format": "UAQE_D4_HYBRID",
            "source_tflite_path": tflite_path,
            "compressed_bin_path": output_bin_path,
            "total_weight_tensors": num_tensors,
            "total_weights": total_weights,
            "nonzero_weights": total_nonzero_weights,
            "overall_weight_sparsity": float(overall_sparsity),
            "raw_weight_bytes": total_raw_weight_bytes,
            "compressed_weight_bytes": len(all_blocks),
            "actual_archive_size_bytes": actual_bin_size,
            "actual_archive_size_mb": round(actual_bin_size / (1024 * 1024), 4),
            "storage_reduction_vs_c4_pct": round(overall_reduction, 2),
            "overall_compression_ratio": round(baseline_size / max(1, actual_bin_size), 4),
            "tensor_breakdown": tensor_reports
        }
        return summary

    def decompress_archive(self, bin_path: str) -> Dict[str, Any]:
        """Reads UAQE_D4 binary archive and returns list of decompressed tensor arrays."""
        with open(bin_path, "rb") as f:
            raw = f.read()

        magic, version, num_tensors, total_raw_bytes, _ = struct.unpack_from("<8sBIII", raw, 0)
        if magic != self.MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}. Expected {self.MAGIC_HEADER}")

        offset = 21  # 8 + 1 + 4 + 4 + 4
        decompressed_tensors = []

        for _ in range(num_tensors):
            t_idx, b_idx, strat_code, p_len = struct.unpack_from("<IIBI", raw, offset)
            offset += 13  # 4 + 4 + 1 + 4
            payload = raw[offset:offset + p_len]
            offset += p_len

            strat_name = self.REV_STRAT_CODES.get(strat_code, "dense")

            # Decode payload
            if strat_code == 0:  # Dense
                arr = np.frombuffer(payload, dtype=np.int8)
                t_name = f"tensor_{t_idx}"
            elif strat_code == 1:  # Sparse
                arr, t_name, _ = SparseEncoder.decode_from_bytes(payload, 0)
            elif strat_code == 2:  # Sparse + RLE
                arr, t_name, _ = RLECompressor.decode_from_bytes(payload, 0)
            elif strat_code in [3, 4, 5, 6]:  # Cluster
                arr, t_name, _ = WeightClusterer.decode_from_bytes(payload, 0)
            else:
                arr = np.frombuffer(payload, dtype=np.int8)
                t_name = f"tensor_{t_idx}"

            decompressed_tensors.append({
                "tensor_index": t_idx,
                "buffer_index": b_idx,
                "strategy": strat_name,
                "data": arr
            })

        return {
            "num_tensors": num_tensors,
            "total_raw_bytes": total_raw_bytes,
            "tensors": decompressed_tensors
        }

    def reconstruct_to_tflite_flatbuffer(
        self,
        bin_path: str,
        template_tflite_path: Optional[str] = None
    ) -> bytearray:
        """Decompresses binary archive and patches weight buffers into a runnable TFLite FlatBuffer."""
        target_template = template_tflite_path or self.base_tflite_path
        with open(target_template, "rb") as f:
            template_buf = bytearray(f.read())

        decompressed = self.decompress_archive(bin_path)

        # Extract buffer offsets from template
        model = schema_fb.Model.GetRootAsModel(template_buf, 0)
        subgraph = model.Subgraphs(0)

        for item in decompressed["tensors"]:
            b_idx = item["buffer_index"]
            arr = item["data"]
            
            b = model.Buffers(b_idx)
            offset = b._tab.Vector(b._tab.Offset(4))
            data_len = b.DataLength()

            patch_len = min(data_len, arr.nbytes)
            template_buf[offset:offset + patch_len] = arr.tobytes()[:patch_len]

        return template_buf

    def reconstruct_to_tflite_file(
        self,
        bin_path: str,
        output_tflite_path: str,
        template_tflite_path: Optional[str] = None
    ) -> str:
        """Decompresses binary archive and writes runnable TFLite model to disk."""
        fb_bytes = self.reconstruct_to_tflite_flatbuffer(bin_path, template_tflite_path)
        os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
        with open(output_tflite_path, "wb") as f:
            f.write(fb_bytes)
        return output_tflite_path

    def verify_reconstruction_integrity(
        self,
        orig_tflite_path: str,
        bin_path: str,
        output_json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verifies mathematical reconstruction accuracy (MAE, RMSE, Max Error, Cosine Sim)."""
        orig_tensors = {t["name"]: t["data"] for t in self.extract_weight_tensors(orig_tflite_path)}
        decompressed = self.decompress_archive(bin_path)

        # Get tensor names from template
        template_tensors = self.extract_weight_tensors(orig_tflite_path)
        idx_to_name = {t["buffer_index"]: t["name"] for t in template_tensors}

        tensor_metrics = []
        all_orig = []
        all_reconst = []

        for item in decompressed["tensors"]:
            b_idx = item["buffer_index"]
            name = idx_to_name.get(b_idx, f"buffer_{b_idx}")
            reconst_arr = item["data"].astype(np.float32)

            if name in orig_tensors:
                orig_arr = orig_tensors[name].astype(np.float32).flatten()
                reconst_flat = reconst_arr.flatten()
                
                # Align lengths if needed
                min_len = min(len(orig_arr), len(reconst_flat))
                orig_sub = orig_arr[:min_len]
                rec_sub = reconst_flat[:min_len]

                diff = np.abs(orig_sub - rec_sub)
                mae = float(np.mean(diff))
                max_err = float(np.max(diff)) if diff.size > 0 else 0.0
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                
                denom = (np.linalg.norm(orig_sub) * np.linalg.norm(rec_sub))
                cos_sim = float(1.0 - cosine(orig_sub, rec_sub)) if denom > 1e-12 else 1.0

                all_orig.extend(orig_sub.tolist())
                all_reconst.extend(rec_sub.tolist())

                tensor_metrics.append({
                    "tensor_name": name,
                    "buffer_index": b_idx,
                    "strategy": item["strategy"],
                    "mae": round(mae, 6),
                    "max_error": round(max_err, 4),
                    "rmse": round(rmse, 6),
                    "cosine_similarity": round(cos_sim, 6)
                })

        all_orig_np = np.array(all_orig)
        all_reconst_np = np.array(all_reconst)
        overall_diff = np.abs(all_orig_np - all_reconst_np)
        overall_mae = float(np.mean(overall_diff))
        overall_max_err = float(np.max(overall_diff)) if overall_diff.size > 0 else 0.0
        overall_rmse = float(np.sqrt(np.mean(overall_diff ** 2)))
        denom = (np.linalg.norm(all_orig_np) * np.linalg.norm(all_reconst_np))
        overall_cos = float(1.0 - cosine(all_orig_np, all_reconst_np)) if denom > 1e-12 else 1.0

        verification = {
            "overall_mae": round(overall_mae, 6),
            "overall_max_error": round(overall_max_err, 4),
            "overall_rmse": round(overall_rmse, 6),
            "overall_cosine_similarity": round(overall_cos, 6),
            "is_bit_level_lossless": bool(overall_max_err == 0.0),
            "tensor_count_verified": len(tensor_metrics),
            "tensors": tensor_metrics
        }

        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(verification, f, indent=2)

        return verification
