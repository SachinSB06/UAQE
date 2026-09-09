"""UAQE Phase D.2 Model Packager and Binary Archive Engine.
Packages pruned and quantized INT8 models into custom compressed .bin archives,
measures exact serialized disk footprint, and reconstructs runnable TFLite FlatBuffers
for live inference and numerical evaluation.
"""

from __future__ import annotations

import os
import struct
import copy
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

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


class ModelPackager:
    """Packages and reconstructs compressed INT8 models using Sparse, RLE, and Clustering strategies."""

    MAGIC_HEADER = b"UAQE_D2\x01"

    def __init__(self, base_tflite_path: str = "output/phase_d1/models/d1_best_sensitive_int8.tflite") -> None:
        self.base_tflite_path = base_tflite_path
        if os.path.exists(base_tflite_path):
            with open(base_tflite_path, "rb") as f:
                self.template_bytes = bytearray(f.read())
        else:
            self.template_bytes = bytearray()

    def extract_weight_tensors(self, tflite_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extracts all INT8 weight tensors and their FlatBuffer buffer offsets from a TFLite model."""
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
            
            if b and b.DataLength() > 0 and t.Type() == 9: # Type 9 = INT8
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

    def extract_int8_weights_from_tflite(self, tflite_path: str) -> Dict[str, np.ndarray]:
        """Extracts map of name -> numpy array for all INT8 weight tensors."""
        tensors = self.extract_weight_tensors(tflite_path)
        return {t["name"]: t["data"] for t in tensors}

    def package_model(
        self,
        tflite_path: str,
        output_bin_path: str,
        strategy: str = "sparse",  # "sparse", "sparse_rle", "cluster4", "cluster8", "cluster16", "cluster32"
        compression_method: Optional[str] = None,
        num_clusters: int = 8
    ) -> Dict[str, Any]:
        """Compresses all INT8 weights of a model into a standalone binary archive on disk."""
        if compression_method is not None:
            strategy = compression_method
        if strategy == "cluster":
            strategy = f"cluster{num_clusters}"

        weight_tensors = self.extract_weight_tensors(tflite_path)
        os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
        
        if strategy == "sparse":
            strat_code = 1
        elif strategy == "sparse_rle":
            strat_code = 2
        elif strategy.startswith("cluster"):
            strat_code = 3
        else:
            strat_code = 1
        
        packed_tensor_records: List[bytes] = []
        tensor_reports: List[Dict[str, Any]] = []
        
        total_raw_weight_bytes = 0
        total_nonzero_weights = 0
        total_weights = 0
        
        for t_info in weight_tensors:
            name = t_info["name"]
            t_idx = t_info["tensor_index"]
            b_idx = t_info["buffer_index"]
            arr = t_info["data"]
            shape = t_info["shape"]
            raw_len = t_info["byte_length"]
            
            total_raw_weight_bytes += raw_len
            total_weights += arr.size
            nz_count = int(np.sum(arr != 0))
            total_nonzero_weights += nz_count
            
            # Compress based on strategy
            if strategy == "sparse":
                res = SparseEncoder.encode_tensor(arr, tensor_name=name)
                payload = res["serialized_payload"]
                mae = 0.0
                max_err = 0.0
                rmse = 0.0
                cos_sim = 1.0
                is_lossless = True
                
            elif strategy == "sparse_rle":
                res = RLECompressor.encode_sparse_rle_tensor(arr, tensor_name=name)
                payload = res["serialized_payload"]
                mae = 0.0
                max_err = 0.0
                rmse = 0.0
                cos_sim = 1.0
                is_lossless = True
                
            elif strategy.startswith("cluster"):
                k_val = num_clusters
                if strategy == "cluster4": k_val = 4
                elif strategy == "cluster8": k_val = 8
                elif strategy == "cluster16": k_val = 16
                elif strategy == "cluster32": k_val = 32
                
                res = WeightClusterer.cluster_tensor(arr, num_clusters=k_val, tensor_name=name)
                payload = res["serialized_payload"]
                mae = res["mae"]
                max_err = res["max_error"]
                rmse = res["rmse"]
                cos_sim = res["cosine_similarity"]
                is_lossless = res["is_lossless"]
            else:
                raise ValueError(f"Unknown compression strategy: {strategy}")
                
            # Block header: [tensor_index: 4B (I)] [buffer_index: 4B (I)] [payload_len: 4B (I)] [payload]
            block_hdr = struct.pack("<III", t_idx, b_idx, len(payload))
            packed_tensor_records.append(block_hdr + payload)
            
            tensor_reports.append({
                "tensor_index": t_idx,
                "buffer_index": b_idx,
                "tensor_name": name,
                "shape": str(shape),
                "total_elements": raw_len,
                "nonzero_elements": nz_count,
                "sparsity": float((raw_len - nz_count) / max(1, raw_len)),
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

        # Master Archive Layout:
        # [MAGIC_HEADER: 8B] [strategy_code: 1B (B)] [num_tensors: 4B (I)]
        # [total_raw_bytes: 4B (I)] [payload_offset: 4B (I)]
        num_tensors = len(weight_tensors)
        archive_header = struct.pack(
            f"<8sBIII",
            self.MAGIC_HEADER,
            strat_code,
            num_tensors,
            total_raw_weight_bytes,
            0  # Reserved
        )
        
        # Assemble final binary stream
        all_blocks = b"".join(packed_tensor_records)
        full_archive = archive_header + all_blocks
        
        with open(output_bin_path, "wb") as f:
            f.write(full_archive)
            
        actual_bin_size = os.path.getsize(output_bin_path)
        source_tflite_size = os.path.getsize(tflite_path)
        baseline_tflite_size = 1855680  # Phase C4 verified baseline size in bytes
        
        overall_sparsity = (total_weights - total_nonzero_weights) / max(1, total_weights)
        
        summary = {
            "strategy": strategy,
            "num_clusters": num_clusters if strategy.startswith("cluster") else 0,
            "source_tflite_path": tflite_path,
            "compressed_bin_path": output_bin_path,
            "total_weight_tensors": num_tensors,
            "total_weights": total_weights,
            "nonzero_weights": total_nonzero_weights,
            "overall_weight_sparsity": float(overall_sparsity),
            "raw_weight_bytes": total_raw_weight_bytes,
            "compressed_weight_bytes": len(all_blocks),
            "actual_bin_file_bytes": actual_bin_size,
            "source_tflite_bytes": source_tflite_size,
            "baseline_tflite_bytes": baseline_tflite_size,
            "weight_compression_ratio": float(total_raw_weight_bytes / max(1, len(all_blocks))),
            "weight_storage_reduction_pct": float((1.0 - len(all_blocks) / max(1, total_raw_weight_bytes)) * 100.0),
            "model_compression_ratio_vs_baseline": float(baseline_tflite_size / max(1, actual_bin_size)),
            "model_storage_reduction_pct_vs_baseline": float((1.0 - actual_bin_size / max(1, baseline_tflite_size)) * 100.0),
            "is_lossless": bool(strategy in ["sparse", "sparse_rle"]),
            "tensor_reports": tensor_reports
        }
        
        return summary

    def unpackage_and_reconstruct_tflite(
        self,
        bin_path: str,
        template_tflite_path: Optional[str] = None
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        """Reads a compressed .bin archive and reconstructs the full executable TFLite FlatBuffer."""
        with open(bin_path, "rb") as f:
            archive_data = f.read()
            
        magic, strat_code, num_tensors, total_raw_bytes, _ = struct.unpack_from("<8sBIII", archive_data, 0)
        if magic != self.MAGIC_HEADER:
            raise ValueError(f"Invalid UAQE D2 binary header: {magic}")
            
        tpl_path = template_tflite_path or self.base_tflite_path
        with open(tpl_path, "rb") as f:
            reconstructed_tflite = bytearray(f.read())
            
        tpl_model = schema_fb.Model.GetRootAsModel(reconstructed_tflite, 0)
        
        offset = 8 + 1 + 4 + 4 + 4
        reconstruction_records: List[Dict[str, Any]] = []
        
        for _ in range(num_tensors):
            t_idx, b_idx, payload_len = struct.unpack_from("<III", archive_data, offset)
            offset += 12
            payload = archive_data[offset:offset + payload_len]
            offset += payload_len
            
            # Decode based on strategy
            if strat_code == 1: # sparse
                rec_tensor, name, _ = SparseEncoder.decode_from_bytes(payload, 0)
            elif strat_code == 2: # sparse_rle
                rec_tensor, name, _ = RLECompressor.decode_from_bytes(payload, 0)
            else: # cluster (strat_code in 3..6)
                rec_tensor, name, _ = WeightClusterer.decode_from_bytes(payload, 0)
                
            # Patch decompressed dense bytes into TFLite FlatBuffer buffer memory
            buf_obj = tpl_model.Buffers(b_idx)
            buf_offset = buf_obj._tab.Vector(buf_obj._tab.Offset(4))
            raw_int8_bytes = rec_tensor.astype(np.int8).tobytes()
            reconstructed_tflite[buf_offset:buf_offset + len(raw_int8_bytes)] = raw_int8_bytes
            
            reconstruction_records.append({
                "tensor_index": t_idx,
                "buffer_index": b_idx,
                "tensor_name": name,
                "shape": list(rec_tensor.shape),
                "byte_length": len(raw_int8_bytes)
            })
            
        return bytes(reconstructed_tflite), reconstruction_records

    def reconstruct_tflite_interpreter(
        self,
        bin_path: str,
        template_tflite_path: Optional[str] = None
    ) -> tf.lite.Interpreter:
        """Reconstructs in-memory TFLite model and instantiates a runnable Interpreter."""
        tflite_bytes, _ = self.unpackage_and_reconstruct_tflite(bin_path, template_tflite_path)
        interpreter = tf.lite.Interpreter(
            model_content=tflite_bytes,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        return interpreter

    def unpackage_and_verify(
        self,
        bin_path: str,
        original_tflite_path: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Unpackages archive, reconstructs tensors, and verifies metrics against original model."""
        orig_weights = self.extract_int8_weights_from_tflite(original_tflite_path)
        
        with open(bin_path, "rb") as f:
            archive_data = f.read()
            
        magic, strat_code, num_tensors, total_raw_bytes, _ = struct.unpack_from("<8sBIII", archive_data, 0)
        offset = 8 + 1 + 4 + 4 + 4
        
        decomp_weights: Dict[str, np.ndarray] = {}
        all_orig = []
        all_decomp = []
        
        for _ in range(num_tensors):
            t_idx, b_idx, payload_len = struct.unpack_from("<III", archive_data, offset)
            offset += 12
            payload = archive_data[offset:offset + payload_len]
            offset += payload_len
            
            if strat_code == 1:
                rec_tensor, name, _ = SparseEncoder.decode_from_bytes(payload, 0)
            elif strat_code == 2:
                rec_tensor, name, _ = RLECompressor.decode_from_bytes(payload, 0)
            else:
                rec_tensor, name, _ = WeightClusterer.decode_from_bytes(payload, 0)
                
            decomp_weights[name] = rec_tensor
            if name in orig_weights:
                all_orig.append(orig_weights[name].flatten().astype(np.float32))
                all_decomp.append(rec_tensor.flatten().astype(np.float32))
                
        concat_orig = np.concatenate(all_orig)
        concat_decomp = np.concatenate(all_decomp)
        
        diff = np.abs(concat_orig - concat_decomp)
        mae = float(np.mean(diff))
        max_err = float(np.max(diff))
        rmse = float(np.sqrt(np.mean((concat_orig - concat_decomp) ** 2)))
        
        norm_orig = np.linalg.norm(concat_orig)
        norm_decomp = np.linalg.norm(concat_decomp)
        if norm_orig > 0 and norm_decomp > 0:
            cos_sim = float(np.dot(concat_orig, concat_decomp) / (norm_orig * norm_decomp))
        else:
            cos_sim = 1.0
            
        metrics = {
            "mean_abs_error": mae,
            "max_abs_error": max_err,
            "rmse": rmse,
            "cosine_similarity": cos_sim,
            "is_lossless": bool(max_err == 0.0)
        }
        
        return decomp_weights, metrics
