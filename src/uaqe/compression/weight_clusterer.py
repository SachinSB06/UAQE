"""Weight Clustering and Codebook Quantization for Sparse INT8 Tensors.
Implements K-Means centroid clustering (K=4, 8, 16, 32) with bit-packed cluster IDs
and reconstruction error metric analysis.
"""

from __future__ import annotations

import struct
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans

try:
    from src.uaqe.compression.sparse_encoder import SparseEncoder
except ImportError:
    from uaqe.compression.sparse_encoder import SparseEncoder


class WeightClusterer:
    """Performs K-Means weight clustering on surviving non-zero INT8 parameters."""

    @staticmethod
    def _pack_bits(indices: np.ndarray, bits_per_item: int) -> bytes:
        """Packs an array of integer indices into a minimal byte array using specified bit width."""
        n = len(indices)
        if bits_per_item == 2:
            # 4 indices per byte
            padded_len = (n + 3) // 4 * 4
            padded = np.zeros(padded_len, dtype=np.uint8)
            padded[:n] = indices & 0x03
            packed = (
                (padded[0::4] << 6) |
                (padded[1::4] << 4) |
                (padded[2::4] << 2) |
                padded[3::4]
            )
            return packed.tobytes()
            
        elif bits_per_item == 4:
            # 2 indices per byte
            padded_len = (n + 1) // 2 * 2
            padded = np.zeros(padded_len, dtype=np.uint8)
            padded[:n] = indices & 0x0F
            packed = (padded[0::2] << 4) | padded[1::2]
            return packed.tobytes()
            
        elif bits_per_item == 8:
            return (indices.astype(np.uint8)).tobytes()
            
        else:
            # General bit packing
            total_bits = n * bits_per_item
            total_bytes = (total_bits + 7) // 8
            out = bytearray(total_bytes)
            bit_pos = 0
            for idx in indices:
                val = int(idx) & ((1 << bits_per_item) - 1)
                for b in range(bits_per_item):
                    bit = (val >> (bits_per_item - 1 - b)) & 1
                    byte_idx = bit_pos // 8
                    bit_idx = 7 - (bit_pos % 8)
                    if bit:
                        out[byte_idx] |= (1 << bit_idx)
                    bit_pos += 1
            return bytes(out)

    @staticmethod
    def _unpack_bits(data: bytes, n_items: int, bits_per_item: int) -> np.ndarray:
        """Unpacks compact bit-packed bytes back to integer cluster indices."""
        if bits_per_item == 2:
            arr = np.frombuffer(data, dtype=np.uint8)
            b0 = (arr >> 6) & 0x03
            b1 = (arr >> 4) & 0x03
            b2 = (arr >> 2) & 0x03
            b3 = arr & 0x03
            unpacked = np.column_stack((b0, b1, b2, b3)).flatten()
            return unpacked[:n_items]
            
        elif bits_per_item == 4:
            arr = np.frombuffer(data, dtype=np.uint8)
            hi = (arr >> 4) & 0x0F
            lo = arr & 0x0F
            unpacked = np.column_stack((hi, lo)).flatten()
            return unpacked[:n_items]
            
        elif bits_per_item == 8:
            return np.frombuffer(data, dtype=np.uint8)[:n_items]
            
        else:
            out = np.zeros(n_items, dtype=np.int32)
            bit_pos = 0
            for i in range(n_items):
                val = 0
                for b in range(bits_per_item):
                    byte_idx = bit_pos // 8
                    bit_idx = 7 - (bit_pos % 8)
                    bit = (data[byte_idx] >> bit_idx) & 1
                    val = (val << 1) | bit
                    bit_pos += 1
                out[i] = val
            return out

    @classmethod
    def cluster_tensor(
        cls,
        tensor: np.ndarray,
        num_clusters: int = 8,
        tensor_name: str = "",
        seed: int = 42
    ) -> Dict[str, Any]:
        """Performs K-Means clustering on the non-zero INT8 values of a tensor.
        
        Args:
            tensor: Dense INT8 tensor.
            num_clusters: Number of centroids K in {4, 8, 16, 32}.
            tensor_name: Name of tensor.
            seed: Random seed for deterministic k-means.
            
        Returns:
            Dict containing codebook, cluster IDs, reconstruction, and compression metrics.
        """
        flat = tensor.astype(np.int8).flatten()
        total_elements = flat.size
        
        nz_mask = (flat != 0)
        nonzero_count = int(np.sum(nz_mask))
        
        # Determine bit-width for cluster indices
        if num_clusters <= 4:
            bits_per_id = 2
            k = 4
        elif num_clusters <= 8:
            bits_per_id = 3
            k = 8
        elif num_clusters <= 16:
            bits_per_id = 4
            k = 16
        else:
            bits_per_id = 5
            k = 32

        # Sparse bitmask
        bitmask = np.packbits(nz_mask).tobytes()

        if nonzero_count == 0:
            centroids = np.zeros(k, dtype=np.int8)
            cluster_ids_packed = b""
            reconstructed_flat = np.zeros(total_elements, dtype=np.int8)
        else:
            nz_vals = flat[nz_mask].astype(np.float32).reshape(-1, 1)
            unique_vals = np.unique(nz_vals)
            
            if len(unique_vals) <= k:
                # If unique values <= K, direct exact mapping without loss!
                cents = np.zeros(k, dtype=np.int8)
                cents[:len(unique_vals)] = np.round(unique_vals).astype(np.int8)
                val_to_id = {v: i for i, v in enumerate(unique_vals)}
                labels = np.array([val_to_id[v[0]] for v in nz_vals], dtype=np.uint8)
                centroids = cents
            else:
                # Deterministic K-Means clustering
                kmeans = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=5,
                    max_iter=50,
                    random_state=seed
                )
                kmeans.fit(nz_vals)
                # Sort centroids for deterministic ordering
                raw_cents = kmeans.cluster_centers_.flatten()
                sort_order = np.argsort(raw_cents)
                sorted_cents = np.round(raw_cents[sort_order]).astype(np.int8)
                
                # Remap cluster labels to sorted centroid order
                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sort_order)}
                labels = np.array([old_to_new[lbl] for lbl in kmeans.labels_], dtype=np.uint8)
                centroids = sorted_cents

            # Pack cluster IDs
            cluster_ids_packed = cls._pack_bits(labels, bits_per_id)
            
            # Reconstruct non-zero values from codebook
            reconstructed_nz = centroids[labels]
            reconstructed_flat = np.zeros(total_elements, dtype=np.int8)
            reconstructed_flat[nz_mask] = reconstructed_nz

        reconstructed_tensor = reconstructed_flat.reshape(tensor.shape)
        
        # Reconstruction Error Metrics
        orig_nz = flat[nz_mask].astype(np.float32) if nonzero_count > 0 else np.array([0.0])
        rec_nz = reconstructed_flat[nz_mask].astype(np.float32) if nonzero_count > 0 else np.array([0.0])
        
        diff = np.abs(orig_nz - rec_nz)
        mae = float(np.mean(diff)) if len(diff) > 0 else 0.0
        max_err = float(np.max(diff)) if len(diff) > 0 else 0.0
        rmse = float(np.sqrt(np.mean((orig_nz - rec_nz) ** 2))) if len(diff) > 0 else 0.0
        
        if np.std(orig_nz) > 1e-6 and np.std(rec_nz) > 1e-6:
            cos_sim = float(1.0 - cosine(orig_nz, rec_nz))
        else:
            cos_sim = 1.0 if np.array_equal(orig_nz, rec_nz) else 0.0
            
        # Serialized payload format:
        # [name_len: 2B (H)] [name: UTF-8] [ndim: 1B (B)] [shape: ndim * 4B (I)]
        # [total_elements: 4B (I)] [nonzero_count: 4B (I)] [K: 2B (H)] [bits_per_id: 1B (B)]
        # [bitmask_len: 4B (I)] [centroids: K * 1B (int8)] [bitmask] [packed_ids]
        name_bytes = tensor_name.encode("utf-8")
        shape = list(tensor.shape)
        header = struct.pack(
            f"<H{len(name_bytes)}sB{len(shape)}IIIHBII",
            len(name_bytes),
            name_bytes,
            len(shape),
            *shape,
            total_elements,
            nonzero_count,
            k,
            bits_per_id,
            len(bitmask),
            len(centroids)
        )
        serialized_bytes = header + centroids.tobytes() + bitmask + cluster_ids_packed
        
        raw_dense_bytes = total_elements * 1
        compressed_bytes = len(serialized_bytes)
        sparsity = (total_elements - nonzero_count) / max(1, total_elements)
        
        return {
            "tensor_name": tensor_name,
            "original_shape": shape,
            "dtype": "int8",
            "num_clusters": k,
            "bits_per_id": bits_per_id,
            "total_elements": total_elements,
            "nonzero_count": nonzero_count,
            "sparsity": float(sparsity),
            "centroids": centroids.tolist(),
            "bitmask": bitmask,
            "cluster_ids_packed": cluster_ids_packed,
            "reconstructed_tensor": reconstructed_tensor,
            "raw_dense_bytes": raw_dense_bytes,
            "payload_bytes": len(bitmask) + len(centroids) + len(cluster_ids_packed),
            "serialized_bytes": compressed_bytes,
            "compression_ratio": float(raw_dense_bytes / max(1, compressed_bytes)),
            "storage_reduction_pct": float((1.0 - compressed_bytes / max(1, raw_dense_bytes)) * 100.0),
            "mae": mae,
            "max_error": max_err,
            "rmse": rmse,
            "cosine_similarity": cos_sim,
            "is_lossless": bool(max_err == 0.0),
            "serialized_payload": serialized_bytes
        }

    @classmethod
    def decode_clustered_tensor(cls, encoded_data: Dict[str, Any]) -> np.ndarray:
        """Reconstructs dense INT8 tensor from codebook centroids and bit-packed cluster IDs."""
        shape = tuple(encoded_data["original_shape"])
        total_elements = encoded_data["total_elements"]
        nonzero_count = encoded_data["nonzero_count"]
        centroids = np.array(encoded_data["centroids"], dtype=np.int8)
        bits_per_id = encoded_data["bits_per_id"]
        bitmask = encoded_data["bitmask"]
        packed_ids = encoded_data["cluster_ids_packed"]
        
        # Unpack bitmask
        nz_mask = np.unpackbits(np.frombuffer(bitmask, dtype=np.uint8))[:total_elements].astype(bool)
        
        reconstructed = np.zeros(total_elements, dtype=np.int8)
        if nonzero_count > 0:
            labels = cls._unpack_bits(packed_ids, nonzero_count, bits_per_id)
            reconstructed_nz = centroids[labels]
            reconstructed[nz_mask] = reconstructed_nz
            
        return reconstructed.reshape(shape)

    @classmethod
    def decode_from_bytes(cls, data: bytes, offset: int = 0) -> Tuple[np.ndarray, str, int]:
        """Decodes clustered tensor directly from binary stream."""
        name_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        name = struct.unpack_from(f"<{name_len}s", data, offset)[0].decode("utf-8")
        offset += name_len
        ndim = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        shape = list(struct.unpack_from(f"<{ndim}I", data, offset))
        offset += ndim * 4
        total_elements, nonzero_count, k, bits_per_id, bitmask_len, cent_len = struct.unpack_from("<IIHBII", data, offset)
        offset += 19
        centroids = np.frombuffer(data[offset:offset + cent_len], dtype=np.int8)
        offset += cent_len
        bitmask = data[offset:offset + bitmask_len]
        offset += bitmask_len
        
        if bits_per_id == 2:
            packed_len = (nonzero_count + 3) // 4
        elif bits_per_id == 4:
            packed_len = (nonzero_count + 1) // 2
        elif bits_per_id == 8:
            packed_len = nonzero_count
        else:
            packed_len = (nonzero_count * bits_per_id + 7) // 8
            
        cluster_ids_packed = data[offset:offset + packed_len]
        offset += packed_len
        
        encoded_dict = {
            "original_shape": shape,
            "total_elements": total_elements,
            "nonzero_count": nonzero_count,
            "centroids": centroids.tolist(),
            "bits_per_id": bits_per_id,
            "bitmask": bitmask,
            "cluster_ids_packed": cluster_ids_packed
        }
        rec = cls.decode_clustered_tensor(encoded_dict)
        return rec, name, offset
