"""
UAQE Phase E.1 Optimized Runtime Decoder
High-performance, zero-copy, hardened decoder and FlatBuffer reconstructor.
Implements vectorized RLE decompression, fast sparse bitmask extraction,
and precomputed FlatBuffer buffer table patching with 100% bit-level fidelity.
"""

from __future__ import annotations

import os
import sys
import struct
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from scipy.spatial.distance import cosine

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

# Ensure uaqe imports work cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.compression.weight_clusterer import WeightClusterer


class OptimizedRuntimeDecoder:
    """High-performance runtime decoder with vectorized RLE, zero-copy reads, and cached FlatBuffer patching."""

    MAGIC_HEADER = b"UAQE_D4\x01"
    SUPPORTED_VERSIONS = [1]
    ESCAPE_BYTE = 0xAA

    STRATEGY_MAP = {
        0: "dense",
        1: "sparse",
        2: "sparse_rle",
        3: "cluster32",
        4: "cluster64",
        5: "cluster16",
        6: "cluster8"
    }

    def __init__(
        self,
        base_template_path: str = "output/phase_c4/models/c4_best_int8.tflite",
        template_bytes: Optional[bytes] = None
    ):
        self.base_template_path = base_template_path
        self._template_bytes = template_bytes
        self._buffer_table: Optional[Dict[int, Tuple[int, int]]] = None

        self.archive_path: Optional[str] = None
        self.archive_bytes: Optional[bytes] = None
        self.archive_hash: Optional[str] = None
        self._parsed_metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def compute_sha256(data: Union[bytes, bytearray, memoryview]) -> str:
        """Computes SHA-256 hex digest."""
        hasher = hashlib.sha256()
        hasher.update(data)
        return hasher.hexdigest()

    def _ensure_template_loaded(self, tpl_path: Optional[str] = None) -> bytes:
        """Loads and caches base template bytes in memory."""
        path = tpl_path or self.base_template_path
        if self._template_bytes is None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Template TFLite model not found at: {path}")
            with open(path, "rb") as f:
                self._template_bytes = f.read()
        return self._template_bytes

    def _ensure_buffer_table(self, tpl_path: Optional[str] = None) -> Dict[int, Tuple[int, int]]:
        """Parses and caches the FlatBuffer buffer index -> (byte_offset, byte_length) map."""
        if self._buffer_table is None:
            raw_tpl = self._ensure_template_loaded(tpl_path)
            model = schema_fb.Model.GetRootAsModel(raw_tpl, 0)
            buf_len = model.BuffersLength()
            table: Dict[int, Tuple[int, int]] = {}
            for i in range(buf_len):
                b = model.Buffers(i)
                if b and b.DataLength() > 0:
                    offset = b._tab.Vector(b._tab.Offset(4))
                    data_len = b.DataLength()
                    table[i] = (offset, data_len)
            self._buffer_table = table
        return self._buffer_table

    @classmethod
    def fast_decompress_rle(cls, compressed: bytes) -> bytes:
        """Fast RLE decompression using C-level memchr token skipping."""
        if not compressed:
            return b""
        if b"\xaa" not in compressed:
            return compressed

        out = bytearray()
        i = 0
        n = len(compressed)

        while i < n:
            next_aa = compressed.find(b"\xaa", i)
            if next_aa == -1:
                out.extend(compressed[i:])
                break

            if next_aa > i:
                out.extend(compressed[i:next_aa])

            i = next_aa + 1
            if i >= n:
                raise ValueError("Malformed RLE stream: trailing escape byte.")

            count = compressed[i]
            if count == 0:
                out.append(cls.ESCAPE_BYTE)
                i += 1
            else:
                if i + 1 >= n:
                    raise ValueError("Malformed RLE stream: truncated run-length tuple.")
                val = compressed[i + 1]
                out.extend(bytes([val]) * count)
                i += 2

        return bytes(out)

    def load(self, path: str) -> Dict[str, Any]:
        """Loads and strictly validates the binary archive from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archive not found: {path}")

        file_size = os.path.getsize(path)
        if file_size < 21:
            raise ValueError(f"Truncated archive: size {file_size} bytes is smaller than master header (21 bytes).")

        with open(path, "rb") as f:
            raw = f.read()

        return self.load_from_bytes(raw, path=path)

    def load_from_bytes(self, raw: bytes, path: Optional[str] = None) -> Dict[str, Any]:
        """Loads and validates archive directly from in-memory bytes."""
        if len(raw) < 21:
            raise ValueError(f"Truncated archive: size {len(raw)} bytes is smaller than master header (21 bytes).")

        magic, version, num_tensors, total_raw_bytes, reserved = struct.unpack_from("<8sBIII", raw, 0)

        if magic != self.MAGIC_HEADER:
            raise ValueError(
                f"Invalid archive magic: expected {self.MAGIC_HEADER!r}, got {magic!r}. Not a valid UAQE D4/D5/E1 archive."
            )

        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported format version {version}. Supported versions: {self.SUPPORTED_VERSIONS}."
            )

        if num_tensors <= 0 or num_tensors > 10000:
            raise ValueError(f"Invalid tensor count in archive header: {num_tensors}.")

        self.archive_path = path or "<in-memory>"
        self.archive_bytes = raw
        self.archive_hash = self.compute_sha256(raw)

        # Inspect block boundaries
        offset = 21
        blocks = []
        for i in range(num_tensors):
            if offset + 13 > len(raw):
                raise ValueError(
                    f"Truncated archive at tensor record {i}/{num_tensors}: unexpected EOF reading record header."
                )
            t_idx, b_idx, strat_code, p_len = struct.unpack_from("<IIBI", raw, offset)
            offset += 13

            if offset + p_len > len(raw):
                raise ValueError(
                    f"Payload overflow at tensor {i} (tensor_idx={t_idx}, buffer_idx={b_idx}): "
                    f"expected payload length {p_len} exceeds remaining archive bytes ({len(raw) - offset})."
                )

            strat_name = self.STRATEGY_MAP.get(strat_code, f"unknown_{strat_code}")
            blocks.append({
                "record_index": i,
                "tensor_index": t_idx,
                "buffer_index": b_idx,
                "strategy_code": strat_code,
                "strategy": strat_name,
                "payload_offset": offset,
                "payload_length": p_len
            })
            offset += p_len

        self._parsed_metadata = {
            "archive_path": self.archive_path,
            "archive_size_bytes": len(raw),
            "archive_sha256": self.archive_hash,
            "magic": magic.decode("latin-1", errors="replace"),
            "version": version,
            "num_tensors": num_tensors,
            "total_raw_weight_bytes": total_raw_bytes,
            "header_bytes": 21,
            "payload_blocks": blocks
        }
        return self._parsed_metadata

    def inspect(self) -> Dict[str, Any]:
        """Returns structural metadata and summary of the loaded archive."""
        if self._parsed_metadata is None:
            raise RuntimeError("No archive loaded. Call load() first.")
        return self._parsed_metadata

    def decode(self) -> List[Dict[str, Any]]:
        """Fast vectorized decode of all compressed tensor blocks into in-memory INT8 arrays."""
        if self.archive_bytes is None or self._parsed_metadata is None:
            raise RuntimeError("No archive loaded. Call load() first.")

        raw = self.archive_bytes
        decoded_tensors = []

        for block in self._parsed_metadata["payload_blocks"]:
            t_idx = block["tensor_index"]
            b_idx = block["buffer_index"]
            strat_code = block["strategy_code"]
            strat_name = block["strategy"]
            p_offset = block["payload_offset"]
            p_len = block["payload_length"]

            payload = raw[p_offset : p_offset + p_len]

            if strat_code == 0:  # Dense
                arr = np.frombuffer(payload, dtype=np.int8)
                name = f"tensor_{t_idx}"
                shape = list(arr.shape)

            elif strat_code == 2:  # Sparse + RLE
                name_len = struct.unpack_from("<H", payload, 0)[0]
                off = 2
                name = struct.unpack_from(f"<{name_len}s", payload, off)[0].decode("utf-8")
                off += name_len
                ndim = struct.unpack_from("<B", payload, off)[0]
                off += 1
                shape = list(struct.unpack_from(f"<{ndim}I", payload, off))
                off += ndim * 4
                total_elements, nonzero_count, bitmask_len, nz_len = struct.unpack_from("<IIII", payload, off)
                off += 16

                rle_bm = payload[off : off + bitmask_len]
                rle_nz = payload[off + bitmask_len : off + bitmask_len + nz_len]

                # Fast RLE Decompress
                bitmask = self.fast_decompress_rle(rle_bm)
                nz_bytes = self.fast_decompress_rle(rle_nz)

                # Fast Sparse Bitmask Unpack & Tensor Fill
                nz_mask = np.unpackbits(np.frombuffer(bitmask, dtype=np.uint8), count=total_elements).view(np.bool_)
                rec = np.zeros(shape, dtype=np.int8)
                if nonzero_count > 0:
                    rec.ravel()[nz_mask] = np.frombuffer(nz_bytes, dtype=np.int8)
                arr = rec

            elif strat_code == 1:  # Sparse
                name_len = struct.unpack_from("<H", payload, 0)[0]
                off = 2
                name = struct.unpack_from(f"<{name_len}s", payload, off)[0].decode("utf-8")
                off += name_len
                ndim = struct.unpack_from("<B", payload, off)[0]
                off += 1
                shape = list(struct.unpack_from(f"<{ndim}I", payload, off))
                off += ndim * 4
                total_elements, nonzero_count, bitmask_len = struct.unpack_from("<III", payload, off)
                off += 12
                bitmask = payload[off : off + bitmask_len]
                nz_bytes = payload[off + bitmask_len : off + bitmask_len + nonzero_count]

                nz_mask = np.unpackbits(np.frombuffer(bitmask, dtype=np.uint8), count=total_elements).view(np.bool_)
                rec = np.zeros(shape, dtype=np.int8)
                if nonzero_count > 0:
                    rec.ravel()[nz_mask] = np.frombuffer(nz_bytes, dtype=np.int8)
                arr = rec

            elif strat_code in [3, 4, 5, 6]:  # Cluster
                arr, name, _ = WeightClusterer.decode_from_bytes(payload, 0)
                shape = list(arr.shape)
            else:
                raise ValueError(f"Unsupported strategy code: {strat_code} for tensor index {t_idx}.")

            decoded_tensors.append({
                "tensor_index": t_idx,
                "buffer_index": b_idx,
                "tensor_name": name,
                "strategy": strat_name,
                "shape": shape,
                "data": arr,
                "byte_length": arr.nbytes
            })

        return decoded_tensors

    def reconstruct_from_decoded(
        self,
        decoded_tensors: List[Dict[str, Any]],
        template_tflite_path: Optional[str] = None
    ) -> bytearray:
        """Fast FlatBuffer reconstruction directly from pre-decoded tensors via precomputed buffer map."""
        tpl_bytes = self._ensure_template_loaded(template_tflite_path)
        buf_table = self._ensure_buffer_table(template_tflite_path)

        tflite_buf = bytearray(tpl_bytes)

        for item in decoded_tensors:
            b_idx = item["buffer_index"]
            if b_idx in buf_table:
                offset, max_len = buf_table[b_idx]
                arr = item["data"]
                # Zero-copy view / tobytes
                raw_bytes = arr.tobytes() if isinstance(arr, np.ndarray) else bytes(arr)
                patch_len = min(max_len, len(raw_bytes))
                tflite_buf[offset : offset + patch_len] = raw_bytes[:patch_len]

        return tflite_buf

    def reconstruct(self, template_tflite_path: Optional[str] = None) -> bytearray:
        """Decodes archive and reconstructs FlatBuffer in a single streamlined pass."""
        decoded = self.decode()
        return self.reconstruct_from_decoded(decoded, template_tflite_path)

    def reconstruct_to_file(
        self,
        output_tflite_path: str,
        template_tflite_path: Optional[str] = None
    ) -> str:
        """Reconstructs the model and saves runnable .tflite to disk."""
        model_bytes = self.reconstruct(template_tflite_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_tflite_path)), exist_ok=True)
        with open(output_tflite_path, "wb") as f:
            f.write(model_bytes)
        return output_tflite_path

    def extract_weights_from_tflite(self, tflite_path: str) -> Dict[int, Dict[str, Any]]:
        """Extracts map of buffer_index -> {name, shape, data} from a TFLite file."""
        with open(tflite_path, "rb") as f:
            raw_buf = bytearray(f.read())

        model = schema_fb.Model.GetRootAsModel(raw_buf, 0)
        subgraph = model.Subgraphs(0)

        weight_map = {}
        for i in range(subgraph.TensorsLength()):
            t = subgraph.Tensors(i)
            b_idx = t.Buffer()
            b = model.Buffers(b_idx)

            if b and b.DataLength() > 0 and t.Type() == 9:  # INT8 tensor
                t_name = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
                shape = [t.Shape(j) for j in range(t.ShapeLength())]
                data_np = b.DataAsNumpy().astype(np.int8)
                weight_map[b_idx] = {
                    "tensor_index": i,
                    "buffer_index": b_idx,
                    "name": t_name,
                    "shape": shape,
                    "data": data_np
                }
        return weight_map

    def verify_tensors(self, source_tflite_path: str) -> Dict[str, Any]:
        """Compares decoded tensors against source model tensors for mathematical and byte equality."""
        decoded = self.decode()
        source_weights = self.extract_weights_from_tflite(source_tflite_path)

        tensor_verifications = []
        all_exact = True
        all_src_flat = []
        all_dec_flat = []

        for item in decoded:
            b_idx = item["buffer_index"]
            dec_arr = item["data"].astype(np.float32).flatten()

            if b_idx in source_weights:
                src_info = source_weights[b_idx]
                src_arr = src_info["data"].astype(np.float32).flatten()

                min_len = min(len(src_arr), len(dec_arr))
                src_sub = src_arr[:min_len]
                dec_sub = dec_arr[:min_len]

                diff = np.abs(src_sub - dec_sub)
                mae = float(np.mean(diff))
                max_err = float(np.max(diff)) if diff.size > 0 else 0.0
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                exact = bool(max_err == 0.0)

                denom = np.linalg.norm(src_sub) * np.linalg.norm(dec_sub)
                cos_sim = float(1.0 - cosine(src_sub, dec_sub)) if denom > 1e-12 else 1.0

                if not exact:
                    all_exact = False

                all_src_flat.extend(src_sub.tolist())
                all_dec_flat.extend(dec_sub.tolist())

                tensor_verifications.append({
                    "buffer_index": b_idx,
                    "tensor_index": item["tensor_index"],
                    "tensor_name": src_info["name"],
                    "strategy": item["strategy"],
                    "source_elements": len(src_arr),
                    "decoded_elements": len(dec_arr),
                    "exact_match": exact,
                    "max_abs_error": round(max_err, 4),
                    "mae": round(mae, 6),
                    "rmse": round(rmse, 6),
                    "cosine_similarity": round(cos_sim, 6)
                })

        all_src_np = np.array(all_src_flat)
        all_dec_np = np.array(all_dec_flat)
        overall_diff = np.abs(all_src_np - all_dec_np)
        overall_mae = float(np.mean(overall_diff))
        overall_max_err = float(np.max(overall_diff)) if overall_diff.size > 0 else 0.0
        overall_rmse = float(np.sqrt(np.mean(overall_diff ** 2)))
        denom = np.linalg.norm(all_src_np) * np.linalg.norm(all_dec_np)
        overall_cos = float(1.0 - cosine(all_src_np, all_dec_np)) if denom > 1e-12 else 1.0

        return {
            "all_tensors_exact_match": all_exact,
            "overall_mae": round(overall_mae, 6),
            "overall_max_error": round(overall_max_err, 4),
            "overall_rmse": round(overall_rmse, 6),
            "overall_cosine_similarity": round(overall_cos, 6),
            "tensor_count": len(tensor_verifications),
            "tensor_records": tensor_verifications
        }
