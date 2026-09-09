"""Lossless Run-Length Encoding (RLE) for Sparse Quantized Tensors.
Implements adaptive byte-stream and index-run RLE with exact bit-for-bit decompression.
"""

from __future__ import annotations

import struct
from typing import Dict, Any, Tuple, List
import numpy as np

try:
    from src.uaqe.compression.sparse_encoder import SparseEncoder
except ImportError:
    from uaqe.compression.sparse_encoder import SparseEncoder


class RLECompressor:
    """Provides lossless RLE compression and decompression on sparse byte streams."""

    ESCAPE_BYTE = 0xAA  # Special escape byte for run-length markers

    @classmethod
    def compress_bytes(cls, data: bytes) -> bytes:
        """Compresses a byte stream using byte-level escape-based Run-Length Encoding.
        
        Runs of length >= 3 are encoded as: [ESCAPE_BYTE, run_length, byte_value].
        Occurrences of ESCAPE_BYTE are escaped as: [ESCAPE_BYTE, 0x00].
        Single or short runs (<3) of other bytes are written verbatim.
        """
        if not data:
            return b""
            
        out = bytearray()
        i = 0
        n = len(data)
        
        while i < n:
            b = data[i]
            # Count identical consecutive bytes (up to max run of 255)
            run = 1
            while i + run < n and data[i + run] == b and run < 255:
                run += 1
                
            if run >= 3:
                out.append(cls.ESCAPE_BYTE)
                out.append(run)
                out.append(b)
                i += run
            elif b == cls.ESCAPE_BYTE:
                out.append(cls.ESCAPE_BYTE)
                out.append(0x00) # Escape marker for literal ESCAPE_BYTE
                i += 1
            else:
                out.append(b)
                i += 1
                
        return bytes(out)

    @classmethod
    def decompress_bytes(cls, compressed: bytes) -> bytes:
        """Decompresses an escape-based RLE byte stream back to original bytes."""
        if not compressed:
            return b""
            
        out = bytearray()
        i = 0
        n = len(compressed)
        
        while i < n:
            b = compressed[i]
            if b == cls.ESCAPE_BYTE:
                if i + 1 >= n:
                    raise ValueError("Malformed RLE stream: trailing escape byte.")
                count = compressed[i + 1]
                if count == 0x00:
                    out.append(cls.ESCAPE_BYTE)
                    i += 2
                else:
                    if i + 2 >= n:
                        raise ValueError("Malformed RLE stream: truncated run-length tuple.")
                    val = compressed[i + 2]
                    out.extend([val] * count)
                    i += 3
            else:
                out.append(b)
                i += 1
                
        return bytes(out)

    @classmethod
    def encode_sparse_rle_tensor(cls, tensor: np.ndarray, tensor_name: str = "") -> Dict[str, Any]:
        """Encodes an INT8 tensor with Sparse Bitmask + RLE compression."""
        sparse_res = SparseEncoder.encode_tensor(tensor, tensor_name=tensor_name)
        
        bitmask = sparse_res["bitmask"]
        nz_vals = sparse_res["nonzero_values"]
        
        # Apply RLE on bitmask and non-zero values
        rle_bitmask = cls.compress_bytes(bitmask)
        rle_nz_vals = cls.compress_bytes(nz_vals)
        
        name_bytes = tensor_name.encode("utf-8")
        shape = list(tensor.shape)
        header = struct.pack(
            f"<H{len(name_bytes)}sB{len(shape)}IIIII",
            len(name_bytes),
            name_bytes,
            len(shape),
            *shape,
            sparse_res["total_elements"],
            sparse_res["nonzero_count"],
            len(rle_bitmask),
            len(rle_nz_vals)
        )
        serialized_bytes = header + rle_bitmask + rle_nz_vals
        
        raw_dense_bytes = sparse_res["raw_dense_bytes"]
        compressed_bytes = len(serialized_bytes)
        
        return {
            "tensor_name": tensor_name,
            "original_shape": shape,
            "dtype": "int8",
            "total_elements": sparse_res["total_elements"],
            "nonzero_count": sparse_res["nonzero_count"],
            "sparsity": sparse_res["sparsity"],
            "rle_bitmask": rle_bitmask,
            "rle_nonzero_values": rle_nz_vals,
            "raw_dense_bytes": raw_dense_bytes,
            "payload_bytes": len(rle_bitmask) + len(rle_nz_vals),
            "serialized_bytes": compressed_bytes,
            "compression_ratio": float(raw_dense_bytes / max(1, compressed_bytes)),
            "storage_reduction_pct": float((1.0 - compressed_bytes / max(1, raw_dense_bytes)) * 100.0),
            "serialized_payload": serialized_bytes
        }

    @classmethod
    def decode_sparse_rle_tensor(cls, encoded_data: Dict[str, Any]) -> np.ndarray:
        """Decodes Sparse + RLE representation back to exact original dense INT8 tensor."""
        rle_bitmask = encoded_data["rle_bitmask"]
        rle_nz = encoded_data["rle_nonzero_values"]
        
        # Decompress RLE
        bitmask = cls.decompress_bytes(rle_bitmask)
        nonzero_values = cls.decompress_bytes(rle_nz)
        
        sparse_dict = {
            "original_shape": encoded_data["original_shape"],
            "total_elements": encoded_data["total_elements"],
            "nonzero_count": encoded_data["nonzero_count"],
            "bitmask": bitmask,
            "nonzero_values": nonzero_values
        }
        return SparseEncoder.decode_tensor(sparse_dict)

    @classmethod
    def decode_from_bytes(cls, data: bytes, offset: int = 0) -> Tuple[np.ndarray, str, int]:
        """Decodes tensor directly from binary stream."""
        name_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        name = struct.unpack_from(f"<{name_len}s", data, offset)[0].decode("utf-8")
        offset += name_len
        ndim = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        shape = list(struct.unpack_from(f"<{ndim}I", data, offset))
        offset += ndim * 4
        total_elements, nonzero_count, bitmask_len, nz_len = struct.unpack_from("<IIII", data, offset)
        offset += 16
        rle_bm = data[offset:offset + bitmask_len]
        offset += bitmask_len
        rle_nz = data[offset:offset + nz_len]
        offset += nz_len
        encoded_dict = {
            "original_shape": shape,
            "total_elements": total_elements,
            "nonzero_count": nonzero_count,
            "rle_bitmask": rle_bm,
            "rle_nonzero_values": rle_nz
        }
        rec = cls.decode_sparse_rle_tensor(encoded_dict)
        return rec, name, offset
