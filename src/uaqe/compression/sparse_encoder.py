"""Lossless Sparse Encoding for Quantized INT8 Tensors and IMR Representations.
Implements bitmask and coordinate-based (COO) sparse representations with exact
bit-for-bit lossless decompression.
"""

from __future__ import annotations

import struct
from typing import Dict, Any, Tuple, Optional, Sequence, Union
import numpy as np

try:
    from src.uaqe.common.imr import IMRTensor
except ImportError:
    try:
        from uaqe.common.imr import IMRTensor
    except ImportError:
        IMRTensor = None


class SparseEncoder:
    """Encodes and decodes sparse tensors losslessly using bitmask or COO representations."""

    @staticmethod
    def dense_byte_size(tensor: Any) -> int:
        """Computes dense byte size of an IMRTensor or numpy array."""
        if hasattr(tensor, "data"):
            return len(tensor.data)
        if isinstance(tensor, np.ndarray):
            return tensor.nbytes
        return 0

    @staticmethod
    def sparse_byte_size(tensor: Any, kept_count: int) -> int:
        """Computes COO sparse byte size (4B header + 4B per index + value bytes)."""
        dtype = getattr(tensor, "dtype", "int8")
        itemsize = 2 if "16" in dtype else (4 if "32" in dtype else 1)
        return 4 + (4 * kept_count) + (itemsize * kept_count)

    @classmethod
    def encode_tensor(
        cls,
        tensor: Union[np.ndarray, Any],
        kept_indices: Optional[Sequence[int]] = None,
        tensor_name: str = ""
    ) -> Union[Dict[str, Any], Any]:
        """Encodes tensor into either D2 bitmask dictionary or COO IMRTensor."""
        # Check if called as IMRTensor COO encoder
        if IMRTensor is not None and isinstance(tensor, IMRTensor):
            return cls._encode_imr_coo(tensor, kept_indices)

        # Also support passing list as second arg for IMRTensor without explicit type
        if kept_indices is not None and not isinstance(tensor, np.ndarray):
            return cls._encode_imr_coo(tensor, kept_indices)

        # Standard D2 INT8 NumPy bitmask encoding
        flat = tensor.astype(np.int8).flatten()
        total_elements = flat.size
        
        nz_mask = (flat != 0)
        nonzero_count = int(np.sum(nz_mask))
        nonzero_values = flat[nz_mask].tobytes()
        bitmask = np.packbits(nz_mask).tobytes()
        
        name_bytes = tensor_name.encode("utf-8")
        shape = list(tensor.shape)
        header = struct.pack(
            f"<H{len(name_bytes)}sB{len(shape)}IIII",
            len(name_bytes),
            name_bytes,
            len(shape),
            *shape,
            total_elements,
            nonzero_count,
            len(bitmask)
        )
        serialized_bytes = header + bitmask + nonzero_values
        
        raw_dense_bytes = total_elements * 1
        compressed_bytes = len(serialized_bytes)
        sparsity = (total_elements - nonzero_count) / max(1, total_elements)
        
        return {
            "tensor_name": tensor_name,
            "original_shape": shape,
            "dtype": "int8",
            "total_elements": total_elements,
            "nonzero_count": nonzero_count,
            "sparsity": float(sparsity),
            "bitmask": bitmask,
            "nonzero_values": nonzero_values,
            "raw_dense_bytes": raw_dense_bytes,
            "payload_bytes": len(bitmask) + len(nonzero_values),
            "serialized_bytes": compressed_bytes,
            "compression_ratio": float(raw_dense_bytes / max(1, compressed_bytes)),
            "storage_reduction_pct": float((1.0 - compressed_bytes / max(1, raw_dense_bytes)) * 100.0),
            "serialized_payload": serialized_bytes
        }

    @staticmethod
    def _encode_imr_coo(tensor: Any, kept_indices: Optional[Sequence[int]]) -> Any:
        """Encodes an IMRTensor into sparse COO format."""
        if tensor.dtype == "float16":
            arr = np.frombuffer(tensor.data, dtype=np.float16)
            out_dtype = "sparse_coo_float16"
            item_dtype = np.float16
        elif tensor.dtype == "int8":
            arr = np.frombuffer(tensor.data, dtype=np.int8)
            out_dtype = "sparse_coo_int8"
            item_dtype = np.int8
        elif tensor.dtype == "float32":
            arr = np.frombuffer(tensor.data, dtype=np.float32)
            out_dtype = "sparse_coo_float32"
            item_dtype = np.float32
        else:
            arr = np.frombuffer(tensor.data, dtype=np.float32)
            out_dtype = f"sparse_coo_{tensor.dtype}"
            item_dtype = np.float32

        if kept_indices is None:
            indices = np.where(arr != 0)[0].astype(np.uint32)
        else:
            indices = np.array(kept_indices, dtype=np.uint32)

        vals = arr[indices].astype(item_dtype)
        header = struct.pack("<I", len(arr))
        idx_bytes = indices.tobytes()
        val_bytes = vals.tobytes()
        payload = header + idx_bytes + val_bytes

        from uaqe.common.imr import IMRTensor
        return IMRTensor(shape=tensor.shape, dtype=out_dtype, data=payload)

    @classmethod
    def decode_tensor(cls, encoded_data: Union[Dict[str, Any], Any]) -> Union[np.ndarray, Any]:
        """Decodes either a D2 bitmask dictionary or an IMRTensor COO tensor."""
        if IMRTensor is not None and isinstance(encoded_data, IMRTensor):
            return cls._decode_imr_coo(encoded_data)
        if hasattr(encoded_data, "dtype") and str(encoded_data.dtype).startswith("sparse_coo_"):
            return cls._decode_imr_coo(encoded_data)

        shape = tuple(encoded_data["original_shape"])
        total_elements = encoded_data["total_elements"]
        nonzero_count = encoded_data["nonzero_count"]
        bitmask = encoded_data["bitmask"]
        nonzero_bytes = encoded_data["nonzero_values"]
        
        nz_mask = np.unpackbits(np.frombuffer(bitmask, dtype=np.uint8))[:total_elements].astype(bool)
        
        reconstructed = np.zeros(total_elements, dtype=np.int8)
        if nonzero_count > 0:
            nz_vals = np.frombuffer(nonzero_bytes, dtype=np.int8)
            reconstructed[nz_mask] = nz_vals
            
        return reconstructed.reshape(shape)

    @staticmethod
    def _decode_imr_coo(tensor: Any) -> Any:
        """Decodes a COO IMRTensor back to dense IMRTensor."""
        dtype_str = tensor.dtype.replace("sparse_coo_", "")
        if dtype_str == "float16":
            item_dtype = np.float16
        elif dtype_str == "int8":
            item_dtype = np.int8
        elif dtype_str == "float32":
            item_dtype = np.float32
        else:
            item_dtype = np.float32

        data = tensor.data
        total_elements = struct.unpack_from("<I", data, 0)[0]
        
        # Calculate kept count from remaining bytes
        rem_bytes = len(data) - 4
        itemsize = np.dtype(item_dtype).itemsize
        kept_count = rem_bytes // (4 + itemsize)
        
        idx_end = 4 + (kept_count * 4)
        indices = np.frombuffer(data[4:idx_end], dtype=np.uint32)
        vals = np.frombuffer(data[idx_end:idx_end + (kept_count * itemsize)], dtype=item_dtype)
        
        dense = np.zeros(total_elements, dtype=item_dtype)
        dense[indices] = vals

        from uaqe.common.imr import IMRTensor
        return IMRTensor(shape=tensor.shape, dtype=dtype_str, data=dense.tobytes())

    @staticmethod
    def decode_from_bytes(data: bytes, offset: int = 0) -> Tuple[np.ndarray, str, int]:
        """Decodes a tensor directly from a binary stream."""
        name_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        name = struct.unpack_from(f"<{name_len}s", data, offset)[0].decode("utf-8")
        offset += name_len
        
        ndim = struct.unpack_from("<B", data, offset)[0]
        offset += 1
        
        shape = struct.unpack_from(f"<{ndim}I", data, offset)
        offset += ndim * 4
        
        total_elements, nonzero_count, bitmask_len = struct.unpack_from("<III", data, offset)
        offset += 12
        
        bitmask = data[offset:offset + bitmask_len]
        offset += bitmask_len
        
        nonzero_bytes = data[offset:offset + nonzero_count]
        offset += nonzero_count
        
        encoded_dict = {
            "original_shape": list(shape),
            "total_elements": total_elements,
            "nonzero_count": nonzero_count,
            "bitmask": bitmask,
            "nonzero_values": nonzero_bytes
        }
        reconstructed = SparseEncoder.decode_tensor(encoded_dict)
        return reconstructed, name, offset
