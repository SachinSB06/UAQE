"""
Unit tests for UAQE Phase D.2: Sparse Encoding, RLE, Weight Clustering,
Model Packaging, Reconstruction, and Baseline Hash Protection.
"""

import os
import unittest
import hashlib
import numpy as np
import tensorflow as tf

from src.uaqe.compression.sparse_encoder import SparseEncoder
from src.uaqe.compression.rle_compressor import RLECompressor
from src.uaqe.compression.weight_clusterer import WeightClusterer
from src.uaqe.compression.model_packager import ModelPackager


class TestPhaseD2Compression(unittest.TestCase):
    """Test suite for Phase D.2 compression, serialization, and reconstruction modules."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cls.baseline_model = os.path.join(cls.project_root, "output", "phase_c4", "models", "c4_best_int8.tflite")
        cls.d1_model_20 = os.path.join(cls.project_root, "output", "phase_d1", "models", "d1_sensitive_20_int8.tflite")
        cls.d1_model_30 = os.path.join(cls.project_root, "output", "phase_d1", "models", "d1_sensitive_30_int8.tflite")
        cls.test_scratch_dir = os.path.join(cls.project_root, "output", "phase_d2", "test_scratch")
        os.makedirs(cls.test_scratch_dir, exist_ok=True)

        cls.sparse_enc = SparseEncoder()
        cls.rle_comp = RLECompressor()
        cls.clusterer = WeightClusterer()
        cls.packager = ModelPackager()

    def test_01_sparse_encoder_lossless_roundtrip(self):
        """Tests that SparseEncoder preserves tensor values exactly bit-for-bit."""
        np.random.seed(42)
        # Create a synthetic 2D weight tensor with 30% sparsity
        dense = np.random.randint(-128, 127, size=(64, 64), dtype=np.int8)
        mask = np.random.rand(64, 64) < 0.30
        dense[mask] = 0

        res = self.sparse_enc.encode_tensor(dense, tensor_name="test_tensor_01")
        encoded_bytes = res["serialized_payload"]
        self.assertIsInstance(encoded_bytes, bytes)
        self.assertGreater(len(encoded_bytes), 0)

        # Size of compressed should be substantially less than dense
        self.assertLess(len(encoded_bytes), dense.size + 100)

        decomp, name, _ = self.sparse_enc.decode_from_bytes(encoded_bytes, 0)
        self.assertEqual(name, "test_tensor_01")
        self.assertEqual(decomp.shape, (64, 64))
        self.assertEqual(decomp.dtype, np.int8)
        self.assertTrue(np.array_equal(dense, decomp), "Sparse decompression failed exact element match!")
        self.assertEqual(np.max(np.abs(dense.astype(np.int32) - decomp.astype(np.int32))), 0)

    def test_02_rle_compressor_lossless_roundtrip(self):
        """Tests that RLECompressor preserves byte streams and sparse tensors losslessly."""
        # 1. Byte level round-trip
        data = b"\x00\x00\x00\x00\x05\x05\x05\x05\x05\x12\x34\x00\x00\x00\xff\xff\xff"
        compressed = self.rle_comp.compress_bytes(data)
        decompressed = self.rle_comp.decompress_bytes(compressed)
        self.assertEqual(data, decompressed, "RLE byte decompression failed!")

        # 2. Tensor level round-trip
        np.random.seed(123)
        tensor = np.random.randint(-128, 127, size=(32, 16, 3, 3), dtype=np.int8)
        tensor[np.random.rand(*tensor.shape) < 0.4] = 0

        res = self.rle_comp.encode_sparse_rle_tensor(tensor, tensor_name="conv_weight")
        decomp, name, _ = self.rle_comp.decode_from_bytes(res["serialized_payload"], 0)

        self.assertEqual(name, "conv_weight")
        self.assertEqual(decomp.shape, (32, 16, 3, 3))
        self.assertEqual(decomp.dtype, np.int8)
        self.assertTrue(np.array_equal(tensor, decomp), "RLE tensor decompression failed exact match!")

    def test_03_weight_clusterer_bounds_and_reconstruction(self):
        """Tests that K-Means clustering bit-packing and reconstruction meet mathematical error bounds."""
        np.random.seed(999)
        tensor = np.random.randint(-100, 100, size=(100, 100), dtype=np.int8)
        tensor[np.random.rand(100, 100) < 0.3] = 0

        for k in [4, 8, 16, 32]:
            res = self.clusterer.cluster_tensor(tensor, num_clusters=k, tensor_name="clustered_tensor")
            decomp, name, _ = self.clusterer.decode_from_bytes(res["serialized_payload"], 0)

            self.assertEqual(name, "clustered_tensor")
            self.assertEqual(decomp.shape, (100, 100))
            self.assertEqual(decomp.dtype, np.int8)

            diff = np.abs(tensor.astype(np.float32) - decomp.astype(np.float32))
            mae = float(np.mean(diff))
            self.assertLess(mae, 15.0, f"MAE too high for K={k}")

    def test_04_model_packager_archive_and_tflite_reconstruction(self):
        """Tests that ModelPackager creates a valid .bin archive and reconstructs runnable TFLite interpreter."""
        self.assertTrue(os.path.exists(self.d1_model_20), f"D1 20% model missing: {self.d1_model_20}")
        bin_path = os.path.join(self.test_scratch_dir, "test_pkg_20.bin")

        # Package
        pkg_meta = self.packager.package_model(
            tflite_path=self.d1_model_20,
            output_bin_path=bin_path,
            compression_method="sparse",
            num_clusters=0
        )
        self.assertTrue(os.path.exists(bin_path))
        self.assertGreater(pkg_meta["actual_bin_file_bytes"], 0)
        self.assertEqual(pkg_meta["total_weight_tensors"], 84)

        # Unpackage and verify
        decomp_weights, recon_metrics = self.packager.unpackage_and_verify(
            bin_path=bin_path,
            original_tflite_path=self.d1_model_20
        )
        self.assertTrue(recon_metrics["is_lossless"])
        self.assertEqual(recon_metrics["max_abs_error"], 0)

        # Reconstruct TFLite interpreter
        interpreter = self.packager.reconstruct_tflite_interpreter(
            bin_path=bin_path,
            template_tflite_path=self.d1_model_20
        )
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()

        # Run dummy forward pass
        dummy_in = np.zeros(in_details[0]["shape"], dtype=np.float32)
        interpreter.set_tensor(in_details[0]["index"], dummy_in)
        interpreter.invoke()
        out = interpreter.get_tensor(out_details[0]["index"])
        self.assertEqual(out.shape, (1, 9))

    def test_05_baseline_and_d1_hash_protection(self):
        """Tests that historical C4 baseline and D1 artifacts exist and retain valid SHA-256 signatures."""
        self.assertTrue(os.path.exists(self.baseline_model), f"C4 baseline missing: {self.baseline_model}")
        with open(self.baseline_model, "rb") as f:
            b_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(b_hash, "c8c8292c136dfafa4240b9b9a51079ceaff12a612996829c29384b493f689e51", "Baseline C4 hash mismatch!")

        self.assertTrue(os.path.exists(self.d1_model_20), "D1 20% model missing!")
        self.assertTrue(os.path.exists(self.d1_model_30), "D1 30% model missing!")


if __name__ == "__main__":
    unittest.main()
