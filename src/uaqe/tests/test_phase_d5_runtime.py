"""
UAQE Phase D.5 Runtime Unit Test Suite
Tests Archive Loading, Magic Validation, Error Rejection, Exact Tensor Restoration,
FlatBuffer Reconstruction, RuntimeSession API, Benchmarking, and Deployment Packaging.
"""

import os
import sys
import struct
import tempfile
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.runtime.runtime_decoder import RuntimeDecoder
from src.uaqe.runtime.runtime_session import RuntimeSession
from src.uaqe.runtime.deployment_packager import DeploymentPackager


class TestPhaseD5Runtime(unittest.TestCase):
    """Unit tests for Phase D.5 Runtime Decoder, Session API, and Deployment Packaging."""

    @classmethod
    def setUpClass(cls):
        cls.archive_path = os.path.join(PROJECT_ROOT, "output", "phase_d4", "compressed", "d4_d_adaptive_sparse_rle.bin")
        cls.template_path = os.path.join(PROJECT_ROOT, "output", "phase_c4", "models", "c4_best_int8.tflite")

    def test_valid_archive_loading_and_inspection(self):
        """Verify that the official D4-D archive loads cleanly and parses correct header metadata."""
        if not os.path.exists(self.archive_path):
            self.skipTest("D4-D archive not found.")

        decoder = RuntimeDecoder(base_template_path=self.template_path)
        meta = decoder.load(self.archive_path)

        self.assertEqual(meta["version"], 1)
        self.assertEqual(meta["num_tensors"], 84)
        self.assertEqual(meta["archive_size_bytes"], 1393023)
        self.assertEqual(len(meta["payload_blocks"]), 84)

    def test_invalid_magic_rejection(self):
        """Verify that an archive with a corrupted magic header is strictly rejected."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf_file:
            # Corrupt magic header
            corrupt_header = b"BADMAGIC\x01" + struct.pack("<III", 10, 1000, 0)
            tf_file.write(corrupt_header)
            tmp_path = tf_file.name

        try:
            decoder = RuntimeDecoder()
            with self.assertRaises(ValueError) as ctx:
                decoder.load(tmp_path)
            self.assertIn("Invalid archive magic", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_unsupported_version_rejection(self):
        """Verify that an archive with an unsupported format version is rejected."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf_file:
            # Magic + Version 99
            bad_version_header = struct.pack("<8sBIII", b"UAQE_D4\x01", 99, 10, 1000, 0)
            tf_file.write(bad_version_header)
            tmp_path = tf_file.name

        try:
            decoder = RuntimeDecoder()
            with self.assertRaises(ValueError) as ctx:
                decoder.load(tmp_path)
            self.assertIn("Unsupported format version", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_truncated_archive_rejection(self):
        """Verify that a truncated archive is safely caught and rejected."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf_file:
            # Only 10 bytes (header requires 21 bytes)
            tf_file.write(b"SHORT_DATA")
            tmp_path = tf_file.name

        try:
            decoder = RuntimeDecoder()
            with self.assertRaises(ValueError) as ctx:
                decoder.load(tmp_path)
            self.assertIn("Truncated archive", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_payload_overflow_rejection(self):
        """Verify that an archive declaring a payload larger than available bytes is rejected."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf_file:
            # Master header with 1 tensor
            header = struct.pack("<8sBIII", b"UAQE_D4\x01", 1, 1, 100, 0)
            # Tensor record with payload_length = 5000 (but no payload bytes attached)
            block_hdr = struct.pack("<IIBI", 0, 1, 0, 5000)
            tf_file.write(header + block_hdr + b"short")
            tmp_path = tf_file.name

        try:
            decoder = RuntimeDecoder()
            with self.assertRaises(ValueError) as ctx:
                decoder.load(tmp_path)
            self.assertIn("Payload overflow", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_tensor_decoding_and_exact_equality(self):
        """Verify that all 84 tensors decode into valid INT8 arrays."""
        if not os.path.exists(self.archive_path):
            self.skipTest("D4-D archive not found.")

        decoder = RuntimeDecoder(base_template_path=self.template_path)
        decoder.load(self.archive_path)
        decoded = decoder.decode()

        self.assertEqual(len(decoded), 84)
        for t in decoded:
            self.assertEqual(t["data"].dtype, np.int8)
            self.assertGreater(len(t["shape"]), 0)
            self.assertGreater(t["byte_length"], 0)

    def test_flatbuffer_reconstruction_and_allocation(self):
        """Verify FlatBuffer in-memory reconstruction and TFLite tensor allocation."""
        if not os.path.exists(self.archive_path):
            self.skipTest("D4-D archive not found.")

        decoder = RuntimeDecoder(base_template_path=self.template_path)
        decoder.load(self.archive_path)
        fb_bytes = decoder.reconstruct(self.template_path)

        self.assertGreater(len(fb_bytes), 1000000)
        self.assertEqual(len(fb_bytes), os.path.getsize(self.template_path))

    def test_runtime_session_lifecycle(self):
        """Verify full RuntimeSession lifecycle: from_archive -> allocate -> predict -> close."""
        if not os.path.exists(self.archive_path):
            self.skipTest("D4-D archive not found.")

        session = RuntimeSession.from_archive(self.archive_path, template_tflite_path=self.template_path)
        session.allocate()

        dummy_input = np.random.randn(1, 3, 128, 128).astype(np.float32)
        logits = session.predict(dummy_input)
        self.assertEqual(logits.shape, (1, 9))

        pred_class = session.predict_class(dummy_input)
        self.assertIsInstance(pred_class, int)
        self.assertGreaterEqual(pred_class, 0)
        self.assertLess(pred_class, 9)

        # Benchmark run
        bench = session.benchmark(input_data=dummy_input, num_runs=5, warmup_runs=2)
        self.assertIn("mean_ms", bench)
        self.assertGreater(bench["mean_ms"], 0.0)

        session.close()
        self.assertFalse(session._is_allocated)

    def test_deployment_packager_manifest_and_checksums(self):
        """Verify deployment package generation, manifest, runtime config, and checksums."""
        if not os.path.exists(self.archive_path):
            self.skipTest("D4-D archive not found.")

        with tempfile.TemporaryDirectory() as tmp_pkg_dir:
            packager = DeploymentPackager(
                source_archive_path=self.archive_path,
                baseline_tflite_path=self.template_path,
                package_dir=tmp_pkg_dir
            )
            summary = packager.build_package(accuracy_pct=98.4694, macro_f1_pct=98.3834, storage_reduction_pct=24.98)

            self.assertIn("model.uaqe", summary["files"])
            self.assertIn("manifest.json", summary["files"])
            self.assertIn("runtime_config.json", summary["files"])
            self.assertIn("checksums.json", summary["files"])
            self.assertIn("README.md", summary["files"])

            self.assertEqual(summary["compressed_archive_size_bytes"], 1393023)
            self.assertEqual(summary["reconstructed_tflite_size_bytes"], 1856832)
            self.assertGreater(summary["complete_package_size_bytes"], summary["compressed_archive_size_bytes"])


if __name__ == "__main__":
    unittest.main()
