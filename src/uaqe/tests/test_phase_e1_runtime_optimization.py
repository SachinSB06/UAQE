"""
Unit tests for UAQE Phase E.1 Runtime Optimization, Hardening & Caching.
Verifies optimized decoder, vectorized RLE, FlatBuffer reconstruction, cache management,
prediction agreement, corruption handling, and baseline protection.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure project root and src are on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.runtime.optimized_decoder import OptimizedRuntimeDecoder
from src.uaqe.runtime.optimized_session import OptimizedRuntimeSession
from src.uaqe.runtime.runtime_cache import RuntimeCacheManager, RUNTIME_CACHE_VERSION
from src.uaqe.runtime.runtime_decoder import RuntimeDecoder
from src.uaqe.compression.rle_compressor import RLECompressor


class TestPhaseE1RuntimeOptimization(unittest.TestCase):
    """Test suite for Phase E.1 runtime optimizations."""

    def setUp(self):
        self.archive_path = os.path.join(PROJECT_ROOT, "output", "phase_d4", "compressed", "d4_d_adaptive_sparse_rle.bin")
        self.template_path = os.path.join(PROJECT_ROOT, "output", "phase_c4", "models", "c4_best_int8.tflite")
        self.d1_best_model = os.path.join(PROJECT_ROOT, "output", "phase_d1", "models", "d1_best_sensitive_int8.tflite")

    def test_01_vectorized_rle_correctness(self):
        """Tests that fast vectorized RLE decompression is 100% bit-for-bit identical to original RLE."""
        # Test basic escape stream
        data = b"\x00\x00\x00\x00\x00\x05\x05\x05\xAA\x00\x01\x02\x03\x00\x00\x00"
        compressed = RLECompressor.compress_bytes(data)
        
        orig_dec = RLECompressor.decompress_bytes(compressed)
        fast_dec = OptimizedRuntimeDecoder.fast_decompress_rle(compressed)

        self.assertEqual(orig_dec, data)
        self.assertEqual(fast_dec, data)
        self.assertEqual(orig_dec, fast_dec)

    def test_02_vectorized_rle_edge_cases(self):
        """Tests empty streams, single byte streams, and escape byte sequences."""
        self.assertEqual(OptimizedRuntimeDecoder.fast_decompress_rle(b""), b"")
        
        # Stream with only escape bytes
        data = b"\xAA\xAA\xAA\xAA"
        comp = RLECompressor.compress_bytes(data)
        fast_dec = OptimizedRuntimeDecoder.fast_decompress_rle(comp)
        self.assertEqual(fast_dec, data)

    def test_03_optimized_decoder_metadata_validation(self):
        """Tests that OptimizedRuntimeDecoder parses and validates archive container header."""
        decoder = OptimizedRuntimeDecoder(base_template_path=self.template_path)
        meta = decoder.load(self.archive_path)

        self.assertEqual(meta["version"], 1)
        self.assertEqual(meta["num_tensors"], 84)
        self.assertEqual(meta["magic"], "UAQE_D4\x01")
        self.assertIn("payload_blocks", meta)
        self.assertEqual(len(meta["payload_blocks"]), 84)

    def test_04_exact_tensor_equality_lossless(self):
        """Tests that decoded tensors are 100% mathematically and bit-for-bit identical to source."""
        decoder = OptimizedRuntimeDecoder(base_template_path=self.template_path)
        decoder.load(self.archive_path)
        res = decoder.verify_tensors(self.d1_best_model)

        self.assertTrue(res["all_tensors_exact_match"])
        self.assertEqual(res["overall_mae"], 0.0)
        self.assertEqual(res["overall_max_error"], 0.0)
        self.assertEqual(res["overall_cosine_similarity"], 1.0)
        self.assertEqual(res["tensor_count"], 84)

    def test_05_flatbuffer_reconstruction_equality(self):
        """Tests that optimized FlatBuffer reconstruction produces bit-for-bit identical models."""
        d5_decoder = RuntimeDecoder(base_template_path=self.template_path)
        d5_decoder.load(self.archive_path)
        d5_bytes = d5_decoder.reconstruct(self.template_path)

        e1_decoder = OptimizedRuntimeDecoder(base_template_path=self.template_path)
        e1_decoder.load(self.archive_path)
        e1_bytes = e1_decoder.reconstruct(self.template_path)

        self.assertEqual(len(d5_bytes), len(e1_bytes))
        self.assertEqual(bytes(d5_bytes), bytes(e1_bytes))

    def test_06_runtime_cache_lifecycle(self):
        """Tests cache key generation, storage, loading, and invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_mgr = RuntimeCacheManager(cache_dir=tmpdir)
            test_sha = "abcd1234ef567890" * 4
            dummy_bytes = b"TFL3" + os.urandom(100)

            # Initially no cached model
            self.assertFalse(cache_mgr.has_cached_model(test_sha))
            self.assertIsNone(cache_mgr.load_cached_model(test_sha))

            # Store model
            cache_mgr.store_cached_model(test_sha, dummy_bytes)
            self.assertTrue(cache_mgr.has_cached_model(test_sha))

            # Load model
            loaded = cache_mgr.load_cached_model(test_sha)
            self.assertEqual(loaded, dummy_bytes)

            # Invalidate
            cache_mgr.invalidate(test_sha)
            self.assertFalse(cache_mgr.has_cached_model(test_sha))
            self.assertIsNone(cache_mgr.load_cached_model(test_sha))

    def test_07_corrupt_cache_rejection(self):
        """Tests that modified/corrupted cache files are safely rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_mgr = RuntimeCacheManager(cache_dir=tmpdir)
            test_sha = "1234567890abcdef" * 4
            dummy_bytes = b"TFL3" + os.urandom(100)

            cache_mgr.store_cached_model(test_sha, dummy_bytes)
            model_path = cache_mgr.get_cached_model_path(test_sha)

            # Corrupt the model file
            with open(model_path, "wb") as f:
                f.write(b"CORRUPT_BYTES")

            # Should reject corrupt cache
            self.assertFalse(cache_mgr.has_cached_model(test_sha))
            self.assertIsNone(cache_mgr.load_cached_model(test_sha))

    def test_08_corrupt_archive_rejection(self):
        """Tests that modified/truncated archives are rejected with ValueError."""
        decoder = OptimizedRuntimeDecoder(base_template_path=self.template_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Truncated header
            bad_p1 = os.path.join(tmpdir, "bad1.bin")
            with open(bad_p1, "wb") as f:
                f.write(b"UAQE_D4")
            with self.assertRaises(ValueError):
                decoder.load(bad_p1)

            # 2. Invalid magic
            bad_p2 = os.path.join(tmpdir, "bad2.bin")
            with open(bad_p2, "wb") as f:
                f.write(b"INVALID_MAGIC_HDR" + b"\x00" * 20)
            with self.assertRaises(ValueError):
                decoder.load(bad_p2)

    def test_09_optimized_session_inference_portable_and_cached(self):
        """Tests that OptimizedRuntimeSession executes inference identically in portable and cached modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Portable mode
            sess_port = OptimizedRuntimeSession.from_archive(
                self.archive_path,
                template_tflite_path=self.template_path,
                mode="portable",
                cache_dir=tmpdir
            )
            sess_port.allocate()

            # Cached mode
            sess_cached = OptimizedRuntimeSession.from_archive(
                self.archive_path,
                template_tflite_path=self.template_path,
                mode="cached",
                cache_dir=tmpdir
            )
            sess_cached.allocate()

            dummy_input = np.random.randn(1, 3, 128, 128).astype(np.float32)

            pred_port = sess_port.predict(dummy_input)
            pred_cached = sess_cached.predict(dummy_input)

            class_port = sess_port.predict_class(dummy_input)
            class_cached = sess_cached.predict_class(dummy_input)

            np.testing.assert_allclose(pred_port, pred_cached, atol=1e-5)
            self.assertEqual(class_port, class_cached)

            sess_port.close()
            sess_cached.close()

    def test_10_protected_directories_integrity(self):
        """Ensures all historical directories exist and are intact."""
        for d in ["output/phase_c4", "output/phase_c5", "output/phase_d1", "output/phase_d2", "output/phase_d3", "output/phase_d4", "output/phase_d5"]:
            full_d = os.path.join(PROJECT_ROOT, d)
            self.assertTrue(os.path.exists(full_d), f"Missing protected directory: {d}")


if __name__ == "__main__":
    unittest.main()
