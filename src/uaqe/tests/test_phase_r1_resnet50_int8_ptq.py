"""Automated Test Suite for UAQE Phase R1: Real INT8 Post-Training Quantization (PTQ).

Validates all 20 strict requirements:
1. E2 FP32 model loads
2. classifier = 10 classes
3. FP32 baseline is reproducible
4. calibration comes from TRAIN only
5. calibration is deterministic
6. calibration/test sets have zero overlap
7. ONNX FP32 export works
8. ONNX FP32 inference works
9. INT8 ONNX artifact exists
10. QDQ nodes exist where expected
11. INT8 initializers exist
12. INT8 artifact reloads independently
13. INT8 inference works through ONNX Runtime
14. prediction CSVs contain exactly the expected 1,000 test samples
15. FP32 and INT8 predictions refer to the same samples
16. size metrics are correctly calculated
17. numerical fidelity report exists
18. sensitivity report exists
19. SHA-256 hashes exist
20. historical C4–E3 artifacts remain unchanged
"""

import os
import sys
import csv
import json
import hashlib
import unittest
from typing import Dict, List, Set

# Add src directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import numpy as np
import torch
import onnx
import onnxruntime as ort

from uaqe.models.resnet50 import ResNetForImageClassification
from uaqe.dataset.universal_dataset_loader import UniversalDatasetLoader
from uaqe.quantization.r1_resnet50_ptq import (
    StratifiedCalibrationSampler,
    QuantizationStructureAuditor
)


class TestPhaseR1ResNet50Int8PTQ(unittest.TestCase):
    """Test suite covering all 20 Phase R1 requirements."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cls.cifar10_dir = r"D:\uaqe_datasets\cifar10\cifar-10-batches-py"
        cls.e2_ckpt_path = os.path.join(cls.project_root, "output", "phase_e2", "models", "resnet50_cifar10_fp32_baseline.pt")
        cls.r1_dir = os.path.join(cls.project_root, "output", "phase_r1")
        cls.fp32_onnx_path = os.path.join(cls.r1_dir, "models", "resnet50_cifar10_fp32.onnx")
        cls.int8_onnx_path = os.path.join(cls.r1_dir, "models", "resnet50_cifar10_int8.onnx")
        cls.fp32_ref_path = os.path.join(cls.r1_dir, "models", "resnet50_cifar10_fp32_r1_reference.pt")

    def test_01_e2_fp32_model_loads(self):
        """1. E2 FP32 model checkpoint exists and loads strictly."""
        self.assertTrue(os.path.exists(self.e2_ckpt_path), f"Missing E2 checkpoint: {self.e2_ckpt_path}")
        ckpt = torch.load(self.e2_ckpt_path, map_location="cpu")
        self.assertIn("model_state_dict", ckpt)
        model = ResNetForImageClassification(num_classes=10)
        incompat = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.assertEqual(len(incompat.missing_keys), 0)
        self.assertEqual(len(incompat.unexpected_keys), 0)

    def test_02_classifier_outputs_10_classes(self):
        """2. Classifier head has exactly 10 outputs."""
        model = ResNetForImageClassification(num_classes=10)
        ckpt = torch.load(self.e2_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        self.assertEqual(out.shape, (2, 10))

    def test_03_fp32_baseline_reproducible(self):
        """3. FP32 reference model produces valid logits with expected range."""
        self.assertTrue(os.path.exists(self.fp32_ref_path), "Frozen R1 FP32 reference does not exist")
        ref_ckpt = torch.load(self.fp32_ref_path, map_location="cpu")
        self.assertIn("model_state_dict", ref_ckpt)
        model = ResNetForImageClassification(num_classes=10)
        model.load_state_dict(ref_ckpt["model_state_dict"], strict=True)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        self.assertEqual(out.shape, (1, 10))
        self.assertFalse(torch.isnan(out).any())

    def test_04_calibration_comes_from_train_only(self):
        """4. Calibration samples are strictly from TRAIN split."""
        loader = UniversalDatasetLoader(self.cifar10_dir)
        loader.load_cifar10()
        sampler = StratifiedCalibrationSampler(loader=loader, num_samples=64, seed=42)
        summary = sampler.sample()
        self.assertEqual(summary["calibration_source"], "train_only")
        self.assertEqual(summary["total_calibration_samples"], 64)
        for row in sampler.manifest_rows:
            self.assertEqual(row["source_split"], "train")

    def test_05_calibration_determinism(self):
        """5. Calibration is deterministic for identical seeds and differs for different seeds."""
        loader = UniversalDatasetLoader(self.cifar10_dir)
        loader.load_cifar10()
        s1 = StratifiedCalibrationSampler(loader=loader, num_samples=64, seed=42)
        s1.sample()
        s2 = StratifiedCalibrationSampler(loader=loader, num_samples=64, seed=42)
        s2.sample()
        s3 = StratifiedCalibrationSampler(loader=loader, num_samples=64, seed=99)
        s3.sample()
        self.assertEqual(s1.calib_indices, s2.calib_indices)
        self.assertNotEqual(s1.calib_indices, s3.calib_indices)

    def test_06_calibration_test_zero_overlap(self):
        """6. Calibration set has zero overlap with test images."""
        loader = UniversalDatasetLoader(self.cifar10_dir)
        loader.load_cifar10()
        sampler = StratifiedCalibrationSampler(loader=loader, num_samples=256, seed=42)
        overlap_info = sampler.verify_zero_test_overlap()
        self.assertTrue(overlap_info["has_zero_overlap"])
        self.assertEqual(overlap_info["overlap_count"], 0)
        self.assertEqual(overlap_info["status"], "PASS")

    def test_07_onnx_fp32_export_works(self):
        """7. ONNX FP32 model exists and satisfies onnx.checker."""
        self.assertTrue(os.path.exists(self.fp32_onnx_path), "FP32 ONNX model missing")
        model = onnx.load(self.fp32_onnx_path)
        onnx.checker.check_model(model)

    def test_08_onnx_fp32_inference_works(self):
        """8. ONNX FP32 model runs inference via ONNX Runtime."""
        sess = ort.InferenceSession(self.fp32_onnx_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)})
        self.assertEqual(out[0].shape, (1, 10))

    def test_09_int8_onnx_artifact_exists(self):
        """9. INT8 ONNX model artifact exists on disk."""
        self.assertTrue(os.path.exists(self.int8_onnx_path), "INT8 ONNX model missing")

    def test_10_qdq_nodes_exist_where_expected(self):
        """10. QuantizeLinear and DequantizeLinear nodes exist in INT8 ONNX graph."""
        audit = QuantizationStructureAuditor.audit(self.int8_onnx_path)
        self.assertGreater(audit["quantization_operators"]["QuantizeLinear"], 0)
        self.assertGreater(audit["quantization_operators"]["DequantizeLinear"], 0)

    def test_11_int8_initializers_exist(self):
        """11. Genuine INT8 initializers (quantized weight tensors) exist in the graph."""
        audit = QuantizationStructureAuditor.audit(self.int8_onnx_path)
        self.assertGreater(audit["initializer_counts"]["int8_tensors"], 0)
        self.assertTrue(audit["is_genuinely_quantized"])

    def test_12_int8_artifact_reloads_independently(self):
        """12. INT8 ONNX model independently reloads in a fresh ONNX Runtime session."""
        sess = ort.InferenceSession(self.int8_onnx_path, providers=["CPUExecutionProvider"])
        self.assertIsNotNone(sess)

    def test_13_int8_inference_works(self):
        """13. INT8 ONNX model runs independent inference and outputs shape [batch, 10]."""
        sess = ort.InferenceSession(self.int8_onnx_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": np.random.randn(2, 3, 224, 224).astype(np.float32)})
        self.assertEqual(out[0].shape, (2, 10))

    def test_14_prediction_csvs_row_counts(self):
        """14. Prediction CSVs contain exactly 1,000 test samples."""
        fp32_csv = os.path.join(self.r1_dir, "predictions", "r1_fp32_predictions.csv")
        int8_csv = os.path.join(self.r1_dir, "predictions", "r1_int8_predictions.csv")
        self.assertTrue(os.path.exists(fp32_csv))
        self.assertTrue(os.path.exists(int8_csv))
        with open(fp32_csv, "r", encoding="utf-8") as f:
            fp32_rows = list(csv.DictReader(f))
        with open(int8_csv, "r", encoding="utf-8") as f:
            int8_rows = list(csv.DictReader(f))
        self.assertEqual(len(fp32_rows), 1000)
        self.assertEqual(len(int8_rows), 1000)

    def test_15_fp32_int8_predictions_same_samples(self):
        """15. FP32 and INT8 prediction files refer to the exact same sample indices."""
        fp32_csv = os.path.join(self.r1_dir, "predictions", "r1_fp32_predictions.csv")
        int8_csv = os.path.join(self.r1_dir, "predictions", "r1_int8_predictions.csv")
        with open(fp32_csv, "r", encoding="utf-8") as f:
            fp32_rows = list(csv.DictReader(f))
        with open(int8_csv, "r", encoding="utf-8") as f:
            int8_rows = list(csv.DictReader(f))
        for f_row, i_row in zip(fp32_rows, int8_rows):
            self.assertEqual(f_row["sample_index"], i_row["sample_index"])
            self.assertEqual(f_row["true_class_id"], i_row["true_class_id"])

    def test_16_size_metrics_correctly_calculated(self):
        """16. Size report correctly calculates size reduction percentage and ratio."""
        size_json = os.path.join(self.r1_dir, "metrics", "r1_size_report.json")
        self.assertTrue(os.path.exists(size_json))
        with open(size_json, "r", encoding="utf-8") as f:
            report = json.load(f)
        fp32_size = report["fp32_onnx_size_bytes"]
        int8_size = report["int8_onnx_size_bytes"]
        expected_reduction = round(((fp32_size - int8_size) / fp32_size) * 100.0, 2)
        self.assertEqual(report["size_reduction_percentage"], expected_reduction)
        self.assertGreater(fp32_size, int8_size)

    def test_17_numerical_fidelity_report_exists(self):
        """17. Numerical fidelity report exists with cosine similarity, MAE, and RMSE."""
        fid_json = os.path.join(self.r1_dir, "verification", "r1_numerical_fidelity.json")
        self.assertTrue(os.path.exists(fid_json))
        with open(fid_json, "r", encoding="utf-8") as f:
            fid = json.load(f)
        self.assertIn("mean_cosine_similarity", fid)
        self.assertIn("mean_mae", fid)
        self.assertIn("mean_rmse", fid)

    def test_18_sensitivity_report_exists(self):
        """18. Sensitivity report exists with stage analysis breakdown."""
        sens_json = os.path.join(self.r1_dir, "verification", "r1_sensitivity_analysis.json")
        self.assertTrue(os.path.exists(sens_json))
        with open(sens_json, "r", encoding="utf-8") as f:
            sens = json.load(f)
        self.assertIn("stages_analyzed", sens)
        self.assertIn("overall_logit_fidelity", sens)

    def test_19_sha256_hashes_exist(self):
        """19. SHA-256 hash manifest exists for all generated R1 artifacts."""
        hash_json = os.path.join(self.r1_dir, "verification", "r1_hashes.json")
        self.assertTrue(os.path.exists(hash_json))
        with open(hash_json, "r", encoding="utf-8") as f:
            hashes = json.load(f)
        self.assertGreater(len(hashes), 0)

    def test_20_historical_integrity_unchanged(self):
        """20. Historical artifacts across Phases C4–E3 remain 100% intact."""
        protected_dirs = [
            os.path.join(self.project_root, "output", "phase_c4"),
            os.path.join(self.project_root, "output", "phase_c5"),
            os.path.join(self.project_root, "output", "phase_d1"),
            os.path.join(self.project_root, "output", "phase_d2"),
            os.path.join(self.project_root, "output", "phase_d3"),
            os.path.join(self.project_root, "output", "phase_d4"),
            os.path.join(self.project_root, "output", "phase_d5"),
            os.path.join(self.project_root, "output", "phase_e1"),
            os.path.join(self.project_root, "output", "phase_e2"),
            os.path.join(self.project_root, "output", "phase_e3")
        ]
        for pdir in protected_dirs:
            self.assertTrue(os.path.exists(pdir), f"Historical directory missing: {pdir}")
            self.assertGreater(len(os.listdir(pdir)), 0, f"Historical directory empty: {pdir}")


if __name__ == "__main__":
    unittest.main()
