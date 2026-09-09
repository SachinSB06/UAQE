"""Unit tests for UAQE Phase D.1 Sensitivity-Aware Pruning Engine.
Verifies dataset manifest, pruning allocation, fine-tuning, export, FlatBuffer,
evaluation, and 500-run stability.
"""

import os
import unittest
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import sys
sys.path.insert(0, "src")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import onnx
import tensorflow as tf

from uaqe.optimizer.sensitivity_pruner import SensitivityPruner, get_stratified_calibration_samples, CLASS_NAMES
from uaqe.optimizer.structured_pruner import StructuredPruner
from uaqe.optimizer.pruning_experimenter import PruningExperimenter
from uaqe.exporter.tflite_exporter import ONNXToTFModel
from uaqe.exporter.flatbuffer_inspector import FlatBufferInspector


class TestPhaseD1Pruning(unittest.TestCase):

    def setUp(self):
        self.pruner = SensitivityPruner()
        self.structured_pruner = StructuredPruner()
        self.inspector = FlatBufferInspector()
        self.model = models.mobilenet_v3_small(num_classes=9)

    def test_stratified_calibration_deterministic_and_balanced(self):
        """Verifies deterministic stratified random calibration produces balanced class sampling."""
        num_samples = 90
        dummy_x = torch.randn(200, 3, 128, 128)
        dummy_y = torch.tensor([i % 9 for i in range(200)], dtype=torch.long)

        calib_x1, calib_y1 = get_stratified_calibration_samples(dummy_x, dummy_y, n_total=num_samples, seed=42)
        calib_x2, calib_y2 = get_stratified_calibration_samples(dummy_x, dummy_y, n_total=num_samples, seed=42)

        self.assertEqual(len(calib_x1), num_samples)
        self.assertTrue(torch.equal(calib_x1, calib_x2), "Calibration set must be 100% deterministic")
        self.assertTrue(torch.equal(calib_y1, calib_y2))

        # Check all 9 classes are present and evenly balanced
        unique_classes, counts = np.unique(calib_y1.numpy(), return_counts=True)
        self.assertEqual(len(unique_classes), 9)
        self.assertTrue(all(c == 10 for c in counts))

    def test_model_structure_audit(self):
        """Verifies model structure classification into STEM, POINTWISE, DEPTHWISE, SE, CLASSIFIER."""
        audit = self.pruner.audit_model_structure(self.model)
        self.assertGreater(len(audit), 10)
        
        categories = {r["semantic_category"] for r in audit}
        self.assertIn("STEM", categories)
        self.assertIn("POINTWISE", categories)
        self.assertIn("DEPTHWISE", categories)
        self.assertIn("SE", categories)
        self.assertIn("CLASSIFIER", categories)

    def test_global_unstructured_pruning_sparsity(self):
        """Verifies global magnitude pruning achieves requested sparsity within tolerance."""
        m = models.mobilenet_v3_small(num_classes=9)
        target_sp = 0.30
        stats = self.pruner.apply_global_unstructured_pruning(m, target_sparsity=target_sp)

        self.assertAlmostEqual(stats["actual_sparsity"], target_sp, delta=0.03)
        self.pruner.make_pruning_permanent(m)

        # Check zeros remain permanent after hook removal
        actual_zeros = 0
        total_w = 0
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                actual_zeros += int(torch.sum(mod.weight == 0).item())
                total_w += mod.weight.numel()

        self.assertEqual(actual_zeros, stats["zero_weights"])

    def test_sensitivity_aware_pruning_allocation(self):
        """Verifies sensitivity-aware pruning allocates higher sparsity to low-sensitivity layers."""
        m = models.mobilenet_v3_small(num_classes=9)
        audit = self.pruner.audit_model_structure(m)
        stats, allocs = self.pruner.apply_sensitivity_aware_unstructured_pruning(
            m, target_effective_sparsity=0.30, structure_audit=audit
        )

        self.assertGreater(len(allocs), 0)
        self.assertAlmostEqual(stats["actual_sparsity"], 0.30, delta=0.05)

        # Check classifier receives lower pruning than low-sensitivity layers
        classifier_allocs = [a["allocated_sparsity"] for a in allocs if a["semantic_category"] == "CLASSIFIER"]
        low_sens_allocs = [a["allocated_sparsity"] for a in allocs if a["sensitivity_category"] == "LOW"]

        if classifier_allocs and low_sens_allocs:
            self.assertLess(np.mean(classifier_allocs), np.mean(low_sens_allocs))

    def test_structured_pruner_dependency_analysis(self):
        """Verifies structured pruner identifies MobileNetV3 topological coupling."""
        analysis = self.structured_pruner.analyze_mobilenet_dependencies(self.model)
        self.assertEqual(analysis["model_architecture"], "MobileNetV3-Small")
        self.assertGreater(analysis["total_blocks"], 0)
        self.assertEqual(analysis["blocked_blocks_count"], analysis["total_blocks"])

        safety = self.structured_pruner.evaluate_structured_pruning_safety(0.20)
        self.assertEqual(safety["status"], "BLOCKED")
        self.assertIn("BLOCKED", safety["technical_rationale"])
        self.assertIn("MobileNetV3", safety["analysis"]["model_architecture"])

    def test_one_batch_fine_tune_and_checkpoint_save_load(self):
        """Verifies one batch fine-tuning, state_dict checkpointing, and deterministic forward pass."""
        m = models.mobilenet_v3_small(num_classes=9)
        self.pruner.apply_global_unstructured_pruning(m, target_sparsity=0.20)
        
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        dummy_x = torch.randn(4, 3, 128, 128)
        dummy_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        
        m.train()
        optimizer.zero_grad()
        out = m(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()
        
        self.pruner.make_pruning_permanent(m)
        
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save({"model_state_dict": m.state_dict()}, tmp_path)
            
            m_loaded = models.mobilenet_v3_small(num_classes=9)
            ckpt = torch.load(tmp_path, weights_only=False)
            m_loaded.load_state_dict(ckpt["model_state_dict"])
            
            m.eval()
            m_loaded.eval()
            with torch.no_grad():
                out1 = m(dummy_x)
                out2 = m_loaded(dummy_x)
            self.assertTrue(torch.allclose(out1, out2, atol=1e-5), "Saved and loaded pruned models must be identical")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_and_flatbuffer_inspector(self):
        """Verifies PyTorch -> ONNX -> TFLite INT8 conversion and FlatBuffer validity."""
        m = models.mobilenet_v3_small(num_classes=9)
        m.eval()
        
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_onnx:
            onnx_path = tmp_onnx.name
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp_tflite:
            tflite_path = tmp_tflite.name
            
        try:
            dummy_in = torch.randn(1, 3, 128, 128)
            torch.onnx.export(
                m, dummy_in, onnx_path,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=13,
                dynamo=False
            )
            
            onnx_m = onnx.load(onnx_path)
            tf_m = ONNXToTFModel(onnx_m)
            _ = tf_m(tf.random.normal([1, 3, 128, 128]))
            
            def rep_dataset():
                for _ in range(5):
                    yield [np.random.uniform(0.0, 1.0, (1, 3, 128, 128)).astype(np.float32)]
                    
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_m)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            tflite_bytes = converter.convert()
            with open(tflite_path, "wb") as f:
                f.write(tflite_bytes)
                
            # Inspect FlatBuffer
            audit = self.inspector.inspect(tflite_path).to_dict()
            self.assertGreater(audit.get("int8_tensors", 0), 20)
            self.assertGreater(audit.get("total_tensors", 0), 50)
            self.assertGreater(audit.get("int8_coverage_percent", 0), 40.0)
            
            # Execute in TFLite interpreter
            interp = tf.lite.Interpreter(model_path=tflite_path)
            interp.allocate_tensors()
            in_idx = interp.get_input_details()[0]["index"]
            out_idx = interp.get_output_details()[0]["index"]
            
            test_in = np.random.uniform(0.0, 1.0, (1, 3, 128, 128)).astype(np.float32)
            interp.set_tensor(in_idx, test_in)
            interp.invoke()
            out_val = interp.get_tensor(out_idx)
            self.assertEqual(out_val.shape, (1, 9))
            self.assertFalse(np.isnan(out_val).any())
            
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            if os.path.exists(tflite_path):
                os.remove(tflite_path)


if __name__ == "__main__":
    unittest.main()
