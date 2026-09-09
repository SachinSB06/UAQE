import os
import sys
import unittest
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "src")

from uaqe.quantization.qat_policy import QATPolicy
from uaqe.quantization.activation_range_analyzer import ActivationRangeAnalyzer
from uaqe.quantization.qat_trainer import QATTrainer

class TestPhaseC2QAT(unittest.TestCase):
    """Unit and integration tests for Phase C.2 Quantization-Aware Training,
    sensitivity policies, activation range analysis, INT8 conversion, and regression validation.
    """

    @classmethod
    def setUpClass(cls):
        cls.fp32_ckpt = "output/phase_c1/models/mobilenetv3_sem_9class_fp32.pth"
        cls.dataset_root = r"D:\semiconductor_dataset\dataset"
        cls.benchmark_path = "datasets/hackathon_test_dataset"
        cls.trainer = QATTrainer(
            fp32_checkpoint_path=cls.fp32_ckpt,
            dataset_root=cls.dataset_root,
            benchmark_path=cls.benchmark_path
        )
        cls.policy = QATPolicy()
        cls.analyzer = ActivationRangeAnalyzer()

    def test_01_checkpoint_loading_and_shape(self):
        """Test 1: FP32 checkpoint loads and produces 9-class output [batch, 9]."""
        model = self.trainer.load_base_fp32_model()
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 9))

    def test_02_class_mapping_validity(self):
        """Test 2: Exactly 9 physical classes mapped in deterministic alphabetical order."""
        self.assertEqual(len(self.trainer.class_names), 9)
        expected = ["bridge", "clean", "cmp", "crack", "opens", "other", "particle", "scratch", "vias"]
        self.assertEqual(self.trainer.class_names, expected)

    def test_03_dataset_counts_and_isolation(self):
        """Test 3: Dataset split counts match and benchmark is isolated."""
        self.assertEqual(len(self.trainer.train_y), 877)
        self.assertEqual(len(self.trainer.val_y), 184)
        self.assertEqual(len(self.trainer.test_y), 197)
        self.assertEqual(len(self.trainer.bench_y), 296)

    def test_04_qat_policy_sensitivity_consumption(self):
        """Test 4: QAT Policy correctly parses sensitive SE fc2 and Depthwise layers."""
        self.assertGreater(len(self.policy.se_fc2_layers), 0)
        self.assertGreater(len(self.policy.depthwise_layers), 0)
        qconfig_se = self.policy.get_sensitivity_aware_qconfig("features.11.block.2.fc2")
        self.assertIsNotNone(qconfig_se)

    def test_05_qat_model_preparation(self):
        """Test 5: Model preparation attaches FakeQuantize modules and observers."""
        model = self.trainer.load_base_fp32_model()
        qat_model = self.policy.apply_qat_policy(model, policy_mode="sensitivity_aware")
        
        has_fake_quant = False
        for name, m in qat_model.named_modules():
            if "fake_quant" in name.lower() or "observer" in name.lower() or "FakeQuantize" in type(m).__name__:
                has_fake_quant = True
                break
        self.assertTrue(has_fake_quant)

    def test_06_qat_training_step(self):
        """Test 6: QAT model executes forward and backward pass for one batch."""
        model = self.trainer.load_base_fp32_model()
        qat_model = self.policy.apply_qat_policy(model, policy_mode="standard")
        qat_model.train()
        
        optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-4)
        x = torch.randn(4, 3, 128, 128)
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        
        optimizer.zero_grad()
        out = qat_model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        optimizer.step()
        
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_07_activation_range_extraction(self):
        """Test 7: ActivationRangeAnalyzer extracts intermediate layer activations."""
        model = self.trainer.load_base_fp32_model()
        sample_x = torch.randn(4, 3, 128, 128)
        activations = self.analyzer.extract_activations(model, sample_x)
        
        self.assertIn("classifier.0", activations)
        self.assertIn("features.1.block.1.fc2", activations)
        ranges = self.analyzer.analyze_ranges(activations)
        self.assertGreater(len(ranges), 0)

    def test_08_sensitivity_recovery_comparison(self):
        """Test 8: Sensitivity recovery comparison computes Cosine, MAE, and status."""
        model = self.trainer.load_base_fp32_model()
        sample_x = torch.randn(4, 3, 128, 128)
        acts1 = self.analyzer.extract_activations(model, sample_x)
        acts2 = self.analyzer.extract_activations(model, sample_x)
        
        recovery = self.analyzer.compare_sensitivity_recovery(acts1, acts2)
        self.assertGreater(len(recovery), 0)
        self.assertAlmostEqual(recovery[0]["qat_cosine_after"], 1.0, places=4)

    def test_09_tflite_conversion_and_flatbuffer_audit(self):
        """Test 9: Model converts to genuine INT8 TFLite with >80% INT8 coverage."""
        import tensorflow as tf
        from uaqe.exporter.tflite_exporter import ONNXToTFModel
        from uaqe.exporter.flatbuffer_inspector import FlatBufferInspector
        import onnx

        onnx_path = "output/phase_c1/models/mobilenetv3_sem_9class_fp32.onnx"
        self.assertTrue(os.path.exists(onnx_path))
        onnx_model = onnx.load(onnx_path)
        tf_model = ONNXToTFModel(onnx_model)
        _ = tf_model(tf.random.normal([1, 3, 128, 128]))

        def rep_gen():
            for i in range(5):
                yield [np.random.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        tflite_bytes = converter.convert()
        self.assertGreater(len(tflite_bytes), 1000000)

        test_path = "scratch/test_tflite_unit_test.tflite"
        os.makedirs("scratch", exist_ok=True)
        with open(test_path, "wb") as f:
            f.write(tflite_bytes)

        inspector = FlatBufferInspector()
        audit = inspector.inspect(test_path)
        self.assertGreater(audit.int8_coverage_percent, 80.0)

    def test_10_tflite_interpreter_execution(self):
        """Test 10: TFLite interpreter executes inference without error or NaN/Inf."""
        import tensorflow as tf
        test_path = "scratch/test_tflite_unit_test.tflite"
        if not os.path.exists(test_path):
            return

        interpreter = tf.lite.Interpreter(
            model_path=test_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter.allocate_tensors()
        in_idx = interpreter.get_input_details()[0]["index"]
        out_idx = interpreter.get_output_details()[0]["index"]

        dummy_x = np.random.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32)
        interpreter.set_tensor(in_idx, dummy_x)
        interpreter.invoke()
        out = interpreter.get_tensor(out_idx)

        self.assertEqual(out.shape, (1, 9))
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())

if __name__ == "__main__":
    unittest.main()
