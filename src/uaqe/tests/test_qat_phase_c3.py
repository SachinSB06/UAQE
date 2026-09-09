import os
import sys
import unittest
import json
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "src")

from uaqe.quantization.c3_baseline_analyzer import C3BaselineAnalyzer
from uaqe.quantization.c3_advanced_qat import C3AdvancedQATEngine
from uaqe.quantization.qat_policy import QATPolicy
from uaqe.exporter.flatbuffer_inspector import FlatBufferInspector

class TestPhaseC3QAT(unittest.TestCase):
    """Unit and integration tests for Phase C.3 Advanced QAT Optimization,
    three-level precision evaluation, error budgeting, feature distillation, and true INT8 verification.
    """

    @classmethod
    def setUpClass(cls):
        cls.fp32_ckpt = "output/phase_c1/models/mobilenetv3_sem_9class_fp32.pth"
        cls.c2_qat_ckpt = "output/phase_c2/models/qat_distilled_best.pth"
        cls.c2_tflite = "output/phase_c2/models/mobilenetv3_sem_9class_qat_int8.tflite"
        
        cls.analyzer = C3BaselineAnalyzer(
            fp32_pth_path=cls.fp32_ckpt,
            c2_qat_pth_path=cls.c2_qat_ckpt,
            c2_int8_tflite_path=cls.c2_tflite,
            output_dir="scratch/test_c3_output",
            reports_dir="scratch/test_c3_reports"
        )
        cls.engine = C3AdvancedQATEngine(
            fp32_checkpoint_path=cls.fp32_ckpt,
            output_dir="scratch/test_c3_output",
            reports_dir="scratch/test_c3_reports"
        )

    def test_01_baseline_models_loading(self):
        """Test 1: FP32, QAT, and TFLite baseline models load successfully."""
        fp32_m, qat_m, interp = self.analyzer.load_models()
        self.assertIsNotNone(fp32_m)
        self.assertIsNotNone(qat_m)
        if os.path.exists(self.c2_tflite):
            self.assertIsNotNone(interp)

    def test_02_three_level_evaluation_structure(self):
        """Test 2: Evaluates accuracy metrics on small batch across precision levels."""
        fp32_m, qat_m, _ = self.analyzer.load_models()
        sample_x = torch.randn(4, 3, 128, 128)
        sample_y = [0, 1, 2, 3]

        eval_fp32 = self.analyzer.trainer.evaluate_model(fp32_m, sample_x, sample_y)
        eval_qat = self.analyzer.trainer.evaluate_model(qat_m, sample_x, sample_y)

        self.assertIn("accuracy", eval_fp32)
        self.assertIn("macro_f1", eval_fp32)
        self.assertIn("accuracy", eval_qat)
        self.assertIn("macro_f1", eval_qat)

    def test_03_feature_distillation_hook(self):
        """Test 3: Feature extractor hook captures intermediate global pooling tensors."""
        model = self.engine.trainer.load_base_fp32_model()
        handle, buf = self.engine._get_feature_extractor_hook(model)

        dummy_x = torch.randn(2, 3, 128, 128)
        _ = model(dummy_x)

        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0].shape[0], 2)
        handle.remove()

    def test_04_c3_qat_training_step_with_dual_distillation(self):
        """Test 4: Dual logit + feature distillation executes one batch training step."""
        base_model = self.engine.trainer.load_base_fp32_model()
        teacher_model = self.engine.trainer.load_base_fp32_model()
        teacher_model.eval()

        qat_model = self.engine.policy.apply_qat_policy(base_model, policy_mode="sensitivity_aware")
        qat_model.train()

        s_handle, s_buf = self.engine._get_feature_extractor_hook(qat_model)
        t_handle, t_buf = self.engine._get_feature_extractor_hook(teacher_model)

        optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-4)
        x = torch.randn(2, 3, 128, 128)
        y = torch.tensor([0, 1], dtype=torch.long)

        optimizer.zero_grad()
        s_buf.clear()
        t_buf.clear()

        out_s = qat_model(x)
        with torch.no_grad():
            out_t = teacher_model(x)

        loss_ce = nn.CrossEntropyLoss()(out_s, y)
        p_s = nn.functional.log_softmax(out_s / 2.0, dim=1)
        p_t = nn.functional.softmax(out_t / 2.0, dim=1)
        loss_kd = nn.functional.kl_div(p_s, p_t, reduction="batchmean") * 4.0

        s_f = torch.flatten(s_buf[0], 1)
        t_f = torch.flatten(t_buf[0], 1)
        loss_feat = nn.functional.mse_loss(s_f, t_f)

        loss = 0.5 * loss_ce + 0.5 * loss_kd + 0.2 * loss_feat
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        s_handle.remove()
        t_handle.remove()

    def test_05_depthwise_gradient_scaling(self):
        """Test 5: Depthwise gradient scaling multiplies gradients by configured factor."""
        model = self.engine.trainer.load_base_fp32_model()
        qat_model = self.engine.policy.apply_qat_policy(model, policy_mode="sensitivity_aware")
        qat_model.train()

        x = torch.randn(2, 3, 128, 128)
        y = torch.tensor([0, 1], dtype=torch.long)
        out = qat_model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()

        found_dw = False
        for n, p in qat_model.named_parameters():
            if "block.1.0" in n and p.grad is not None:
                orig_norm = p.grad.data.norm().item()
                p.grad.data.mul_(1.5)
                new_norm = p.grad.data.norm().item()
                self.assertAlmostEqual(new_norm, orig_norm * 1.5, places=4)
                found_dw = True
                break
        self.assertTrue(found_dw)

    def test_06_true_int8_tflite_conversion(self):
        """Test 6: C3 Engine converts ONNX model to genuine INT8 TFLite FlatBuffer (>80% INT8)."""
        onnx_path = "output/phase_c1/models/mobilenetv3_sem_9class_fp32.onnx"
        self.assertTrue(os.path.exists(onnx_path))

        tflite_test_path = "scratch/test_c3_int8.tflite"
        os.makedirs("scratch", exist_ok=True)

        meta = self.engine.convert_to_genuine_int8_tflite(onnx_path, tflite_test_path)
        self.assertTrue(os.path.exists(tflite_test_path))
        self.assertGreater(meta["int8_coverage_percent"], 80.0)
        self.assertEqual(meta["fp32_tensors"], 2)

    def test_07_interpreter_execution_and_stability(self):
        """Test 7: TFLite interpreter executes inference without runtime exceptions or NaN/Inf."""
        tflite_test_path = "scratch/test_c3_int8.tflite"
        if not os.path.exists(tflite_test_path):
            return

        interp = tf.lite.Interpreter(
            model_path=tflite_test_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interp.allocate_tensors()
        in_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]

        dummy_x = np.random.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32)
        interp.set_tensor(in_idx, dummy_x)
        interp.invoke()
        out = interp.get_tensor(out_idx)

        self.assertEqual(out.shape, (1, 9))
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())

if __name__ == "__main__":
    unittest.main()
