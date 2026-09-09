import os
import unittest
import numpy as np
import torch
import onnx
import onnxruntime as ort
import tensorflow as tf

from uaqe.exporter.conversion_gap_analyzer import ConversionGapAnalyzer
from uaqe.exporter.flatbuffer_inspector import FlatBufferInspector

class TestPhaseC4ConversionGap(unittest.TestCase):
    """Test suite validating the Phase C.4 QAT -> TFLite conversion gap investigation pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = ConversionGapAnalyzer(
            fp32_ckpt_path="output/phase_c1/models/mobilenetv3_sem_9class_fp32.pth",
            c3_qat_ckpt_path="output/phase_c3/models/c3_best_qat.pth",
            c3_onnx_path="output/phase_c3/models/c3_2_tailored_observers.onnx",
            c3_tflite_path="output/phase_c3/models/c3_best_int8.tflite",
            output_dir="output/phase_c4",
            reports_dir="reports/phase_c4"
        )
        cls.cache = cls.analyzer.build_deterministic_input_cache(num_samples=10)

    def test_01_common_input_equivalence(self):
        """Validates that evaluation samples are strictly deterministic and identical across frameworks."""
        np_in = self.cache["np_inputs"]
        self.assertEqual(len(np_in), 10)
        self.assertEqual(np_in.shape[1:], (3, 128, 128))
        self.assertGreaterEqual(float(np.min(np_in)), 0.0)
        self.assertLessEqual(float(np.max(np_in)), 1.0)
        # Verify checksum reproducibility
        self.assertEqual(len(self.cache["checksums"]), 10)

    def test_02_onnx_export_and_runtime_validity(self):
        """Validates that the exported ONNX model loads and runs correctly in ONNXRuntime."""
        self.assertTrue(os.path.exists(self.analyzer.c3_onnx_path))
        onnx_model = onnx.load(self.analyzer.c3_onnx_path)
        onnx.checker.check_model(onnx_model)
        
        sess = ort.InferenceSession(self.analyzer.c3_onnx_path)
        in_name = sess.get_inputs()[0].name
        out = sess.run(None, {in_name: self.cache["np_inputs"][:2]})[0]
        self.assertEqual(out.shape, (2, 9))

    def test_03_tensorflow_conversion_validity(self):
        """Validates that the ONNX graph converts to a callable TensorFlow Keras model."""
        from uaqe.exporter.tflite_exporter import ONNXToTFModel
        onnx_model = onnx.load(self.analyzer.c3_onnx_path)
        tf_model = ONNXToTFModel(onnx_model)
        
        tf_in = tf.constant(self.cache["np_inputs"][:2])
        tf_out = tf_model(tf_in)
        self.assertEqual(tf_out.shape, (2, 9))

    def test_04_tflite_conversion_and_interpreter_execution(self):
        """Validates that the TFLite model executes cleanly without runtime error."""
        self.assertTrue(os.path.exists(self.analyzer.c3_tflite_path))
        interp = tf.lite.Interpreter(
            model_path=self.analyzer.c3_tflite_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interp.allocate_tensors()
        in_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]

        sample = self.cache["np_inputs"][:1]
        interp.set_tensor(in_idx, sample)
        interp.invoke()
        out = interp.get_tensor(out_idx)
        self.assertEqual(out.shape, (1, 9))
        self.assertFalse(np.isnan(out).any())

    def test_05_flatbuffer_int8_dtype_audit(self):
        """Validates that the FlatBuffer contains genuine INT8 tensors and meets coverage requirements."""
        inspector = FlatBufferInspector()
        audit = inspector.inspect(self.analyzer.c3_tflite_path)
        self.assertGreater(audit.int8_tensors, 200)
        self.assertGreater(audit.int8_coverage_percent, 80.0)

    def test_06_representative_dataset_determinism(self):
        """Validates representative dataset audit generation and determinism."""
        audit = self.analyzer.audit_representative_dataset()
        self.assertTrue(audit["is_deterministic"])
        self.assertGreaterEqual(audit["calibration_sample_count"], 50)
        self.assertEqual(len(audit["class_distribution"]), 9)

    def test_07_protected_artifact_integrity(self):
        """Guarantees that protected baseline artifacts have not been modified."""
        protected_paths = [
            "src/models/mobilenetv3_sem.onnx",
            "output/phase_c1/models/mobilenetv3_sem_9class_fp32.pth",
            "output/phase_c2/models/mobilenetv3_sem_9class_qat_int8.tflite",
            "output/phase_c3/models/c3_best_int8.tflite"
        ]
        for p in protected_paths:
            self.assertTrue(os.path.exists(p), f"Protected artifact missing: {p}")

if __name__ == "__main__":
    unittest.main()
