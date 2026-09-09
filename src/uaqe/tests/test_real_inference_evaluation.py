import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uaqe.evaluation.real_dataset_adapter import RealDatasetAdapter
from uaqe.evaluation.real_inference_evaluator import RealInferenceEvaluator

class TestRealInferenceEvaluation(unittest.TestCase):
    
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        
        # Create a small dataset fixture
        self.class_dirs = ["Clean", "Bridge", "CMP"]
        self.dataset_path = os.path.join(self.tmp_dir, "dataset")
        os.makedirs(self.dataset_path)
        
        self.class_mapping = {"Clean": 0, "Bridge": 1, "CMP": 2}
        
        # Save a valid test image in each directory
        self.image_files = []
        for name in self.class_dirs:
            cls_dir = os.path.join(self.dataset_path, name)
            os.makedirs(cls_dir)
            
            # Create a simple 8x8 grayscale image
            img_path = os.path.join(cls_dir, f"{name.lower()}_01.png")
            img = Image.new("L", (8, 8), color=128)
            img.save(img_path)
            self.image_files.append(img_path)

        # Create a sample evaluation config
        self.eval_config = {
            "preprocessing": "rgb_0_1",
            "class_mapping": self.class_mapping,
            "input_layout": "NCHW",
            "expected_num_classes": 10,
            "evaluation_thresholds": {
                "min_cosine_similarity": 0.95,
                "max_mae": 0.05,
                "max_rmse": 0.1,
                "min_prediction_agreement": 0.90
            }
        }
        self.config_path = os.path.join(self.tmp_dir, "config.json")
        with open(self.config_path, "w") as f:
            json.dump(self.eval_config, f)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_tflite_dataset_loading(self) -> None:
        """Test dataset discovery and loading."""
        adapter = RealDatasetAdapter(
            dataset_path=self.dataset_path,
            class_mapping=self.class_mapping,
            preprocessing_mode="rgb_0_1",
            input_shape=(1, 3, 128, 128)
        )
        
        self.assertEqual(len(adapter), 3)
        self.assertEqual(adapter.layout, "NCHW")
        self.assertEqual(adapter.channels, 3)
        self.assertEqual(adapter.height, 128)
        self.assertEqual(adapter.width, 128)
        self.assertEqual(adapter.class_counts["Clean"], 1)

    def test_tflite_preprocessing_consistency(self) -> None:
        """Test that preprocessing correctly resizes and transforms to NCHW."""
        adapter = RealDatasetAdapter(
            dataset_path=self.dataset_path,
            class_mapping=self.class_mapping,
            preprocessing_mode="rgb_0_1",
            input_shape=(1, 3, 32, 32)
        )
        
        sample = adapter[0]
        tensor = sample["tensor"]
        self.assertEqual(tensor.shape, (3, 32, 32))
        self.assertTrue((tensor >= 0.0).all() and (tensor <= 1.0).all())

    def test_tflite_label_mapping(self) -> None:
        """Test that labels match correct folder indices."""
        adapter = RealDatasetAdapter(
            dataset_path=self.dataset_path,
            class_mapping=self.class_mapping,
            preprocessing_mode="rgb_0_1",
            input_shape=(1, 3, 128, 128)
        )
        
        # Test mapping order sorting
        self.assertEqual(adapter.samples[0][2], "Bridge")
        self.assertEqual(adapter.samples[0][1], 1)
        self.assertEqual(adapter.samples[1][2], "CMP")
        self.assertEqual(adapter.samples[1][1], 2)
        self.assertEqual(adapter.samples[2][2], "Clean")
        self.assertEqual(adapter.samples[2][1], 0)

    def test_tflite_output_shape(self) -> None:
        """Test model-agnostic layout layout parsing."""
        # NCHW shape
        adapter = RealDatasetAdapter(
            dataset_path=self.dataset_path,
            class_mapping=self.class_mapping,
            preprocessing_mode="rgb_0_1",
            input_shape=(1, 3, 64, 64)
        )
        self.assertEqual(adapter.layout, "NCHW")
        
        # NHWC shape
        adapter_nhwc = RealDatasetAdapter(
            dataset_path=self.dataset_path,
            class_mapping=self.class_mapping,
            preprocessing_mode="rgb_0_1",
            input_shape=(1, 64, 64, 3)
        )
        self.assertEqual(adapter_nhwc.layout, "NHWC")

    def test_tflite_prediction_generation(self) -> None:
        """Test classification metrics calculations (precision, recall, F1, confusion matrix)."""
        evaluator = RealInferenceEvaluator.__new__(RealInferenceEvaluator)
        evaluator.config = self.eval_config
        
        y_true = [0, 0, 1, 1, 2]
        y_pred = [0, 1, 1, 1, 2]
        class_names = ["Clean", "Bridge", "CMP"]
        
        report = evaluator._compute_class_metrics(y_true, y_pred, class_names, num_classes=3)
        
        # Verify CM
        cm = np.array(report["confusion_matrix"])
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[0, 1], 1) # actual Clean predicted as Bridge
        
        # Verify Clean metrics
        self.assertEqual(report["per_class"]["Clean"]["precision"], 1.0)
        self.assertEqual(report["per_class"]["Clean"]["recall"], 0.5)
        self.assertEqual(report["per_class"]["Clean"]["f1"], 2/3)

    def test_tflite_numerical_metrics(self) -> None:
        """Test logits comparison similarity calculations."""
        evaluator = RealInferenceEvaluator.__new__(RealInferenceEvaluator)
        
        a = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])
        b = np.array([[1.0, 2.1, 0.0], [3.0, 3.9, 5.0]])
        pred_a = [1, 2]
        pred_b = [1, 2]
        
        sim = evaluator._compute_similarity(a, b, pred_a, pred_b)
        self.assertGreater(sim["cosine_similarity"], 0.99)
        self.assertAlmostEqual(sim["mean_absolute_error"], 0.2 / 6)
        self.assertEqual(sim["prediction_agreement"], 1.0)

    # Negative Tests
    def test_missing_dataset(self) -> None:
        """Assert FileNotFoundError raised for invalid dataset directory."""
        with self.assertRaises(FileNotFoundError):
            RealDatasetAdapter(
                dataset_path=os.path.join(self.tmp_dir, "nonexistent"),
                class_mapping=self.class_mapping
            )

    def test_unknown_class(self) -> None:
        """Assert folder skipped when it is not present in class mapping."""
        # Add a folder that is not in class mapping
        extra_dir = os.path.join(self.dataset_path, "ExtraFolder")
        os.makedirs(extra_dir)
        img = Image.new("L", (8, 8))
        img.save(os.path.join(extra_dir, "img.png"))
        
        adapter = RealDatasetAdapter(
            dataset_path=self.dataset_path,
            class_mapping=self.class_mapping
        )
        # ExtraFolder is skipped, sample count remains 3
        self.assertEqual(len(adapter), 3)
        self.assertEqual(len(adapter.skipped_files), 1)

    def test_corrupt_image(self) -> None:
        """Assert corrupt files are caught and tracked instead of crashing."""
        corrupt_file = os.path.join(self.dataset_path, "Clean", "corrupt.png")
        with open(corrupt_file, "w") as f:
            f.write("not an image file")
            
        adapter = RealDatasetAdapter(
            dataset_path=self.dataset_path,
            class_mapping=self.class_mapping
        )
        self.assertEqual(len(adapter), 3) # ignores the corrupt file
        self.assertEqual(len(adapter.corrupt_files), 1)

    def test_missing_model(self) -> None:
        """Assert FileNotFoundError raised if model path is invalid."""
        adapter = MagicMock()
        with self.assertRaises(FileNotFoundError):
            RealInferenceEvaluator(
                fp32_model_path=os.path.join(self.tmp_dir, "nonexistent_fp32.onnx"),
                onnx_model_path=os.path.join(self.tmp_dir, "nonexistent_opt.onnx"),
                tflite_model_path=os.path.join(self.tmp_dir, "nonexistent.tflite"),
                dataset=adapter,
                config=self.eval_config
            )

    def test_invalid_preprocessing(self) -> None:
        """Assert ValueError raised for invalid preprocessing modes."""
        adapter = RealDatasetAdapter(
            dataset_path=self.dataset_path,
            class_mapping=self.class_mapping,
            preprocessing_mode="rgb_0_1"
        )
        # Manually alter preprocessing to invalid string
        adapter.preprocessing_mode = "invalid_mode"
        with self.assertRaises(ValueError):
            _ = adapter[0]

if __name__ == "__main__":
    unittest.main()
