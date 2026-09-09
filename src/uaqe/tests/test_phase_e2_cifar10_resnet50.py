"""Unit and Integration Tests for UAQE Phase E.2: CIFAR-10 & ResNet-50."""

import os
import sys
import unittest
import tempfile
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uaqe.dataset.universal_dataset_loader import UniversalDatasetLoader
from uaqe.dataset.cifar10_adapter import CIFAR10PickleAdapter, CIFAR10TorchDataset
from uaqe.models.resnet50 import (
    ResNetForImageClassification,
    load_safetensors_state_dict,
    build_resnet50_cifar10
)
from uaqe.trainer.cifar10_resnet50_trainer import CIFAR10ResNet50Trainer

CIFAR10_DIR = r"D:\uaqe_datasets\cifar10\cifar-10-batches-py"
SAFETENSORS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "resnet50",
    "model.safetensors"
)


class TestPhaseE2CIFAR10ResNet50(unittest.TestCase):
    """Test suite for Phase E.2 dataset ingestion, model adaptation, and trainer."""

    def test_cifar10_format_detection_and_adapter_resolution(self):
        """Verify UniversalDatasetLoader format auto-detection and CIFAR-10 adapter resolution."""
        loader = UniversalDatasetLoader(CIFAR10_DIR)
        self.assertEqual(loader.detected_format, UniversalDatasetLoader.FORMAT_CIFAR10_PICKLE)
        self.assertIsInstance(loader.adapter, CIFAR10PickleAdapter)

        detected_nonexistent = UniversalDatasetLoader.detect_format("C:/nonexistent_dir_12345")
        self.assertEqual(detected_nonexistent, UniversalDatasetLoader.FORMAT_UNKNOWN)

    def test_cifar10_deterministic_split(self):
        """Verify deterministic 45k / 5k / 10k split and balanced class counts."""
        loader = UniversalDatasetLoader(CIFAR10_DIR)
        splits = loader.load_cifar10(train_val_split=(45000, 5000), seed=42)

        self.assertEqual(len(splits["train"]["images"]), 45000)
        self.assertEqual(len(splits["val"]["images"]), 5000)
        self.assertEqual(len(splits["test"]["images"]), 10000)

        self.assertEqual(len(splits["train"]["labels"]), 45000)
        self.assertEqual(len(splits["val"]["labels"]), 5000)
        self.assertEqual(len(splits["test"]["labels"]), 10000)

        # Verify per-class balance
        for c in range(10):
            self.assertEqual((splits["train"]["labels"] == c).sum(), 4500)
            self.assertEqual((splits["val"]["labels"] == c).sum(), 500)
            self.assertEqual((splits["test"]["labels"] == c).sum(), 1000)

        # Test seed determinism
        loader2 = UniversalDatasetLoader(CIFAR10_DIR)
        splits2 = loader2.load_cifar10(train_val_split=(45000, 5000), seed=42)
        self.assertTrue(np.array_equal(splits["train"]["labels"], splits2["train"]["labels"]))
        self.assertTrue(np.array_equal(splits["val"]["labels"], splits2["val"]["labels"]))

    def test_model_adaptive_preprocessing(self):
        """Verify preprocessing transforms 32x32 uint8 images into normalized 224x224 float32 tensors."""
        dummy_imgs = np.zeros((4, 3, 32, 32), dtype=np.uint8)
        dummy_labels = np.array([0, 1, 2, 3], dtype=np.int64)

        dataset = CIFAR10TorchDataset(
            images=dummy_imgs,
            labels=dummy_labels,
            target_size=(224, 224),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

        self.assertEqual(len(dataset), 4)
        tensor, label = dataset[0]
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(label, 0)

        # For zero input (0.0): (0.0 - mean) / std -> negative values
        expected_c0 = (0.0 - 0.485) / 0.229
        self.assertAlmostEqual(tensor[0, 0, 0].item(), expected_c0, places=4)

    def test_resnet50_checkpoint_model_building_and_head_replacement(self):
        """Verify ResNet-50 loading matching downloaded checkpoint specification."""
        model, meta = build_resnet50_cifar10(SAFETENSORS_PATH, num_classes=10, device="cpu")

        self.assertEqual(meta["architecture"], "ResNetForImageClassification")
        self.assertEqual(meta["model_type"], "resnet")
        self.assertEqual(meta["num_classes"], 10)
        self.assertEqual(meta["total_parameters"], 23528522)
        self.assertEqual(meta["classifier_parameters"], 20490)
        self.assertEqual(meta["checkpoint_config"]["depths"], [3, 4, 6, 3])
        self.assertEqual(meta["checkpoint_config"]["hidden_sizes"], [256, 512, 1024, 2048])

        # Forward pass test
        dummy = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(dummy)

        self.assertEqual(out.shape, (2, 10))

    def test_dataset_ingestion_report_generation(self):
        """Verify dataset ingestion report output structure and content."""
        loader = UniversalDatasetLoader(CIFAR10_DIR)
        loader.load_cifar10()

        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = os.path.join(tmpdir, "ingestion_report.json")
            report = loader.generate_ingestion_report(report_file)

            self.assertTrue(os.path.exists(report_file))
            self.assertEqual(report["dataset_name"], "CIFAR-10")
            self.assertEqual(report["class_count"], 10)
            self.assertEqual(report["splits"]["train_count"], 45000)
            self.assertEqual(report["splits"]["val_count"], 5000)
            self.assertEqual(report["splits"]["test_count"], 10000)
            self.assertEqual(len(report["raw_batch_manifest"]), 6)
            self.assertFalse(report["original_files_modified"])


if __name__ == "__main__":
    unittest.main()
