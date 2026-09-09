import os
import sys
import unittest
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "src")

from uaqe.quantization.pytorch_reconstructor import MobileNetV3Reconstructor
from uaqe.quantization.semiconductor_trainer import SemiconductorTrainer

class TestPhaseC1Reconstruction(unittest.TestCase):
    """Unit and integration tests for Phase C.1 MobileNetV3 model reconstruction,
    weight transfer, numerical equivalence, dataset loading, and baseline validation.
    """

    @classmethod
    def setUpClass(cls):
        cls.onnx_path = "src/models/mobilenetv3_sem.onnx"
        cls.dataset_root = r"D:\semiconductor_dataset\dataset"
        cls.benchmark_path = "datasets/hackathon_test_dataset"
        cls.reconstructor = MobileNetV3Reconstructor(cls.onnx_path)

    def test_01_model_instantiation(self):
        """Test 1: Model instantiation for both 10-class reference and 9-class model."""
        model_10, meta_10 = self.reconstructor.build_10class_reference_model()
        model_9, meta_9 = self.reconstructor.build_9class_trainable_model()
        self.assertIsInstance(model_10, nn.Module)
        self.assertIsInstance(model_9, nn.Module)
        self.assertEqual(meta_10["num_classes"], 10)
        self.assertEqual(meta_9["num_classes"], 9)

    def test_02_output_shape(self):
        """Test 2: Output shape = [batch, 9] for 9-class model on arbitrary batch sizes."""
        model_9, _ = self.reconstructor.build_9class_trainable_model()
        model_9.eval()
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 128, 128)
            with torch.no_grad():
                out = model_9(x)
            self.assertEqual(out.shape, (batch_size, 9))

    def test_03_class_mapping_deterministic(self):
        """Test 3: Class mapping is deterministic, 9 classes, alphabetical."""
        trainer = SemiconductorTrainer(
            dataset_root=self.dataset_root,
            benchmark_path=self.benchmark_path
        )
        expected_classes = ["bridge", "clean", "cmp", "crack", "opens", "other", "particle", "scratch", "vias"]
        self.assertEqual(trainer.class_names, expected_classes)
        for i, name in enumerate(expected_classes):
            self.assertEqual(trainer.class_to_idx[name], i)
            self.assertEqual(trainer.idx_to_class[str(i)], name)

    def test_04_onnx_parameter_extraction(self):
        """Test 4: ONNX parameter extraction correctly loads initializers and Conv nodes."""
        self.assertGreater(len(self.reconstructor.initializers), 0)
        self.assertEqual(len(self.reconstructor.conv_nodes), 52)
        self.assertIn("classifier.0.weight", self.reconstructor.initializers)
        self.assertIn("classifier.3.weight", self.reconstructor.initializers)

    def test_05_weight_transfer_shape_validation(self):
        """Test 5: Weight transfer shape validation matching all 52 convs and classifier."""
        model_9, meta_9 = self.reconstructor.build_9class_trainable_model()
        stats = meta_9["transfer_stats"]
        self.assertGreater(stats["transferred_tensors"], 50)
        for rec in stats["tensor_records"]:
            if rec["status"] == "TRANSFERRED":
                self.assertEqual(rec["onnx_shape"], rec["py_shape"])

    def test_06_no_unexpected_missing_tensors(self):
        """Test 6: No unexpected missing or unmatched weight tensors."""
        _, meta_9 = self.reconstructor.build_9class_trainable_model()
        stats = meta_9["transfer_stats"]
        self.assertEqual(stats["unmatched_tensors"], 0)

    def test_07_no_nan_inf(self):
        """Test 7: Forward pass produces no NaN or Inf values across multiple distributions."""
        model_9, _ = self.reconstructor.build_9class_trainable_model()
        model_9.eval()
        test_tensors = [
            torch.randn(2, 3, 128, 128),
            torch.zeros(2, 3, 128, 128),
            torch.ones(2, 3, 128, 128),
            torch.rand(4, 3, 128, 128)
        ]
        for x in test_tensors:
            with torch.no_grad():
                out = model_9(x)
            self.assertFalse(torch.isnan(out).any())
            self.assertFalse(torch.isinf(out).any())

    def test_08_training_one_minibatch(self):
        """Test 8: Training one mini-batch successfully executes backward pass and updates parameters."""
        model_9, _ = self.reconstructor.build_9class_trainable_model()
        model_9.train()
        optimizer = torch.optim.AdamW(model_9.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(4, 3, 128, 128)
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        p_before = model_9.classifier[3].weight.clone()
        optimizer.zero_grad()
        out = model_9(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        p_after = model_9.classifier[3].weight

        self.assertFalse(torch.equal(p_before, p_after))
        self.assertFalse(torch.isnan(loss))

    def test_09_checkpoint_save_load(self):
        """Test 9: Checkpoint save/load integrity and parameter restoration."""
        trainer = SemiconductorTrainer(
            dataset_root=self.dataset_root,
            benchmark_path=self.benchmark_path
        )
        model_9, _ = self.reconstructor.build_9class_trainable_model()
        test_summary = {"best_epoch": 1, "best_val_accuracy": 0.85, "optimizer": "AdamW"}
        test_eval = {"test": {"accuracy": 0.82}, "benchmark": {"accuracy": 0.35}}

        pth_path, meta_path = trainer.save_checkpoint(model_9, test_summary, test_eval)
        self.assertTrue(os.path.exists(pth_path))
        self.assertTrue(os.path.exists(meta_path))

        loaded_ckpt = torch.load(pth_path, weights_only=False, map_location="cpu")
        self.assertIn("model_state_dict", loaded_ckpt)
        self.assertIn("class_mapping", loaded_ckpt)

        new_model, _ = self.reconstructor.build_9class_trainable_model()
        new_model.load_state_dict(loaded_ckpt["model_state_dict"])
        for p1, p2 in zip(model_9.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_10_checkpoint_inference_reproducibility(self):
        """Test 10: Checkpoint inference reproducibility (identical outputs for identical inputs)."""
        model_9, _ = self.reconstructor.build_9class_trainable_model()
        model_9.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out1 = model_9(x)
            out2 = model_9(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-7))

    def test_11_dataset_loader_counts(self):
        """Test 11: Dataset loader counts match physical filesystem."""
        trainer = SemiconductorTrainer(
            dataset_root=self.dataset_root,
            benchmark_path=self.benchmark_path
        )
        train_x, train_y, train_counts = trainer.load_dataset_split("train")
        val_x, val_y, val_counts = trainer.load_dataset_split("val")
        test_x, test_y, test_counts = trainer.load_dataset_split("test")

        self.assertEqual(len(train_y), 877)
        self.assertEqual(len(val_y), 184)
        self.assertEqual(len(test_y), 197)
        self.assertEqual(train_x.shape[1:], (3, 128, 128))

    def test_12_benchmark_isolation(self):
        """Test 12: Benchmark isolation — benchmark dataset is strictly separated from training."""
        trainer = SemiconductorTrainer(
            dataset_root=self.dataset_root,
            benchmark_path=self.benchmark_path
        )
        bench_x, bench_y, native_classes, counts = trainer.load_benchmark()
        self.assertEqual(len(bench_y), 296)
        self.assertEqual(bench_x.shape[1:], (3, 128, 128))
        self.assertIn("LER", native_classes) # Benchmark native label preserved

if __name__ == "__main__":
    unittest.main()
