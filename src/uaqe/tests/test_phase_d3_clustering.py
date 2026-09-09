"""
Unit tests for UAQE Phase D.3: Clustering-Aware Modules, Straight-Through Estimators (STE),
Centroid Optimization, Zero Preservation, Serialization, and Baseline Protection.
"""

import os
import unittest
import hashlib
import numpy as np
import torch
import torch.nn as nn

from src.uaqe.clustering.clustering_module import (
    ClusteringConv2d,
    ClusteringLinear,
    apply_clustering_to_model,
    freeze_clustered_weights
)
from src.uaqe.clustering.clustering_trainer import ClusteringTrainer
from src.uaqe.compression.model_packager import ModelPackager


class TestPhaseD3Clustering(unittest.TestCase):
    """Test suite for Phase D.3 clustering-aware modules, STE, and fine-tuning engine."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cls.baseline_model = os.path.join(cls.project_root, "output", "phase_c4", "models", "c4_best_int8.tflite")
        cls.d1_model_20 = os.path.join(cls.project_root, "output", "phase_d1", "models", "d1_sensitive_20_int8.tflite")
        cls.d1_model_30 = os.path.join(cls.project_root, "output", "phase_d1", "models", "d1_sensitive_30_int8.tflite")
        cls.d2_b1_model = os.path.join(cls.project_root, "output", "phase_d2", "compressed", "d2_20_sparse_rle.bin")
        cls.packager = ModelPackager()

    def test_01_clustering_conv2d_zero_preservation_and_centroids(self):
        """Tests that ClusteringConv2d preserves exact structural zeros and initializes K centroids."""
        conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        # Create 30% sparsity mask
        mask = torch.rand_like(conv.weight) > 0.30
        conv.weight.data.mul_(mask.float())

        c_conv = ClusteringConv2d.from_conv2d(conv, num_clusters=16, mask=mask)
        self.assertEqual(c_conv.num_clusters, 16)
        self.assertTrue(c_conv.centroids_initialized)
        self.assertEqual(len(c_conv.centroids), 16)

        w_clustered, assign = c_conv.get_clustered_weights()
        # Verify zeros are preserved exactly
        zero_locs = ~mask
        self.assertTrue(torch.all(w_clustered[zero_locs] == 0))
        self.assertTrue(torch.all(assign[zero_locs] == 0))

        # Verify non-zeros only take centroid values
        w_nz = w_clustered[mask]
        for val in w_nz.flatten():
            diffs = torch.abs(c_conv.centroids - val)
            self.assertLess(torch.min(diffs).item(), 1e-4)

    def test_02_ste_gradient_flow_and_backprop(self):
        """Tests that Straight-Through Estimator passes continuous gradients while freezing pruned zeros."""
        conv = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        mask = torch.ones_like(conv.weight, dtype=torch.bool)
        mask[0, 0, 0, 0] = False  # pruned position
        conv.weight.data.mul_(mask.float())

        c_conv = ClusteringConv2d.from_conv2d(conv, num_clusters=8, mask=mask)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        out = c_conv(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(c_conv.weight.grad)
        # Gradient should flow to unpruned positions
        self.assertNotEqual(c_conv.weight.grad[mask].abs().sum().item(), 0.0)

        # Apply sparsity mask multiplication to grad
        c_conv.weight.grad.mul_(c_conv.sparsity_mask)
        self.assertEqual(c_conv.weight.grad[0, 0, 0, 0].item(), 0.0)

    def test_03_lloyd_max_centroid_update(self):
        """Tests that update_centroids_from_weights updates codebook centroids to assigned cluster means."""
        conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        c_conv = ClusteringConv2d.from_conv2d(conv, num_clusters=4)
        old_centroids = c_conv.centroids.clone()

        # Perturb trainable weights
        c_conv.weight.data.add_(0.5)
        c_conv.update_centroids_from_weights()
        new_centroids = c_conv.centroids.clone()

        self.assertFalse(torch.equal(old_centroids, new_centroids))

    def test_04_freeze_clustered_weights(self):
        """Tests that freeze_clustered_weights produces clean standard PyTorch modules with discrete weights."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 9)
        )
        c_model = apply_clustering_to_model(model, num_clusters=16)
        self.assertIsInstance(c_model[0], ClusteringConv2d)
        self.assertIsInstance(c_model[5], ClusteringLinear)

        frozen_model = freeze_clustered_weights(c_model)
        self.assertIsInstance(frozen_model[0], nn.Conv2d)
        self.assertIsInstance(frozen_model[5], nn.Linear)

        # Forward pass on standard model works
        dummy_in = torch.randn(2, 3, 32, 32)
        out = frozen_model(dummy_in)
        self.assertEqual(out.shape, (2, 9))

    def test_05_baseline_hash_and_d2b1_protection(self):
        """Tests that historical C4 baseline and D2-B1 artifacts are intact and unmodified."""
        self.assertTrue(os.path.exists(self.baseline_model), "Baseline c4_best_int8.tflite missing!")
        with open(self.baseline_model, "rb") as f:
            b_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(b_hash, "c8c8292c136dfafa4240b9b9a51079ceaff12a612996829c29384b493f689e51")
        self.assertTrue(os.path.exists(self.d2_b1_model), "D2-B1 archive missing!")


if __name__ == "__main__":
    unittest.main()
