"""
UAQE Phase D.4 Unit Test Suite
Tests Layer Profiling, Sensitivity Normalization, Risk-Aware Strategy Rules,
Multi-Objective Scoring, Greedy Optimization & Validation Gating, Hybrid Packaging,
Exact FlatBuffer Reconstruction, and Pareto Frontier Logic.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.optimizer.layer_profiler import LayerProfiler
from src.uaqe.optimizer.adaptive_planner import AdaptivePlanner, StrategyType, MultiObjectiveScorer
from src.uaqe.compression.adaptive_packager import AdaptiveModelPackager


class TestPhaseD4AdaptiveOptimization(unittest.TestCase):
    """Unit tests for Phase D.4 Adaptive Multi-Objective Optimization."""

    def setUp(self):
        self.profiler = LayerProfiler()
        self.planner = AdaptivePlanner()
        self.packager = AdaptiveModelPackager()

    def test_entropy_computation(self):
        """Test Shannon entropy calculation on uniform vs skewed distributions."""
        uniform_arr = np.arange(16, dtype=np.int8)
        entropy_uniform = LayerProfiler.compute_entropy(uniform_arr)
        self.assertAlmostEqual(entropy_uniform, 4.0, delta=0.01)

        constant_arr = np.zeros(100, dtype=np.int8)
        entropy_const = LayerProfiler.compute_entropy(constant_arr)
        self.assertAlmostEqual(entropy_const, 0.0, delta=0.01)

    def test_layer_classification(self):
        """Test operator type and semantic category classification."""
        t_type, cat = self.profiler.classify_layer_type("features/features.1/block/block.1/Conv", [1, 3, 3, 16])
        self.assertEqual(cat, "depthwise")

        t_type, cat = self.profiler.classify_layer_type("features/features.11/block/block.2/fc2/Conv", [82944])
        self.assertEqual(cat, "SE")

        t_type, cat = self.profiler.classify_layer_type("classifier/classifier.1/Linear", [9, 1024])
        self.assertEqual(cat, "classifier")

        t_type, cat = self.profiler.classify_layer_type("features/features.10/block/block.3/block.3.0/Conv", [55296, 1, 1, 160])
        self.assertEqual(cat, "pointwise")

    def test_sensitivity_score_bounds(self):
        """Test sensitivity score calculation is strictly within [0.0, 1.0]."""
        dummy_weights = np.random.randint(-128, 127, size=(64, 64), dtype=np.int8)
        
        # High sensitivity category (depthwise)
        s_high = self.profiler.calculate_sensitivity_score("depthwise_conv", "depthwise", dummy_weights)
        self.assertGreaterEqual(s_high, 0.0)
        self.assertLessEqual(s_high, 1.0)
        self.assertGreater(s_high, 0.50)

        # Low sensitivity category (pointwise)
        s_low = self.profiler.calculate_sensitivity_score("pointwise_conv", "pointwise", dummy_weights)
        self.assertGreaterEqual(s_low, 0.0)
        self.assertLessEqual(s_low, 1.0)
        self.assertLess(s_low, s_high)

    def test_strategy_selection_and_clustering_safety(self):
        """Verify that high-sensitivity, depthwise, SE, and classifier layers are protected from clustering."""
        # 1. Depthwise Layer -> Should NEVER be clustered
        dw_profile = {
            "layer_name": "features.1.block.1.0",
            "semantic_category": "depthwise",
            "sensitivity_score": 0.85
        }
        dw_strat = self.planner.select_rule_based_strategy(dw_profile, allow_clustering=True)
        self.assertNotEqual(dw_strat["compression_strategy"], "cluster32")
        self.assertIn(dw_strat["compression_strategy"], ["dense", "sparse_rle"])

        # 2. Classifier Layer -> Must be PROTECTED (KEEP_INT8)
        cls_profile = {
            "layer_name": "classifier.1",
            "semantic_category": "classifier",
            "sensitivity_score": 0.80
        }
        cls_strat = self.planner.select_rule_based_strategy(cls_profile, allow_clustering=True)
        self.assertEqual(cls_strat["strategy_label"], StrategyType.KEEP_INT8)
        self.assertEqual(cls_strat["pruning_ratio"], 0.0)
        self.assertEqual(cls_strat["compression_strategy"], "dense")

        # 3. Low-sensitivity Pointwise Layer -> Permitted for clustering when enabled
        pw_profile = {
            "layer_name": "features.10.project_conv",
            "semantic_category": "pointwise",
            "sensitivity_score": 0.20
        }
        pw_strat = self.planner.select_rule_based_strategy(pw_profile, allow_clustering=True)
        self.assertEqual(pw_strat["compression_strategy"], "cluster32")
        self.assertEqual(pw_strat["cluster_count"], 32)

    def test_multi_objective_scorer(self):
        """Verify multi-objective scoring weights and presets."""
        scorer_acc = MultiObjectiveScorer(mode="accuracy_first")
        score_acc = scorer_acc.compute_score(accuracy_ratio=1.0, storage_reduction=0.25, latency_benefit=0.10, memory_benefit=0.05)
        # 0.70*1.0 + 0.15*0.25 + 0.10*0.10 + 0.05*0.05 = 0.70 + 0.0375 + 0.010 + 0.0025 = 0.7500
        self.assertAlmostEqual(score_acc, 0.750, places=3)

        scorer_bal = MultiObjectiveScorer(mode="balanced")
        score_bal = scorer_bal.compute_score(accuracy_ratio=1.0, storage_reduction=0.25, latency_benefit=0.10, memory_benefit=0.05)
        # 0.50*1.0 + 0.25*0.25 + 0.15*0.10 + 0.10*0.05 = 0.50 + 0.0625 + 0.015 + 0.005 = 0.5825
        self.assertAlmostEqual(score_bal, 0.5825, places=3)

    def test_deterministic_layer_pruning(self):
        """Verify magnitude pruning zeros the exact requested percentage of weights."""
        arr = np.array([-10, 20, -30, 40, -50, 60, -70, 80, -90, 100], dtype=np.int8)
        pruned = AdaptiveModelPackager.apply_layer_pruning(arr, prune_ratio=0.30)
        zero_count = int(np.sum(pruned == 0))
        self.assertEqual(zero_count, 3)
        # Smallest absolute values (| -10 |, | 20 |, | -30 |) should be zeroed
        self.assertEqual(pruned[0], 0)
        self.assertEqual(pruned[1], 0)
        self.assertEqual(pruned[2], 0)
        self.assertEqual(pruned[3], 40)

    def test_pareto_frontier_logic(self):
        """Verify 3D Pareto frontier filtering correctly eliminates dominated candidates."""
        from src.uaqe.optimizer.adaptive_experimenter import AdaptiveExperimenter
        exp = AdaptiveExperimenter.__new__(AdaptiveExperimenter)
        
        test_df = pd.DataFrame([
            {"Candidate": "Cand1", "Accuracy (%)": 98.0, "Actual Size (Bytes)": 1000, "Host Latency (ms)": 2.0},
            {"Candidate": "Cand2", "Accuracy (%)": 97.5, "Actual Size (Bytes)": 1000, "Host Latency (ms)": 2.5},  # Dominated by Cand1
            {"Candidate": "Cand3", "Accuracy (%)": 97.0, "Actual Size (Bytes)": 800,  "Host Latency (ms)": 1.5},  # Non-dominated (smaller & faster)
        ])
        pareto = exp.compute_pareto_frontier(test_df)
        pareto_cands = pareto["Candidate"].tolist()
        self.assertIn("Cand1", pareto_cands)
        self.assertIn("Cand3", pareto_cands)
        self.assertNotIn("Cand2", pareto_cands)


if __name__ == "__main__":
    unittest.main()
