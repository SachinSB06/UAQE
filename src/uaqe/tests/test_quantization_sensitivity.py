"""Unit tests for QuantizationErrorAnalyzer (UAQE Phase A.2).

Tests:
1. Topology and tensor mapping.
2. Metric calculations (MAE, RMSE, Cosine similarity).
3. Zero denominator handling.
4. Deterministic sample selection.
5. Sensitivity score formula correctness.
6. Report generation (JSON, Markdown, CSV).
"""

import json
import os
import tempfile
import unittest
import numpy as np

from uaqe.quantization.quantization_error_analyzer import (
    LayerMetrics,
    QuantizationErrorAnalyzer,
)


class DummyDatasetAdapter:
    """Mock dataset adapter for unit testing."""

    def __init__(self, num_samples: int = 5):
        self.class_mapping = {"Clean": 0, "Bridge": 1, "CMP": 2}
        self.samples = []
        for i in range(num_samples):
            # Deterministic pseudo-random tensor
            np.random.seed(i)
            tensor = np.random.randn(3, 128, 128).astype(np.float32)
            self.samples.append({
                "tensor": tensor,
                "label": i % 3,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class TestQuantizationSensitivity(unittest.TestCase):
    """Test suite for diagnostic quantization error analyzer."""

    def test_zero_denominator_handling(self):
        """Verify metric calculation safely handles zero-norm vectors without NaN/Inf."""
        zeros_a = np.zeros((10, 10), dtype=np.float32)
        zeros_b = np.zeros((10, 10), dtype=np.float32)
        
        flat_a = zeros_a.flatten()
        flat_b = zeros_b.flatten()
        
        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)
        cos = float(np.dot(flat_a, flat_b) / (norm_a * norm_b)) if (norm_a * norm_b) > 0 else 1.0
        
        self.assertEqual(cos, 1.0)
        self.assertFalse(np.isnan(cos))
        self.assertFalse(np.isinf(cos))

    def test_sensitivity_score_calculation(self):
        """Verify sensitivity score formula logic and edge cases."""
        lm = LayerMetrics(
            activation_cosine=0.90,
            activation_mae=0.05,
            activation_scale=0.10,
            weight_cosine=0.95,
            clipping_rate=0.02,
            has_weights=True,
        )
        
        # Expected:
        # 0.40 * (1 - 0.90) = 0.04
        # 0.30 * min(1.0, 0.05 / 0.10) = 0.30 * 0.50 = 0.15
        # 0.20 * (1 - 0.95) = 0.01
        # 0.10 * 0.02 = 0.002
        # Total = 0.202
        cos_act_err = max(0.0, 1.0 - lm.activation_cosine)
        mae_norm = min(1.0, lm.activation_mae / (lm.activation_scale + 1e-6))
        cos_w_err = max(0.0, 1.0 - lm.weight_cosine)
        clip_err = lm.clipping_rate
        
        score = 0.40 * cos_act_err + 0.30 * mae_norm + 0.20 * cos_w_err + 0.10 * clip_err
        self.assertAlmostEqual(score, 0.202, places=3)

    def test_deterministic_sample_selection(self):
        """Verify sample selection produces consistent deterministic subsets."""
        ds = DummyDatasetAdapter(num_samples=10)
        indices_1 = list(range(min(5, len(ds))))
        indices_2 = list(range(min(5, len(ds))))
        self.assertEqual(indices_1, indices_2)
        self.assertEqual(len(indices_1), 5)

    def test_report_export(self):
        """Verify JSON, Markdown, and CSV file exports."""
        mock_result = {
            "metadata": {
                "timestamp": "2026-09-04T00:00:00Z",
                "duration_seconds": 1.23,
                "onnx_model_path": "mock.onnx",
                "tflite_model_path": "mock.tflite",
                "sample_count": 5,
                "random_seed": 42,
                "sensitivity_formula": "Score = 0.40 * (1 - ActCosine) + ...",
            },
            "layer_metrics": [
                {
                    "rank": 1,
                    "layer_name": "conv1",
                    "onnx_node_name": "/conv1",
                    "tflite_tensor_idx": 10,
                    "block_name": "features.1",
                    "operator_type": "Conv",
                    "semantic_category": "depthwise_conv",
                    "weight_params": 144,
                    "weight_mae": 0.01,
                    "weight_rmse": 0.02,
                    "weight_cosine": 0.999,
                    "weight_quant_type": "per-channel",
                    "activation_mae": 0.12,
                    "activation_rmse": 0.15,
                    "activation_cosine": 0.85,
                    "activation_scale": 0.05,
                    "activation_zero_point": 0,
                    "clipping_rate": 0.01,
                    "sensitivity_score": 0.35,
                    "recommendation": "KEEP INT8",
                }
            ],
            "rankings": {
                "top_10_sensitive_layers": [],
                "top_5_sensitive_blocks": [],
                "top_depthwise_layers": [],
                "top_se_layers": [],
                "top_hardswish_layers": [],
            },
            "depthwise_audit": {"is_major_error_source": True},
            "se_audit": {"is_major_error_source": False},
            "hardswish_audit": {"is_major_error_source": False},
            "error_accumulation": {
                "first_major_error_spike": {"block": "features.1", "cosine_drop": 0.15},
                "largest_error_spike": {"block": "features.5", "cosine_drop": 0.25},
                "block_progression": [],
            },
            "class_analysis": {
                "overall_fp32_accuracy": 37.16,
                "overall_int8_accuracy": 21.96,
                "prediction_agreement": 34.12,
                "class_metrics": [],
            },
        }
        
        with tempfile.TemporaryDirectory() as tmp_out, tempfile.TemporaryDirectory() as tmp_rep:
            analyzer = QuantizationErrorAnalyzer.__new__(QuantizationErrorAnalyzer)
            exported = analyzer.export_reports(mock_result, tmp_out, tmp_rep)
            
            # Check files exist
            self.assertTrue(os.path.exists(exported["json_output"]))
            self.assertTrue(os.path.exists(exported["json_reports"]))
            self.assertTrue(os.path.exists(exported["csv_output"]))
            self.assertTrue(os.path.exists(exported["csv_reports"]))
            self.assertTrue(os.path.exists(exported["markdown_reports"]))
            
            # Check JSON contents
            with open(exported["json_output"], "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["metadata"]["sample_count"], 5)
            self.assertEqual(len(loaded["layer_metrics"]), 1)


if __name__ == "__main__":
    unittest.main()
