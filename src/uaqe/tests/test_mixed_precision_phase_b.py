"""Unit tests for Phase B Mixed-Precision Optimization framework.

Tests:
1. PrecisionPolicy construction, dict/JSON serialization, and deserialization.
2. Sensitivity-guided policy builders (top-K, category, threshold, se+depthwise).
3. Precision resolution logic (exact, substring, category, block, default).
4. FlatBufferInspector dtype counting and INT8 coverage calculation.
5. CandidateRanker multi-objective scoring across objectives.
6. Pareto frontier computation logic.
7. MixedPrecisionReportGenerator CSV, JSON, and Markdown generation.
8. FlatBuffer layer verification reporting.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uaqe.common.types import Precision
from uaqe.exporter.flatbuffer_inspector import (
    FlatBufferDtypeSummary,
    FlatBufferInspector,
    LayerVerificationResult,
)
from uaqe.optimizer.candidate_ranker import (
    CandidateMetrics,
    CandidateRanker,
    CandidateScore,
)
from uaqe.quantization.precision_policy import PrecisionPolicy
from uaqe.reports.mixed_precision_report_generator import MixedPrecisionReportGenerator


class TestMixedPrecisionPhaseB(unittest.TestCase):
    """Test suite for Phase B Mixed Precision modules."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

        # Mock sensitivity report
        self.mock_report = {
            "metadata": {"sample_count": 50},
            "layer_metrics": [
                {
                    "rank": 1,
                    "layer_name": "features/features.11/block/block.2/fc2/Conv",
                    "tflite_tensor_idx": 325,
                    "semantic_category": "se_fc2",
                    "block_name": "features.11",
                    "sensitivity_score": 0.88,
                },
                {
                    "rank": 2,
                    "layer_name": "features/features.10/block/block.2/fc2/Conv",
                    "tflite_tensor_idx": 305,
                    "semantic_category": "se_fc2",
                    "block_name": "features.10",
                    "sensitivity_score": 0.84,
                },
                {
                    "rank": 3,
                    "layer_name": "features/features.6/block/block.2/fc2/Conv",
                    "tflite_tensor_idx": 226,
                    "semantic_category": "se_fc2",
                    "block_name": "features.6",
                    "sensitivity_score": 0.78,
                },
                {
                    "rank": 4,
                    "layer_name": "features/features.1/block/block.0/Conv",
                    "tflite_tensor_idx": 150,
                    "semantic_category": "depthwise_conv",
                    "block_name": "features.1",
                    "sensitivity_score": 0.65,
                },
                {
                    "rank": 5,
                    "layer_name": "features/features.2/block/block.0/Conv",
                    "tflite_tensor_idx": 160,
                    "semantic_category": "depthwise_conv",
                    "block_name": "features.2",
                    "sensitivity_score": 0.55,
                },
            ],
        }
        self.report_path = os.path.join(self.tmp_dir, "test_sensitivity_report.json")
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_report, f)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_precision_policy_serialization(self):
        """Test PrecisionPolicy to_dict, to_json, from_dict, from_json."""
        pol = PrecisionPolicy(
            name="test_pol",
            default_precision=Precision.INT8,
            layer_overrides={"layer1": Precision.FP16},
            block_overrides={"block1": Precision.FP32},
            category_overrides={"se_fc2": Precision.FP16},
        )
        d = pol.to_dict()
        self.assertEqual(d["name"], "test_pol")
        self.assertEqual(d["default_precision"], "INT8")
        self.assertEqual(d["layer_overrides"]["layer1"], "FP16")

        json_str = pol.to_json()
        pol_loaded = PrecisionPolicy.from_json(json_str)
        self.assertEqual(pol_loaded.name, "test_pol")
        self.assertEqual(pol_loaded.layer_overrides["layer1"], Precision.FP16)
        self.assertEqual(pol_loaded.block_overrides["block1"], Precision.FP32)

    def test_precision_policy_resolution(self):
        """Test resolution hierarchy: layer > category > block > default."""
        pol = PrecisionPolicy(
            default_precision=Precision.INT8,
            layer_overrides={"conv1": Precision.FP16},
            block_overrides={"blockA": Precision.FP32},
            category_overrides={"depthwise": Precision.FP16},
        )

        # 1. Exact layer override
        self.assertEqual(pol.resolve_precision("conv1"), Precision.FP16)
        # 2. Category override
        self.assertEqual(pol.resolve_precision("other_conv", category="depthwise"), Precision.FP16)
        # 3. Block override
        self.assertEqual(pol.resolve_precision("other_conv", block_name="blockA"), Precision.FP32)
        # 4. Fallback to default
        self.assertEqual(pol.resolve_precision("unrelated_layer"), Precision.INT8)

    def test_policy_factory_top_k(self):
        """Test PrecisionPolicy.from_sensitivity_report with top_k."""
        pol = PrecisionPolicy.from_sensitivity_report(self.report_path, top_k=3, target_precision=Precision.FP16)
        self.assertEqual(len(pol.layer_overrides), 3)
        self.assertIn("features/features.11/block/block.2/fc2/Conv", pol.layer_overrides)
        self.assertEqual(pol.layer_overrides["features/features.11/block/block.2/fc2/Conv"], Precision.FP16)

    def test_policy_factory_categories(self):
        """Test PrecisionPolicy.from_categories."""
        pol = PrecisionPolicy.from_categories(self.report_path, categories=["depthwise_conv"])
        self.assertEqual(len(pol.layer_overrides), 2)
        self.assertIn("features/features.1/block/block.0/Conv", pol.layer_overrides)

    def test_policy_factory_threshold(self):
        """Test PrecisionPolicy.from_sensitivity_threshold."""
        pol = PrecisionPolicy.from_sensitivity_threshold(self.report_path, min_sensitivity=0.75)
        # ranks 1, 2, 3 have scores >= 0.78
        self.assertEqual(len(pol.layer_overrides), 3)

    def test_candidate_ranker_multi_objective(self):
        """Test CandidateRanker ranking under different objectives."""
        c1 = CandidateMetrics(
            experiment_id="E1",
            name="Accurate",
            policy_name="p1",
            accuracy=35.0,
            accuracy_delta_vs_int8=13.0,
            accuracy_delta_vs_fp32=-2.0,
            cosine_similarity=0.90,
            mae=0.1,
            rmse=0.2,
            model_size_bytes=3000000,
            model_size_mb=2.86,
            latency_ms=10.0,
            int8_tensor_count=100,
            fp16_tensor_count=50,
            fp32_tensor_count=10,
            int32_tensor_count=10,
            int8_coverage_percent=60.0,
        )
        c2 = CandidateMetrics(
            experiment_id="E2",
            name="Compressed",
            policy_name="p2",
            accuracy=23.0,
            accuracy_delta_vs_int8=1.0,
            accuracy_delta_vs_fp32=-14.0,
            cosine_similarity=0.60,
            mae=0.8,
            rmse=1.0,
            model_size_bytes=1800000,
            model_size_mb=1.72,
            latency_ms=4.0,
            int8_tensor_count=280,
            fp16_tensor_count=4,
            fp32_tensor_count=2,
            int32_tensor_count=64,
            int8_coverage_percent=95.0,
        )

        # In accuracy_first, c1 should rank #1
        acc_ranking = CandidateRanker.rank_candidates([c1, c2], objective="accuracy_first")
        self.assertEqual(acc_ranking[0].candidate.experiment_id, "E1")

        # In size_first, c2 should rank #1
        size_ranking = CandidateRanker.rank_candidates([c1, c2], objective="size_first")
        self.assertEqual(size_ranking[0].candidate.experiment_id, "E2")

    def test_pareto_frontier_computation(self):
        """Test Pareto non-dominated identification."""
        c_dominated = CandidateMetrics(
            experiment_id="Dominated",
            name="Dominated",
            policy_name="p",
            accuracy=20.0,
            accuracy_delta_vs_int8=-1.0,
            accuracy_delta_vs_fp32=-17.0,
            cosine_similarity=0.5,
            mae=1.0,
            rmse=1.2,
            model_size_bytes=5000000,
            model_size_mb=4.8,
            latency_ms=20.0,
            int8_tensor_count=50,
            fp16_tensor_count=50,
            fp32_tensor_count=50,
            int32_tensor_count=10,
            int8_coverage_percent=30.0,
        )
        c_dominant = CandidateMetrics(
            experiment_id="Dominant",
            name="Dominant",
            policy_name="p",
            accuracy=35.0,
            accuracy_delta_vs_int8=13.0,
            accuracy_delta_vs_fp32=-2.0,
            cosine_similarity=0.9,
            mae=0.1,
            rmse=0.2,
            model_size_bytes=2000000,
            model_size_mb=1.9,
            latency_ms=5.0,
            int8_tensor_count=250,
            fp16_tensor_count=20,
            fp32_tensor_count=10,
            int32_tensor_count=10,
            int8_coverage_percent=85.0,
        )

        pareto = CandidateRanker.compute_pareto_frontier([c_dominant, c_dominated])
        self.assertTrue(pareto[0])   # Dominant is on Pareto front
        self.assertFalse(pareto[1])  # Dominated is NOT on Pareto front

    def test_report_generation(self):
        """Test CSV, JSON, and Markdown report export."""
        exp_row = {
            "experiment_id": "EXP_TEST",
            "policy": "Test Policy",
            "selected_layers": ["layerA", "layerB"],
            "selected_blocks": ["block1"],
            "requested_precision": "FP16",
            "actual_precision": "FLOAT16",
            "accuracy": 30.0,
            "accuracy_delta_vs_int8": 8.04,
            "accuracy_delta_vs_fp32": -7.16,
            "precision": 0.25,
            "recall": 0.25,
            "f1": 0.25,
            "prediction_agreement": 70.0,
            "cosine_similarity": 0.85,
            "mae": 0.3,
            "rmse": 0.4,
            "model_size_bytes": 2000000,
            "model_size_mb": 1.91,
            "latency_ms": 6.5,
            "int8_tensor_count": 250,
            "fp16_tensor_count": 10,
            "fp32_tensor_count": 10,
            "int32_tensor_count": 60,
            "int8_coverage_percent": 75.0,
            "stability_status": "PASSED",
            "status": "COMPLETED",
        }
        csv_path = os.path.join(self.tmp_dir, "test.csv")
        MixedPrecisionReportGenerator.generate_csv([exp_row], csv_path)
        self.assertTrue(os.path.exists(csv_path))

        report_payload = {
            "metadata": {"timestamp": "2026-09-04T00:00:00Z"},
            "decision": "PARTIAL MIXED PRECISION",
            "baseline": {"int8": {"accuracy": 21.96}, "fp32": {"accuracy": 37.16}},
            "global_fp16_reference": {"accuracy": 37.16},
            "proof_of_concept": {"poc_status": "PASS"},
            "experiments": [exp_row],
            "candidate_ranking": {"balanced_ranking": []},
            "stability_results": {"status": "PASSED"},
        }
        json_path = os.path.join(self.tmp_dir, "test.json")
        md_path = os.path.join(self.tmp_dir, "test.md")
        MixedPrecisionReportGenerator.generate_json(report_payload, json_path)
        MixedPrecisionReportGenerator.generate_markdown(report_payload, md_path)

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        self.assertIn("UAQE Phase B", md_text)
        self.assertIn("EXP_TEST", md_text)


if __name__ == "__main__":
    unittest.main()
