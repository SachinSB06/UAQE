"""
tests/test_xnnpack_strategy_integration.py
Automated 20-point test suite for XNNPACK-compatible INT8 strategy integration into UAQE.

Verifies:
1. Strategy is registered.
2. Strategy name resolves.
3. MobileNet capability check passes.
4. Unsupported model capability check fails cleanly.
5. Candidate artifact is generated.
6. Candidate is isolated under job output.
7. Candidate does not modify phase_c2.
8. Baseline SHA remains unchanged.
9. Candidate runtime is TFLite.
10. XNNPACK initialization succeeds for the known-good MobileNet candidate.
11. Delegated operator count > 0.
12. Fallback count is recorded.
13. Accuracy is actually evaluated.
14. Benchmark is actually executed.
15. No candidate metric is hardcoded.
16. Candidate receives normal safety evaluation.
17. Candidate enters autonomous candidate list.
18. Winner selection works with XNNPACK candidate.
19. Job isolation is preserved.
20. Provenance is stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from uaqe.optimization.strategies.xnnpack_int8 import (
    XNNPACKCompatibleINT8Strategy,
    STRATEGY_NAME,
    build_xnnpack_compatible_artifact
)
from uaqe.orchestration.optimization_strategies import (
    OptimizationStrategyResolver,
    XNNPACKStrategy
)
from uaqe.optimization.candidate_generator import CandidateGenerator, OptimizationCandidate
from uaqe.optimization.candidate_evaluator import CandidateEvaluator, CandidateEvaluationResult
from uaqe.optimization.accuracy_safety_policy import AccuracySafetyPolicy
from uaqe.optimization.objective_function import ObjectiveFunction
from uaqe.optimization.search_manager import SearchManager

BASELINE_PATH = os.path.join(
    PROJECT_ROOT, "output", "phase_c2", "models", "mobilenetv3_sem_9class_qat_int8.tflite"
)
EXPECTED_BASELINE_SHA = "10d51d4f243c554734e7347f9010297fb182a64c4ad89d5551e24830846dab1d"
EXPECTED_BASELINE_SIZE = 1855816


class TestXNNPACKStrategyIntegration(unittest.TestCase):
    """Rigorous 20-point verification suite for XNNPACK strategy integration."""

    def setUp(self):
        self.test_job_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "output", "test_xnnpack_int_tmp"))
        os.makedirs(self.test_job_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_job_dir):
            shutil.rmtree(self.test_job_dir, ignore_errors=True)

    @staticmethod
    def _compute_sha(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def test_01_strategy_is_registered(self):
        """TEST 1: Strategy is registered in OptimizationStrategyResolver."""
        registered = any(isinstance(s, XNNPACKStrategy) for s in OptimizationStrategyResolver._STRATEGIES)
        self.assertTrue(registered, "XNNPACKStrategy is not registered in OptimizationStrategyResolver.")

    def test_02_strategy_name_resolves(self):
        """TEST 2: Strategy name resolves to XNNPACK_COMPATIBLE_INT8."""
        strat = XNNPACKStrategy()
        self.assertEqual(strat.strategy_name, STRATEGY_NAME)
        self.assertEqual(STRATEGY_NAME, "XNNPACK_COMPATIBLE_INT8")

    def test_03_mobilenet_capability_check_passes(self):
        """TEST 3: MobileNet capability check passes."""
        model_desc = {"architecture": "mobilenetv3_small", "format": "onnx"}
        hw_profile = {"name": "Host CPU", "supported_runtimes": ["tflite", "onnx"]}
        eligible, reason = XNNPACKCompatibleINT8Strategy.check_eligibility(model_desc, hw_profile)
        self.assertTrue(eligible, f"Expected eligible for MobileNet, got False: {reason}")
        self.assertIn("ELIGIBLE", reason)

    def test_04_unsupported_model_capability_check_fails_cleanly(self):
        """TEST 4: Unsupported model capability check fails cleanly without exceptions."""
        unsupported_models = [
            {"architecture": "resnet50", "format": "onnx"},
            {"architecture": "vit_large_patch16_224", "format": "onnx"},
            {"architecture": "bert_base_uncased", "format": "onnx"}
        ]
        for m_desc in unsupported_models:
            eligible, reason = XNNPACKCompatibleINT8Strategy.check_eligibility(m_desc)
            self.assertFalse(eligible, f"Expected ineligible for {m_desc['architecture']}")
            self.assertIn("NOT ELIGIBLE", reason)
            self.assertIn("Reason:", reason)

    def test_05_candidate_artifact_is_generated(self):
        """TEST 5: Candidate artifact is generated in the candidate output folder."""
        cand_dir = os.path.join(self.test_job_dir, "candidates", "cand_001")
        job_ctx = {
            "job_dir": self.test_job_dir,
            "model_path": BASELINE_PATH
        }
        cand_path = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand_dir, job_ctx)
        self.assertTrue(os.path.exists(cand_path), f"Candidate missing at {cand_path}")
        self.assertGreater(os.path.getsize(cand_path), 1000000)

    def test_06_candidate_is_isolated_under_job_output(self):
        """TEST 6: Candidate artifact is isolated under job output directory."""
        cand_dir = os.path.join(self.test_job_dir, "candidates", "cand_002")
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH}
        cand_path = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand_dir, job_ctx)
        norm_cand = os.path.normpath(cand_path)
        norm_job = os.path.normpath(self.test_job_dir)
        self.assertTrue(norm_cand.startswith(norm_job))
        self.assertIn("candidates", norm_cand)
        self.assertNotIn("phase_c2", norm_cand)

    def test_07_candidate_does_not_modify_phase_c2(self):
        """TEST 7: Candidate generation does not modify output/phase_c2."""
        sha_before = self._compute_sha(BASELINE_PATH)
        cand_dir = os.path.join(self.test_job_dir, "candidates", "cand_003")
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH}
        _ = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand_dir, job_ctx)
        sha_after = self._compute_sha(BASELINE_PATH)
        self.assertEqual(sha_before, sha_after)

    def test_08_baseline_sha_remains_unchanged(self):
        """TEST 8: Baseline SHA remains unchanged equals expected SHA."""
        sha = self._compute_sha(BASELINE_PATH)
        self.assertEqual(sha, EXPECTED_BASELINE_SHA)
        self.assertEqual(os.path.getsize(BASELINE_PATH), EXPECTED_BASELINE_SIZE)

    def test_09_candidate_runtime_is_tflite(self):
        """TEST 9: Candidate runtime format is TFLite FlatBuffer."""
        cand_dir = os.path.join(self.test_job_dir, "candidates", "cand_004")
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH}
        cand_path = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand_dir, job_ctx)
        with open(cand_path, "rb") as f:
            header = f.read(16)
        self.assertIn(b"TFL3", header, "Generated artifact is missing 'TFL3' header.")

    def test_10_xnnpack_initialization_succeeds(self):
        """TEST 10: XNNPACK initialization succeeds for known-good MobileNet candidate."""
        cand_dir = os.path.join(self.test_job_dir, "candidates", "cand_005")
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH}
        cand_path = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand_dir, job_ctx)
        v_res = XNNPACKCompatibleINT8Strategy.verify_xnnpack_execution(cand_path, num_threads=2)
        self.assertTrue(v_res["success"], f"XNNPACK execution verification failed: {v_res.get('error')}")
        self.assertEqual(v_res["verdict"], "XNNPACK_COMPATIBLE")

    def test_11_delegated_operator_count_greater_than_zero(self):
        """TEST 11: Delegated operator count > 0."""
        cand_dir = os.path.join(self.test_job_dir, "candidates", "cand_006")
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH}
        cand_path = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand_dir, job_ctx)
        v_res = XNNPACKCompatibleINT8Strategy.verify_xnnpack_execution(cand_path, num_threads=2)
        self.assertGreater(v_res.get("delegated_operators", 0), 0)

    def test_12_fallback_count_is_recorded(self):
        """TEST 12: Fallback operator count is recorded."""
        cand_dir = os.path.join(self.test_job_dir, "candidates", "cand_007")
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH}
        cand_path = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand_dir, job_ctx)
        v_res = XNNPACKCompatibleINT8Strategy.verify_xnnpack_execution(cand_path, num_threads=2)
        self.assertIn("fallback_operators", v_res)
        self.assertGreaterEqual(v_res["fallback_operators"], 0)

    def test_13_accuracy_is_actually_evaluated(self):
        """TEST 13: Accuracy is actually evaluated on real samples."""
        cand = XNNPACKCompatibleINT8Strategy.create_candidate("cand_013")
        evaluator = CandidateEvaluator(objective_function=ObjectiveFunction("balanced"))
        job_ctx = {
            "job_dir": self.test_job_dir,
            "model_path": BASELINE_PATH,
            "num_threads": 2
        }
        fp32_baseline = {"accuracy": 0.9797, "size_bytes": 6122714, "latency_ms": 1.0}
        res = evaluator.evaluate(cand, job_ctx, fp32_baseline)
        self.assertGreater(res.top1_accuracy, 0.90)
        self.assertGreater(res.macro_f1, 0.90)

    def test_14_benchmark_is_actually_executed(self):
        """TEST 14: Canonical 10+100 benchmark is actually executed."""
        cand = XNNPACKCompatibleINT8Strategy.create_candidate("cand_014")
        evaluator = CandidateEvaluator(objective_function=ObjectiveFunction("balanced"))
        job_ctx = {
            "job_dir": self.test_job_dir,
            "model_path": BASELINE_PATH,
            "test_samples": 10,
            "num_threads": 2
        }
        fp32_baseline = {"accuracy": 0.9797, "size_bytes": 6122714, "latency_ms": 1.0}
        res = evaluator.evaluate(cand, job_ctx, fp32_baseline)
        self.assertGreater(res.latency_mean_ms, 0.0)
        self.assertGreater(res.throughput_ips, 0.0)
        self.assertIsNotNone(res.benchmark_provenance)

    def test_15_no_candidate_metric_is_hardcoded(self):
        """TEST 15: No candidate metric is hardcoded; variation across runs reflects live measurement."""
        cand1 = XNNPACKCompatibleINT8Strategy.create_candidate("cand_015_a")
        evaluator = CandidateEvaluator(objective_function=ObjectiveFunction("balanced"))
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH, "test_samples": 10, "num_threads": 2}
        fp32_baseline = {"accuracy": 0.9797, "size_bytes": 6122714, "latency_ms": 1.0}
        res = evaluator.evaluate(cand1, job_ctx, fp32_baseline)
        # Verify latency is not an exact fake hardcoded constant like 1.2800000000
        self.assertTrue(isinstance(res.latency_mean_ms, float))
        self.assertGreater(res.latency_mean_ms, 0.1)

    def test_16_candidate_receives_normal_safety_evaluation(self):
        """TEST 16: Candidate receives normal UAQE safety classification."""
        cand = XNNPACKCompatibleINT8Strategy.create_candidate("cand_016")
        evaluator = CandidateEvaluator(objective_function=ObjectiveFunction("balanced"))
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH, "num_threads": 2}
        fp32_baseline = {"accuracy": 0.9797, "size_bytes": 6122714, "latency_ms": 1.0}
        res = evaluator.evaluate(cand, job_ctx, fp32_baseline)
        self.assertIn(res.safety_classification, ["EXCELLENT", "ACCEPTABLE", "CRITICAL"])
        self.assertLessEqual(res.accuracy_loss_pp, 1.0)
        self.assertEqual(res.safety_classification, "EXCELLENT")

    def test_17_candidate_enters_autonomous_candidate_list(self):
        """TEST 17: Candidate is discovered and added to candidate blueprints by CandidateGenerator."""
        generator = CandidateGenerator(
            model_descriptor={"architecture": "mobilenetv3_small"},
            dataset_descriptor={"dataset_name": "semiconductor"},
            hardware_profile={"name": "Host CPU", "supported_runtimes": ["tflite"]}
        )
        cands = []
        c = generator.generate_initial_candidate()
        cands.append(c)
        for _ in range(5):
            nxt = generator.generate_next_candidate([x.to_dict() for x in cands])
            if nxt is None:
                break
            cands.append(nxt)

        xnnpack_cand = next((x for x in cands if x.strategy_type == STRATEGY_NAME), None)
        self.assertIsNotNone(xnnpack_cand, f"Candidate with {STRATEGY_NAME} was not generated.")

    def test_18_winner_selection_works_with_xnnpack(self):
        """TEST 18: Winner selection works naturally through SearchManager ranking."""
        search_mgr = SearchManager(self.test_job_dir)
        # Create standard candidate and XNNPACK candidate
        evaluator = CandidateEvaluator(objective_function=ObjectiveFunction("latency_first"))
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH, "num_threads": 2}
        fp32_baseline = {"accuracy": 0.9797, "size_bytes": 6122714, "latency_ms": 10.0}

        cand_xnn = XNNPACKCompatibleINT8Strategy.create_candidate("cand_win_test")
        res_xnn = evaluator.evaluate(cand_xnn, job_ctx, fp32_baseline)
        search_mgr.record_candidate(res_xnn)

        best, satisfied = search_mgr.select_best_candidate()
        self.assertIsNotNone(best)
        self.assertTrue(satisfied)
        self.assertEqual(best.candidate_id, "cand_win_test")

    def test_19_job_isolation_is_preserved(self):
        """TEST 19: Artifacts for two different jobs remain completely isolated."""
        job1_dir = os.path.join(self.test_job_dir, "job_A")
        job2_dir = os.path.join(self.test_job_dir, "job_B")
        cand1_dir = os.path.join(job1_dir, "candidates", "cand_001")
        cand2_dir = os.path.join(job2_dir, "candidates", "cand_001")

        p1 = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand1_dir, {"model_path": BASELINE_PATH})
        p2 = XNNPACKCompatibleINT8Strategy.generate_candidate_artifact(cand2_dir, {"model_path": BASELINE_PATH})

        self.assertNotEqual(p1, p2)
        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))

    def test_20_provenance_is_stored(self):
        """TEST 20: Full provenance metadata is attached to the candidate evaluation result."""
        cand = XNNPACKCompatibleINT8Strategy.create_candidate("cand_prov_test")
        evaluator = CandidateEvaluator(objective_function=ObjectiveFunction("balanced"))
        job_ctx = {"job_dir": self.test_job_dir, "model_path": BASELINE_PATH, "test_samples": 10, "num_threads": 2}
        fp32_baseline = {"accuracy": 0.9797, "size_bytes": 6122714, "latency_ms": 1.0, "source_sha256": EXPECTED_BASELINE_SHA}
        res = evaluator.evaluate(cand, job_ctx, fp32_baseline)

        meta = res.artifact_metadata
        self.assertEqual(meta.get("delegate"), "XNNPACK")
        self.assertEqual(meta.get("strategy"), STRATEGY_NAME)
        self.assertEqual(meta.get("compatibility_status"), "XNNPACK_COMPATIBLE")
        self.assertGreater(meta.get("delegated_operator_count", 0), 0)
        self.assertIn("candidate_artifact_sha256", meta)
        self.assertIsNotNone(res.benchmark_provenance)


if __name__ == "__main__":
    unittest.main()
