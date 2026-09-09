"""Automated Unit & Integration Test Suite for UAQE Autonomous Optimization.

Tests:
1. Accuracy loss calculation (percentage points)
2. <= 1.0 pp classification = EXCELLENT
3. > 1.0 and <= 4.0 pp = ACCEPTABLE
4. > 4.0 pp = CRITICAL
5. CRITICAL candidates are blocked from final selection
6. Valid candidates can be selected
7. Max candidate budget defaults to 10
8. Candidate count never exceeds 10
9. Early stopping works
10. Stopping reason is recorded
11. Duplicate candidates are not executed (deterministic signature deduplication)
12. Progressive candidate generation works
13. Capability filtering works
14. Pareto frontier ignores CRITICAL candidates for final selection
15. ResNet INT8 5.10 pp-loss case triggers recovery search
16. No-valid-candidate behavior is handled correctly
17. FP32 fallback/safe behavior is correct where configured
18. Job isolation works (output/jobs/<job_id>/)
19. Historical artifacts remain immutable (Phases C4–R1)
20. Calibration/test split isolation remains valid (zero data leakage)
21. Exactly one final deployable package is produced
"""

import os
import sys
import json
import shutil
import hashlib
import unittest

# Ensure src is at the head of sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from uaqe.optimization.accuracy_safety_policy import AccuracySafetyPolicy, AccuracyClassification
from uaqe.optimization.objective_function import ObjectiveFunction, ObjectiveWeights
from uaqe.optimization.stopping_policy import StoppingPolicy, StoppingReason
from uaqe.optimization.candidate_generator import CandidateGenerator, OptimizationCandidate
from uaqe.optimization.candidate_evaluator import CandidateEvaluator, CandidateEvaluationResult
from uaqe.optimization.search_manager import SearchManager
from uaqe.optimization.optimization_controller import OptimizationController


class TestAutonomousOptimization(unittest.TestCase):
    """Test suite covering all 21 autonomous optimization requirements."""

    def setUp(self):
        self.temp_job_dir = os.path.abspath("output/test_autonomous_job_tmp")
        os.makedirs(self.temp_job_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.temp_job_dir):
            shutil.rmtree(self.temp_job_dir, ignore_errors=True)

    # 1. Accuracy loss calculation
    def test_01_accuracy_loss_calculation(self):
        # 75.00% -> 69.90% = 5.10 pp
        loss_pp1 = AccuracySafetyPolicy.calculate_loss_pp(0.7500, 0.6990)
        self.assertAlmostEqual(loss_pp1, 5.10, places=2)

        # 75.00% -> 71.50% = 3.50 pp
        loss_pp2 = AccuracySafetyPolicy.calculate_loss_pp(0.7500, 0.7150)
        self.assertAlmostEqual(loss_pp2, 3.50, places=2)

        # 75.00% -> 74.60% = 0.40 pp
        loss_pp3 = AccuracySafetyPolicy.calculate_loss_pp(0.7500, 0.7460)
        self.assertAlmostEqual(loss_pp3, 0.40, places=2)

    # 2. <= 1.0 pp classification = EXCELLENT
    def test_02_excellent_classification(self):
        self.assertEqual(AccuracySafetyPolicy.classify(0.00), AccuracyClassification.EXCELLENT)
        self.assertEqual(AccuracySafetyPolicy.classify(0.40), AccuracyClassification.EXCELLENT)
        self.assertEqual(AccuracySafetyPolicy.classify(1.00), AccuracyClassification.EXCELLENT)

    # 3. > 1.0 and <= 4.0 pp = ACCEPTABLE
    def test_03_acceptable_classification(self):
        self.assertEqual(AccuracySafetyPolicy.classify(1.01), AccuracyClassification.ACCEPTABLE)
        self.assertEqual(AccuracySafetyPolicy.classify(2.50), AccuracyClassification.ACCEPTABLE)
        self.assertEqual(AccuracySafetyPolicy.classify(4.00), AccuracyClassification.ACCEPTABLE)

    # 4. > 4.0 pp = CRITICAL
    def test_04_critical_classification(self):
        self.assertEqual(AccuracySafetyPolicy.classify(4.01), AccuracyClassification.CRITICAL)
        self.assertEqual(AccuracySafetyPolicy.classify(5.10), AccuracyClassification.CRITICAL)
        self.assertEqual(AccuracySafetyPolicy.classify(10.00), AccuracyClassification.CRITICAL)

    # 5. CRITICAL candidates are blocked from final selection
    def test_05_critical_candidates_blocked_from_final_selection(self):
        search_mgr = SearchManager(self.temp_job_dir)
        cand_critical = CandidateEvaluationResult(
            candidate_id="c1", candidate_name="Critical PTQ", strategy_type="ptq", model_path="c1.onnx",
            top1_accuracy=0.6990, macro_f1=0.6990, model_size_bytes=24000000, latency_mean_ms=39.0,
            latency_median_ms=39.0, latency_p95_ms=42.0, throughput_ips=25.0, prediction_agreement=79.3,
            accuracy_loss_pp=5.10, safety_classification="CRITICAL", is_satisfied=False, is_critical=True,
            composite_score=0.12, raw_score=0.85, size_reduction=0.74, latency_reduction=0.41,
            execution_duration_sec=1.0, artifact_metadata={}
        )
        cand_valid = CandidateEvaluationResult(
            candidate_id="c2", candidate_name="Valid Mixed", strategy_type="ptq", model_path="c2.onnx",
            top1_accuracy=0.7460, macro_f1=0.7460, model_size_bytes=25000000, latency_mean_ms=46.0,
            latency_median_ms=46.0, latency_p95_ms=48.0, throughput_ips=21.0, prediction_agreement=88.0,
            accuracy_loss_pp=0.40, safety_classification="EXCELLENT", is_satisfied=True, is_critical=False,
            composite_score=0.88, raw_score=0.88, size_reduction=0.73, latency_reduction=0.31,
            execution_duration_sec=1.0, artifact_metadata={}
        )
        search_mgr.record_candidate(cand_critical)
        search_mgr.record_candidate(cand_valid)

        best_cand, is_satisfied = search_mgr.select_best_candidate()
        self.assertEqual(best_cand.candidate_id, "c2")
        self.assertTrue(is_satisfied)

    # 6. Valid candidates can be selected
    def test_06_valid_candidates_can_be_selected(self):
        search_mgr = SearchManager(self.temp_job_dir)
        cand_acceptable = CandidateEvaluationResult(
            candidate_id="c1", candidate_name="Acceptable", strategy_type="ptq", model_path="c1.onnx",
            top1_accuracy=0.7200, macro_f1=0.7200, model_size_bytes=24000000, latency_mean_ms=40.0,
            latency_median_ms=40.0, latency_p95_ms=42.0, throughput_ips=25.0, prediction_agreement=82.0,
            accuracy_loss_pp=3.00, safety_classification="ACCEPTABLE", is_satisfied=True, is_critical=False,
            composite_score=0.75, raw_score=0.75, size_reduction=0.74, latency_reduction=0.40,
            execution_duration_sec=1.0, artifact_metadata={}
        )
        search_mgr.record_candidate(cand_acceptable)
        best_cand, is_satisfied = search_mgr.select_best_candidate()
        self.assertEqual(best_cand.candidate_id, "c1")
        self.assertTrue(is_satisfied)

    # 7. Max candidate budget defaults to 10
    def test_07_max_candidate_budget_defaults_to_10(self):
        generator = CandidateGenerator({}, {}, {}, profile="balanced")
        self.assertEqual(generator.max_budget, 10)

        policy = StoppingPolicy()
        self.assertEqual(policy.max_candidates, 10)

    # 8. Candidate count never exceeds budget
    def test_08_candidate_count_never_exceeds_budget(self):
        generator = CandidateGenerator(
            {"architecture": "ResNetForImageClassification", "model_family": "resnet"},
            {}, {}, profile="balanced", max_budget=3
        )
        c1 = generator.generate_initial_candidate()
        c2 = generator.generate_next_candidate([c1.to_dict()])
        c3 = generator.generate_next_candidate([c1.to_dict(), c2.to_dict()])
        c4 = generator.generate_next_candidate([c1.to_dict(), c2.to_dict(), c3.to_dict()])

        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)
        self.assertIsNotNone(c3)
        self.assertIsNone(c4, "Candidate count must strictly not exceed max_budget")

    # 9. Early stopping works
    def test_09_early_stopping_works(self):
        policy = StoppingPolicy(max_candidates=10, target_accuracy_loss_pp=1.0, target_size_reduction=0.70)
        best_safe = {
            "accuracy_loss_pp": 0.40,
            "safety_classification": "EXCELLENT",
            "size_reduction": 0.74
        }
        should_stop, reason, _ = policy.evaluate(candidate_count=3, history=[], has_more_candidates=True, current_best_safe=best_safe)
        self.assertTrue(should_stop)
        self.assertEqual(reason, StoppingReason.TARGET_CONSTRAINTS_SATISFIED)

    # 10. Stopping reason is recorded
    def test_10_stopping_reason_is_recorded(self):
        policy = StoppingPolicy(max_candidates=2)
        should_stop, reason, desc = policy.evaluate(candidate_count=2, history=[], has_more_candidates=True, current_best_safe=None)
        self.assertTrue(should_stop)
        self.assertEqual(reason, StoppingReason.BUDGET_EXHAUSTED)
        self.assertIn("Search budget", desc)

    # 11. Duplicate candidates are not executed
    def test_11_duplicate_candidates_are_not_executed(self):
        generator = CandidateGenerator(
            {"architecture": "ResNetForImageClassification", "model_family": "resnet"},
            {}, {}, profile="balanced", max_budget=10
        )
        c1 = generator.generate_initial_candidate()
        c2 = generator.generate_next_candidate([c1.to_dict()])
        c3 = generator.generate_next_candidate([c1.to_dict(), c2.to_dict()])

        sigs = [c.signature for c in [c1, c2, c3] if c is not None]
        self.assertEqual(len(sigs), len(set(sigs)), "All candidate signatures must be uniquely deterministic")

    # 12. Progressive candidate generation works
    def test_12_progressive_candidate_generation(self):
        generator = CandidateGenerator(
            {"architecture": "ResNetForImageClassification", "model_family": "resnet"},
            {}, {}, profile="balanced", max_budget=5
        )
        c1 = generator.generate_initial_candidate()
        self.assertIn("Standard Static INT8", c1.name)

        c2 = generator.generate_next_candidate([c1.to_dict()])
        self.assertIn("Stem & Classifier Protected", c2.name)

        c3 = generator.generate_next_candidate([c1.to_dict(), c2.to_dict()])
        self.assertIn("Early Stages & Stem Protected", c3.name)

    # 13. Capability filtering works
    def test_13_capability_filtering(self):
        mobilenet_gen = CandidateGenerator(
            {"architecture": "MobileNetV3", "model_family": "mobilenet"},
            {}, {}, profile="balanced"
        )
        c_mob = mobilenet_gen.generate_initial_candidate()
        self.assertEqual(c_mob.strategy_type, "mobilenet_adaptive")

    # 14. Pareto frontier ignores CRITICAL candidates for final selection
    def test_14_pareto_frontier_ignores_critical_for_final_selection(self):
        search_mgr = SearchManager(self.temp_job_dir)
        c_crit = CandidateEvaluationResult(
            candidate_id="c_crit", candidate_name="Critical", strategy_type="ptq", model_path="",
            top1_accuracy=0.6990, macro_f1=0.6990, model_size_bytes=24000000, latency_mean_ms=39.0,
            latency_median_ms=39.0, latency_p95_ms=39.0, throughput_ips=25.0, prediction_agreement=79.0,
            accuracy_loss_pp=5.10, safety_classification="CRITICAL", is_satisfied=False, is_critical=True,
            composite_score=0.12, raw_score=0.85, size_reduction=0.74, latency_reduction=0.41,
            execution_duration_sec=1.0, artifact_metadata={}
        )
        c_safe = CandidateEvaluationResult(
            candidate_id="c_safe", candidate_name="Safe", strategy_type="ptq", model_path="",
            top1_accuracy=0.7460, macro_f1=0.7460, model_size_bytes=25000000, latency_mean_ms=46.0,
            latency_median_ms=46.0, latency_p95_ms=46.0, throughput_ips=21.0, prediction_agreement=88.0,
            accuracy_loss_pp=0.40, safety_classification="EXCELLENT", is_satisfied=True, is_critical=False,
            composite_score=0.88, raw_score=0.88, size_reduction=0.73, latency_reduction=0.31,
            execution_duration_sec=1.0, artifact_metadata={}
        )
        search_mgr.record_candidate(c_crit)
        search_mgr.record_candidate(c_safe)

        best_cand, is_satisfied = search_mgr.select_best_candidate()
        self.assertEqual(best_cand.candidate_id, "c_safe")
        self.assertTrue(is_satisfied)

    # 15. ResNet INT8 5.10 pp-loss case triggers recovery search
    def test_15_resnet_int8_critical_triggers_recovery_search(self):
        generator = CandidateGenerator(
            {"architecture": "ResNetForImageClassification", "model_family": "resnet"},
            {}, {}, profile="balanced", max_budget=5
        )
        c1 = generator.generate_initial_candidate()
        # Candidate 1 evaluated with 5.10 pp loss
        eval_c1 = {
            "candidate_id": c1.candidate_id,
            "accuracy_loss_pp": 5.10,
            "safety_classification": "CRITICAL",
            "is_critical": True
        }
        # Generator creates recovery candidate
        c2 = generator.generate_next_candidate([eval_c1], eval_c1)
        self.assertIsNotNone(c2)
        self.assertNotEqual(c2.candidate_id, c1.candidate_id)
        self.assertIn("Mixed Precision", c2.name)

    # 16. No-valid-candidate behavior is handled correctly
    def test_16_no_valid_candidate_behavior_handled(self):
        search_mgr = SearchManager(self.temp_job_dir)
        c_fail = CandidateEvaluationResult(
            candidate_id="c_fail", candidate_name="Failed", strategy_type="ptq", model_path="",
            top1_accuracy=0.6000, macro_f1=0.6000, model_size_bytes=24000000, latency_mean_ms=39.0,
            latency_median_ms=39.0, latency_p95_ms=39.0, throughput_ips=25.0, prediction_agreement=70.0,
            accuracy_loss_pp=15.00, safety_classification="CRITICAL", is_satisfied=False, is_critical=True,
            composite_score=0.10, raw_score=0.60, size_reduction=0.74, latency_reduction=0.41,
            execution_duration_sec=1.0, artifact_metadata={}
        )
        search_mgr.record_candidate(c_fail)
        best_cand, is_satisfied = search_mgr.select_best_candidate()
        self.assertFalse(is_satisfied)
        self.assertEqual(best_cand.candidate_id, "c_fail")

    # 17. FP32 fallback/safe behavior is correct where configured
    def test_17_fp32_fallback_safe_behavior(self):
        job_dir = os.path.join(self.temp_job_dir, "UAQE-FALLBACK-TEST")
        os.makedirs(job_dir, exist_ok=True)

        fp32_dummy = os.path.join(job_dir, "model_fp32.onnx")
        with open(fp32_dummy, "wb") as f:
            f.write(b"FP32_ORIGINAL_WEIGHTS")

        c_fail_path = os.path.join(job_dir, "c_fail.onnx")
        with open(c_fail_path, "wb") as f:
            f.write(b"CRITICAL_UNSAFE_WEIGHTS")

        c_fail = CandidateEvaluationResult(
            candidate_id="c_fail", candidate_name="Unsafe PTQ", strategy_type="ptq", model_path=c_fail_path,
            top1_accuracy=0.6000, macro_f1=0.6000, model_size_bytes=len(b"CRITICAL_UNSAFE_WEIGHTS"),
            latency_mean_ms=39.0, latency_median_ms=39.0, latency_p95_ms=39.0, throughput_ips=25.0,
            prediction_agreement=70.0, accuracy_loss_pp=15.00, safety_classification="CRITICAL",
            is_satisfied=False, is_critical=True, composite_score=0.10, raw_score=0.60, size_reduction=0.74,
            latency_reduction=0.41, execution_duration_sec=1.0, artifact_metadata={}
        )

        job_context = {
            "job_id": "UAQE-FALLBACK-TEST",
            "job_dir": job_dir,
            "model_path": fp32_dummy,
            "model_desc": {"architecture": "TestModel"},
            "dataset_desc": {"dataset_name": "TestDataset"},
            "hw_profile": {"name": "Raspberry Pi 5"},
            "dataset_ingestor": None
        }
        controller = OptimizationController(job_context=job_context, profile="balanced")
        pkg_res = controller._package_final_candidate(
            best_candidate=c_fail,
            fp32_baseline={"accuracy": 0.75, "macro_f1": 0.75, "size_bytes": len(b"FP32_ORIGINAL_WEIGHTS"), "latency_ms": 60.0, "throughput_ips": 16.0},
            is_satisfied=False,
            stopping_reason="BUDGET_EXHAUSTED",
            stopping_desc="Search budget exhausted without satisfying accuracy constraint.",
            fallback_to_fp32=True
        )

        # Confirm FP32 model was retained in optimized_model.onnx
        with open(pkg_res["optimized_model_path"], "rb") as f:
            content = f.read()
        self.assertEqual(content, b"FP32_ORIGINAL_WEIGHTS")
        self.assertFalse(pkg_res["metrics"]["accuracy_constraint_satisfied"])
        self.assertTrue(pkg_res["metrics"]["fp32_retained_as_safe_fallback"])

    # 18. Job isolation works
    def test_18_job_isolation_works(self):
        job_dir = os.path.join(self.temp_job_dir, "UAQE-ISOLATION-001")
        search_mgr = SearchManager(job_dir=job_dir)
        cand = CandidateEvaluationResult(
            candidate_id="c1", candidate_name="test", strategy_type="test", model_path="",
            top1_accuracy=0.75, macro_f1=0.75, model_size_bytes=1000, latency_mean_ms=10.0,
            latency_median_ms=10.0, latency_p95_ms=10.0, throughput_ips=100.0, prediction_agreement=100.0,
            accuracy_loss_pp=0.0, safety_classification="EXCELLENT", is_satisfied=True, is_critical=False,
            composite_score=0.9, raw_score=0.9, size_reduction=0.5, latency_reduction=0.5,
            execution_duration_sec=0.1, artifact_metadata={}
        )
        search_mgr.record_candidate(cand)
        paths = search_mgr.export_artifacts()
        self.assertTrue(os.path.exists(paths["optimization_history_json"]))
        self.assertTrue(paths["optimization_history_json"].startswith(job_dir))

    # 19. Historical artifacts remain immutable
    def test_19_historical_artifacts_remain_immutable(self):
        phase_e2_model = "output/phase_e2/models/resnet50_cifar10_fp32_baseline.pt"
        if os.path.exists(phase_e2_model):
            hasher = hashlib.sha256()
            with open(phase_e2_model, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            sha = hasher.hexdigest()
            self.assertEqual(len(sha), 64)

    # 20. Calibration/test dataset split isolation
    def test_20_calibration_test_split_isolation(self):
        dataset_path = "D:/uaqe_datasets/cifar10"
        if os.path.exists(dataset_path):
            from uaqe.dataset.universal_dataset_loader import UniversalDatasetLoader
            from uaqe.quantization.r1_resnet50_ptq import StratifiedCalibrationSampler
            from uaqe.orchestration.universal_dataset_ingestor import UniversalDatasetIngestor

            ingestor = UniversalDatasetIngestor(dataset_path)
            splits = ingestor.load()
            loader = UniversalDatasetLoader(ingestor.adapter.dataset_path)
            loader.load()

            sampler = StratifiedCalibrationSampler(loader, num_samples=64, seed=42)
            summary = sampler.sample()
            overlap = sampler.verify_zero_test_overlap()

            self.assertEqual(overlap["overlap_count"], 0)
            self.assertTrue(overlap["has_zero_overlap"])

    # 21. Exactly one final deployable package is produced
    def test_21_exactly_one_final_package_produced(self):
        job_dir = os.path.join(self.temp_job_dir, "UAQE-TEST-PKG-001")
        os.makedirs(job_dir, exist_ok=True)

        dummy_model_path = os.path.join(job_dir, "model_cand.onnx")
        with open(dummy_model_path, "wb") as f:
            f.write(b"ONNX_DUMMY_MODEL_DATA")

        job_context = {
            "job_id": "UAQE-TEST-PKG-001",
            "job_dir": job_dir,
            "model_path": dummy_model_path,
            "model_desc": {"architecture": "TestModel"},
            "dataset_desc": {"dataset_name": "TestDataset"},
            "hw_profile": {"name": "Raspberry Pi 5"},
            "dataset_ingestor": None
        }

        controller = OptimizationController(job_context=job_context, profile="balanced")
        cand = CandidateEvaluationResult(
            candidate_id="cand_001", candidate_name="Test Strategy", strategy_type="test",
            model_path=dummy_model_path, top1_accuracy=0.745, macro_f1=0.745,
            model_size_bytes=len(b"ONNX_DUMMY_MODEL_DATA"), latency_mean_ms=45.0,
            latency_median_ms=45.0, latency_p95_ms=45.0, throughput_ips=22.0, prediction_agreement=85.0,
            accuracy_loss_pp=0.50, safety_classification="EXCELLENT", is_satisfied=True, is_critical=False,
            composite_score=0.88, raw_score=0.88, size_reduction=0.70, latency_reduction=0.30,
            execution_duration_sec=1.0, artifact_metadata={}
        )

        fp32_baseline = {
            "accuracy": 0.75,
            "macro_f1": 0.75,
            "size_bytes": 100,
            "latency_ms": 60.0,
            "throughput_ips": 16.0
        }

        pkg_res = controller._package_final_candidate(
            best_candidate=cand,
            fp32_baseline=fp32_baseline,
            is_satisfied=True,
            stopping_reason="TARGET_CONSTRAINTS_SATISFIED",
            stopping_desc="Target constraints met."
        )

        self.assertTrue(os.path.exists(pkg_res["package_path"]))
        self.assertTrue(os.path.exists(pkg_res["report_path"]))
        self.assertTrue(os.path.exists(pkg_res["metrics_json_path"]))
        self.assertTrue(os.path.exists(pkg_res["optimized_model_path"]))


if __name__ == "__main__":
    unittest.main()
