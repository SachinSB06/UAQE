"""
run_mobilenet_autonomous_xnnpack_e2e.py
Executes a real MobileNet autonomous candidate search workflow incorporating XNNPACK_COMPATIBLE_INT8.
"""

import os
import sys
import json
import shutil
import hashlib

# Ensure src is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from uaqe.optimization.candidate_generator import CandidateGenerator
from uaqe.optimization.candidate_evaluator import CandidateEvaluator, ObjectiveFunction
from uaqe.optimization.search_manager import SearchManager
from uaqe.optimization.strategies.xnnpack_int8 import XNNPACKCompatibleINT8Strategy

BASELINE_PATH = os.path.join(PROJECT_ROOT, "output", "phase_c2", "models", "mobilenetv3_sem_9class_qat_int8.tflite")
EXPECTED_BASELINE_SHA = "10d51d4f243c554734e7347f9010297fb182a64c4ad89d5551e24830846dab1d"


def main():
    print("====================================================")
    print("UAQE AUTONOMOUS E2E OPTIMIZATION: MOBILENETV3")
    print("====================================================")

    # 1. Verify baseline pre-flight
    with open(BASELINE_PATH, "rb") as f:
        pre_sha = hashlib.sha256(f.read()).hexdigest()
    assert pre_sha == EXPECTED_BASELINE_SHA, f"Baseline SHA mismatch! {pre_sha}"
    print(f"Pre-flight baseline verified: {pre_sha}")

    # 2. Setup isolated job directory
    job_id = "JOB-MOBILENET-XNNPACK-E2E"
    job_dir = os.path.join(PROJECT_ROOT, "output", "jobs", job_id)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
    os.makedirs(job_dir, exist_ok=True)

    # 3. Autonomous Candidate Generation
    print("\n--- Generating Autonomous Candidates ---")
    model_desc = {"architecture": "mobilenetv3_small", "model_path": BASELINE_PATH}
    dataset_desc = {"dataset_name": "semiconductor"}
    hw_profile = {"name": "Host CPU", "supported_runtimes": ["tflite"]}

    generator = CandidateGenerator(model_desc, dataset_desc, hw_profile)
    candidates = []
    c0 = generator.generate_initial_candidate()
    candidates.append(c0)
    for _ in range(4):
        c_next = generator.generate_next_candidate([c.to_dict() for c in candidates])
        if c_next is None:
            break
        candidates.append(c_next)

    print(f"Discovered {len(candidates)} candidate strategies:")
    for idx, c in enumerate(candidates, 1):
        print(f"  [{idx}] {c.candidate_id}: {c.name} ({c.strategy_type})")

    xnn_cand = next((c for c in candidates if c.strategy_type == "XNNPACK_COMPATIBLE_INT8"), None)
    assert xnn_cand is not None, "XNNPACK_COMPATIBLE_INT8 not discovered by autonomous generator!"

    # 4. Evaluate Candidates
    print("\n--- Evaluating Candidates with Profile: BALANCED ---")
    evaluator = CandidateEvaluator(ObjectiveFunction("balanced"))
    search_mgr = SearchManager(job_dir)
    job_ctx = {
        "job_id": job_id,
        "job_dir": job_dir,
        "model_path": BASELINE_PATH,
        "num_threads": 2
    }
    fp32_baseline = {
        "accuracy": 0.9797,
        "size_bytes": 6122714,
        "latency_ms": 1.0
    }

    # Evaluate the XNNPACK candidate
    res_xnn = evaluator.evaluate(xnn_cand, job_ctx, fp32_baseline)
    search_mgr.record_candidate(res_xnn)

    print(f"\nXNNPACK Candidate Result:")
    print(f"  Candidate ID:        {res_xnn.candidate_id}")
    print(f"  Strategy:            {res_xnn.strategy_type}")
    print(f"  Accuracy:            {res_xnn.top1_accuracy * 100:.2f}%")
    print(f"  Accuracy Loss:       {res_xnn.accuracy_loss_pp:.4f} pp")
    print(f"  Safety:              {res_xnn.safety_classification}")
    print(f"  Latency Mean:        {res_xnn.latency_mean_ms:.2f} ms")
    print(f"  Throughput:          {res_xnn.throughput_ips:.1f} img/s")
    print(f"  Delegated Ops:       {res_xnn.artifact_metadata.get('delegated_operator_count')}")
    print(f"  Fallback Ops:        {res_xnn.artifact_metadata.get('fallback_operator_count')}")
    print(f"  Candidate Path:      {res_xnn.model_path}")

    # 5. Winner Selection
    best, satisfied = search_mgr.select_best_candidate()
    print(f"\n--- Winner Selection ---")
    print(f"  Selected Winner:     {best.candidate_name} ({best.candidate_id})")
    print(f"  Satisfied:           {satisfied}")
    print(f"  Composite Score:     {best.composite_score:.4f}")

    # 6. Verify Baseline Integrity Post-Execution
    with open(BASELINE_PATH, "rb") as f:
        post_sha = hashlib.sha256(f.read()).hexdigest()
    assert post_sha == EXPECTED_BASELINE_SHA, f"Baseline altered! {post_sha}"
    print(f"\nPost-flight baseline SHA verified unchanged: {post_sha}")

    # Output Summary JSON
    summary = {
        "status": "SUCCESS",
        "job_id": job_id,
        "xnnpack_candidate": {
            "id": res_xnn.candidate_id,
            "accuracy": res_xnn.top1_accuracy,
            "accuracy_loss_pp": res_xnn.accuracy_loss_pp,
            "latency_ms": res_xnn.latency_mean_ms,
            "throughput_ips": res_xnn.throughput_ips,
            "safety": res_xnn.safety_classification,
            "delegated_operators": res_xnn.artifact_metadata.get("delegated_operator_count"),
            "fallback_operators": res_xnn.artifact_metadata.get("fallback_operator_count")
        },
        "winner": {
            "id": best.candidate_id,
            "name": best.candidate_name,
            "strategy": best.strategy_type
        },
        "baseline_sha_unchanged": (pre_sha == post_sha == EXPECTED_BASELINE_SHA)
    }

    out_file = os.path.join(job_dir, "autonomous_run_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {out_file}")

    return summary


if __name__ == "__main__":
    main()
