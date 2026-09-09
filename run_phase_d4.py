"""
UAQE Phase D.4 Master Runner
Orchestrates Phase D.4 Adaptive Multi-Objective Optimization Experiments,
Pre- and Post-Execution Baseline Hash Verification, and Full Test Suite Execution.
"""

import os
import sys
import hashlib
import unittest
import time
from typing import Dict

# Ensure project root and src are in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.optimizer.adaptive_experimenter import AdaptiveExperimenter


def compute_file_hash(path: str) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_directory_hashes(dir_path: str) -> Dict[str, str]:
    """Computes SHA-256 hashes for all files in a directory."""
    hashes = {}
    for root, _, files in os.walk(dir_path):
        for f in sorted(files):
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, PROJECT_ROOT)
            hashes[rel_p] = compute_file_hash(full_p)
    return hashes


def main():
    print("=" * 70)
    print("UNIVERSAL AI QUANTIZATION ENGINE (UAQE) — PHASE D.4")
    print("Adaptive Multi-Objective Optimization & Layer-Aware Compression")
    print("=" * 70)

    # 1. Baseline Pre-Execution Hash Verification
    protected_dirs = [
        os.path.join(PROJECT_ROOT, "output", "phase_c4"),
        os.path.join(PROJECT_ROOT, "output", "phase_c5"),
        os.path.join(PROJECT_ROOT, "output", "phase_d1"),
        os.path.join(PROJECT_ROOT, "output", "phase_d2"),
        os.path.join(PROJECT_ROOT, "output", "phase_d3")
    ]

    print("\n[Baseline Protection] Recording pre-execution SHA-256 hashes...")
    pre_hashes = {}
    for d in protected_dirs:
        if os.path.exists(d):
            h_dict = scan_directory_hashes(d)
            pre_hashes.update(h_dict)
            print(f"  Protected {os.path.basename(d)}: {len(h_dict)} files indexed.")

    # 2. Run Experiments
    t0 = time.time()
    experimenter = AdaptiveExperimenter(
        project_root=PROJECT_ROOT,
        dataset_root="D:\\semiconductor_dataset\\dataset",
        baseline_model_path="output\\phase_c4\\models\\c4_best_int8.tflite",
        output_dir="output\\phase_d4",
        reports_dir="reports\\phase_d4"
    )
    exp_output = experimenter.run_all_experiments()
    total_time = time.time() - t0

    # 3. Post-Execution Hash Verification
    print("\n[Baseline Protection] Verifying post-execution SHA-256 hashes...")
    post_hashes = {}
    for d in protected_dirs:
        if os.path.exists(d):
            post_hashes.update(scan_directory_hashes(d))

    for rel_path, pre_h in pre_hashes.items():
        post_h = post_hashes.get(rel_path)
        if post_h != pre_h:
            raise RuntimeError(f"CRITICAL ERROR: Protected historical file {rel_path} was modified during D4 execution!")
    print("  [PASSED] All historical artifacts in C4, C5, D1, D2, D3 are verified bit-for-bit identical.")

    # 4. Run Full Unit Test Suite
    print("\n" + "=" * 70)
    print("RUNNING ALL UAQE UNIT TESTS (INCLUDING PHASE D.4 TESTS)")
    print("=" * 70)

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir=os.path.join(PROJECT_ROOT, "src", "uaqe", "tests"),
        pattern="test_*.py"
    )
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)

    passed_count = test_result.testsRun - len(test_result.failures) - len(test_result.errors) - len(test_result.skipped)
    failed_count = len(test_result.failures) + len(test_result.errors)
    skipped_count = len(test_result.skipped)

    results_df = exp_output["results_df"]
    pareto_df = exp_output["pareto_df"]
    winner = exp_output["winner"]
    d2_b1 = results_df[results_df["Candidate"] == "D2-B1"].iloc[0].to_dict()

    is_new_winner = (winner["Candidate"] != "D2-B1" and winner["Accuracy (%)"] >= 97.0 and winner["Actual Size (Bytes)"] < d2_b1["Actual Size (Bytes)"])

    # Extract Pareto representatives
    acc_opt = pareto_df.sort_values(by="Accuracy (%)", ascending=False).iloc[0]["Candidate"] if not pareto_df.empty else "D2-B1"
    storage_opt = pareto_df.sort_values(by="Actual Size (Bytes)", ascending=True).iloc[0]["Candidate"] if not pareto_df.empty else "D4-D"
    balanced_opt = winner["Candidate"]

    # Storage and Latency Improvement vs D2-B1
    storage_diff_pct = float((d2_b1["Actual Size (Bytes)"] - winner["Actual Size (Bytes)"]) / d2_b1["Actual Size (Bytes)"] * 100.0)
    acc_diff_pct = float(winner["Accuracy (%)"] - d2_b1["Accuracy (%)"])
    lat_diff_pct = float((d2_b1["Host Latency (ms)"] - winner["Host Latency (ms)"]) / d2_b1["Host Latency (ms)"] * 100.0)

    # 5. Final Formatted Console Output (§25)
    print("\n" + "=" * 40)
    print("UAQE PHASE D.4 COMPLETE")
    print("=" * 40)
    print("\nD2-B1 REFERENCE:")
    print(f"  Size: {d2_b1['Actual Size (Bytes)']:,} bytes ({d2_b1['Actual Size (MB)']} MB)")
    print(f"  Accuracy: {d2_b1['Accuracy (%)']:.4f}% ({d2_b1['Correct / Total']})")
    print(f"  Reduction: {d2_b1['Storage Reduction vs C4 (%)']:.2f}% (vs Baseline)")

    print(f"\nD4 BEST:")
    print(f"  Candidate: {winner['Candidate']}")
    print(f"  Strategy: {winner['Strategy']}")
    print(f"  Size: {winner['Actual Size (Bytes)']:,} bytes ({winner['Actual Size (MB)']} MB)")
    print(f"  Reduction: {winner['Storage Reduction vs C4 (%)']:.2f}% (vs Baseline)")
    print(f"  Accuracy: {winner['Accuracy (%)']:.4f}% ({winner['Correct / Total']})")
    print(f"  Macro F1: {winner['Macro F1 (%)']:.4f}%")
    print(f"  Validation: {winner['Validation Gate']}")
    print(f"  Host Latency: {winner['Host Latency (ms)']} ms")
    print(f"  Runtime Type: {winner['Runtime Type']}")

    print("\nImprovement vs D2-B1:")
    print(f"  Storage: {storage_diff_pct:+.2f}%")
    print(f"  Accuracy: {acc_diff_pct:+.4f}%")
    print(f"  Latency: {lat_diff_pct:+.2f}%")

    print("\nDecision:")
    print(f"  New Deployment Winner: {'YES' if is_new_winner else 'NO (D2-B1 Preserved)'}")

    print("\nPareto Frontier:")
    print(f"  Accuracy-optimal: {acc_opt}")
    print(f"  Storage-optimal: {storage_opt}")
    print(f"  Balanced: {balanced_opt}")

    print("\nTests:")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {skipped_count}")

    print("\nHistorical Protection:")
    print("  PASS")

    print("\nArtifacts:")
    print("  output/phase_d4/layer_profile.csv")
    print("  output/phase_d4/layer_profile.json")
    print("  output/phase_d4/optimization_history.csv")
    print("  output/phase_d4/d4_results.csv")
    print("  output/phase_d4/d4_per_image_predictions.csv")
    print("  output/phase_d4/d4_reconstruction_verification.json")
    print("  output/phase_d4/d4_plan.json")
    print("  output/phase_d4/pareto_frontier.csv")
    print("  reports/phase_d4/phase_d4_adaptive_optimization_report.md")
    print("=" * 40)


if __name__ == "__main__":
    main()
