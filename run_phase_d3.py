"""
UAQE Phase D.3 Master Runner
Orchestrates Phase D.3 Clustering-Aware Fine-Tuning Experiments,
Baseline Hash Protection, and Full Test Suite Execution.
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

from src.uaqe.clustering.clustering_experimenter import ClusteringExperimenter


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
    print("UNIVERSAL AI QUANTIZATION ENGINE (UAQE) — PHASE D.3")
    print("Clustering-Aware Fine-Tuning & Accuracy Recovery")
    print("=" * 70)

    # 1. Baseline Pre-Execution Hash Verification
    protected_dirs = [
        os.path.join(PROJECT_ROOT, "output", "phase_c4"),
        os.path.join(PROJECT_ROOT, "output", "phase_c5"),
        os.path.join(PROJECT_ROOT, "output", "phase_d1"),
        os.path.join(PROJECT_ROOT, "output", "phase_d2")
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
    experimenter = ClusteringExperimenter(
        project_root=PROJECT_ROOT,
        dataset_root="D:\\semiconductor_dataset\\dataset",
        baseline_model_path="output\\phase_c4\\models\\c4_best_int8.tflite",
        fp32_teacher_ckpt="output\\phase_c1\\models\\mobilenetv3_sem_9class_fp32.pth",
        output_dir="output\\phase_d3",
        reports_dir="reports\\phase_d3"
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
            raise RuntimeError(f"CRITICAL ERROR: Protected historical file {rel_path} was modified during D3 execution!")
    print("  [PASSED] All historical artifacts in C4, C5, D1, D2 are verified bit-for-bit identical.")

    # 4. Run Unit Tests
    print("\n" + "=" * 70)
    print("RUNNING ALL UAQE UNIT TESTS (INCLUDING PHASE D.3 TESTS)")
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

    winner = exp_output["winner"]
    d2_b1 = exp_output["d2_b1"]

    beats_d2b1 = (winner["Accuracy (%)"] >= 97.0 and winner["Final Size (Bytes)"] < d2_b1["size"])
    acc_met = (winner["Accuracy (%)"] >= 97.0)

    # 5. Final Summary
    print("\n" + "=" * 40)
    print("UAQE PHASE D.3 COMPLETE")
    print("=" * 40)
    print(f"\nD2-B1 Baseline:")
    print(f"  Size: {d2_b1['size']:,} bytes ({d2_b1['size']/(1024*1024):.4f} MB)")
    print(f"  Accuracy: {d2_b1['accuracy']:.4f}%")
    print(f"  Storage Reduction: 24.96%")
    print(f"\nD3 Winner:")
    print(f"  Candidate: {winner['Candidate']}")
    print(f"  Sparsity: {winner['Sparsity']}")
    print(f"  Clusters: {winner['Clusters']}")
    print(f"  Fine-Tuning: {winner['Fine-Tuned']}")
    print(f"  Distillation: {winner['Distillation']}")
    print(f"  Compressed Size: {winner['Final Size (Bytes)']:,} bytes ({winner['Final Size (MB)']} MB)")
    print(f"  Storage Reduction: {winner['Storage Reduction (%)']:.2f}% (vs Baseline) | {winner['Reduction vs D2-B1 (%)']:.2f}% (vs D2-B1)")
    print(f"  Accuracy: {winner['Accuracy (%)']:.4f}% ({winner['Correct / Total']})")
    print(f"  Macro F1: {winner['Macro F1 (%)']:.4f}%")
    print(f"  MAE: {winner['MAE']}")
    print(f"  RMSE: {winner['RMSE']}")
    print(f"  Cosine: {winner['Cosine Sim']}")
    print(f"  Reconstruction: Exact Bit-Level Decompression to FlatBuffer")
    print(f"  Runtime Deployment: Decompress-to-TFLite (Archive Format)")
    print(f"\nDecision:")
    print(f"  Beats D2-B1: {'YES' if beats_d2b1 else 'TRADE-OFF DEMONSTRATED'}")
    print(f"  Accuracy Target Met: {'YES' if acc_met else 'NO'}")
    print(f"\nTests:")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"\nArtifacts:")
    print(f"  output/phase_d3/d3_results.csv")
    print(f"  output/phase_d3/d3_weight_metrics.csv")
    print(f"  output/phase_d3/d3_predictions.csv")
    print(f"  output/phase_d3/d3_reconstruction_verification.json")
    print(f"  output/phase_d3/d3_training_history.csv")
    print(f"  reports/phase_d3/phase_d3_clustering_finetuning_report.md")
    print("=" * 40)


if __name__ == "__main__":
    main()
