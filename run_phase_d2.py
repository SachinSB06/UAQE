"""
UAQE Phase D.2 Master Runner
Orchestrates Phase D.2 Storage Reduction Experiments, Verification, Baseline Protection,
and Unit Testing.
"""

import os
import sys
import hashlib
import unittest
import time
from typing import Dict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.uaqe.compression.compression_experimenter import CompressionExperimenter


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
    print("UNIVERSAL AI QUANTIZATION ENGINE (UAQE) — PHASE D.2")
    print("Turn Pruning Sparsity Into Real Storage Reduction")
    print("=" * 70)

    # 1. Baseline Pre-Execution Hash Verification
    baseline_path = os.path.join(PROJECT_ROOT, "output", "phase_c4", "models", "c4_best_int8.tflite")
    d1_dir = os.path.join(PROJECT_ROOT, "output", "phase_d1")

    assert os.path.exists(baseline_path), f"Baseline model missing: {baseline_path}"
    assert os.path.exists(d1_dir), f"Phase D.1 directory missing: {d1_dir}"

    print("\n[Baseline Protection] Recording pre-execution SHA-256 hashes...")
    pre_baseline_hash = compute_file_hash(baseline_path)
    pre_d1_hashes = scan_directory_hashes(d1_dir)
    print(f"  Protected Baseline SHA-256: {pre_baseline_hash}")
    print(f"  Protected Phase D.1 Files Count: {len(pre_d1_hashes)}")

    # 2. Run Experiments
    t0 = time.time()
    experimenter = CompressionExperimenter(
        project_root=PROJECT_ROOT,
        dataset_root="D:\\semiconductor_dataset\\dataset",
        baseline_model_path="output\\phase_c4\\models\\c4_best_int8.tflite",
        output_dir="output\\phase_d2",
        reports_dir="reports\\phase_d2"
    )
    exp_output = experimenter.run_all_experiments()
    total_time = time.time() - t0

    # 3. Post-Execution Hash Verification
    print("\n[Baseline Protection] Verifying post-execution SHA-256 hashes...")
    post_baseline_hash = compute_file_hash(baseline_path)
    post_d1_hashes = scan_directory_hashes(d1_dir)

    if pre_baseline_hash != post_baseline_hash:
        raise RuntimeError("CRITICAL ERROR: Baseline model c4_best_int8.tflite was modified during execution!")
    if pre_d1_hashes != post_d1_hashes:
        raise RuntimeError("CRITICAL ERROR: Phase D.1 artifacts were modified during execution!")
    print("  [PASSED] Baseline and Phase D.1 artifacts are unmodified and verified bit-for-bit.")

    # 4. Run Unit Tests
    print("\n" + "=" * 70)
    print("RUNNING ALL UAQE UNIT TESTS (INCLUDING PHASE D.2 TESTS)")
    print("=" * 70)

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir=os.path.join(PROJECT_ROOT, "src", "uaqe", "tests"),
        pattern="test_*.py"
    )
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)

    passed_count = test_result.testsRun - len(test_result.failures) - len(test_result.errors)
    failed_count = len(test_result.failures) + len(test_result.errors)

    winner = exp_output["winner"]
    b_info = exp_output["baseline"]

    # 5. Final Summary
    print("\n" + "=" * 40)
    print("UAQE PHASE D.2 COMPLETE")
    print("=" * 40)
    print(f"\nBaseline:")
    print(f"  Size: {b_info['size_bytes']:,} bytes ({b_info['size_bytes']/(1024*1024):.4f} MB)")
    print(f"  Accuracy: {b_info['accuracy']*100:.4f}%")
    print(f"\nBest D2 candidate:")
    print(f"  Candidate ID: {winner['Candidate']}")
    print(f"  Method: {winner['Compression']}")
    print(f"  Sparsity: {winner['Sparsity']}")
    print(f"  Clusters: {winner['Clusters']}")
    print(f"  Compressed size: {winner['Actual Size (Bytes)']:,} bytes ({winner['Actual Size (MB)']} MB)")
    print(f"  Storage reduction: {winner['Storage Reduction (%)']:.2f}% (Ratio: {winner['Compression Ratio']:.2f}x)")
    print(f"  Accuracy: {winner['Accuracy (%)']:.4f}% ({winner['Correct / Total']})")
    print(f"  Macro F1: {winner['Macro F1 (%)']:.4f}%")
    print(f"  Reconstruction: Lossless={winner['Lossless']}, MAE={winner['Mean Abs Error']:.4f}, CosSim={winner['Cosine Sim']:.6f}")
    print(f"  Deployment status: {winner['Deployable Runtime']}")
    print(f"\nTests:")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")
    print(f"\nArtifacts:")
    print(f"  output/phase_d2/d2_compression_results.csv")
    print(f"  output/phase_d2/d2_tensor_compression_report.csv")
    print(f"  output/phase_d2/d2_reconstruction_verification.json")
    print(f"  output/phase_d2/d2_per_image_predictions.csv")
    print(f"  reports/phase_d2/phase_d2_compression_report.md")
    print("=" * 40)


if __name__ == "__main__":
    main()
