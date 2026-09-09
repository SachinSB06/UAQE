"""CLI Accuracy Ablation Experiment Runner.

Orchestrates controlled experiments to analyze accuracy vs. compression trade-offs
for INT8 quantization, pruning, and weight clustering.
"""

from __future__ import annotations

import os
import sys
import json
import time
import copy
import argparse
import subprocess
import traceback
import hashlib
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import onnxruntime as ort
import tensorflow as tf

from uaqe.evaluation.real_dataset_adapter import RealDatasetAdapter
from uaqe.evaluation.real_inference_evaluator import RealInferenceEvaluator

EXP_DIR_MAP = {
    "fp32_baseline": "ablation_exp_fp32",
    "int8_only": "ablation_exp_int8",
    "int8_pruning_5": "ablation_exp_int8_pruning_5",
    "int8_pruning_10": "ablation_exp_int8_pruning_10",
    "int8_pruning_15": "ablation_exp_int8_pruning_15",
    "int8_pruning_20": "ablation_exp_int8_pruning_20",
    "int8_pruning_30": "ablation_exp_int8_pruning_30",
    "int8_clustering": "ablation_exp_int8_clustering",
    "int8_best_pruning_clustering": "ablation_exp_int8_pruning_clustering",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UAQE Accuracy Ablation Experiment Runner")
    parser.add_argument(
        "--experiment",
        dest="experiment",
        default=None,
        choices=[
            "fp32_baseline",
            "int8_only",
            "int8_pruning_5",
            "int8_pruning_10",
            "int8_pruning_15",
            "int8_pruning_20",
            "int8_pruning_30",
            "int8_clustering",
            "int8_best_pruning_clustering",
        ],
        help="Run a specific experiment by name.",
    )
    parser.add_argument(
        "--all",
        dest="all",
        action="store_true",
        help="Run the entire ablation matrix (A to I).",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        default=os.path.join(PROJECT_ROOT, "datasets", "hackathon_test_dataset"),
        help="Path to the real labeled dataset directory.",
    )
    parser.add_argument(
        "--eval-config",
        dest="eval_config",
        default=os.path.join(PROJECT_ROOT, "src", "config", "mobilenetv3_sem_eval.json"),
        help="Path to the evaluation configuration file.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=os.path.join(PROJECT_ROOT, "src", "models", "mobilenetv3_sem.onnx"),
        help="Path to the reference FP32 ONNX model file.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=os.path.join(PROJECT_ROOT, "reports", "accuracy_ablation"),
        help="Directory to save experiment reports.",
    )
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        help="Continue running remaining experiments if one fails.",
    )
    return parser.parse_args()


def load_config_snapshot(config_dir: str) -> Dict[str, Any]:
    snapshot = {}
    for name in ["quantization.json", "compression.json", "optimization.json"]:
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                snapshot[name] = json.load(f)
        else:
            snapshot[name] = None
    return snapshot


def verify_config_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    is_changed = False
    details = {}
    for name in ["quantization.json", "compression.json", "optimization.json"]:
        if before[name] != after[name]:
            is_changed = True
            details[name] = "CHANGED"
        else:
            details[name] = "UNCHANGED"
    status = "CHANGED — FAILURE" if is_changed else "UNCHANGED"
    return status, details


def get_production_artifacts_snapshot(exports_dir: str) -> Dict[str, str]:
    snapshot = {}
    if os.path.exists(exports_dir):
        for root, dirs, files in os.walk(exports_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, exports_dir)
                try:
                    sha256 = hashlib.sha256()
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha256.update(chunk)
                    snapshot[rel_path] = sha256.hexdigest()
                except Exception:
                    snapshot[rel_path] = "ERROR_READING"
    return snapshot


def verify_artifacts_snapshots(before: Dict[str, str], after: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    is_changed = False
    details = {}
    all_keys = set(before.keys()).union(set(after.keys()))
    for key in all_keys:
        if key not in before:
            is_changed = True
            details[key] = "ADDED"
        elif key not in after:
            is_changed = True
            details[key] = "DELETED"
        elif before[key] != after[key]:
            is_changed = True
            details[key] = "MODIFIED"
        else:
            details[key] = "UNCHANGED"
    status = "CHANGED — FAILURE" if is_changed else "UNCHANGED"
    return status, details


def get_experiment_overrides(name: str, best_pruning: float = 0.05) -> Dict[str, Any]:
    """Define the overrides dictionary for each experiment name."""
    if name == "fp32_baseline":
        return {
            "quantization.default_precision": "FP32",
            "compression.enabled_types": [],
        }
    elif name == "int8_only":
        return {
            "quantization.default_precision": "INT8",
            "compression.enabled_types": [],
        }
    elif name == "int8_pruning_5":
        return {
            "quantization.default_precision": "INT8",
            "compression.enabled_types": ["PRUNING"],
            "compression.pruning_sparsity": 0.05,
        }
    elif name == "int8_pruning_10":
        return {
            "quantization.default_precision": "INT8",
            "compression.enabled_types": ["PRUNING"],
            "compression.pruning_sparsity": 0.10,
        }
    elif name == "int8_pruning_15":
        return {
            "quantization.default_precision": "INT8",
            "compression.enabled_types": ["PRUNING"],
            "compression.pruning_sparsity": 0.15,
        }
    elif name == "int8_pruning_20":
        return {
            "quantization.default_precision": "INT8",
            "compression.enabled_types": ["PRUNING"],
            "compression.pruning_sparsity": 0.20,
        }
    elif name == "int8_pruning_30":
        return {
            "quantization.default_precision": "INT8",
            "compression.enabled_types": ["PRUNING"],
            "compression.pruning_sparsity": 0.30,
        }
    elif name == "int8_clustering":
        return {
            "quantization.default_precision": "INT8",
            "compression.enabled_types": ["WEIGHT_CLUSTERING"],
            "compression.clustering_clusters": 256,
        }
    elif name == "int8_best_pruning_clustering":
        return {
            "quantization.default_precision": "INT8",
            "compression.enabled_types": ["PRUNING", "WEIGHT_CLUSTERING"],
            "compression.pruning_sparsity": best_pruning,
            "compression.clustering_clusters": 256,
        }
    else:
        raise ValueError(f"Unknown experiment name: {name}")


def validate_and_hash_artifact(file_path: str, format_type: str) -> Tuple[bool, str, str]:
    """Validate model file and return validation_status, sha256_hash."""
    if not os.path.exists(file_path):
        return False, "NOT GENERATED", "NOT GENERATED"
        
    size_bytes = os.path.getsize(file_path)
    if size_bytes == 0:
        return False, "EMPTY_FILE", "NOT GENERATED"
        
    try:
        if format_type == "ONNX":
            import onnx
            model = onnx.load(file_path)
            onnx.checker.check_model(model)
            import onnxruntime as ort
            _ = ort.InferenceSession(file_path)
        elif format_type == "TFLite":
            import tensorflow as tf
            import contextlib
            with contextlib.redirect_stdout(None):
                interpreter = tf.lite.Interpreter(model_path=file_path)
                interpreter.allocate_tensors()
        else:
            raise ValueError(f"Unknown format: {format_type}")
        
        # Calculate SHA256
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return True, "VALIDATED", sha256.hexdigest()
        
    except Exception as e:
        return False, f"VALIDATION_FAILED: {str(e)}", "NOT GENERATED"


def extract_sensitivity_info_from_logs(name: str, start_offset: int = 0) -> List[Dict[str, Any]]:
    log_path = os.path.join(PROJECT_ROOT, "src", "logs", "composition-root", "run.log")
    results = []
    if not os.path.exists(log_path):
        return results
    import re
    with open(log_path, "r", encoding="utf-8") as f:
        if start_offset > 0:
            f.seek(start_offset)
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("stage_name") == "sensitivity_analyzer":
                    msg = data.get("message", "")
                    match = re.search(r"Layer '([^']+)' flagged as quantization-sensitive \(score=([\d.]+) > threshold=([\d.]+)\)", msg)
                    if match:
                        results.append({
                            "layer": match.group(1),
                            "score": float(match.group(2)),
                            "threshold": float(match.group(3))
                        })
            except Exception:
                pass
    return results


def run_experiment(
    name: str,
    overrides: Dict[str, Any],
    model_path: str,
    dataset: RealDatasetAdapter,
    eval_config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f"RUNNING EXPERIMENT: {name}")
    print(f"==========================================")
    
    isolated_dir_name = EXP_DIR_MAP[name]
    exp_output_subdir = f"outputs/{isolated_dir_name}"
    
    # Inject output dir override directly into overrides
    overrides = copy.deepcopy(overrides)
    overrides["exporter.output_dir"] = exp_output_subdir
    print(f"Overrides: {json.dumps(overrides)}")

    src_dir = os.path.join(PROJECT_ROOT, "src")
    
    # Execute main.py subprocess
    cmd = [
        sys.executable,
        "main.py",
        "--model", os.path.relpath(model_path, src_dir),
        "--hardware", "raspberrypi5",
        "--runtime", "tflite-runtime",
        "--output", exp_output_subdir,
        "--run-id-prefix", f"ablation_{name}",
        "--config-overrides", json.dumps(overrides),
        "--calibration-dataset", "../datasets/calibration/calibration_manifest.json"
    ]

    print(f"Command: {' '.join(cmd)}")
    start_time = time.monotonic()
    
    log_path = os.path.join(PROJECT_ROOT, "src", "logs", "composition-root", "run.log")
    log_start_offset = os.path.getsize(log_path) if os.path.exists(log_path) else 0

    res = subprocess.run(
        cmd,
        cwd=src_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    duration = time.monotonic() - start_time
    print(f"Subprocess finished in {duration:.2f}s with return code {res.returncode}")

    exp_report_dir = os.path.join(output_dir, name)
    os.makedirs(exp_report_dir, exist_ok=True)

    if res.returncode != 0:
        # Write failure logs
        with open(os.path.join(exp_report_dir, "stdout.log"), "w", encoding="utf-8") as f:
            f.write(res.stdout)
        with open(os.path.join(exp_report_dir, "stderr.log"), "w", encoding="utf-8") as f:
            f.write(res.stderr)
        
        failure_info = {
            "experiment": name,
            "status": "FAILED",
            "returncode": res.returncode,
            "error_msg": "Subprocess exited with non-zero code.",
            "overrides": overrides,
        }
        with open(os.path.join(exp_report_dir, "failure.json"), "w", encoding="utf-8") as f:
            json.dump(failure_info, f, indent=2)
            
        raise RuntimeError(f"Experiment {name} failed compiling: exit code {res.returncode}")

    # Resolve output artifact locations (these are written directly under the isolated subdirectory)
    onnx_path = os.path.join(src_dir, exp_output_subdir, "raspberrypi5", "model.onnx")
    tflite_path = os.path.join(src_dir, exp_output_subdir, "raspberrypi5", "model.tflite")

    # Validate generated artifacts
    onnx_ok, onnx_status, onnx_hash = validate_and_hash_artifact(onnx_path, "ONNX")
    if not onnx_ok:
        raise RuntimeError(f"Optimized ONNX artifact failed validation: {onnx_status} at {onnx_path}")
        
    tflite_ok, tflite_status, tflite_hash = validate_and_hash_artifact(tflite_path, "TFLite")
    eval_tflite_path = tflite_path if tflite_ok else "NOT GENERATED"

    # Run real inference evaluation using the exact verified paths
    print("Executing Real Inference Evaluation...")
    evaluator = RealInferenceEvaluator(
        fp32_model_path=model_path,
        onnx_model_path=onnx_path,
        tflite_model_path=eval_tflite_path,
        dataset=dataset,
        config=eval_config,
    )

    acc_results, class_results, num_results = evaluator.evaluate_accuracy_and_numerical_metrics()
    size_results = evaluator.audit_model_size()
    
    # Run at least 100 benchmark runs after 10 warm-up runs
    latency_results = evaluator.benchmark_latency(warmup_runs=10, benchmark_runs=100)
    memory_results = evaluator.benchmark_memory()

    # Re-verify artifact hash after evaluation to ensure no cross-run mutation
    onnx_ok_post, _, onnx_hash_post = validate_and_hash_artifact(onnx_path, "ONNX")
    if not onnx_ok_post or onnx_hash != onnx_hash_post:
        raise RuntimeError("ONNX artifact hash mutated during evaluation!")
    if tflite_ok:
        tflite_ok_post, _, tflite_hash_post = validate_and_hash_artifact(tflite_path, "TFLite")
        if not tflite_ok_post or tflite_hash != tflite_hash_post:
            raise RuntimeError("TFLite artifact hash mutated during evaluation!")

    # Record artifact hash and verification details
    artifact_details = {
        "onnx": {
            "path": onnx_path,
            "size_bytes": os.path.getsize(onnx_path) if (onnx_ok and os.path.exists(onnx_path)) else 0,
            "sha256": onnx_hash,
            "validation_status": onnx_status
        },
        "tflite": {
            "path": tflite_path if tflite_ok else "NOT GENERATED",
            "size_bytes": os.path.getsize(tflite_path) if (tflite_ok and os.path.exists(tflite_path)) else "NOT GENERATED",
            "sha256": tflite_hash,
            "validation_status": tflite_status
        }
    }

    # Extract sensitivity reports if any
    sens_layers = extract_sensitivity_info_from_logs(name, start_offset=log_start_offset)

    # Save detailed files
    with open(os.path.join(exp_report_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)
    with open(os.path.join(exp_report_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(class_results, f, indent=2)
    with open(os.path.join(exp_report_dir, "size.json"), "w", encoding="utf-8") as f:
        json.dump(size_results, f, indent=2)
    with open(os.path.join(exp_report_dir, "numerical.json"), "w", encoding="utf-8") as f:
        json.dump(num_results, f, indent=2)
    with open(os.path.join(exp_report_dir, "performance.json"), "w", encoding="utf-8") as f:
        json.dump(latency_results, f, indent=2)

    # Compile summary.json
    summary = {
        "name": name,
        "status": "SUCCESS",
        "overrides": overrides,
        "accuracy": acc_results,
        "size": size_results,
        "performance": latency_results,
        "memory": memory_results,
        "numerical": num_results,
        "artifacts": artifact_details,
        "sensitivity": sens_layers
    }
    with open(os.path.join(exp_report_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save experiment summary.md
    with open(os.path.join(exp_report_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Experiment Summary — {name}\n\n")
        f.write(f"- **Overall Accuracy**: {acc_results.get('tflite_accuracy', 'NOT GENERATED')}\n")
        f.write(f"- **TFLite File Size**: {size_results.get('tflite_size_bytes', 'NOT GENERATED')}\n")
        f.write(f"- **TFLite SHA256**: {tflite_hash}\n")
        f.write(f"- **Throughput**: {latency_results.get('tflite', {}).get('throughput_ips', 'FAILED')}\n")
        tflite_sim = num_results.get('fp32_vs_tflite')
        sim_val = tflite_sim.get('cosine_similarity', 'NOT GENERATED') if isinstance(tflite_sim, dict) else 'NOT GENERATED'
        f.write(f"- **Cosine Similarity vs FP32**: {sim_val}\n")

    print(f"SUCCESS: Experiment {name} evaluated.")
    return summary


def select_best_pruning_sparsity(results: Dict[str, Any]) -> float:
    """Select best pruning level based on accuracy trade-off.
    INT8-only accuracy is the control.
    We want to find the highest pruning sparsity that keeps the accuracy drop <= 2.0 pp vs INT8-only.
    If none, select the one with the highest accuracy.
    """
    int8_acc = results["int8_only"]["accuracy"].get("tflite_accuracy", 0.0)
    candidates = []
    for p_level in [5, 10, 15, 20, 30]:
        exp_name = f"int8_pruning_{p_level}"
        if exp_name in results and results[exp_name].get("status") == "SUCCESS":
            acc = results[exp_name]["accuracy"].get("tflite_accuracy", 0.0)
            drop = (int8_acc - acc) * 100.0
            candidates.append((p_level, acc, drop))
            
    if not candidates:
        return 0.05
        
    good_candidates = [c for c in candidates if c[2] <= 2.0]
    if good_candidates:
        best = max(good_candidates, key=lambda x: x[0])
    else:
        best = max(candidates, key=lambda x: x[1])
        
    return best[0] / 100.0


def resolve_best_pruning_from_reports(output_dir: str) -> float:
    results = {}
    for name in ["int8_only", "int8_pruning_5", "int8_pruning_10", "int8_pruning_15", "int8_pruning_20", "int8_pruning_30"]:
        path = os.path.join(output_dir, name, "summary.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results[name] = json.load(f)
            except Exception:
                pass
    if "int8_only" not in results:
        return 0.05
    return select_best_pruning_sparsity(results)


def main() -> int:
    args = parse_args()

    # 1. Paths Resolution
    config_dir = os.path.join(PROJECT_ROOT, "src", "config")
    exports_dir = os.path.join(PROJECT_ROOT, "src", "outputs", "exports")
    os.makedirs(args.output, exist_ok=True)

    # 2. Production Config & Artifacts Snapshots Before
    print("Snapshotting production configuration & artifacts...")
    config_before = load_config_snapshot(config_dir)
    artifacts_before = get_production_artifacts_snapshot(exports_dir)
    
    with open(os.path.join(args.output, "production_config_before.json"), "w", encoding="utf-8") as f:
        json.dump(config_before, f, indent=2)
    with open(os.path.join(args.output, "production_artifacts_before.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts_before, f, indent=2)

    # 3. Setup Dataset Adapter
    if not os.path.exists(args.eval_config):
        print(f"CRITICAL ERROR: Eval configuration not found: {args.eval_config}")
        return 1
    with open(args.eval_config, "r", encoding="utf-8") as f:
        eval_config = json.load(f)

    print("Loading test dataset...")
    try:
        sess = ort.InferenceSession(args.model)
        model_input_shape = sess.get_inputs()[0].shape
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load reference FP32 ONNX model: {e}")
        return 1

    dataset = RealDatasetAdapter(
        dataset_path=args.dataset,
        class_mapping=eval_config["class_mapping"],
        preprocessing_mode=eval_config["preprocessing"],
        input_shape=tuple(model_input_shape),
    )
    print(f"Loaded {len(dataset)} valid samples.")

    # 4. Resolve Experiment List
    if args.all:
        experiments = [
            "fp32_baseline",
            "int8_only",
            "int8_pruning_5",
            "int8_pruning_10",
            "int8_pruning_15",
            "int8_pruning_20",
            "int8_pruning_30",
            "int8_clustering",
        ]
    elif args.experiment:
        experiments = [args.experiment]
    else:
        print("ERROR: Please specify --experiment <name> or --all.")
        return 1

    results = {}
    failures = {}
    best_pruning = 0.05

    for exp_name in experiments:
        overrides = get_experiment_overrides(exp_name)
        try:
            summary = run_experiment(
                name=exp_name,
                overrides=overrides,
                model_path=args.model,
                dataset=dataset,
                eval_config=eval_config,
                output_dir=args.output,
            )
            results[exp_name] = summary
        except Exception as exc:
            print(f"FAILURE on experiment {exp_name}: {exc}")
            traceback.print_exc()
            failures[exp_name] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            exp_report_dir = os.path.join(args.output, exp_name)
            os.makedirs(exp_report_dir, exist_ok=True)
            with open(os.path.join(exp_report_dir, "failure.json"), "w", encoding="utf-8") as f:
                json.dump({"experiment": exp_name, "status": "FAILED", "error": str(exc)}, f, indent=2)
            with open(os.path.join(exp_report_dir, "stderr.log"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            
            if not args.continue_on_error:
                print("Stopping ablation due to experiment failure.")
                break

    # If --all and previous sweep completed, run best pruning + clustering dynamically
    if args.all and not failures:
        best_pruning = select_best_pruning_sparsity(results)
        print(f"\nDYNAMICALLY DETERMINED BEST PRUNING SPARSITY: {best_pruning * 100:.1f}%")
        exp_name = "int8_best_pruning_clustering"
        overrides = get_experiment_overrides(exp_name, best_pruning=best_pruning)
        try:
            summary = run_experiment(
                name=exp_name,
                overrides=overrides,
                model_path=args.model,
                dataset=dataset,
                eval_config=eval_config,
                output_dir=args.output,
            )
            results[exp_name] = summary
        except Exception as exc:
            print(f"FAILURE on experiment {exp_name}: {exc}")
            failures[exp_name] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
    elif args.experiment == "int8_best_pruning_clustering":
        best_pruning = resolve_best_pruning_from_reports(args.output)
        print(f"\nRESOLVED BEST PRUNING SPARSITY: {best_pruning * 100:.1f}%")
        overrides = get_experiment_overrides(args.experiment, best_pruning=best_pruning)
        try:
            summary = run_experiment(
                name=args.experiment,
                overrides=overrides,
                model_path=args.model,
                dataset=dataset,
                eval_config=eval_config,
                output_dir=args.output,
            )
            results[args.experiment] = summary
        except Exception as exc:
            print(f"FAILURE on experiment {args.experiment}: {exc}")
            failures[args.experiment] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    # 5. Production Config & Artifacts Snapshots After & Verify
    print("\nVerifying production configuration remains unchanged...")
    config_after = load_config_snapshot(config_dir)
    artifacts_after = get_production_artifacts_snapshot(exports_dir)
    
    with open(os.path.join(args.output, "production_config_after.json"), "w", encoding="utf-8") as f:
        json.dump(config_after, f, indent=2)
    with open(os.path.join(args.output, "production_artifacts_after.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts_after, f, indent=2)

    verify_status, verify_details = verify_config_snapshots(config_before, config_after)
    verification_info = {
        "status": verify_status,
        "details": verify_details,
    }
    with open(os.path.join(args.output, "production_config_verification.json"), "w", encoding="utf-8") as f:
        json.dump(verification_info, f, indent=2)

    art_status, art_details = verify_artifacts_snapshots(artifacts_before, artifacts_after)
    art_verification_info = {
        "status": art_status,
        "details": art_details,
    }
    with open(os.path.join(args.output, "production_artifacts_verification.json"), "w", encoding="utf-8") as f:
        json.dump(art_verification_info, f, indent=2)

    print(f"Production configuration status: {verify_status}")
    print(f"Production artifacts status: {art_status}")

    # 6. Generate Aggregated Final Reports (Only if we have results)
    if results:
        generate_aggregated_reports(results, failures, args.output, verify_status, art_status, best_pruning)

    return 0 if not failures else 1


def generate_aggregated_reports(
    results: Dict[str, Dict[str, Any]],
    failures: Dict[str, Dict[str, Any]],
    output_dir: str,
    verify_status: str,
    art_status: str,
    best_pruning: float,
) -> None:
    # 1. Save ablation_summary.json
    aggregated = {
        "results": results,
        "failures": {k: {"status": "FAILED", "error": v["error"]} for k, v in failures.items()},
        "production_config_status": verify_status,
        "production_artifacts_status": art_status,
        "best_pruning_sparsity": best_pruning,
    }
    with open(os.path.join(output_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    # Helper helper to extract metrics cleanly
    def get_lat_fps_sim(summary):
        lat = "FAILED"
        fps = "FAILED"
        sim = "NOT GENERATED"
        
        # Check if tflite was generated
        has_tflite = summary["size"].get("tflite_size_bytes") != "NOT GENERATED"
        perf = summary.get("performance", {})
        
        if has_tflite:
            tflite_perf = perf.get("tflite", {})
            if tflite_perf.get("benchmark_status") == "PASS":
                lat = tflite_perf.get("mean_ms", "FAILED")
                fps = tflite_perf.get("throughput_ips", "FAILED")
            sim = summary.get("numerical", {}).get("fp32_vs_tflite", {}).get("cosine_similarity", "NOT GENERATED")
        else:
            onnx_perf = perf.get("onnx", {})
            if onnx_perf.get("benchmark_status") == "PASS":
                lat = onnx_perf.get("mean_ms", "FAILED")
                fps = onnx_perf.get("throughput_ips", "FAILED")
            sim = summary.get("numerical", {}).get("fp32_vs_onnx", {}).get("cosine_similarity", "NOT GENERATED")
            
        return lat, fps, sim

    # 2. Save ablation_comparison.csv
    import csv
    csv_path = os.path.join(output_dir, "ablation_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment",
            "Status",
            "Accuracy",
            "Accuracy_Delta_PP",
            "Size_MB",
            "Parameter_Bytes",
            "Compression_Ratio",
            "Latency_MS",
            "Throughput_FPS",
            "Cosine_Similarity",
        ])
        
        fp32_acc = results.get("fp32_baseline", {}).get("accuracy", {}).get("tflite_accuracy", 0.0)
        if isinstance(fp32_acc, str):
            fp32_acc = 0.0
        
        for name in [
            "fp32_baseline",
            "int8_only",
            "int8_pruning_5",
            "int8_pruning_10",
            "int8_pruning_15",
            "int8_pruning_20",
            "int8_pruning_30",
            "int8_clustering",
            "int8_best_pruning_clustering",
        ]:
            if name in results:
                summary = results[name]
                acc = summary["accuracy"].get("tflite_accuracy", 0.0)
                
                # Delta calculations
                if acc == "NOT GENERATED":
                    acc_str = "NOT GENERATED"
                    delta_str = "NOT GENERATED"
                else:
                    acc_str = f"{acc * 100:.2f}%"
                    delta = (acc - fp32_acc) * 100.0 if name != "fp32_baseline" else 0.0
                    delta_str = f"{delta:+.2f} pp"
                
                # Size details
                has_tflite = summary["size"].get("tflite_size_bytes") != "NOT GENERATED"
                if has_tflite:
                    size_bytes = summary["size"].get("tflite_size_bytes", 0)
                    param_bytes = summary["size"].get("tflite_parameter_bytes", 0)
                    comp_ratio = summary["size"].get("fp32_to_tflite_compression_ratio", 1.0)
                else:
                    size_bytes = summary["size"].get("optimized_onnx_size_bytes", 0)
                    param_bytes = summary["size"].get("optimized_onnx_parameter_bytes", 0)
                    comp_ratio = summary["size"].get("fp32_to_onnx_compression_ratio", 1.0)
                
                size_mb = f"{size_bytes / (1024 * 1024):.2f}"
                comp_ratio_str = f"{comp_ratio:.2f}x"
                
                lat, fps, sim = get_lat_fps_sim(summary)
                lat_str = f"{lat:.2f}" if isinstance(lat, (int, float)) else str(lat)
                fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
                sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
                
                writer.writerow([
                    name,
                    "SUCCESS",
                    acc_str,
                    delta_str,
                    size_mb,
                    param_bytes,
                    comp_ratio_str,
                    lat_str,
                    fps_str,
                    sim_str,
                ])
            elif name in failures:
                writer.writerow([name, "FAILED", "", "", "", "", "", "", "", ""])

    # 3. Save pruning_sweep.csv
    sweep_csv_path = os.path.join(output_dir, "pruning_sweep.csv")
    with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sparsity",
            "Accuracy",
            "Accuracy_Delta_vs_FP32_PP",
            "Accuracy_Delta_vs_INT8_PP",
            "Size_MB",
            "Parameter_Bytes",
            "Compression_Ratio",
            "Latency_MS",
            "Throughput_FPS",
            "Cosine_Similarity",
            "MAE",
            "RMSE",
            "Prediction_Agreement",
        ])
        
        fp32_acc = results.get("fp32_baseline", {}).get("accuracy", {}).get("tflite_accuracy", 0.0)
        int8_acc = results.get("int8_only", {}).get("accuracy", {}).get("tflite_accuracy", 0.0)
        
        sweep_mapping = [
            ("0%", "int8_only"),
            ("5%", "int8_pruning_5"),
            ("10%", "int8_pruning_10"),
            ("15%", "int8_pruning_15"),
            ("20%", "int8_pruning_20"),
            ("30%", "int8_pruning_30"),
        ]
        for pct, name in sweep_mapping:
            if name in results:
                summary = results[name]
                acc = summary["accuracy"].get("tflite_accuracy", 0.0)
                
                if acc == "NOT GENERATED":
                    acc_str = "NOT GENERATED"
                    delta_fp32 = "NOT GENERATED"
                    delta_int8 = "NOT GENERATED"
                else:
                    acc_str = f"{acc * 100:.2f}%"
                    delta_fp32 = f"{(acc - fp32_acc) * 100.0:+.2f} pp" if isinstance(fp32_acc, (int, float)) else "N/A"
                    delta_int8 = f"{(acc - int8_acc) * 100.0:+.2f} pp" if isinstance(int8_acc, (int, float)) else "N/A"
                    
                has_tflite = summary["size"].get("tflite_size_bytes") != "NOT GENERATED"
                if has_tflite:
                    size_bytes = summary["size"].get("tflite_size_bytes", 0)
                    param_bytes = summary["size"].get("tflite_parameter_bytes", 0)
                    comp_ratio = summary["size"].get("fp32_to_tflite_compression_ratio", 1.0)
                    tflite_vs_fp32 = summary.get("numerical", {}).get("fp32_vs_tflite", {})
                    mae = tflite_vs_fp32.get("mean_absolute_error", "N/A")
                    rmse = tflite_vs_fp32.get("rmse", "N/A")
                    agree = tflite_vs_fp32.get("prediction_agreement", "N/A")
                else:
                    size_bytes = summary["size"].get("optimized_onnx_size_bytes", 0)
                    param_bytes = summary["size"].get("optimized_onnx_parameter_bytes", 0)
                    comp_ratio = summary["size"].get("fp32_to_onnx_compression_ratio", 1.0)
                    onnx_vs_fp32 = summary.get("numerical", {}).get("fp32_vs_onnx", {})
                    mae = onnx_vs_fp32.get("mean_absolute_error", "N/A")
                    rmse = onnx_vs_fp32.get("rmse", "N/A")
                    agree = onnx_vs_fp32.get("prediction_agreement", "N/A")
                    
                size_mb = f"{size_bytes / (1024 * 1024):.2f}"
                comp_ratio_str = f"{comp_ratio:.2f}x"
                
                lat, fps, sim = get_lat_fps_sim(summary)
                lat_str = f"{lat:.2f}" if isinstance(lat, (int, float)) else str(lat)
                fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
                sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
                mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
                rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
                agree_str = f"{agree * 100:.1f}%" if isinstance(agree, (int, float)) else str(agree)
                
                writer.writerow([
                    pct,
                    acc_str,
                    delta_fp32,
                    delta_int8,
                    size_mb,
                    param_bytes,
                    comp_ratio_str,
                    lat_str,
                    fps_str,
                    sim_str,
                    mae_str,
                    rmse_str,
                    agree_str
                ])

    # 4. Save ablation_summary.md
    md_path = os.path.join(output_dir, "ablation_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# UAQE Accuracy Ablation & Compression Trade-Off Report\n\n")
        
        f.write("## 1. Experimental Setup\n")
        f.write("- **Hardware Target Profile**: Raspberry Pi 5\n")
        f.write("- **Execution Environment**: Local Windows Host (Inference Emulation)\n")
        f.write("- **Software Compiler Version**: UAQE v1.0\n")
        f.write("- **Run Timestamp**: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        f.write("## 2. Dataset & Preprocessing Information\n")
        f.write("- **Dataset**: Labeled Hackathon Test Dataset (`datasets/hackathon_test_dataset`)\n")
        f.write("- **Total Image Count**: 296 valid images\n")
        f.write("- **Preprocessing**: Normalized to `[0.0, 1.0]` (RGB layout NCHW, bilinear resize to `(128, 128)`)\n\n")

        f.write("## 3. Configuration Matrix & Overrides\n")
        f.write("| Configuration | DTO Dotted Overrides |\n")
        f.write("|---|---|\n")
        f.write("| **FP32** | `default_precision=FP32`, no compression |\n")
        f.write("| **INT8** | `default_precision=INT8`, no compression |\n")
        for p_level in [5, 10, 15, 20, 30]:
            f.write(f"| **INT8 + {p_level}% pruning** | `default_precision=INT8`, `enabled_types=[PRUNING]`, `pruning_sparsity={p_level/100:.2f}` |\n")
        f.write("| **INT8 + clustering** | `default_precision=INT8`, `enabled_types=[WEIGHT_CLUSTERING]`, `clustering_clusters=256` |\n")
        f.write(f"| **INT8 + best pruning + clustering** | `default_precision=INT8`, `enabled_types=[PRUNING, WEIGHT_CLUSTERING]`, `pruning_sparsity={best_pruning:.2f}`, `clustering_clusters=256` |\n\n")

        f.write("## 4. Overall Accuracy & Performance Comparison Table\n\n")
        f.write("```text\n")
        f.write("Configuration                         Accuracy    ΔAccuracy    Size    Compression    Latency    FPS\n")
        f.write("------------------------------------------------------------------------------------------------------\n")
        
        fp32_acc = results.get("fp32_baseline", {}).get("accuracy", {}).get("tflite_accuracy", 0.0)
        if isinstance(fp32_acc, str):
            fp32_acc = 0.0
            
        mapping_details = [
            ("FP32", "fp32_baseline"),
            ("INT8", "int8_only"),
            ("INT8 + 5% pruning", "int8_pruning_5"),
            ("INT8 + 10% pruning", "int8_pruning_10"),
            ("INT8 + 15% pruning", "int8_pruning_15"),
            ("INT8 + 20% pruning", "int8_pruning_20"),
            ("INT8 + 30% pruning", "int8_pruning_30"),
            ("INT8 + clustering", "int8_clustering"),
            (f"INT8 + best pruning ({best_pruning*100:.0f}%) + clustering", "int8_best_pruning_clustering")
        ]
        
        for label, name in mapping_details:
            if name in results:
                summary = results[name]
                acc = summary["accuracy"].get("tflite_accuracy", 0.0)
                
                if acc == "NOT GENERATED":
                    acc_str = "NOT GENERATED"
                    delta_str = "NOT GENERATED"
                else:
                    acc_str = f"{acc * 100:.2f}%"
                    delta = (acc - fp32_acc) * 100.0 if name != "fp32_baseline" else 0.0
                    delta_str = f"{delta:+.2f} pp"
                    
                has_tflite = summary["size"].get("tflite_size_bytes") != "NOT GENERATED"
                if has_tflite:
                    size_bytes = summary["size"].get("tflite_size_bytes", 0)
                    comp_ratio = summary["size"].get("fp32_to_tflite_compression_ratio", 1.0)
                else:
                    size_bytes = summary["size"].get("optimized_onnx_size_bytes", 0)
                    comp_ratio = summary["size"].get("fp32_to_onnx_compression_ratio", 1.0)
                    
                size_mb_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                comp_ratio_str = f"{comp_ratio:.2f}x"
                
                lat, fps, _ = get_lat_fps_sim(summary)
                lat_str = f"{lat:.2f} ms" if isinstance(lat, (int, float)) else str(lat)
                fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
                
                f.write(f"{label:<38} {acc_str:<11} {delta_str:<12} {size_mb_str:<8} {comp_ratio_str:<14} {lat_str:<10} {fps_str:<5}\n")
            elif name in failures:
                f.write(f"{label:<38} FAILED\n")
                
        f.write("```\n\n")

        f.write("## 5. Accuracy Loss Attribution\n")
        if "int8_only" in results and "int8_pruning_30" in results and "int8_best_pruning_clustering" in results:
            acc_fp32 = fp32_acc
            acc_int8 = results["int8_only"]["accuracy"].get("tflite_accuracy", 0.0)
            acc_prun_30 = results["int8_pruning_30"]["accuracy"].get("tflite_accuracy", 0.0)
            acc_clust = results["int8_clustering"]["accuracy"].get("tflite_accuracy", 0.0) if "int8_clustering" in results else None
            acc_final = results["int8_best_pruning_clustering"]["accuracy"].get("tflite_accuracy", 0.0)
            
            loss_int8 = acc_fp32 - acc_int8
            loss_prun_30 = acc_int8 - acc_prun_30
            
            f.write(f"- **INT8 Quantization Loss**: {loss_int8 * 100:.2f} pp (from FP32 baseline of {acc_fp32 * 100:.2f}% to {acc_int8 * 100:.2f}%)\n")
            f.write(f"- **30% Pruning Incremental Loss**: {loss_prun_30 * 100:.2f} pp (from INT8 baseline to INT8 + 30% Pruning of {acc_prun_30 * 100:.2f}%)\n")
            if acc_clust is not None:
                loss_clust_ind = acc_int8 - acc_clust
                f.write(f"- **Clustering Incremental Loss (Independent)**: {loss_clust_ind * 100:.2f} pp (from INT8 baseline to INT8+Clustering of {acc_clust * 100:.2f}%)\n")
            if acc_final != "NOT GENERATED" and acc_int8 != "NOT GENERATED":
                loss_final = acc_int8 - acc_final
                f.write(f"- **Best Pruning + Clustering stacked loss vs INT8**: {loss_final * 100:.2f} pp (from INT8 baseline to dynamic candidate of {acc_final * 100:.2f}%)\n\n")

        f.write("## 6. Production Safety Verification\n")
        f.write(f"- **Production Configuration Status**: `{verify_status}`\n")
        f.write(f"- **Production Artifacts Integrity status**: `{art_status}`\n\n")
        f.write("> [!IMPORTANT]\n")
        if verify_status == "UNCHANGED" and art_status == "UNCHANGED":
            f.write("> **Verification Passed**: Production environment remained completely unchanged.\n\n")
        else:
            f.write("> **Verification Failed**: Production modifications detected! Verify snapshotted changes.\n\n")

        # Add artifact tracking
        f.write("## 7. Artifact Trackings & Hashes\n")
        f.write("| Experiment | Artifact Path | SHA-256 Hash | Size (Bytes) | Validation |\n")
        f.write("|---|---|---|---|---|\n")
        for label, name in mapping_details:
            if name in results:
                summary = results[name]
                arts = summary.get("artifacts", {})
                tflite_art = arts.get("tflite", {})
                f.write(f"| {label} (TFLite) | `{tflite_art.get('path')}` | `{tflite_art.get('sha256')}` | {tflite_art.get('size_bytes')} | `{tflite_art.get('validation_status')}` |\n")
                onnx_art = arts.get("onnx", {})
                f.write(f"| {label} (ONNX) | `{onnx_art.get('path')}` | `{onnx_art.get('sha256')}` | {onnx_art.get('size_bytes')} | `{onnx_art.get('validation_status')}` |\n")
        f.write("\n")

        # Transparent selection rule
        f.write("## 8. Dynamic Pruning Selection Rule & Best Candidates\n")
        f.write("- **Selection Rule**: The selection algorithm sweeps the pruning configurations (`5%`, `10%`, `15%`, `20%`, `30%`) and selects the highest pruning level that keeps accuracy loss within `2.0 percentage points` compared to the INT8-only control baseline. If all configurations exceed this threshold, the one yielding the highest overall accuracy is selected as the candidate.\n")
        f.write(f"- **Selected Pruning Sparsity**: **{best_pruning * 100:.0f}%**\n\n")
        
        # Sensitivity Analysis Section
        f.write("## 9. Sensitivity Analyzer Logs Details\n")
        f.write("Below are the quantization-sensitive layers detected by the `SensitivityAnalyzer` during the calibration stage:\n\n")
        f.write("| Layer Name | Sensitivity Score | Threshold |\n")
        f.write("|---|---|---|\n")
        sens_data = results.get("int8_only", {}).get("sensitivity", [])
        if not sens_data:
            sens_data = extract_sensitivity_info_from_logs("int8_only")
        if sens_data:
            for item in sens_data[:20]:
                f.write(f"| {item['layer']} | {item['score']:.4f} | {item['threshold']:.4f} |\n")
        else:
            f.write("| No sensitive layers detected or calibration log unavailable. | - | - |\n")
        f.write("\n")

    # 5. Save pruning_sweep.md
    sweep_md_path = os.path.join(output_dir, "pruning_sweep.md")
    with open(sweep_md_path, "w", encoding="utf-8") as f:
        f.write("# Pruning Sensitivity Sweep Report\n\n")
        f.write("## Pruning Sparsity vs. Accuracy & Performance Table\n\n")
        f.write("| Sparsity | Accuracy | Δ vs FP32 | Δ vs INT8 | Size (MB) | Parameters (Bytes) | Latency (ms) | Throughput (FPS) | Cosine Sim | Prediction Agreement |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        
        for pct, name in sweep_mapping:
            if name in results:
                summary = results[name]
                acc = summary["accuracy"].get("tflite_accuracy", 0.0)
                if acc == "NOT GENERATED":
                    acc_str = "NOT GENERATED"
                    delta_fp32 = "NOT GENERATED"
                    delta_int8 = "NOT GENERATED"
                else:
                    acc_str = f"{acc * 100:.2f}%"
                    delta_fp32 = f"{(acc - fp32_acc) * 100.0:+.2f} pp" if isinstance(fp32_acc, (int, float)) else "N/A"
                    delta_int8 = f"{(acc - int8_acc) * 100.0:+.2f} pp" if isinstance(int8_acc, (int, float)) else "N/A"
                    
                has_tflite = summary["size"].get("tflite_size_bytes") != "NOT GENERATED"
                if has_tflite:
                    size_bytes = summary["size"].get("tflite_size_bytes", 0)
                    param_bytes = summary["size"].get("tflite_parameter_bytes", 0)
                    agree = summary.get("numerical", {}).get("fp32_vs_tflite", {}).get("prediction_agreement", 0.0)
                else:
                    size_bytes = summary["size"].get("optimized_onnx_size_bytes", 0)
                    param_bytes = summary["size"].get("optimized_onnx_parameter_bytes", 0)
                    agree = summary.get("numerical", {}).get("fp32_vs_onnx", {}).get("prediction_agreement", 0.0)
                    
                size_mb = f"{size_bytes / (1024 * 1024):.2f} MB"
                
                lat, fps, sim = get_lat_fps_sim(summary)
                lat_str = f"{lat:.2f} ms" if isinstance(lat, (int, float)) else str(lat)
                fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
                sim_str = f"{sim:.4f}" if isinstance(sim, (int, float)) else str(sim)
                agree_str = f"{agree * 100:.1f}%" if isinstance(agree, (int, float)) else str(agree)
                
                f.write(f"| {pct} | {acc_str} | {delta_fp32} | {delta_int8} | {size_mb} | {param_bytes} | {lat_str} | {fps_str} | {sim_str} | {agree_str} |\n")
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
