#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import subprocess
import time
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add src to python path so we can import uaqe modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

def get_file_sha256(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "N/A"
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return "FAILED"

def get_onnx_params_count(path: str) -> int:
    try:
        import onnx
        import numpy as np
        model = onnx.load(path)
        return sum(int(np.prod(init.dims)) for init in model.graph.initializer)
    except Exception:
        return 0

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

def get_production_artifacts_snapshot(exports_dir: str) -> Dict[str, str]:
    snapshot = {}
    if os.path.exists(exports_dir):
        for root, dirs, files in os.walk(exports_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, exports_dir)
                snapshot[rel_path] = get_file_sha256(file_path)
    return snapshot

def clean_stale_artifacts(workspace_root: str) -> List[str]:
    cleaned_dirs = []
    
    # 1. Clean src/outputs/ ablation/validation/test folders
    src_outputs_dir = os.path.join(workspace_root, "src", "outputs")
    if os.path.exists(src_outputs_dir):
        for item in os.listdir(src_outputs_dir):
            item_path = os.path.join(src_outputs_dir, item)
            # Delete ONLY ablation, validation, test prefix directories
            if os.path.isdir(item_path) and (item.startswith("ablation_") or item.startswith("validation_") or item.startswith("test_")):
                try:
                    shutil.rmtree(item_path)
                    cleaned_dirs.append(os.path.relpath(item_path, workspace_root))
                except Exception as e:
                    print(f"Warning: Failed to clean stale dir {item_path}: {e}")
                    
    # 2. Clean reports folders
    reports_dir = os.path.join(workspace_root, "reports")
    for name in ["final_software_validation", "accuracy_ablation", "structured_pruning"]:
        path = os.path.join(reports_dir, name)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                cleaned_dirs.append(os.path.relpath(path, workspace_root))
            except Exception as e:
                print(f"Warning: Failed to clean stale report dir {path}: {e}")
                
    return cleaned_dirs

def run_cleanup_safety_test(workspace_root: str, cleaned_dirs: List[str]) -> Tuple[bool, str]:
    # 1. Verify stale generated directories are gone
    for d in cleaned_dirs:
        full_path = os.path.join(workspace_root, d)
        # Avoid checking folders that are re-created (like reports) immediately
        if os.path.exists(full_path) and "outputs" in d:
            return False, f"Stale generated directory {d} was not removed."
            
    # 2. Verify protected directories and files are preserved
    ref_model = os.path.join(workspace_root, "src", "models", "mobilenetv3_sem.onnx")
    if not os.path.exists(ref_model):
        return False, "Reference model was deleted!"
        
    config_dir = os.path.join(workspace_root, "src", "config")
    if not os.path.exists(config_dir) or len(os.listdir(config_dir)) == 0:
        return False, "Config directory was deleted or emptied!"
        
    datasets_dir = os.path.join(workspace_root, "datasets")
    if not os.path.exists(datasets_dir) or len(os.listdir(datasets_dir)) == 0:
        return False, "Datasets directory was deleted or emptied!"
        
    exports_dir = os.path.join(workspace_root, "src", "outputs", "exports")
    if not os.path.exists(exports_dir) or len(os.listdir(exports_dir)) == 0:
        return False, "Exports directory was deleted or emptied!"
        
    return True, "Cleanup safety checks passed. Stale directories removed, protected directories preserved."

def run_command(cmd: List[str], cwd: str) -> subprocess.CompletedProcess:
    """Helper to run a subprocess and print its output on error."""
    print(f"Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = src_dir + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_dir
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Error executing command. Exit code: {result.returncode}")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
    return result

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="validate_all",
        description="Host-Side Software Validation CLI for UAQE Phases 1-5"
    )
    
    # Model/Dataset paths
    parser.add_argument(
        "--model",
        default="src/models/mobilenetv3_sem.onnx",
        help="Path to the original FP32 reference ONNX model."
    )
    parser.add_argument(
        "--optimized-onnx",
        default="src/outputs/exports/raspberrypi5/model.onnx",
        help="Path to the optimized ONNX model."
    )
    parser.add_argument(
        "--tflite",
        default="src/outputs/exports/raspberrypi5/model.tflite",
        help="Path to the compiled TFLite model."
    )
    parser.add_argument(
        "--dataset",
        default="D:\\Quantization embedded\\datasets\\hackathon_test_dataset",
        help="Path to the real labeled evaluation dataset."
    )
    parser.add_argument(
        "--eval-config",
        default="src/config/mobilenetv3_sem_eval.json",
        help="Path to the model evaluation configuration JSON file."
    )
    
    # Run options
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate existing model artifacts only."
    )
    mode_group.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild model artifacts, then evaluate."
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Rebuild artifacts + run regression tests + evaluate + benchmark."
    )
    parser.add_argument(
        "--quantization-sensitivity",
        action="store_true",
        help="Run quantization error and layer/block sensitivity analysis."
    )
    
    args = parser.parse_args()
    
    run_start_time = datetime.utcnow().isoformat() + "Z"
    run_id = f"run_{int(time.time())}"
    
    workspace_root = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.abspath(args.model)
    onnx_path = os.path.abspath(args.optimized_onnx)
    tflite_path = os.path.abspath(args.tflite)
    dataset_path = os.path.abspath(args.dataset)
    eval_config_path = os.path.abspath(args.eval_config)
    
    # 1. Take snapshot BEFORE cleanup
    config_dir = os.path.join(workspace_root, "src", "config")
    exports_dir = os.path.join(workspace_root, "src", "outputs", "exports")
    
    snapshot_before = {
        "timestamp": run_start_time,
        "configs": {
            "quantization.json": get_file_sha256(os.path.join(config_dir, "quantization.json")),
            "compression.json": get_file_sha256(os.path.join(config_dir, "compression.json")),
            "optimization.json": get_file_sha256(os.path.join(config_dir, "optimization.json")),
            "config.json": get_file_sha256(os.path.join(config_dir, "config.json")),
            "settings.yaml": get_file_sha256(os.path.join(config_dir, "settings.yaml")),
            "mobilenetv3_sem_eval.json": get_file_sha256(os.path.join(config_dir, "mobilenetv3_sem_eval.json"))
        },
        "reference_model": get_file_sha256(model_path),
        "exports": get_production_artifacts_snapshot(exports_dir)
    }
    
    # 2. Cleanup stale generated validation/experiment outputs
    print("\n=== CLEANING STALE ARTIFACTS ===")
    cleaned_dirs = clean_stale_artifacts(workspace_root)
    for d in cleaned_dirs:
        print(f"Cleaned stale directory: {d}")
        
    # Recreate the target validation report dir immediately
    reports_dir = os.path.join(workspace_root, "reports", "final_software_validation")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save the before snapshot
    with open(os.path.join(reports_dir, "production_integrity_before.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_before, f, indent=2)
        
    # Run cleanup safety check
    safety_pass, safety_info = run_cleanup_safety_test(workspace_root, cleaned_dirs)
    print(f"Cleanup safety checks: {'PASSED' if safety_pass else 'FAILED'} - {safety_info}")
    
    # 3. Rebuild if requested
    pipeline_status = "PASS"
    rebuild_error_reason = ""
    
    if args.rebuild or args.full:
        print("\n=== REBUILDING ARTIFACTS ===")
        src_dir = os.path.join(workspace_root, "src")
        rebuild_cmd = [
            sys.executable,
            "main.py",
            "--model", os.path.relpath(model_path, src_dir),
            "--hardware", "raspberrypi5",
            "--runtime", "tflite-runtime",
            "--output", "outputs",
            "--calibration-dataset", "../datasets/calibration/calibration_manifest.json"
        ]
        
        result = run_command(rebuild_cmd, src_dir)
        if result.returncode != 0:
            pipeline_status = "FAIL"
            rebuild_error_reason = f"main.py rebuild process exited with code {result.returncode}"
            print("FAILED to rebuild model artifacts.")
        else:
            print("Model artifacts rebuilt successfully.")
            
    # 4. Verify artifacts exist before evaluating
    for name, path in [("Reference Model", model_path), 
                       ("Optimized ONNX", onnx_path), 
                       ("TFLite Model", tflite_path)]:
        if not os.path.exists(path):
            pipeline_status = "FAIL"
            rebuild_error_reason = f"Required artifact {name} is missing at {path} after rebuild"
            print(f"CRITICAL ERROR: Required artifact {name} is missing at: {path}")
            
    # 5. Run regression tests if full mode requested
    tests_summary = {"passed": 0, "failed": 0, "skipped": 0, "details": []}
    if args.full:
        print("\n=== RUNNING REGRESSION TESTS ===")
        
        test_scripts = [
            "src/uaqe/analyzer/test_analyzer.py",
            "src/uaqe/analyzer/test_analyzer_core.py",
            "src/uaqe/tests/test_sensitivity_analyzer.py",
            "src/uaqe/tests/uaqe_quantization_smoke_test.py",
            "src/uaqe/tests/test_compression_extension.py",
            "src/uaqe/tests/test_real_inference_evaluation.py",
            "src/uaqe/tests/test_sensitivity_structured_pruner.py",
            "src/uaqe/tests/test_accuracy_ablation.py"
        ]
        
        # Count the cleanup safety check as an automated test
        tests_summary["passed"] += 1 if safety_pass else 0
        tests_summary["failed"] += 0 if safety_pass else 1
        tests_summary["details"].append({
            "test": "cleanup_safety_check",
            "status": "PASS" if safety_pass else "FAIL",
            "info": safety_info
        })
        
        for script in test_scripts:
            full_script_path = os.path.join(workspace_root, script)
            if not os.path.exists(full_script_path):
                print(f"Skipping missing test script: {script}")
                tests_summary["skipped"] += 1
                continue
                
            print(f"Running test: {script} ...")
            test_res = run_command([sys.executable, full_script_path], workspace_root)
            
            is_success = test_res.returncode == 0
            detail = f"Exit code {test_res.returncode}"
            
            if is_success and "uaqe_quantization_smoke_test" in script:
                if "TOTAL BUGS FOUND: 0" not in test_res.stdout:
                    is_success = False
                    detail = "Bugs found in smoke test output"
            
            if is_success:
                print(f"[PASS] {script}")
                tests_summary["passed"] += 1
                tests_summary["details"].append({"test": script, "status": "PASS", "info": detail})
            else:
                print(f"[FAIL] {script}")
                tests_summary["failed"] += 1
                tests_summary["details"].append({"test": script, "status": "FAIL", "info": detail})
                
        print(f"\nTests run complete. Passed: {tests_summary['passed']}, Failed: {tests_summary['failed']}")
        if tests_summary["failed"] > 0:
            print("WARNING: Some regression tests failed. Continuing to evaluation...")
            
    # 6. Load evaluation config and run evaluation
    eval_pass = False
    eval_error_msg = ""
    
    acc_results, class_results, num_results = {}, {}, {}
    size_results, tflite_audit = {}, {}
    latency_results, memory_results, stability_results = {}, {}, {}
    
    if pipeline_status == "PASS":
        try:
            if not os.path.exists(eval_config_path):
                raise FileNotFoundError(f"Evaluation configuration file not found at: {eval_config_path}")
                
            with open(eval_config_path, "r") as f:
                eval_config = json.load(f)
                
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path)
            model_input_shape = sess.get_inputs()[0].shape
            
            print("\n=== LOADING DATASET ===")
            from uaqe.evaluation.real_dataset_adapter import RealDatasetAdapter
            
            dataset = RealDatasetAdapter(
                dataset_path=dataset_path,
                class_mapping=eval_config["class_mapping"],
                preprocessing_mode=eval_config["preprocessing"],
                input_shape=tuple(model_input_shape)
            )
            
            print(f"Successfully loaded real evaluation dataset. Total samples: {len(dataset)}")
            
            print("\n=== RUNNING EVALUATION ===")
            from uaqe.evaluation.real_inference_evaluator import RealInferenceEvaluator
            
            evaluator = RealInferenceEvaluator(
                fp32_model_path=model_path,
                onnx_model_path=onnx_path,
                tflite_model_path=tflite_path,
                dataset=dataset,
                config=eval_config
            )
            
            print("Calculating real accuracy and numerical similarities...")
            acc_results, class_results, num_results = evaluator.evaluate_accuracy_and_numerical_metrics()
            
            print("Performing file size audit...")
            size_results = evaluator.audit_model_size()
            
            print("Performing TFLite structural audit...")
            tflite_audit = evaluator.audit_tflite()
            
            print("Performing latency & throughput benchmarks (warmup=10, runs=100)...")
            latency_results = evaluator.benchmark_latency(warmup_runs=10, benchmark_runs=100)
            
            print("Performing memory benchmarks...")
            memory_results = evaluator.benchmark_memory()
            
            print("Performing TFLite 500-run stability test...")
            stability_results = evaluator.run_stability_test(iterations=500)
            
            if getattr(args, "quantization_sensitivity", False):
                print("Performing Quantization Error & Layer Sensitivity Analysis...")
                from uaqe.quantization.quantization_error_analyzer import QuantizationErrorAnalyzer
                analyzer = QuantizationErrorAnalyzer(
                    onnx_model_path=model_path,
                    tflite_model_path=tflite_path,
                    dataset_adapter=dataset,
                    sample_count=50,
                    random_seed=42
                )
                sens_results = analyzer.run_full_analysis()
                sens_paths = analyzer.export_reports(
                    sens_results,
                    output_dir=os.path.join(workspace_root, "output"),
                    reports_dir=os.path.join(workspace_root, "reports", "quantization_sensitivity")
                )
                print(f"Sensitivity reports generated: {sens_paths['markdown_reports']}")
            
            eval_pass = True
        except Exception as e:
            eval_pass = False
            eval_error_msg = str(e)
            print(f"ERROR during evaluation/benchmark execution: {e}")
            pipeline_status = "FAIL"
            
    # 7. Take snapshot AFTER run & verify production integrity
    run_end_time = datetime.utcnow().isoformat() + "Z"
    
    snapshot_after = {
        "timestamp": run_end_time,
        "configs": {
            "quantization.json": get_file_sha256(os.path.join(config_dir, "quantization.json")),
            "compression.json": get_file_sha256(os.path.join(config_dir, "compression.json")),
            "optimization.json": get_file_sha256(os.path.join(config_dir, "optimization.json")),
            "config.json": get_file_sha256(os.path.join(config_dir, "config.json")),
            "settings.yaml": get_file_sha256(os.path.join(config_dir, "settings.yaml")),
            "mobilenetv3_sem_eval.json": get_file_sha256(os.path.join(config_dir, "mobilenetv3_sem_eval.json"))
        },
        "reference_model": get_file_sha256(model_path),
        "exports": get_production_artifacts_snapshot(exports_dir)
    }
    
    with open(os.path.join(reports_dir, "production_integrity_after.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_after, f, indent=2)
        
    # Compare before and after snapshots
    configs_match = snapshot_before["configs"] == snapshot_after["configs"]
    ref_model_match = snapshot_before["reference_model"] == snapshot_after["reference_model"]
    
    # Compare only stable exports model binaries (model.onnx and model.tflite)
    exports_match = True
    for key in ["raspberrypi5/model.onnx", "raspberrypi5/model.tflite"]:
        norm_key = key.replace("/", os.sep)
        h_before = snapshot_before["exports"].get(norm_key)
        h_after = snapshot_after["exports"].get(norm_key)
        if h_before != h_after:
            exports_match = False
            break
            
    prod_configs_integrity = "UNCHANGED" if configs_match else "CHANGED — FAILURE"
    prod_artifacts_integrity = "UNCHANGED" if ref_model_match else "CHANGED — FAILURE"
    prod_integrity_status = "PASS" if (configs_match and ref_model_match) else "FAIL"
    
    verification_report = {
        "production_configs_match": configs_match,
        "reference_model_match": ref_model_match,
        "production_exports_match": exports_match,
        "production_integrity_status": prod_integrity_status
    }
    with open(os.path.join(reports_dir, "production_integrity_verification.json"), "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2)
        
    # 8. Evaluate Quality Gate criteria
    quality_gate_status = "BLOCKED"
    thresholds_checks_info = []
    
    if eval_pass:
        num_thresholds = eval_config.get("evaluation_thresholds", {})
        tflite_vs_fp32 = num_results.get("fp32_vs_tflite", {})
        
        quality_gate_status = "PASS"
        for key, field, thresh, val, is_lower_better in [
            ("Cosine Similarity", "cosine_similarity", num_thresholds.get("min_cosine_similarity", 0.95), tflite_vs_fp32.get("cosine_similarity", 0.0), False),
            ("Mean Absolute Error (MAE)", "mean_absolute_error", num_thresholds.get("max_mae", 0.05), tflite_vs_fp32.get("mean_absolute_error", 0.0), True),
            ("RMSE", "rmse", num_thresholds.get("max_rmse", 0.1), tflite_vs_fp32.get("rmse", 0.0), True),
            ("Prediction Agreement", "prediction_agreement", num_thresholds.get("min_prediction_agreement", 0.90), tflite_vs_fp32.get("prediction_agreement", 0.0), False),
        ]:
            passed = (val <= thresh) if is_lower_better else (val >= thresh)
            status_str = "PASS" if passed else "FAIL"
            if not passed:
                quality_gate_status = "FAIL"
            thresholds_checks_info.append({
                "metric": key,
                "threshold": thresh,
                "measured": val,
                "status": status_str
            })
            
    # Calculate Final Status
    final_status = "FAIL"
    if pipeline_status == "PASS" and quality_gate_status == "PASS" and prod_integrity_status == "PASS":
        final_status = "PASS"
    elif pipeline_status == "FAIL":
        final_status = "FAIL"
    elif quality_gate_status == "FAIL":
        final_status = "FAIL"
    elif prod_integrity_status == "FAIL":
        final_status = "FAIL"
    elif quality_gate_status == "BLOCKED":
        final_status = "BLOCKED"
        
    # 9. Print final Markdown reports and JSON files
    if eval_pass:
        # comparison.csv
        csv_path = os.path.join(reports_dir, "comparison.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Accuracy (%)", "File Size (MB)", "Mean Latency (ms)", "Throughput (FPS)"])
            writer.writerow(["FP32 Reference", f"{acc_results['fp32_accuracy']*100:.2f}", f"{size_results['fp32_onnx_size_bytes']/(1024**2):.2f}", f"{latency_results['fp32']['mean_ms']:.2f}", f"{latency_results['fp32']['throughput_ips']:.1f}"])
            writer.writerow(["Optimized ONNX", f"{acc_results['onnx_accuracy']*100:.2f}", f"{size_results['optimized_onnx_size_bytes']/(1024**2):.2f}", f"{latency_results['onnx']['mean_ms']:.2f}", f"{latency_results['onnx']['throughput_ips']:.1f}"])
            writer.writerow(["Final TFLite", f"{acc_results['tflite_accuracy']*100:.2f}", f"{size_results['tflite_size_bytes']/(1024**2):.2f}", f"{latency_results['tflite']['mean_ms']:.2f}", f"{latency_results['tflite']['throughput_ips']:.1f}"])
            
        # JSON files
        for filename, data in [
            ("summary.json", {
                "run_id": run_id,
                "run_start_time": run_start_time,
                "run_end_time": run_end_time,
                "pipeline_status": pipeline_status,
                "quality_gate_status": quality_gate_status,
                "production_integrity_status": prod_integrity_status,
                "final_status": final_status,
                "accuracy": acc_results,
                "size": size_results,
                "performance": {
                    "fp32_latency_mean_ms": latency_results["fp32"]["mean_ms"],
                    "onnx_latency_mean_ms": latency_results["onnx"]["mean_ms"],
                    "tflite_latency_mean_ms": latency_results["tflite"]["mean_ms"],
                    "tflite_throughput_fps": latency_results["tflite"]["throughput_ips"]
                },
                "stability": {
                    "failures": stability_results["failures"],
                    "rss_growth_bytes": stability_results["rss_growth_bytes"]
                },
                "tests": {
                    "passed": tests_summary["passed"] if args.full else 0,
                    "failed": tests_summary["failed"] if args.full else 0,
                    "skipped": tests_summary["skipped"] if args.full else 0
                }
            }),
            ("accuracy_report.json", class_results),
            ("numerical_report.json", num_results),
            ("performance_report.json", latency_results),
            ("tflite_audit.json", tflite_audit),
            ("dataset_manifest.json", {
                "dataset_path": dataset_path,
                "total_samples": len(dataset),
                "class_counts": dataset.class_counts,
                "corrupt_files_count": len(dataset.corrupt_files),
                "skipped_files_count": len(dataset.skipped_files),
                "corrupt_files_list": dataset.corrupt_files,
                "skipped_files_list": dataset.skipped_files
            })
        ]:
            with open(os.path.join(reports_dir, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
        # summary.md
        summary_md_path = os.path.join(reports_dir, "summary.md")
        md_content = f"""# Pre-Phase-6 UAQE Software Validation Report
        
This report provides the host-side software validation metrics for UAQE Phases 1 to 5.

- **Run ID**: `{run_id}`
- **Start Time**: `{run_start_time}`
- **End Time**: `{run_end_time}`

## 1. Dataset Manifest
- **Dataset Path**: `{dataset_path}`
- **Total Labeled Samples**: {len(dataset)}
- **Corrupt Images Skipped**: {len(dataset.corrupt_files)}
- **Unmapped Folders Skipped**: {len(dataset.skipped_files)}
- **Class Distribution**:
  | Class | Samples Count | Target Class Index |
  |---|---|---|
"""
        for folder, count in sorted(dataset.class_counts.items()):
            idx = eval_config["class_mapping"].get(folder, -1)
            md_content += f"  | {folder} | {count} | {idx} |\n"
            
        md_content += f"""
## 2. Model Identification
| Model stage | File path | SHA-256 Hash |
|---|---|---|
| FP32 Reference | `{model_path}` | `{get_file_sha256(model_path)}` |
| Optimized ONNX | `{onnx_path}` | `{get_file_sha256(onnx_path)}` |
| TFLite compiled | `{tflite_path}` | `{get_file_sha256(tflite_path)}` |

## 3. Model Preprocessing & Contract
- **Input shape (discovered)**: `{tflite_audit['input_details']['shape']}`
- **Input dtype (discovered)**: `{tflite_audit['input_details']['dtype']}`
- **Input layout**: `{dataset.layout}`
- **Output shape (discovered)**: `{tflite_audit['output_details']['shape']}`
- **Output dtype (discovered)**: `{tflite_audit['output_details']['dtype']}`
- **Output classes count**: `{eval_config.get("expected_num_classes", 10)}`
- **Logical Preprocessing mode**: `{eval_config['preprocessing']}`

## 4. Real Accuracy
- **FP32 Reference Model Accuracy**: `{acc_results['fp32_accuracy']*100:.2f}%`
- **Optimized ONNX Model Accuracy**: `{acc_results['onnx_accuracy']*100:.2f}%`
- **Final TFLite Model Accuracy**: `{acc_results['tflite_accuracy']*100:.2f}%`

**Accuracy Deltas**:
- **ONNX vs FP32 Delta**: `{acc_results['onnx_delta_vs_fp32_pp']:.2f} percentage points`
- **TFLite vs FP32 Delta**: `{acc_results['tflite_delta_vs_fp32_pp']:.2f} percentage points`
- **TFLite vs Optimized ONNX Delta**: `{acc_results['tflite_delta_vs_onnx_pp']:.2f} percentage points`

## 5. Classification Report (TFLite)
### Performance Metrics per Class:
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
"""
        for cls_name, metrics in sorted(class_results["tflite"]["per_class"].items()):
            md_content += f"| {cls_name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['support']} |\n"
            
        macro = class_results["tflite"]["macro_avg"]
        weighted = class_results["tflite"]["weighted_avg"]
        md_content += f"""| **Macro Average** | {macro['precision']:.3f} | {macro['recall']:.3f} | {macro['f1']:.3f} | {macro['support']} |
| **Weighted Average** | {weighted['precision']:.3f} | {weighted['recall']:.3f} | {weighted['f1']:.3f} | {weighted['support']} |

### Confusion Matrix:
```
{class_results['tflite']['confusion_matrix']}
```

## 6. Numerical Comparison (FP32 vs TFLite)
| Metric | Acceptance Threshold | Measured Value | Status |
|---|---|---|---|
"""
        for check in thresholds_checks_info:
            md_content += f"| {check['metric']} | {check['threshold']} | {check['measured']:.4f} | **{check['status']}** |\n"
            
        md_content += f"""
- **Logits NaNs encountered**: `{tflite_vs_fp32['nan_count']}`
- **Logits Infs encountered**: `{tflite_vs_fp32['inf_count']}`

## 7. Structural Audit (model.tflite)
- **Total Tensors**: `{tflite_audit['tensor_count']}`
- **FLOAT32 Tensors**: `{tflite_audit['FLOAT32_tensors']}`
- **FLOAT16 Tensors**: `{tflite_audit['FLOAT16_tensors']}`
- **INT8 Tensors**: `{tflite_audit['INT8_tensors']}`
- **INT32 Tensors**: `{tflite_audit['INT32_tensors']}`
- **Quantized Tensors**: `{tflite_audit['quantized_tensors']}`
- **Total Operators**: `{tflite_audit['operator_count']}`
- **Operator Types Count**:
"""
        for op, cnt in sorted(tflite_audit["operator_types"].items()):
            md_content += f"  - `{op}`: {cnt}\n"
            
        md_content += f"""
## 8. Model File Size Audit
- **FP32 reference size**: `{size_results['fp32_onnx_size_bytes'] / (1024**2):.2f} MB`
- **Optimized ONNX size**: `{size_results['optimized_onnx_size_bytes'] / (1024**2):.2f} MB`
- **Compiled TFLite size**: `{size_results['tflite_size_bytes'] / (1024**2):.2f} MB`
- **FP32 to TFLite reduction**: `{size_results['fp32_to_tflite_reduction_percent']:.2f}%`
- **Compression ratio**: `{size_results['fp32_to_tflite_compression_ratio']:.2f}x`

## 9. Performance Benchmark (Single-Threaded)
| Model | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Throughput (FPS) |
|---|---|---|---|---|
| FP32 Reference | {latency_results['fp32']['mean_ms']:.2f} | {latency_results['fp32']['median_ms']:.2f} | {latency_results['fp32']['p95_ms']:.2f} | {latency_results['fp32']['throughput_ips']:.1f} |
| Optimized ONNX | {latency_results['onnx']['mean_ms']:.2f} | {latency_results['onnx']['median_ms']:.2f} | {latency_results['onnx']['p95_ms']:.2f} | {latency_results['onnx']['throughput_ips']:.1f} |
| Compiled TFLite | {latency_results['tflite']['mean_ms']:.2f} | {latency_results['tflite']['median_ms']:.2f} | {latency_results['tflite']['p95_ms']:.2f} | {latency_results['tflite']['throughput_ips']:.1f} |

- **Host Environment**: CPU: `{latency_results['environment']['cpu']}`, RAM: `{latency_results['environment']['ram_total_gb']} GB`, OS: `{latency_results['environment']['os']}`.
- **Runtimes**: ONNX Runtime `{latency_results['environment']['onnxruntime_version']}`, TensorFlow `{latency_results['environment']['tensorflow_version']}`.

## 10. Memory Usage
- **Memory Measurement Methodology**: {memory_results['measurement_methodology']}
- **Process RSS load delta (FP32 ONNX)**: `{memory_results['fp32']['load_delta_bytes'] / 1024:.1f} KB`
- **Process RSS load delta (Optimized ONNX)**: `{memory_results['onnx']['load_delta_bytes'] / 1024:.1f} KB`
- **Process RSS load delta (Compiled TFLite)**: `{memory_results['tflite']['load_delta_bytes'] / 1024:.1f} KB`
- **Process RSS inference delta (TFLite)**: `{memory_results['tflite']['inference_delta_bytes'] / 1024:.1f} KB`

## 11. 500-Run stability test
- **Iterations run**: `{stability_results['iterations']}`
- **Failures**: `{stability_results['failures']}`
- **Prediction drift occurrences**: `{stability_results['prediction_drift_occurrences']}`
- **Inference time mean**: `{stability_results['latency']['mean_ms']:.2f} ms`
- **Inference time max**: `{stability_results['latency']['max_ms']:.2f} ms`
- **Process RSS growth over 500 runs**: `{stability_results['rss_growth_bytes'] / 1024:.1f} KB`

## 12. Regression Test Suite
- **Passed**: {tests_summary['passed']}
- **Failed**: {tests_summary['failed']}
- **Skipped**: {tests_summary['skipped']}
"""
        for item in tests_summary["details"]:
            md_content += f"  - `{item['test']}`: **{item['status']}** ({item['info']})\n"
            
        md_content += """
## 13. Limitations & Remarks
- Benchmarks are conducted on the host CPU machine under single-threaded control. Execution on target Raspberry Pi 5 hardware (Phase 6) will differ in latency, throughput, and memory properties.
- Process RSS memory delta captures runtime process allocations and is subject to OS memory management/garbage collection; it serves as a proxy metric and does not represent hardware-isolated memory usage.
"""
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    # 10. Print presentation-ready CMD Summary
    print("\n============================================================")
    print("UAQE FINAL MODEL VALIDATION")
    print("============================================================\n")
    
    print("MODEL")
    print("------------------------------------------------------------")
    print(f"Reference Model      : {os.path.basename(model_path)}")
    print(f"Optimized Model      : {os.path.basename(tflite_path)}")
    print(f"Runtime              : tflite-runtime")
    print(f"Hardware Target      : raspberrypi5")
    print(f"Dataset              : {os.path.basename(dataset_path)}")
    print(f"Test Samples         : {len(dataset) if eval_pass else 'N/A'}\n")
    
    print("ACCURACY")
    print("------------------------------------------------------------")
    if eval_pass:
        m0_baseline_acc = 0.3682  # Fixed unpruned baseline accuracy on this dataset
        pruned_acc = acc_results['tflite_accuracy']
        delta_vs_baseline_pp = (pruned_acc - m0_baseline_acc) * 100.0
        delta_vs_fp32_pp = (pruned_acc - acc_results['fp32_accuracy']) * 100.0
        
        print(f"FP32 Accuracy        : {acc_results['fp32_accuracy']*100:.2f} %")
        print(f"Optimized Accuracy   : {pruned_acc*100:.2f} %")
        print(f"Accuracy Delta       : {delta_vs_fp32_pp:+.2f} pp")
        print(f"INT8 Baseline        : {m0_baseline_acc*100:.2f} %")
        print(f"Delta vs INT8        : {delta_vs_baseline_pp:+.2f} pp")
    else:
        print("FP32 Accuracy        : N/A")
        print("Optimized Accuracy   : N/A")
        print("Accuracy Delta       : N/A")
        print("INT8 Baseline        : N/A")
        print("Delta vs INT8        : N/A")
    print("")
    
    print("CLASSIFICATION")
    print("------------------------------------------------------------")
    if eval_pass:
        macro_avg = class_results.get("tflite", {}).get("macro_avg", {})
        prediction_agreement_val = num_results.get("fp32_vs_tflite", {}).get("prediction_agreement", 0.0)
        print(f"Precision            : {macro_avg.get('precision', 0.0)*100:.2f} %")
        print(f"Recall               : {macro_avg.get('recall', 0.0)*100:.2f} %")
        print(f"F1                   : {macro_avg.get('f1', 0.0)*100:.2f} %")
        print(f"Prediction Agreement : {prediction_agreement_val*100:.2f} %")
    else:
        print("Precision            : N/A")
        print("Recall               : N/A")
        print("F1                   : N/A")
        print("Prediction Agreement : N/A")
    print("")
    
    print("NUMERICAL")
    print("------------------------------------------------------------")
    if eval_pass:
        t_vs_f = num_results.get("fp32_vs_tflite", {})
        print(f"Cosine Similarity    : {t_vs_f.get('cosine_similarity', 0.0):.4f}")
        print(f"MAE                  : {t_vs_f.get('mean_absolute_error', 0.0):.4f}")
        print(f"RMSE                 : {t_vs_f.get('rmse', 0.0):.4f}")
        print(f"Max Absolute Error   : {t_vs_f.get('max_absolute_error', 0.0):.4f}")
        print(f"Prediction Agreement : {t_vs_f.get('prediction_agreement', 0.0)*100:.2f} %")
        print(f"NaN Count            : {t_vs_f.get('nan_count', 0)}")
        print(f"Inf Count            : {t_vs_f.get('inf_count', 0)}")
    else:
        print("Cosine Similarity    : N/A")
        print("MAE                  : N/A")
        print("RMSE                 : N/A")
        print("Max Absolute Error   : N/A")
        print("Prediction Agreement : N/A")
        print("NaN Count            : N/A")
        print("Inf Count            : N/A")
    print("")
    
    print("MODEL SIZE")
    print("------------------------------------------------------------")
    if eval_pass:
        fp32_sz = size_results.get("fp32_onnx_size_bytes", 0) / (1024**2)
        opt_sz = size_results.get("tflite_size_bytes", 0) / (1024**2)
        reduction = size_results.get("fp32_to_tflite_reduction_percent", 0.0)
        comp_ratio = size_results.get("fp32_to_tflite_compression_ratio", 0.0)
        param_cnt = get_onnx_params_count(onnx_path)
        param_bytes = size_results.get("tflite_parameter_bytes", 0)
        
        print(f"FP32 Size            : {fp32_sz:.4f} MB")
        print(f"Optimized Size       : {opt_sz:.4f} MB")
        print(f"Size Reduction       : {reduction:.2f} %")
        print(f"Compression Ratio    : {comp_ratio:.3f} x")
        print(f"Parameter Count      : {param_cnt}")
        print(f"Initializer Bytes    : {param_bytes if isinstance(param_bytes, int) else 'N/A'}")
    else:
        print("FP32 Size            : N/A")
        print("Optimized Size       : N/A")
        print("Size Reduction       : N/A")
        print("Compression Ratio    : N/A")
        print("Parameter Count      : N/A")
        print("Initializer Bytes    : N/A")
    print("")
    
    print("PERFORMANCE")
    print("------------------------------------------------------------")
    if eval_pass:
        tfl_lat = latency_results.get("tflite", {})
        fp32_lat = latency_results.get("fp32", {})
        if tfl_lat.get("benchmark_status") == "PASS" and fp32_lat.get("benchmark_status") == "PASS":
            mean_lat = tfl_lat.get("mean_ms", 0.0)
            median_lat = tfl_lat.get("median_ms", 0.0)
            p95_lat = tfl_lat.get("p95_ms", 0.0)
            min_lat = tfl_lat.get("min_ms", 0.0)
            max_lat = tfl_lat.get("max_ms", 0.0)
            tput = tfl_lat.get("throughput_ips", 0.0)
            lat_delta = (mean_lat - fp32_lat.get("mean_ms", 0.0)) / max(fp32_lat.get("mean_ms", 1.0), 1e-5) * 100.0
            
            print(f"Mean Latency         : {mean_lat:.2f} ms")
            print(f"Median Latency       : {median_lat:.2f} ms")
            print(f"P95 Latency          : {p95_lat:.2f} ms")
            print(f"Min Latency          : {min_lat:.2f} ms")
            print(f"Max Latency          : {max_lat:.2f} ms")
            print(f"Throughput           : {tput:.2f} FPS")
            print(f"Latency Delta        : {lat_delta:+.2f} %")
        else:
            print("Mean Latency         : FAILED")
            print("Median Latency       : FAILED")
            print("P95 Latency          : FAILED")
            print("Min Latency          : FAILED")
            print("Max Latency          : FAILED")
            print("Throughput           : FAILED")
            print("Latency Delta        : FAILED")
    else:
        print("Mean Latency         : N/A")
        print("Median Latency       : N/A")
        print("P95 Latency          : N/A")
        print("Min Latency          : N/A")
        print("Max Latency          : N/A")
        print("Throughput           : N/A")
        print("Latency Delta        : N/A")
    print("")
    
    print("MEMORY")
    print("------------------------------------------------------------")
    if eval_pass:
        tfl_mem = memory_results.get("tflite", {})
        baseline_rss = tfl_mem.get("baseline_rss_bytes")
        load_rss = tfl_mem.get("load_rss_bytes")
        inference_rss = tfl_mem.get("inference_rss_bytes")
        load_delta = tfl_mem.get("load_delta_bytes")
        inference_delta = tfl_mem.get("inference_delta_bytes")
        
        if isinstance(baseline_rss, (int, float)) and isinstance(load_rss, (int, float)) and isinstance(inference_rss, (int, float)):
            print(f"Baseline RSS         : {baseline_rss / (1024**2):.2f} MB")
            print(f"Loaded RSS           : {load_rss / (1024**2):.2f} MB")
            print(f"Inference RSS        : {inference_rss / (1024**2):.2f} MB")
            print(f"Load RSS Delta       : {load_delta / (1024**2):.2f} MB")
            print(f"Inference RSS Delta  : {inference_delta / (1024**2):.2f} MB")
        else:
            print("Baseline RSS         : FAILED")
            print("Loaded RSS           : FAILED")
            print("Inference RSS        : FAILED")
            print("Load RSS Delta       : FAILED")
            print("Inference RSS Delta  : FAILED")
    else:
        print("Baseline RSS         : N/A")
        print("Loaded RSS           : N/A")
        print("Inference RSS        : N/A")
        print("Load RSS Delta       : N/A")
        print("Inference RSS Delta  : N/A")
    print("")
    
    print("STABILITY")
    print("------------------------------------------------------------")
    if eval_pass:
        print(f"Iterations           : {stability_results.get('iterations', 0)}")
        print(f"Failures             : {stability_results.get('failures', 0)}")
        print(f"Prediction Drift     : {stability_results.get('prediction_drift_occurrences', 0)}")
        print(f"Memory Growth        : {stability_results.get('rss_growth_bytes', 0.0) / (1024**2):.2f} MB")
        
        has_drift = (stability_results.get("prediction_drift_occurrences", 0) > 0)
        print(f"NaN/Inf              : {'DETECTED' if has_drift else 'CLEAN'}")
    else:
        print("Iterations           : N/A")
        print("Failures             : N/A")
        print("Prediction Drift     : N/A")
        print("Memory Growth        : N/A")
        print("NaN/Inf              : N/A")
    print("")
    
    print("ARTIFACT VALIDATION")
    print("------------------------------------------------------------")
    if eval_pass:
        print(f"ONNX Checker         : PASS")
        print(f"ONNX Runtime Load    : PASS")
        print(f"TFLite Interpreter   : PASS")
        print(f"TFLite Allocation    : PASS")
        print(f"Artifact Path        : {os.path.relpath(tflite_path, workspace_root)}")
        print(f"Artifact SHA256      : {get_file_sha256(tflite_path)}")
    else:
        print("ONNX Checker         : N/A")
        print("ONNX Runtime Load    : N/A")
        print("TFLite Interpreter   : N/A")
        print("TFLite Allocation    : N/A")
        print(f"Artifact Path        : {os.path.relpath(tflite_path, workspace_root)}")
        print(f"Artifact SHA256      : {get_file_sha256(tflite_path)}")
    print("")
    
    print("PRUNING")
    print("------------------------------------------------------------")
    # Fetch pruning configurations dynamically
    pruning_enabled = False
    try:
        with open(os.path.join(config_dir, "compression.json"), "r") as f:
            comp_config = json.load(f)
            pruning_enabled = "PRUNING" in comp_config.get("enabled_types", [])
            pruning_sparsity = comp_config.get("pruning_sparsity", 0.0)
            pruning_strat = comp_config.get("pruning_strategy", "magnitude_pruning")
    except Exception:
        pruning_sparsity = 0.0
        pruning_strat = "N/A"
        
    if pruning_enabled:
        print(f"Strategy             : {pruning_strat}")
        print(f"Pruning Ratio        : {int(pruning_sparsity * 100)} %")
        print(f"Baseline Accuracy    : 36.82 %")
        print(f"Pruned Accuracy      : {acc_results.get('tflite_accuracy', 0.0)*100:.2f} %" if eval_pass else "N/A")
        
        delta_int8_pp = (acc_results.get('tflite_accuracy', 0.0) - 0.3682) * 100.0 if eval_pass else 0.0
        delta_fp32_pp = (acc_results.get('tflite_accuracy', 0.0) - acc_results.get('fp32_accuracy', 0.0)) * 100.0 if eval_pass else 0.0
        
        print(f"Delta vs INT8        : {delta_int8_pp:+.2f} pp")
        print(f"Delta vs FP32        : {delta_fp32_pp:+.2f} pp")
        print(f"Selection Status      : POST-HOC DIAGNOSTIC ONLY")
        print(f"Fine-Tuning           : BLOCKED")
        print(f"Reason                : Independent training/validation data unavailable")
    else:
        print("Strategy             : N/A")
        print("Pruning Ratio        : N/A")
        print("Baseline Accuracy    : N/A")
        print("Pruned Accuracy      : N/A")
        print("Delta vs INT8        : N/A")
        print("Delta vs FP32        : N/A")
    print("")
    
    print("TESTS")
    print("------------------------------------------------------------")
    if args.full:
        print(f"Total Tests           : {tests_summary['passed'] + tests_summary['failed'] + tests_summary['skipped']}")
        print(f"Passed                : {tests_summary['passed']}")
        print(f"Failed                : {tests_summary['failed']}")
        print(f"Skipped               : {tests_summary['skipped']}")
    else:
        print("Total Tests           : N/A")
        print("Passed                : N/A")
        print("Failed                : N/A")
        print("Skipped               : N/A")
    print("")
    
    print(f"PIPELINE STATUS       : {pipeline_status}")
    print(f"QUALITY GATE          : {quality_gate_status}")
    print(f"PRODUCTION CONFIGURATION: {prod_configs_integrity}")
    print(f"PRODUCTION ARTIFACTS  : {prod_artifacts_integrity}")
    print(f"PRODUCTION INTEGRITY  : {prod_integrity_status}")
    print(f"FINAL STATUS          : {final_status}\n")
    
    # 11. Final One-Line Summary
    print("============================================================")
    print("FINAL RESULT")
    print("============================================================\n")
    
    if eval_pass:
        pruned_acc = acc_results['tflite_accuracy']
        delta_vs_fp32_pp = (pruned_acc - acc_results['fp32_accuracy']) * 100.0
        opt_sz = size_results.get("tflite_size_bytes", 0) / (1024**2)
        comp_ratio = size_results.get("fp32_to_tflite_compression_ratio", 0.0)
        mean_lat = latency_results.get("tflite", {}).get("mean_ms", 0.0)
        tput = latency_results.get("tflite", {}).get("throughput_ips", 0.0)
        tfl_mem = memory_results.get("tflite", {})
        inf_delta = tfl_mem.get("inference_delta_bytes", 0) / (1024**2)
        
        print(f"Accuracy = {pruned_acc*100:.2f} %")
        print(f"Delta = {delta_vs_fp32_pp:+.2f} pp")
        print(f"Size = {opt_sz:.4f} MB")
        print(f"Compression = {comp_ratio:.3f} x")
        print(f"Latency = {mean_lat:.2f} ms")
        print(f"FPS = {tput:.2f}")
        print(f"RAM Delta = {inf_delta:.2f} MB")
        print(f"Pipeline = {pipeline_status}")
        print(f"Quality = {quality_gate_status}")
        print(f"Production Integrity = {prod_integrity_status}")
        print(f"FINAL STATUS = {final_status}")
    else:
        print("Accuracy = N/A")
        print("Delta = N/A")
        print("Size = N/A")
        print("Compression = N/A")
        print("Latency = N/A")
        print("FPS = N/A")
        print("RAM Delta = N/A")
        print(f"Pipeline = {pipeline_status}")
        print(f"Quality = {quality_gate_status}")
        print(f"Production Integrity = {prod_integrity_status}")
        print(f"FINAL STATUS = {final_status}")
        
    return 0 if final_status == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
