"""Optimization Orchestrator for UAQE.

Provides the unified, generic user-facing workflow entry point:
OptimizationOrchestrator.run(job_config)

Orchestrates inspection, capability auditing, compatibility checking, model adaptation,
dry-run planning, user approval gating, calibration, optimization, validation gating,
and deployable artifact packaging in isolated job directories.
"""

import os
import sys
import json
import time
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .universal_model_ingestor import UniversalModelIngestor
from .universal_dataset_ingestor import UniversalDatasetIngestor
from .task_detector import TaskDetector
from .compatibility_checker import CompatibilityChecker
from .preprocessing_resolver import PreprocessingResolver
from .hardware_target_registry import HardwareTargetRegistry
from .model_adaptation_service import ModelAdaptationService
from .optimization_planner import OptimizationPlanner
from .optimization_strategies import OptimizationStrategyResolver


class OptimizationOrchestrator:
    """Master Orchestrator for generic UAQE user jobs."""

    def __init__(self, output_root: str = "output/jobs"):
        self.output_root = os.path.abspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)

    @staticmethod
    def _compute_sha256(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_job_id(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        rand_hex = hashlib.sha256(os.urandom(16)).hexdigest()[:8].upper()
        return f"UAQE-{date_str}-{rand_hex}"

    def run(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute universal optimization workflow."""
        model_path = os.path.abspath(job_config["model_path"])
        dataset_path = os.path.abspath(job_config["dataset_path"])
        target_hardware = job_config.get("target_hardware", "raspberrypi5")
        profile = job_config.get("optimization_profile", "balanced")
        auto_approve = job_config.get("auto_approve", False)
        plan_only = job_config.get("plan_only", False) or job_config.get("dry_run", False)

        # 1. Initialize Job Directory
        job_id = job_config.get("job_id") or self.generate_job_id()
        job_dir = os.path.join(self.output_root, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # 2. Source Hashes & Manifest
        model_sha256 = self._compute_sha256(model_path)
        source_hashes = {
            "model_path": model_path,
            "model_sha256": model_sha256,
            "dataset_path": dataset_path,
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(job_dir, "source_hashes.json"), "w", encoding="utf-8") as f:
            json.dump(source_hashes, f, indent=2)

        input_manifest = {
            "job_id": job_id,
            "model_path": model_path,
            "dataset_path": dataset_path,
            "target_hardware": target_hardware,
            "optimization_profile": profile,
            "auto_approve": auto_approve,
            "plan_only": plan_only,
            "created_at": datetime.now().isoformat()
        }
        with open(os.path.join(job_dir, "input_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(input_manifest, f, indent=2)

        progress_callback = job_config.get("progress_callback", None)

        # 3. Model Ingestion & Inspection
        if progress_callback:
            progress_callback({
                "type": "stage_start",
                "stage": "INGESTING",
                "message": "Ingesting model and dataset"
            })

        model_ingestor = UniversalModelIngestor(model_path)
        model_desc = model_ingestor.generate_descriptor(os.path.join(job_dir, "model_descriptor.json"))
        # Save alias model_inspection.json
        with open(os.path.join(job_dir, "model_inspection.json"), "w", encoding="utf-8") as f:
            json.dump(model_desc, f, indent=2)

        cap_report = model_ingestor.generate_capability_report(os.path.join(job_dir, "capability_report.json"))
        model_caps = model_ingestor.get_capabilities()

        # 4. Dataset Ingestion & Inspection
        dataset_ingestor = UniversalDatasetIngestor(dataset_path)
        dataset_splits = dataset_ingestor.load()
        dataset_desc = dataset_ingestor.generate_descriptor(os.path.join(job_dir, "dataset_descriptor.json"))
        # Save alias dataset_inspection.json
        with open(os.path.join(job_dir, "dataset_inspection.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_desc, f, indent=2)

        # 5. Task Detection
        if progress_callback:
            progress_callback({
                "type": "stage_start",
                "stage": "INSPECTING",
                "message": "Auditing model capabilities and compatibility"
            })

        task_info = TaskDetector.detect_task(model_desc, dataset_desc)
        if not task_info["supported"]:
            raise ValueError(f"Task detection failed: {task_info.get('error')}")

        # 6. Compatibility Check
        compat_report = CompatibilityChecker.check_compatibility(model_desc, dataset_desc, task_info)
        CompatibilityChecker.generate_report(os.path.join(job_dir, "compatibility_report.json"), compat_report)
        # Save alias compatibility.json
        with open(os.path.join(job_dir, "compatibility.json"), "w", encoding="utf-8") as f:
            json.dump(compat_report, f, indent=2)
        if not compat_report["compatible"]:
            raise ValueError(f"Model and dataset incompatible: {compat_report['issues']}")

        # 7. Preprocessing Resolution with Provenance
        if progress_callback:
            progress_callback({
                "type": "stage_start",
                "stage": "CALIBRATING",
                "message": "Resolving preprocessing and calibration dataset"
            })

        model_dir = os.path.dirname(model_path)
        preprocess_config = PreprocessingResolver.resolve(model_desc, dataset_desc, model_dir=model_dir)
        PreprocessingResolver.generate_config(os.path.join(job_dir, "preprocessing_config.json"), preprocess_config)

        # 8. Model Adaptation Inspection & Execution (if needed)
        adaptation_service = ModelAdaptationService()
        target_classes = dataset_desc["class_count"]
        checkpoint_mode = job_config.get("checkpoint_mode", "USER_UPLOAD")
        benchmark_checkpoint_path = job_config.get("benchmark_checkpoint_path", None)
        adapted_model, adaptation_record = adaptation_service.adapt(
            model_descriptor=model_desc,
            model_path=model_path,
            target_classes=target_classes,
            device="cpu",
            checkpoint_mode=checkpoint_mode,
            benchmark_checkpoint_path=benchmark_checkpoint_path
        )
        adaptation_service.generate_report(os.path.join(job_dir, "adaptation_report.json"), adaptation_record)

        # 9. Hardware Profile Resolution
        hw_profile = HardwareTargetRegistry.get_profile(target_hardware)

        # 10. Optimization Planning (Dry-Run Plan)
        opt_plan = OptimizationPlanner.create_plan(
            model_descriptor=model_desc,
            model_capabilities=model_caps,
            dataset_descriptor=dataset_desc,
            compatibility_report=compat_report,
            hardware_profile=hw_profile,
            profile_name=profile
        )
        plan_json, plan_md = OptimizationPlanner.generate_plan_files(job_dir, opt_plan)

        # Write execution log entry
        log_path = os.path.join(job_dir, "execution_log.txt")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now().isoformat()}] Job {job_id} initialized.\n")
            log_file.write(f"[{datetime.now().isoformat()}] Model inspected: {model_desc['architecture']}\n")
            log_file.write(f"[{datetime.now().isoformat()}] Dataset inspected: {dataset_desc['dataset_name']} ({dataset_desc['detected_format']})\n")
            log_file.write(f"[{datetime.now().isoformat()}] Optimization plan created for target: {hw_profile['name']}\n")

        # 11. Approval Gate Check (Dry-Run / Plan-Only)
        if plan_only or not auto_approve:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[{datetime.now().isoformat()}] Plan generated. Awaiting approval or dry-run complete.\n")

            return {
                "job_id": job_id,
                "status": "PLAN_GENERATED",
                "message": "Optimization plan generated successfully. Awaiting user approval.",
                "job_dir": job_dir,
                "optimization_plan_json": plan_json,
                "optimization_plan_md": plan_md,
                "model_descriptor": model_desc,
                "dataset_descriptor": dataset_desc,
                "capability_report": cap_report,
                "compatibility_report": compat_report,
                "adaptation_report": adaptation_record,
                "preprocessing_config": preprocess_config
            }

        # 12. Autonomous Optimization Controller Execution
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now().isoformat()}] Launching Autonomous Optimization Controller...\n")

        job_context = {
            "job_id": job_id,
            "job_dir": job_dir,
            "model_path": model_path,
            "model_desc": model_desc,
            "dataset_ingestor": dataset_ingestor,
            "dataset_desc": dataset_desc,
            "task_info": task_info,
            "preprocess_config": preprocess_config,
            "opt_plan": opt_plan,
            "hw_profile": hw_profile,
            "checkpoint_mode": checkpoint_mode,
            "adapted_model": adapted_model,
            "adaptation_record": adaptation_record,
            "calib_samples": job_config.get("calib_samples", 256),
            "calib_seed": job_config.get("calib_seed", 42),
            "test_samples": job_config.get("test_samples", 1000),
            "runtime_mode": job_config.get("runtime_mode", "performance"),
            "performance_enabled": job_config.get("performance_enabled", True),
            "num_threads": int(job_config.get("num_threads", 2))
        }

        from uaqe.optimization.optimization_controller import OptimizationController

        raw_budget = job_config.get("max_candidates", job_config.get("max_budget", 10))
        bounded_budget = min(max(int(raw_budget), 1), 20)

        controller = OptimizationController(
            job_context=job_context,
            profile=profile,
            max_budget=bounded_budget,
            progress_callback=job_config.get("progress_callback", None)
        )
        results = controller.optimize()

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now().isoformat()}] Optimization completed. Final Verdict: {results.get('verdict')}\n")

        return results
