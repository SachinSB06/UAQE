"""Optimization Planner for UAQE.

Synthesizes user profile, hardware constraints, model capabilities, and dataset properties
into an actionable, auditable optimization plan without modifying source artifacts.
Generates optimization_plan.json and human-readable optimization_plan.md.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any


class OptimizationPlanner:
    """Creates a structured optimization plan based on model capabilities and target constraints."""

    SUPPORTED_PROFILES = ["accuracy_first", "balanced", "storage_first", "latency_first"]

    @classmethod
    def create_plan(
        cls,
        model_descriptor: Dict[str, Any],
        model_capabilities: Dict[str, bool],
        dataset_descriptor: Dict[str, Any],
        compatibility_report: Dict[str, Any],
        hardware_profile: Dict[str, Any],
        profile_name: str = "balanced"
    ) -> Dict[str, Any]:
        """Generate structured optimization plan."""
        profile = profile_name.lower()
        if profile not in cls.SUPPORTED_PROFILES:
            profile = "balanced"

        # 1. Quantization Selection & Fallback Logic
        target_precisions = hardware_profile.get("supported_precisions", ["INT8"])
        supports_int8 = model_capabilities.get("supports_int8", False) and "INT8" in target_precisions
        supports_fp16 = model_capabilities.get("supports_fp16", False) and "FP16" in target_precisions

        fallback_used = False
        fallback_reason = "None"

        if supports_int8:
            selected_precision = "INT8"
            quantization_method = "Sensitivity-Aware Post-Training Quantization / QAT"
        elif supports_fp16:
            selected_precision = "FP16"
            quantization_method = "Float16 Conversion"
            fallback_used = True
            fallback_reason = "INT8 unsupported on target or model; falling back to FP16"
        else:
            selected_precision = "FP32"
            quantization_method = "Passthrough FP32"
            fallback_used = True
            fallback_reason = "Neither INT8 nor FP16 supported; preserving FP32"

        # 2. Pruning & Compression Strategy based on Profile
        pruning_enabled = model_capabilities.get("supports_pruning", False)
        sparse_enabled = model_capabilities.get("supports_sparse", False)
        rle_enabled = model_capabilities.get("supports_rle", False)
        clustering_enabled = model_capabilities.get("supports_clustering", False)

        if profile == "accuracy_first":
            candidate_pruning_ratios = [0.0, 0.10] if pruning_enabled else []
            sparse_compression = "Sparse + RLE" if (sparse_enabled and rle_enabled) else "None"
            clustering_allowed = False
            accuracy_gate_drop_max = 0.01  # Max 1.0% accuracy drop
        elif profile == "balanced":
            candidate_pruning_ratios = [0.10, 0.20] if pruning_enabled else []
            sparse_compression = "Sparse + RLE" if (sparse_enabled and rle_enabled) else "None"
            clustering_allowed = False
            accuracy_gate_drop_max = 0.02  # Max 2.0% accuracy drop
        elif profile == "storage_first":
            candidate_pruning_ratios = [0.20, 0.30] if pruning_enabled else []
            sparse_compression = "Sparse + RLE" if (sparse_enabled and rle_enabled) else "None"
            clustering_allowed = clustering_enabled
            accuracy_gate_drop_max = 0.035  # Max 3.5% accuracy drop
        elif profile == "latency_first":
            candidate_pruning_ratios = [0.10, 0.20] if pruning_enabled else []
            sparse_compression = "Dense/TFLite"
            clustering_allowed = False
            accuracy_gate_drop_max = 0.025
        else:
            candidate_pruning_ratios = [0.10, 0.20]
            sparse_compression = "Sparse + RLE"
            clustering_allowed = False
            accuracy_gate_drop_max = 0.02

        plan = {
            "optimization_profile": profile,
            "target_hardware": {
                "target_id": hardware_profile.get("target_id", "generic"),
                "name": hardware_profile.get("name", "Generic Target"),
                "hardware_class": hardware_profile.get("hardware_class", "UNKNOWN"),
                "is_physical_measurement": False
            },
            "model_summary": {
                "architecture": model_descriptor.get("architecture", "Unknown"),
                "framework": model_descriptor.get("framework", "Unknown"),
                "total_parameters": model_descriptor.get("parameter_count", 0),
                "source_sha256": model_descriptor.get("source_sha256", "")
            },
            "dataset_summary": {
                "dataset_name": dataset_descriptor.get("dataset_name", "Unknown"),
                "format": dataset_descriptor.get("detected_format", "Unknown"),
                "class_count": dataset_descriptor.get("class_count", 0),
                "train_samples": dataset_descriptor.get("splits", {}).get("train_count", 0),
                "val_samples": dataset_descriptor.get("splits", {}).get("val_count", 0),
                "test_samples": dataset_descriptor.get("splits", {}).get("test_count", 0)
            },
            "adaptation_plan": {
                "adaptation_required": compatibility_report.get("class_alignment", {}).get("adaptation_required", False),
                "original_classes": compatibility_report.get("class_alignment", {}).get("model_classes", 0),
                "target_classes": compatibility_report.get("class_alignment", {}).get("dataset_classes", 0)
            },
            "quantization_plan": {
                "selected_precision": selected_precision,
                "quantization_method": quantization_method,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason
            },
            "compression_plan": {
                "pruning_enabled": bool(candidate_pruning_ratios),
                "candidate_pruning_ratios": candidate_pruning_ratios,
                "sparse_compression": sparse_compression,
                "selective_clustering": clustering_allowed
            },
            "validation_policy": {
                "search_split": "validation",
                "final_eval_split": "test",
                "accuracy_gate_max_drop": accuracy_gate_drop_max
            },
            "accuracy_safety_policy": {
                "max_acceptable_loss_pp": 1.0 if profile == "accuracy_first" else 4.0,
                "classifications": {
                    "EXCELLENT": "<= 1.0 pp",
                    "ACCEPTABLE": "1.0 < pp <= 4.0",
                    "CRITICAL": "> 4.0 pp"
                }
            },
            "objective_weights": {
                "accuracy_weight": 0.75 if profile == "accuracy_first" else (0.35 if profile in ("storage_first", "size_first", "latency_first") else 0.50),
                "size_weight": 0.45 if profile in ("storage_first", "size_first") else (0.15 if profile == "accuracy_first" else 0.25),
                "latency_weight": 0.45 if profile == "latency_first" else (0.10 if profile == "accuracy_first" else 0.25)
            },
            "candidate_search_budget": {
                "max_candidates": 10,
                "stopping_policy": "TARGET_CONSTRAINTS_SATISFIED_OR_BUDGET_EXHAUSTED"
            },
            "export_target": {
                "preferred_format": hardware_profile.get("preferred_export_format", "tflite"),
                "runtime_package": "complete_deployment_package"
            }
        }

        return plan

    @classmethod
    def generate_plan_files(cls, output_dir: str, plan: Dict[str, Any]) -> Tuple[str, str]:
        """Write optimization_plan.json and readable optimization_plan.md."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "optimization_plan.json")
        md_path = os.path.join(output_dir, "optimization_plan.md")

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)

        # Generate Markdown
        m_sum = plan["model_summary"]
        d_sum = plan["dataset_summary"]
        q_plan = plan["quantization_plan"]
        c_plan = plan["compression_plan"]
        hw = plan["target_hardware"]
        adapt = plan["adaptation_plan"]

        md_content = rf"""# UAQE Automated Optimization Plan

**Target Profile:** `{hw['name']}`  
**Optimization Profile:** `{plan['optimization_profile']}`  
**Status:** Read-Only Plan Generated (Awaiting User Approval)

---

## 1. Model & Dataset Specification
- **Model Architecture:** {m_sum['architecture']} ({m_sum['framework']})
- **Parameter Count:** {m_sum['total_parameters']:,}
- **Source Checksum (SHA-256):** `{m_sum['source_sha256']}`
- **Dataset:** {d_sum['dataset_name']} ({d_sum['format']})
- **Classes:** {d_sum['class_count']}
- **Dataset Partitions:** Train: {d_sum['train_samples']:,} | Val: {d_sum['val_samples']:,} | Test: {d_sum['test_samples']:,}

## 2. Adaptation Strategy
- **Adaptation Required:** {'YES' if adapt['adaptation_required'] else 'NO'}
- **Original Output Classes:** {adapt['original_classes']}
- **Target Dataset Classes:** {adapt['target_classes']}

## 3. Quantization & Compression Plan
- **Planned Precision:** **{q_plan['selected_precision']}** ({q_plan['quantization_method']})
- **Fallback Used:** {'YES (' + q_plan['fallback_reason'] + ')' if q_plan['fallback_used'] else 'NO'}
- **Pruning Allocation:** {c_plan['candidate_pruning_ratios']}
- **Encoding & Packaging:** {c_plan['sparse_compression']}
- **Selective Clustering:** {'Enabled' if c_plan['selective_clustering'] else 'Disabled'}
- **Target Export Format:** `{plan['export_target']['preferred_format']}`

## 4. Validation Policy
- **Optimization Search Split:** `{plan['validation_policy']['search_split']}`
- **Final Evaluation Split:** `{plan['validation_policy']['final_eval_split']}` (Frozen test set)
- **Accuracy Gating Tolerance:** $\le {plan['validation_policy']['accuracy_gate_max_drop'] * 100:.1f}\%$ drop

> [!NOTE]
> Hardware latency figures represent target profile planning constraints and do not represent physical hardware measurements.
"""

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path
