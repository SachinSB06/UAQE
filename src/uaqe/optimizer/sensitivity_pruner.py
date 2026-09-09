"""Sensitivity-aware and global unstructured pruning for MobileNetV3-Small in UAQE.
Phase D.1 implementation.
"""

from __future__ import annotations

import os
import copy
import json
import csv
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models


CLASS_NAMES = ["bridge", "clean", "cmp", "crack", "opens", "other", "particle", "scratch", "vias"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def get_stratified_calibration_samples(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_total: int = 100,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts a deterministic, class-balanced stratified random calibration set from TRAIN.
    
    Args:
        train_x: Training input tensor of shape (N, C, H, W).
        train_y: Training labels tensor of shape (N,).
        n_total: Total calibration sample count (e.g. 50, 100, 150, 200, 250).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (calib_x, calib_y).
    """
    np_rng = np.random.RandomState(seed)
    num_classes = len(CLASS_NAMES)
    base_per_class = n_total // num_classes
    remainder = n_total % num_classes
    
    selected_indices: List[int] = []
    
    for c_idx in range(num_classes):
        c_indices = np.where(train_y.numpy() == c_idx)[0]
        if len(c_indices) == 0:
            continue
        count_to_pick = base_per_class + (1 if c_idx < remainder else 0)
        count_to_pick = min(count_to_pick, len(c_indices))
        
        # Deterministic stratified random choice
        chosen = np_rng.choice(c_indices, size=count_to_pick, replace=False)
        selected_indices.extend(chosen.tolist())
        
    # Sort indices to preserve reproducible order
    selected_indices = sorted(selected_indices)
    return train_x[selected_indices], train_y[selected_indices]


class SensitivityPruner:
    """Manages layer structure audit, sensitivity scoring, and unstructured pruning."""

    def __init__(
        self,
        sensitivity_csv_path: str = "output/layer_sensitivity.csv",
        sensitivity_json_path: str = "output/quantization_sensitivity_report.json"
    ):
        self.sensitivity_csv_path = sensitivity_csv_path
        self.sensitivity_json_path = sensitivity_json_path
        self.layer_sensitivities: Dict[str, Dict[str, Any]] = {}
        self._load_sensitivity_data()

    def _load_sensitivity_data(self) -> None:
        """Loads and indexes sensitivity metrics from Phase A.2 reports."""
        if os.path.exists(self.sensitivity_csv_path):
            with open(self.sensitivity_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    layer_name = row.get("layer_name", "")
                    # Normalize layer name
                    clean_name = layer_name.replace("/", ".").strip(".")
                    if clean_name.startswith("features."):
                        pass
                    self.layer_sensitivities[clean_name] = {
                        "rank": int(row.get("rank", 999)),
                        "score": float(row.get("sensitivity_score", 0.5)),
                        "semantic_category": row.get("semantic_category", "unknown"),
                        "recommendation": row.get("recommendation", "N/A"),
                        "operator_type": row.get("operator_type", "Conv"),
                    }

    def audit_model_structure(self, model: nn.Module, output_csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Audits model architecture and classifies all operators into semantic categories.
        
        Categories:
            STEM, POINTWISE, DEPTHWISE, SE, HARDSWISH, CLASSIFIER
        """
        records: List[Dict[str, Any]] = []
        
        for name, module in model.named_modules():
            if not name:
                continue
                
            op_type = module.__class__.__name__
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            has_bias = hasattr(module, "bias") and module.bias is not None
            
            in_ch = getattr(module, "in_channels", getattr(module, "in_features", None))
            out_ch = getattr(module, "out_channels", getattr(module, "out_features", None))
            kernel = getattr(module, "kernel_size", None)
            stride = getattr(module, "stride", None)
            groups = getattr(module, "groups", 1)
            weight_shape = list(module.weight.shape) if hasattr(module, "weight") and module.weight is not None else []
            
            # Semantic classification
            if name.startswith("features.0"):
                sem_cat = "STEM"
            elif name.startswith("classifier"):
                sem_cat = "CLASSIFIER"
            elif "block.2" in name or "fc1" in name or "fc2" in name or "avgpool" in name:
                sem_cat = "SE"
            elif isinstance(module, nn.Conv2d) and groups > 1:
                sem_cat = "DEPTHWISE"
            elif isinstance(module, nn.Conv2d) and groups == 1 and kernel in [(1, 1), 1]:
                sem_cat = "POINTWISE"
            elif isinstance(module, (nn.Hardswish, nn.Hardsigmoid, nn.ReLU)):
                sem_cat = "HARDSWISH" if isinstance(module, nn.Hardswish) else "ACTIVATION"
            elif isinstance(module, nn.Conv2d):
                sem_cat = "POINTWISE"
            else:
                sem_cat = "OTHER"

            # Match sensitivity score
            matched_sens = 0.5
            for k, v in self.layer_sensitivities.items():
                if name in k or k in name:
                    matched_sens = v["score"]
                    break
                    
            if sem_cat == "STEM" or sem_cat == "CLASSIFIER":
                sens_tier = "HIGH"
            elif matched_sens > 0.65 or sem_cat == "SE":
                sens_tier = "HIGH"
            elif matched_sens > 0.40:
                sens_tier = "MEDIUM"
            else:
                sens_tier = "LOW"

            # Record if it is a leaf module or has parameters
            if len(list(module.children())) == 0 or param_count > 0:
                records.append({
                    "layer_name": name,
                    "operator_type": op_type,
                    "input_channels": str(in_ch) if in_ch is not None else "N/A",
                    "output_channels": str(out_ch) if out_ch is not None else "N/A",
                    "kernel": str(kernel) if kernel is not None else "N/A",
                    "stride": str(stride) if stride is not None else "N/A",
                    "groups": groups,
                    "parameter_count": param_count,
                    "weight_shape": str(weight_shape),
                    "bias": "YES" if has_bias else "NO",
                    "semantic_category": sem_cat,
                    "sensitivity_category": sens_tier,
                    "sensitivity_score": matched_sens
                })

        if output_csv_path:
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            fieldnames = [
                "layer_name", "operator_type", "input_channels", "output_channels",
                "kernel", "stride", "groups", "parameter_count", "weight_shape",
                "bias", "semantic_category", "sensitivity_category", "sensitivity_score"
            ]
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

        return records

    def apply_global_unstructured_pruning(
        self,
        model: nn.Module,
        target_sparsity: float
    ) -> Dict[str, Any]:
        """Applies global unstructured L1 magnitude pruning across Conv2d and Linear weights.
        
        Args:
            model: PyTorch model.
            target_sparsity: Target overall sparsity (0.0 to 1.0).
            
        Returns:
            Dict containing requested vs actual sparsity statistics.
        """
        # Collect parameters to prune (Conv2d and Linear weights only, excluding bias)
        parameters_to_prune: List[Tuple[nn.Module, str]] = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight") and module.weight is not None:
                    parameters_to_prune.append((module, "weight"))

        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=target_sparsity,
        )

        return self.compute_sparsity_stats(model, target_sparsity=target_sparsity)

    def apply_sensitivity_aware_unstructured_pruning(
        self,
        model: nn.Module,
        target_effective_sparsity: float,
        structure_audit: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Applies sensitivity-aware unstructured pruning.
        
        Layers with HIGH sensitivity receive low pruning (e.g. 0-10%).
        Layers with MEDIUM sensitivity receive moderate pruning (10-30%).
        Layers with LOW sensitivity receive higher pruning (20-60%).
        Classifier and Stem are protected/minimally pruned.
        
        Args:
            model: PyTorch model.
            target_effective_sparsity: Desired global effective sparsity (e.g. 0.20, 0.30, 0.40).
            structure_audit: Audit records from audit_model_structure.
            
        Returns:
            Tuple of (overall_stats, per_layer_allocations).
        """
        if structure_audit is None:
            structure_audit = self.audit_model_structure(model)

        audit_map = {r["layer_name"]: r for r in structure_audit}
        
        # Scaling factor alpha to tune per-tier sparsity towards target_effective_sparsity
        # Base multiplier per sensitivity tier:
        # LOW: 1.5 * base_rate
        # MEDIUM: 1.0 * base_rate
        # HIGH: 0.25 * base_rate
        # CLASSIFIER / STEM: 0.05 * base_rate
        
        prunable_modules: List[Tuple[str, nn.Module, str, str, int]] = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight") and module.weight is not None:
                    info = audit_map.get(name, {})
                    sens_tier = info.get("sensitivity_category", "MEDIUM")
                    sem_cat = info.get("semantic_category", "OTHER")
                    num_weights = module.weight.numel()
                    prunable_modules.append((name, module, sens_tier, sem_cat, num_weights))

        # Binary search / linear solve for base_rate to match target total zero weights
        total_prunable_weights = sum(m[4] for m in prunable_modules)
        target_zero_weights = int(total_prunable_weights * target_effective_sparsity)

        def get_rate(name: str, sens_tier: str, sem_cat: str, alpha: float) -> float:
            if name == "classifier.3":
                # Final 9-class classifier head -> strongly protected
                rate = 0.05 * alpha
            elif sem_cat == "STEM":
                # Stem input conv -> protected
                rate = 0.15 * alpha
            elif sem_cat == "DEPTHWISE":
                # Depthwise 3x3 / 5x5 convs -> protected
                rate = 0.35 * alpha
            elif sem_cat == "SE":
                # SE squeeze-and-excitation channels -> protected
                rate = 0.35 * alpha
            elif name == "classifier.0":
                # FC projection (576 -> 1024) -> moderately prunable
                rate = 0.95 * alpha
            elif sens_tier == "HIGH":
                rate = 0.30 * alpha
            elif sens_tier == "MEDIUM":
                rate = 0.85 * alpha
            else: # LOW
                rate = 1.15 * alpha
            return float(np.clip(rate, 0.0, 0.80))

        # Optimize alpha
        low_a, high_a = 0.0, 2.0
        best_alpha = 0.5
        for _ in range(30):
            mid_a = (low_a + high_a) / 2.0
            cur_zeros = sum(int(m[4] * get_rate(m[0], m[2], m[3], mid_a)) for m in prunable_modules)
            if cur_zeros < target_zero_weights:
                low_a = mid_a
                best_alpha = mid_a
            else:
                high_a = mid_a
                best_alpha = mid_a

        # Apply per-layer pruning
        layer_allocations: List[Dict[str, Any]] = []
        for name, module, sens_tier, sem_cat, num_weights in prunable_modules:
            rate = get_rate(name, sens_tier, sem_cat, best_alpha)
            if rate > 0.001:
                prune.l1_unstructured(module, name="weight", amount=rate)
            else:
                # 0% prune: apply identity mask so model state is uniform
                prune.identity(module, name="weight")

            actual_zeros = int(torch.sum(module.weight == 0).item())
            layer_allocations.append({
                "layer_name": name,
                "semantic_category": sem_cat,
                "sensitivity_category": sens_tier,
                "total_weights": num_weights,
                "allocated_sparsity": rate,
                "zero_weights": actual_zeros,
                "actual_layer_sparsity": actual_zeros / max(1, num_weights)
            })

        stats = self.compute_sparsity_stats(model, target_sparsity=target_effective_sparsity)
        return stats, layer_allocations

    def make_pruning_permanent(self, model: nn.Module) -> None:
        """Removes PyTorch pruning forward hooks and burns zeros permanently into weights."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight_orig"):
                    try:
                        prune.remove(module, "weight")
                    except Exception:
                        pass

    def compute_sparsity_stats(self, model: nn.Module, target_sparsity: float = 0.0) -> Dict[str, Any]:
        """Computes exact parameter counts, zero weights, and sparsity across all weights."""
        total_params = 0
        total_weights = 0
        zero_weights = 0
        nonzero_weights = 0

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight") and module.weight is not None:
                    w = module.weight
                    n_elems = w.numel()
                    total_weights += n_elems
                    z = int(torch.sum(w == 0).item())
                    zero_weights += z
                    nonzero_weights += (n_elems - z)

        for p in model.parameters():
            total_params += p.numel()

        actual_sparsity = zero_weights / max(1, total_weights)
        return {
            "requested_sparsity": target_sparsity,
            "actual_sparsity": float(actual_sparsity),
            "total_weights": total_weights,
            "zero_weights": zero_weights,
            "nonzero_weights": nonzero_weights,
            "total_parameters": total_params,
            "nonzero_parameters": total_params - zero_weights
        }
