"""
UAQE Phase D.4 Adaptive Multi-Objective Optimization Planner
Implements deterministic risk-aware strategy rules, multi-objective scoring,
greedy layer-aware search with validation accuracy gating, and plan serialization.
"""

from __future__ import annotations

import os
import json
import csv
import copy
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np


class StrategyType:
    KEEP_INT8 = "KEEP_INT8"
    INT8_PRUNE_10 = "INT8_PRUNE_10"
    INT8_PRUNE_20 = "INT8_PRUNE_20"
    INT8_PRUNE_30 = "INT8_PRUNE_30"
    INT8_PRUNE_40 = "INT8_PRUNE_40"
    INT8_SPARSE_RLE = "INT8_SPARSE_RLE"
    INT8_CLUSTER_32 = "INT8_CLUSTER_32"
    INT8_CLUSTER_64 = "INT8_CLUSTER_64"


class MultiObjectiveScorer:
    """Calculates multi-objective score over accuracy, storage, latency, and memory."""

    PRESETS = {
        "accuracy_first": {"w_a": 0.70, "w_s": 0.15, "w_l": 0.10, "w_m": 0.05},
        "balanced": {"w_a": 0.50, "w_s": 0.25, "w_l": 0.15, "w_m": 0.10},
        "storage_first": {"w_a": 0.40, "w_s": 0.45, "w_l": 0.10, "w_m": 0.05},
        "latency_first": {"w_a": 0.50, "w_s": 0.15, "w_l": 0.30, "w_m": 0.05}
    }

    def __init__(self, mode: str = "balanced", custom_weights: Optional[Dict[str, float]] = None):
        self.mode = mode
        if custom_weights is not None:
            self.weights = custom_weights
        else:
            self.weights = self.PRESETS.get(mode, self.PRESETS["balanced"])

    def compute_score(
        self,
        accuracy_ratio: float,      # Acc / BaselineAcc (e.g. 0.98 / 0.98 = 1.0)
        storage_reduction: float,   # Storage reduction fraction (e.g. 0.25 for 25%)
        latency_benefit: float,     # Normalized latency benefit (e.g. MAC reduction or speedup fraction)
        memory_benefit: float       # Normalized memory benefit
    ) -> float:
        """Calculates Score = w_a*A + w_s*S + w_l*L + w_m*M."""
        w_a = self.weights.get("w_a", 0.50)
        w_s = self.weights.get("w_s", 0.25)
        w_l = self.weights.get("w_l", 0.15)
        w_m = self.weights.get("w_m", 0.10)
        
        score = (
            w_a * float(np.clip(accuracy_ratio, 0.0, 1.05)) +
            w_s * float(np.clip(storage_reduction, 0.0, 1.0)) +
            w_l * float(np.clip(latency_benefit, 0.0, 1.0)) +
            w_m * float(np.clip(memory_benefit, 0.0, 1.0))
        )
        return float(score)


class AdaptivePlanner:
    """Plans layer-specific optimization and compression strategies."""

    def __init__(
        self,
        sensitivity_threshold_high: float = 0.80,
        sensitivity_threshold_mid: float = 0.60,
        sensitivity_threshold_low: float = 0.30,
        val_accuracy_floor: float = 0.970,  # 97.0% validation gate
        scoring_mode: str = "balanced"
    ):
        self.thresh_high = sensitivity_threshold_high
        self.thresh_mid = sensitivity_threshold_mid
        self.thresh_low = sensitivity_threshold_low
        self.val_accuracy_floor = val_accuracy_floor
        self.scorer = MultiObjectiveScorer(mode=scoring_mode)

    def select_rule_based_strategy(
        self,
        layer_profile: Dict[str, Any],
        allow_clustering: bool = False
    ) -> Dict[str, Any]:
        """Applies deterministic risk-aware heuristics to assign an initial strategy.
        
        Rules:
            S_i >= thresh_high (0.80) -> KEEP_INT8 (protect)
            thresh_mid <= S_i < thresh_high (0.60..0.80) -> PRUNE_10 + SPARSE_RLE
            thresh_low <= S_i < thresh_mid (0.30..0.60) -> PRUNE_20 + SPARSE_RLE
            S_i < thresh_low (0.30) -> PRUNE_30 + SPARSE_RLE (or selective clustering if enabled)
            
        Strict Protection:
            Depthwise, SE, and Classifier layers are NEVER clustered.
        """
        name = layer_profile["layer_name"]
        cat = layer_profile.get("semantic_category", "other")
        score = layer_profile.get("sensitivity_score", 0.5)

        # 1. High Sensitivity: Protect (Dense INT8)
        if score >= self.thresh_high or cat in ["classifier"]:
            return {
                "layer_name": name,
                "category": cat,
                "sensitivity_score": score,
                "pruning_ratio": 0.0,
                "compression_strategy": "dense",
                "cluster_count": 0,
                "strategy_label": StrategyType.KEEP_INT8,
                "rationale": "High sensitivity or critical classifier layer; protected with 0% pruning and dense storage."
            }

        # 2. Moderate-High Sensitivity: Conservative Pruning (10% + Sparse RLE)
        if score >= self.thresh_mid or cat in ["depthwise", "SE"]:
            # Note: Depthwise and SE are always kept at conservative pruning with lossless sparse/rle
            prune_ratio = 0.10 if score >= 0.70 else 0.20
            return {
                "layer_name": name,
                "category": cat,
                "sensitivity_score": score,
                "pruning_ratio": prune_ratio,
                "compression_strategy": "sparse_rle",
                "cluster_count": 0,
                "strategy_label": StrategyType.INT8_PRUNE_10 if prune_ratio == 0.10 else StrategyType.INT8_PRUNE_20,
                "rationale": f"Moderate sensitivity {cat} layer; conservative {int(prune_ratio*100)}% pruning with lossless Sparse+RLE."
            }

        # 3. Moderate-Low Sensitivity: Moderate Pruning (20% or 30% + Sparse RLE)
        if score >= self.thresh_low:
            return {
                "layer_name": name,
                "category": cat,
                "sensitivity_score": score,
                "pruning_ratio": 0.25,
                "compression_strategy": "sparse_rle",
                "cluster_count": 0,
                "strategy_label": StrategyType.INT8_PRUNE_20,
                "rationale": "Moderate-low sensitivity pointwise layer; 25% pruning with lossless Sparse+RLE."
            }

        # 4. Low Sensitivity: Aggressive Pruning or Selective Clustering
        if allow_clustering and cat in ["pointwise", "standard_conv"]:
            return {
                "layer_name": name,
                "category": cat,
                "sensitivity_score": score,
                "pruning_ratio": 0.30,
                "compression_strategy": "cluster32",
                "cluster_count": 32,
                "strategy_label": StrategyType.INT8_CLUSTER_32,
                "rationale": "Low sensitivity pointwise layer; 30% pruning with 32-centroid codebook clustering."
            }
        else:
            return {
                "layer_name": name,
                "category": cat,
                "sensitivity_score": score,
                "pruning_ratio": 0.35,
                "compression_strategy": "sparse_rle",
                "cluster_count": 0,
                "strategy_label": StrategyType.INT8_PRUNE_30,
                "rationale": "Low sensitivity layer; aggressive 35% pruning with lossless Sparse+RLE."
            }

    def generate_plan(
        self,
        layer_profiles: List[Dict[str, Any]],
        allow_clustering: bool = False
    ) -> List[Dict[str, Any]]:
        """Generates complete layer optimization plan for all layers."""
        plan = []
        for lp in layer_profiles:
            strat = self.select_rule_based_strategy(lp, allow_clustering=allow_clustering)
            strat["tensor_index"] = lp.get("tensor_index", -1)
            strat["buffer_index"] = lp.get("buffer_index", -1)
            strat["parameter_count"] = lp.get("parameter_count", 0)
            strat["input_shape"] = lp.get("input_shape", "[]")
            plan.append(strat)
        return plan

    def run_greedy_optimization(
        self,
        base_plan: List[Dict[str, Any]],
        val_eval_fn: Callable[[List[Dict[str, Any]]], Tuple[float, float, float]],
        # val_eval_fn takes a plan and returns (val_accuracy, estimated_size_bytes, estimated_latency_ms)
        baseline_size: int = 1856832,
        baseline_latency: float = 1.0,
        baseline_val_acc: float = 0.98,
        output_dir: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Executes greedy layer-aware search, accepting beneficial transforms that respect the validation gate."""
        current_plan = copy.deepcopy(base_plan)
        history = []
        
        # Initial evaluation
        curr_acc, curr_size, curr_lat = val_eval_fn(current_plan)
        curr_red = float((baseline_size - curr_size) / baseline_size)
        curr_lat_ben = float(max(0.0, (baseline_latency - curr_lat) / baseline_latency))
        curr_score = self.scorer.compute_score(curr_acc / baseline_val_acc, curr_red, curr_lat_ben, curr_red)
        
        iteration = 0
        history.append({
            "iteration": iteration,
            "layer": "INITIAL_BASELINE",
            "previous_strategy": "N/A",
            "new_strategy": "INITIAL_PLAN",
            "estimated_gain": round(curr_red * 100.0, 2),
            "val_accuracy": round(curr_acc * 100.0, 4),
            "objective_score": round(curr_score, 4),
            "accepted": True
        })
        
        candidate_transforms = [
            {"pruning_ratio": 0.30, "compression_strategy": "sparse_rle", "label": "PRUNE_30_SPARSE_RLE"},
            {"pruning_ratio": 0.40, "compression_strategy": "sparse_rle", "label": "PRUNE_40_SPARSE_RLE"},
            {"pruning_ratio": 0.20, "compression_strategy": "sparse_rle", "label": "PRUNE_20_SPARSE_RLE"},
        ]
        
        # Sort layers by sensitivity (try aggressive transforms on least sensitive layers first)
        sorted_indices = sorted(range(len(current_plan)), key=lambda idx: current_plan[idx]["sensitivity_score"])
        
        for idx in sorted_indices:
            layer = current_plan[idx]
            cat = layer.get("category", "")
            if cat in ["classifier", "depthwise", "SE"]:
                # Keep high-sensitivity layers protected
                continue
                
            prev_strat = copy.deepcopy(layer)
            
            for transform in candidate_transforms:
                iteration += 1
                trial_plan = copy.deepcopy(current_plan)
                trial_plan[idx]["pruning_ratio"] = transform["pruning_ratio"]
                trial_plan[idx]["compression_strategy"] = transform["compression_strategy"]
                trial_plan[idx]["strategy_label"] = transform["label"]
                
                trial_acc, trial_size, trial_lat = val_eval_fn(trial_plan)
                trial_red = float((baseline_size - trial_size) / baseline_size)
                trial_lat_ben = float(max(0.0, (baseline_latency - trial_lat) / baseline_latency))
                trial_score = self.scorer.compute_score(trial_acc / baseline_val_acc, trial_red, trial_lat_ben, trial_red)
                
                # Check validation accuracy floor and objective score improvement
                accepted = False
                if trial_acc >= self.val_accuracy_floor and trial_score >= curr_score:
                    accepted = True
                    current_plan = trial_plan
                    curr_acc, curr_size, curr_lat, curr_score = trial_acc, trial_size, trial_lat, trial_score
                
                history.append({
                    "iteration": iteration,
                    "layer": layer["layer_name"],
                    "previous_strategy": prev_strat["strategy_label"],
                    "new_strategy": transform["label"],
                    "estimated_gain": round(trial_red * 100.0, 2),
                    "val_accuracy": round(trial_acc * 100.0, 4),
                    "objective_score": round(trial_score, 4),
                    "accepted": accepted
                })
                
                if accepted:
                    break
                    
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            hist_csv = os.path.join(output_dir, "optimization_history.csv")
            if history:
                with open(hist_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                    writer.writeheader()
                    writer.writerows(history)
            plan_json = os.path.join(output_dir, "d4_plan.json")
            with open(plan_json, "w", encoding="utf-8") as f:
                json.dump(current_plan, f, indent=2)
                
        return current_plan, history
