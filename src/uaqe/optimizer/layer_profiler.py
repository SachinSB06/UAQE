"""
UAQE Phase D.4 Layer Profiler
Audits MobileNetV3 layers, analyzes weight statistics, computes information entropy,
determines activation/weight sensitivity, and calculates normalized layer sensitivity scores.
"""

from __future__ import annotations

import os
import json
import math
import csv
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb


class LayerProfiler:
    """Extracts structural, statistical, and sensitivity profiles for all layers."""

    def __init__(
        self,
        sensitivity_csv_path: str = "output/layer_sensitivity.csv",
        baseline_tflite_path: str = "output/phase_c4/models/c4_best_int8.tflite"
    ):
        self.sensitivity_csv_path = sensitivity_csv_path
        self.baseline_tflite_path = baseline_tflite_path
        self.sensitivity_data: Dict[str, Dict[str, Any]] = {}
        self._load_historical_sensitivity()

    def _load_historical_sensitivity(self) -> None:
        """Loads historical sensitivity analysis metrics from Phase A/C."""
        if os.path.exists(self.sensitivity_csv_path):
            with open(self.sensitivity_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("layer_name", "")
                    clean_name = name.replace("/", ".").strip(".")
                    self.sensitivity_data[clean_name] = {
                        "rank": int(row.get("rank", 999)),
                        "raw_score": float(row.get("sensitivity_score", 0.5)),
                        "semantic_category": row.get("semantic_category", "unknown"),
                        "operator_type": row.get("operator_type", "Conv"),
                        "weight_mae": float(row.get("weight_mae", 0.0)),
                        "weight_cosine": float(row.get("weight_cosine", 1.0)),
                        "activation_mae": float(row.get("activation_mae", 0.0)),
                        "activation_cosine": float(row.get("activation_cosine", 1.0)),
                        "clipping_rate": float(row.get("clipping_rate", 0.0))
                    }

    @staticmethod
    def compute_entropy(arr: np.ndarray) -> float:
        """Computes Shannon entropy of an integer/quantized array in bits."""
        if arr.size == 0:
            return 0.0
        flat = arr.flatten()
        _, counts = np.unique(flat, return_counts=True)
        probs = counts / flat.size
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)

    def classify_layer_type(self, tensor_name: str, shape: List[int]) -> Tuple[str, str]:
        """Classifies layer into structural operator type and semantic category.
        
        Categories:
            depthwise, pointwise, standard_conv, SE, classifier, normalization, activation, other
        """
        lower = tensor_name.lower()
        
        # Classifier
        if "classifier" in lower or "linear" in lower or (len(shape) == 2 and shape[0] == 9):
            return "Gemm", "classifier"
        
        # Squeeze-and-Excitation
        if "fc1" in lower or "fc2" in lower or "se" in lower:
            return "Conv", "SE"
        
        # Depthwise Conv (typically 3x3 or shape with group conv properties in TFLite: [1, H, W, C] or [O, H, W, I] with I=1)
        if len(shape) == 4:
            # TFLite depthwise weights are often [1, 3, 3, Channels] or [1, 5, 5, Channels]
            if shape[0] == 1 and (shape[1] in [3, 5] and shape[2] in [3, 5]):
                return "DepthwiseConv2D", "depthwise"
            if "depthwise" in lower or "block.1.0" in lower:
                return "DepthwiseConv2D", "depthwise"
            # 1x1 Pointwise / Projection / Expansion
            if shape[1] == 1 and shape[2] == 1:
                return "Conv2D", "pointwise"
            # Standard conv (e.g. stem 3x3 with 3 input channels)
            return "Conv2D", "standard_conv"
        
        if "bias" in lower or "add" in lower:
            return "Bias", "other"
            
        return "Unknown", "other"

    def calculate_sensitivity_score(
        self,
        layer_name: str,
        category: str,
        weight_arr: np.ndarray,
        hist_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculates normalized layer sensitivity score S_i in [0.0, 1.0].
        
        Formula:
            S_i = clip(0.40 * S_act + 0.20 * S_weight + 0.40 * S_type, 0.0, 1.0)
            
        where:
            - S_act: normalized activation error and cosine distortion
            - S_weight: weight variation & information entropy
            - S_type: structural sensitivity penalty based on operator category
        """
        # 1. Structural Type Prior
        type_priors = {
            "depthwise": 0.90,       # Highest sensitivity: spatial feature maps
            "SE": 0.85,              # Attention gating: channel recalibration
            "classifier": 0.80,      # Output classification boundary
            "standard_conv": 0.65,   # Stem & head spatial convolutions
            "pointwise": 0.25,       # High redundancy in 1x1 projection/expansion
            "other": 0.30
        }
        s_type = type_priors.get(category, 0.30)
        
        # 2. Activation Sensitivity Prior
        s_act = 0.50
        s_weight = 0.50
        if hist_info is not None:
            act_cos = hist_info.get("activation_cosine", 1.0)
            act_mae = hist_info.get("activation_mae", 0.0)
            # Higher MAE and lower cosine similarity increase sensitivity
            s_act = float(np.clip(0.5 * (1.0 - act_cos) / 1.5 + 0.5 * min(act_mae / 2.0, 1.0), 0.0, 1.0))
            
            w_cos = hist_info.get("weight_cosine", 1.0)
            s_weight = float(np.clip(1.0 - w_cos, 0.0, 1.0) * 10.0)
        else:
            # Fallback based on weight variance and entropy
            std = float(np.std(weight_arr)) if weight_arr.size > 0 else 0.0
            entropy = self.compute_entropy(weight_arr)
            s_weight = float(np.clip(entropy / 8.0 * 0.5 + min(std / 30.0, 0.5), 0.0, 1.0))

        # Composite score
        score = 0.40 * s_act + 0.20 * s_weight + 0.40 * s_type
        return float(np.clip(score, 0.0, 1.0))

    def profile_model(self, tflite_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extracts complete layer profile for all weight tensors in the model."""
        target_path = tflite_path or self.baseline_tflite_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target model for profiling not found: {target_path}")

        with open(target_path, "rb") as f:
            raw_buf = bytearray(f.read())

        model = schema_fb.Model.GetRootAsModel(raw_buf, 0)
        subgraph = model.Subgraphs(0)
        
        profiles = []
        for i in range(subgraph.TensorsLength()):
            t = subgraph.Tensors(i)
            b_idx = t.Buffer()
            b = model.Buffers(b_idx)
            
            if b and b.DataLength() > 0 and t.Type() == 9:  # INT8 weights
                name = t.Name().decode("utf-8") if t.Name() else f"tensor_{i}"
                shape = [t.Shape(j) for j in range(t.ShapeLength())]
                data_np = b.DataAsNumpy().astype(np.int8)
                param_count = int(data_np.size)
                
                op_type, category = self.classify_layer_type(name, shape)
                
                # Find matching historical sensitivity info
                clean_name = name.replace("/", ".").strip(".")
                hist_match = None
                for k, v in self.sensitivity_data.items():
                    if k in clean_name or clean_name in k:
                        hist_match = v
                        break
                
                sensitivity_score = self.calculate_sensitivity_score(name, category, data_np, hist_match)
                
                w_min = int(np.min(data_np)) if param_count > 0 else 0
                w_max = int(np.max(data_np)) if param_count > 0 else 0
                w_mean = float(np.mean(data_np)) if param_count > 0 else 0.0
                w_std = float(np.std(data_np)) if param_count > 0 else 0.0
                zero_count = int(np.sum(data_np == 0))
                zero_fraction = float(zero_count / max(1, param_count))
                entropy = self.compute_entropy(data_np)
                
                # Estimated MACs and memory
                # For conv: MACs roughly param_count * (output_h * output_w)
                est_macs = param_count * 16  # standard estimation proxy
                est_memory = param_count     # 1 byte per INT8 param
                est_size = param_count       # 1 byte per INT8 param
                
                profiles.append({
                    "tensor_index": i,
                    "buffer_index": b_idx,
                    "layer_name": name,
                    "layer_type": op_type,
                    "semantic_category": category,
                    "input_shape": str(shape),
                    "output_shape": str(shape),
                    "parameter_count": param_count,
                    "weight_dtype": "INT8",
                    "weight_min": w_min,
                    "weight_max": w_max,
                    "weight_mean": round(w_mean, 4),
                    "weight_std": round(w_std, 4),
                    "zero_fraction": round(zero_fraction, 4),
                    "weight_entropy": round(entropy, 4),
                    "activation_sensitivity": round(hist_match.get("activation_mae", 0.0) if hist_match else 0.0, 4),
                    "weight_sensitivity": round(hist_match.get("weight_mae", 0.0) if hist_match else 0.0, 4),
                    "mac_count": est_macs,
                    "estimated_memory": est_memory,
                    "estimated_size": est_size,
                    "sensitivity_score": round(sensitivity_score, 4)
                })
                
        return profiles

    def export_profile(self, output_dir: str, tflite_path: Optional[str] = None) -> Tuple[str, str]:
        """Profiles the model and exports layer_profile.csv and layer_profile.json."""
        os.makedirs(output_dir, exist_ok=True)
        profiles = self.profile_model(tflite_path)
        
        csv_path = os.path.join(output_dir, "layer_profile.csv")
        json_path = os.path.join(output_dir, "layer_profile.json")
        
        # Write CSV
        if profiles:
            fieldnames = list(profiles[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(profiles)
                
        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
            
        print(f"[LayerProfiler] Exported profile for {len(profiles)} layers to {csv_path} and {json_path}")
        return csv_path, json_path
