"""Capability-Driven Optimization Candidate Generator for UAQE.

Inspects model graph capabilities, operator support, dataset characteristics,
hardware constraints, optimization profile, and previous execution history
to generate legal, non-duplicate optimization candidate specifications.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set


@dataclass(frozen=True)
class OptimizationCandidate:
    """Specification of an executable optimization candidate."""
    candidate_id: str
    name: str
    strategy_type: str
    description: str
    config: Dict[str, Any] = field(default_factory=dict)
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CandidateGenerator:
    """Generates legal, capability-driven optimization candidates."""

    def __init__(
        self,
        model_descriptor: Dict[str, Any],
        dataset_descriptor: Dict[str, Any],
        hardware_profile: Dict[str, Any],
        profile: str = "balanced",
        max_budget: int = 10
    ):
        self.model_desc = model_descriptor
        self.dataset_desc = dataset_descriptor
        self.hw_profile = hardware_profile
        self.profile = profile.lower().strip()
        self.max_budget = max_budget
        self._generated_signatures: Set[str] = set()
        self._candidate_index = 0

    @staticmethod
    def compute_signature(strategy_type: str, config: Dict[str, Any]) -> str:
        """Compute deterministic SHA-256 signature for a candidate configuration."""
        canonical_json = json.dumps({"strategy": strategy_type, "config": config}, sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]

    def _next_id(self) -> str:
        self._candidate_index += 1
        return f"cand_{self._candidate_index:03d}"

    def generate_initial_candidate(self) -> OptimizationCandidate:
        """Generate the standard initial candidate for the model architecture."""
        arch = self.model_desc.get("architecture", "").lower()
        format_name = self.model_desc.get("format", "").lower()

        if "resnet" in arch or "resnet" in self.model_desc.get("model_family", "").lower():
            config = {
                "nodes_to_exclude": [],
                "quant_format": "QDQ",
                "activation_type": "QInt8",
                "weight_type": "QInt8",
                "per_channel": True,
                "calibrate_method": "MinMax"
            }
            cand_id = self._next_id()
            sig = self.compute_signature("resnet_onnx_static_int8_ptq", config)
            self._generated_signatures.add(sig)
            return OptimizationCandidate(
                candidate_id=cand_id,
                name="Standard Static INT8 PTQ",
                strategy_type="resnet_onnx_ptq",
                description="Uniform static INT8 QDQ quantization with per-channel weights.",
                config=config,
                signature=sig
            )

        elif "mobilenet" in arch:
            config = {
                "strategy": "tflite_int8_ptq",
                "target_sparsity": 0.0,
                "compression": "none"
            }
            cand_id = self._next_id()
            sig = self.compute_signature("mobilenet_adaptive", config)
            self._generated_signatures.add(sig)
            return OptimizationCandidate(
                candidate_id=cand_id,
                name="MobileNetV3 Static INT8 PTQ",
                strategy_type="mobilenet_adaptive",
                description="TFLite INT8 post-training quantization.",
                config=config,
                signature=sig
            )

        else:
            config = {"quant_precision": "INT8"}
            cand_id = self._next_id()
            sig = self.compute_signature("generic_fallback", config)
            self._generated_signatures.add(sig)
            return OptimizationCandidate(
                candidate_id=cand_id,
                name="Generic Standard INT8 PTQ",
                strategy_type="generic_fallback",
                description="Standard INT8 baseline quantization.",
                config=config,
                signature=sig
            )

    def generate_next_candidate(
        self,
        history: List[Dict[str, Any]],
        last_evaluation: Optional[Dict[str, Any]] = None
    ) -> Optional[OptimizationCandidate]:
        """Generate the next candidate based on previous evaluation feedback.
        
        If last candidate was CRITICAL, generates safer mixed-precision or protected-layer candidates.
        """
        if len(history) >= self.max_budget:
            return None

        arch = self.model_desc.get("architecture", "").lower()

        # 1. ResNet Architecture Candidates
        if "resnet" in arch or "resnet" in self.model_desc.get("model_family", "").lower():
            candidate_blueprints = [
                (
                    "Sensitivity-Aware Mixed Precision (Stem & Classifier Protected)",
                    "resnet_onnx_ptq",
                    "Protects initial stem convolution and classification head in FP32; quantizes remaining layers to INT8.",
                    {
                        "nodes_to_exclude": [
                            "/resnet/embedder/convolution/Conv",
                            "/classifier/classifier.1/Gemm"
                        ],
                        "quant_format": "QDQ",
                        "activation_type": "QInt8",
                        "weight_type": "QInt8",
                        "per_channel": True,
                        "calibrate_method": "MinMax"
                    }
                ),
                (
                    "Sensitivity-Aware Mixed Precision (Early Stages & Stem Protected)",
                    "resnet_onnx_ptq",
                    "Preserves high-sensitivity stem and Stage 0 representations in FP32 while quantizing deep compute stages to INT8.",
                    {
                        "nodes_to_exclude_pattern": ["stages.0", "embedder", "classifier"],
                        "quant_format": "QDQ",
                        "activation_type": "QInt8",
                        "weight_type": "QInt8",
                        "per_channel": True,
                        "calibrate_method": "MinMax"
                    }
                ),
                (
                    "Stage-Protected Static INT8 PTQ with Entropy Calibration",
                    "resnet_onnx_ptq",
                    "KL-divergence entropy calibration with sensitive early stage protection.",
                    {
                        "nodes_to_exclude_pattern": ["stages.0", "embedder", "classifier"],
                        "quant_format": "QDQ",
                        "activation_type": "QInt8",
                        "weight_type": "QInt8",
                        "per_channel": True,
                        "calibrate_method": "Entropy"
                    }
                ),
                (
                    "Sensitivity-Aware Mixed Precision (Stem + Stage 0 + Stage 1 Protected)",
                    "resnet_onnx_ptq",
                    "Preserves stem, Stage 0, and Stage 1 in FP32 for conservative accuracy recovery.",
                    {
                        "nodes_to_exclude_pattern": ["stages.0", "stages.1", "embedder", "classifier"],
                        "quant_format": "QDQ",
                        "activation_type": "QInt8",
                        "weight_type": "QInt8",
                        "per_channel": True,
                        "calibrate_method": "MinMax"
                    }
                ),
                (
                    "Stage-Protected Per-Tensor INT8 PTQ",
                    "resnet_onnx_ptq",
                    "Per-tensor quantized variant with sensitive early stage protection.",
                    {
                        "nodes_to_exclude_pattern": ["stages.0", "embedder", "classifier"],
                        "quant_format": "QDQ",
                        "activation_type": "QInt8",
                        "weight_type": "QInt8",
                        "per_channel": False,
                        "calibrate_method": "MinMax"
                    }
                ),
                (
                    "Stage-Protected UINT8 Activation PTQ",
                    "resnet_onnx_ptq",
                    "Unsigned INT8 activation quantization with sensitive early stage protection.",
                    {
                        "nodes_to_exclude_pattern": ["stages.0", "embedder", "classifier"],
                        "quant_format": "QDQ",
                        "activation_type": "QUInt8",
                        "weight_type": "QInt8",
                        "per_channel": True,
                        "calibrate_method": "MinMax"
                    }
                )
            ]

            for name, strat, desc, cfg in candidate_blueprints:
                sig = self.compute_signature(strat, cfg)
                if sig not in self._generated_signatures:
                    self._generated_signatures.add(sig)
                    cand_id = self._next_id()
                    return OptimizationCandidate(
                        candidate_id=cand_id,
                        name=name,
                        strategy_type=strat,
                        description=desc,
                        config=cfg,
                        signature=sig
                    )

        # 2. MobileNet Architecture Candidates
        elif "mobilenet" in arch:
            candidate_blueprints = [
                (
                    "MobileNetV3 INT8 PTQ + Magnitude Pruning",
                    "mobilenet_adaptive",
                    "INT8 quantization with 30% magnitude pruning on compute-heavy depthwise layers.",
                    {"strategy": "tflite_int8_pruned", "target_sparsity": 0.30, "compression": "none"}
                ),
                (
                    "MobileNetV3 INT8 + Adaptive Sparse RLE Packaging",
                    "mobilenet_adaptive",
                    "INT8 quantization combined with run-length encoding for zero-weight compression.",
                    {"strategy": "tflite_int8_rle", "target_sparsity": 0.40, "compression": "sparse_rle"}
                )
            ]

            # Probe capability for XNNPACK-compatible INT8 strategy
            from uaqe.optimization.strategies.xnnpack_int8 import XNNPACKCompatibleINT8Strategy, STRATEGY_NAME
            is_eligible, _ = XNNPACKCompatibleINT8Strategy.check_eligibility(
                model_descriptor=self.model_desc,
                hardware_profile=self.hw_profile
            )
            if is_eligible:
                candidate_blueprints.append((
                    "MobileNetV3 XNNPACK-Compatible INT8",
                    STRATEGY_NAME,
                    "Mathematically normalized INT8 quantization with corrected SE activation scales and re-scaled biases for XNNPACK CPU acceleration.",
                    {
                        "strategy": "xnnpack_compatible_int8",
                        "delegate": "XNNPACK",
                        "num_threads": 2,
                        "quant_precision": "INT8",
                        "compression": "none",
                        "normalize_se_activations": True
                    }
                ))

            for name, strat, desc, cfg in candidate_blueprints:
                sig = self.compute_signature(strat, cfg)
                if sig not in self._generated_signatures:
                    self._generated_signatures.add(sig)
                    cand_id = self._next_id()
                    return OptimizationCandidate(
                        candidate_id=cand_id,
                        name=name,
                        strategy_type=strat,
                        description=desc,
                        config=cfg,
                        signature=sig
                    )

        # 3. Generic Fallback
        return None
