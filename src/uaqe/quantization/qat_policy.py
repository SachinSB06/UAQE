import os
import json
from typing import Dict, Any, List, Optional, Set
import torch
import torch.nn as nn
import torch.ao.quantization as ao_q

class QATPolicy:
    """Policy engine that consumes sensitivity metadata from Phase A.2 to configure
    tailored Quantization-Aware Training (QAT) observer strategies, fake-quantization
    ranges, and layer-specific protection policies.
    """

    def __init__(
        self,
        sensitivity_report_path: str = "output/quantization_sensitivity_report.json"
    ) -> None:
        self.sensitivity_report_path = sensitivity_report_path
        self.sensitivity_data: Dict[str, Any] = {}
        self.sensitive_layers: List[Dict[str, Any]] = []
        self.se_fc2_layers: Set[str] = set()
        self.depthwise_layers: Set[str] = set()
        self.hardswish_layers: Set[str] = set()
        
        self._load_sensitivity_data()

    def _load_sensitivity_data(self) -> None:
        """Loads and parses layer sensitivity metrics from the Phase A.2 diagnostic report."""
        if not os.path.exists(self.sensitivity_report_path):
            return

        with open(self.sensitivity_report_path, "r", encoding="utf-8") as f:
            self.sensitivity_data = json.load(f)

        for m in self.sensitivity_data.get("layer_metrics", []):
            layer_name = m.get("layer_name", "")
            cat = m.get("semantic_category", "")
            score = m.get("sensitivity_score", 0.0)

            if cat == "se_fc2" or "fc2" in layer_name:
                self.se_fc2_layers.add(layer_name)
            elif cat == "depthwise" or "block.1.0" in layer_name or m.get("operator_type") == "DepthwiseConv":
                self.depthwise_layers.add(layer_name)
            elif "Hardswish" in layer_name or "HardSigmoid" in layer_name:
                self.hardswish_layers.add(layer_name)

            if score > 0.5 or m.get("recommendation") == "QAT CANDIDATE":
                self.sensitive_layers.append(m)

    def get_standard_qat_qconfig(self) -> ao_q.QConfig:
        """Standard symmetric INT8 QAT configuration."""
        return ao_q.get_default_qat_qconfig("qnnpack")

    def get_sensitivity_aware_qconfig(self, layer_name: str) -> ao_q.QConfig:
        """Returns a specialized QConfig tuned to the sensitivity profile of the layer.
        For SE fc2 layers, uses smoother observer momentum to avoid activation collapse.
        For depthwise layers, uses per-channel weight fake-quant with unclipped activation tracking.
        """
        # Base QConfig
        if any(se_name in layer_name for se_name in ["fc2", "se", "features.11.block.2.fc2", "features.10.block.2.fc2"]):
            # Sensitive SE fc2 activation: high precision observer with momentum
            act_fake_quant = ao_q.FusedMovingAvgObsFakeQuantize.with_args(
                observer=ao_q.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                averaging_constant=0.01
            )
            weight_fake_quant = ao_q.FusedMovingAvgObsFakeQuantize.with_args(
                observer=ao_q.MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
            return ao_q.QConfig(activation=act_fake_quant, weight=weight_fake_quant)
        
        elif any(dw in layer_name for dw in ["block.1.0", "depthwise"]):
            # Depthwise convolution: per-channel weights with full dynamic range
            act_fake_quant = ao_q.FusedMovingAvgObsFakeQuantize.with_args(
                observer=ao_q.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                averaging_constant=0.05
            )
            weight_fake_quant = ao_q.FusedMovingAvgObsFakeQuantize.with_args(
                observer=ao_q.MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
            return ao_q.QConfig(activation=act_fake_quant, weight=weight_fake_quant)
        
        else:
            return ao_q.get_default_qat_qconfig("qnnpack")

    def apply_qat_policy(
        self,
        model: nn.Module,
        policy_mode: str = "standard"
    ) -> nn.Module:
        """Applies the selected QAT policy and prepares the model for Quantization-Aware Training.
        
        Args:
            model: Trainable PyTorch model in training mode.
            policy_mode: 'standard', 'sensitivity_aware', or 'learnable_range'.
        """
        model.train()

        if policy_mode == "standard":
            model.qconfig = self.get_standard_qat_qconfig()
            for name, module in model.named_modules():
                module.qconfig = model.qconfig

        elif policy_mode == "sensitivity_aware":
            default_qc = self.get_standard_qat_qconfig()
            model.qconfig = default_qc
            for name, module in model.named_modules():
                module.qconfig = self.get_sensitivity_aware_qconfig(name)

        elif policy_mode == "learnable_range":
            # Uses adaptive range scaling for sensitive blocks
            model.qconfig = self.get_standard_qat_qconfig()
            for name, module in model.named_modules():
                if any(se_key in name for se_key in ["fc2", "features.11", "features.10"]):
                    module.qconfig = self.get_sensitivity_aware_qconfig(name)
                else:
                    module.qconfig = model.qconfig
        else:
            model.qconfig = self.get_standard_qat_qconfig()

        prepared_model = ao_q.prepare_qat(model, inplace=False)
        return prepared_model
