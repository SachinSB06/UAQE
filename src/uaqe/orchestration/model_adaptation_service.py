"""Model Adaptation Service for UAQE.

Manages explicit classification head adaptation when target dataset class count
differs from the pretrained model's classifier output dimension.
Ensures deterministic, verifiable, and auditable head replacement.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn

from uaqe.models.resnet50 import build_resnet50_cifar10, ResNetForImageClassification
from uaqe.quantization.pytorch_reconstructor import MobileNetV3Reconstructor


class ModelAdaptationService:
    """Service for inspecting, planning, executing, and auditing classifier head adaptation."""

    def __init__(self):
        pass

    def inspect_adaptation(
        self,
        model_descriptor: Dict[str, Any],
        target_classes: int
    ) -> Dict[str, Any]:
        """Analyze if adaptation is required and whether it is supported."""
        output_shape = model_descriptor.get("output_shape", [1, 1000])
        model_classes = output_shape[-1] if len(output_shape) > 1 else 1000
        arch = model_descriptor.get("architecture", "")

        mismatch = (model_classes != target_classes)
        supported = False
        method = "None"
        reason = "Class counts already match" if not mismatch else ""

        if mismatch:
            if "resnet" in arch.lower():
                supported = True
                method = "Replace Linear(2048, model_classes) with Linear(2048, target_classes) + Kaiming normal init"
                reason = f"Pretrained model has {model_classes} output classes; target dataset has {target_classes} classes."
            elif "mobilenet" in arch.lower():
                supported = True
                method = f"Reconstruct backbone and replace final Linear(1024, {model_classes}) with Linear(1024, {target_classes})"
                reason = f"Pretrained model has {model_classes} output classes; target dataset has {target_classes} classes."
            else:
                supported = False
                method = "Unsupported architecture for automated head adaptation"
                reason = f"Automated head adaptation not registered for architecture: {arch}"

        return {
            "adaptation_required": mismatch,
            "original_output_classes": model_classes,
            "new_output_classes": target_classes,
            "adaptation_supported": supported,
            "adaptation_method": method,
            "adaptation_reason": reason
        }

    def can_adapt(self, model_descriptor: Dict[str, Any], target_classes: int) -> bool:
        """Return True if adaptation is required and supported."""
        info = self.inspect_adaptation(model_descriptor, target_classes)
        return info["adaptation_required"] and info["adaptation_supported"]

    def adapt(
        self,
        model_descriptor: Dict[str, Any],
        model_path: str,
        target_classes: int,
        device: str = "cpu",
        checkpoint_mode: str = "USER_UPLOAD",
        benchmark_checkpoint_path: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply classifier head adaptation deterministically."""
        info = self.inspect_adaptation(model_descriptor, target_classes)
        if not info["adaptation_required"]:
            return None, {
                "adapted": False,
                "reason": "No adaptation required (class counts match)",
                "original_output_classes": info["original_output_classes"],
                "new_output_classes": target_classes,
                "adaptation_verified": True,
                "checkpoint_mode": checkpoint_mode,
                "weight_source": "ORIGINAL_MODEL",
                "adaptation_status": "NOT_REQUIRED",
                "adaptation_note": "Model classes match dataset classes; original model weights preserved."
            }

        if not info["adaptation_supported"]:
            return None, {
                "adapted": False,
                "reason": info["adaptation_reason"],
                "original_output_classes": info["original_output_classes"],
                "new_output_classes": target_classes,
                "adaptation_verified": False,
                "checkpoint_mode": checkpoint_mode,
                "weight_source": "ORIGINAL_MODEL",
                "adaptation_status": "UNSUPPORTED",
                "adaptation_note": info["adaptation_reason"]
            }

        arch = model_descriptor.get("architecture", "")
        param_count_before = model_descriptor.get("parameter_count", 0)

        if "resnet" in arch.lower() or model_descriptor.get("format") == "safetensors":
            # Adapt ResNet-50 safetensors
            adapted_model, meta = build_resnet50_cifar10(
                model_path,
                num_classes=target_classes,
                checkpoint_mode=checkpoint_mode,
                benchmark_checkpoint_path=benchmark_checkpoint_path,
                device=device
            )
            param_count_after = meta["total_parameters"]
            
            verified = self.verify_adaptation(adapted_model, target_classes, input_shape=[1, 3, 224, 224], device=device)
            
            adaptation_record = {
                "adapted": True,
                "architecture": meta["architecture"],
                "original_classifier": f"Linear(2048, {info['original_output_classes']})",
                "adapted_classifier": f"Linear(2048, {target_classes})",
                "original_output_classes": info["original_output_classes"],
                "new_output_classes": target_classes,
                "adaptation_reason": info["adaptation_reason"],
                "adaptation_method": info["adaptation_method"],
                "adaptation_supported": True,
                "adaptation_verified": verified,
                "parameter_count_before": param_count_before,
                "parameter_count_after": param_count_after,
                "checkpoint_mode": meta.get("checkpoint_mode", checkpoint_mode),
                "weight_source": meta.get("weight_source", "RANDOM_INITIALIZATION"),
                "adaptation_status": meta.get("adaptation_status", "RANDOM_HEAD"),
                "adaptation_note": meta.get("adaptation_note", "")
            }
            return adapted_model, adaptation_record

        elif "mobilenet" in arch.lower():
            # Adapt MobileNetV3
            reconstructor = MobileNetV3Reconstructor(model_path)
            if target_classes == 9:
                adapted_model, _ = reconstructor.build_9class_trainable_model()
                weight_source = "TRAINED_ADAPTATION"
                adaptation_status = "TRAINED_ADAPTATION"
                adaptation_note = "Adapted MobileNetV3 with semiconductor trained architecture weights."
            elif target_classes <= info.get("original_output_classes", 10):
                adapted_model, _ = reconstructor.build_10class_reference_model()
                orig_fc = adapted_model.classifier[3]
                new_fc = nn.Linear(1024, target_classes)
                with torch.no_grad():
                    new_fc.weight.copy_(orig_fc.weight[:target_classes])
                    new_fc.bias.copy_(orig_fc.bias[:target_classes])
                adapted_model.classifier[3] = new_fc
                weight_source = "TRAINED_ADAPTATION"
                adaptation_status = "TRAINED_ADAPTATION"
                adaptation_note = f"Adapted MobileNetV3 with sliced trained classifier weights ({target_classes}/{info['original_output_classes']} classes)."
            else:
                adapted_model, _ = reconstructor.build_10class_reference_model()
                adapted_model.classifier[3] = nn.Linear(1024, target_classes)
                nn.init.normal_(adapted_model.classifier[3].weight, 0, 0.01)
                nn.init.zeros_(adapted_model.classifier[3].bias)
                weight_source = "RANDOM_INITIALIZATION"
                adaptation_status = "RANDOM_HEAD"
                adaptation_note = f"Adapted MobileNetV3 classifier head randomly initialized for {target_classes} classes."

            param_count_after = sum(p.numel() for p in adapted_model.parameters())
            verified = self.verify_adaptation(adapted_model, target_classes, input_shape=[1, 3, 128, 128], device=device)

            adaptation_record = {
                "adapted": True,
                "architecture": "MobileNetV3-Small",
                "original_classifier": f"Linear(1024, {info['original_output_classes']})",
                "adapted_classifier": f"Linear(1024, {target_classes})",
                "original_output_classes": info["original_output_classes"],
                "new_output_classes": target_classes,
                "adaptation_reason": info["adaptation_reason"],
                "adaptation_method": info["adaptation_method"],
                "adaptation_supported": True,
                "adaptation_verified": verified,
                "parameter_count_before": param_count_before,
                "parameter_count_after": param_count_after,
                "checkpoint_mode": checkpoint_mode,
                "weight_source": weight_source,
                "adaptation_status": adaptation_status,
                "adaptation_note": adaptation_note
            }
            return adapted_model, adaptation_record

        raise NotImplementedError(f"Adaptation execution for architecture '{arch}' not implemented")

    def verify_adaptation(
        self,
        model: Any,
        target_classes: int,
        input_shape: List[int] = [1, 3, 224, 224],
        device: str = "cpu"
    ) -> bool:
        """Verify that adapted model accepts input and produces [B, target_classes] output."""
        try:
            if isinstance(model, torch.nn.Module):
                model.eval()
                dummy_input = torch.randn(*input_shape, device=device)
                with torch.no_grad():
                    out = model(dummy_input)
                return bool(out.shape[-1] == target_classes)
        except Exception:
            return False
        return False

    def generate_report(self, output_path: str, adaptation_record: Dict[str, Any]) -> None:
        """Write adaptation_report.json."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(adaptation_record, f, indent=2)
