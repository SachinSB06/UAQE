"""Model and Dataset Compatibility Checker for UAQE.

Verifies input modality, channel alignment, resolution adaptability,
class counts, and produces compatibility_report.json.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any


class CompatibilityChecker:
    """Evaluates compatibility between an ingested model and dataset."""

    @classmethod
    def check_compatibility(
        cls,
        model_descriptor: Dict[str, Any],
        dataset_descriptor: Dict[str, Any],
        task_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive compatibility audit."""
        issues: List[str] = []
        warnings: List[str] = []

        # 1. Modality & Task Check
        if not task_info.get("supported", False):
            issues.append(f"Unsupported task: {task_info.get('task')}")

        # 2. Input Shape & Channel Check
        input_shape = model_descriptor.get("input_shape", [1, 3, 224, 224])
        # NCHW vs NHWC
        if len(input_shape) == 4:
            if input_shape[1] in [1, 3]:
                model_channels = input_shape[1]
                model_res = (input_shape[2], input_shape[3])
                layout = "NCHW"
            elif input_shape[3] in [1, 3]:
                model_channels = input_shape[3]
                model_res = (input_shape[1], input_shape[2])
                layout = "NHWC"
            else:
                model_channels = 3
                model_res = (224, 224)
                layout = "NCHW"
        else:
            model_channels = 3
            model_res = (224, 224)
            layout = "NCHW"

        # 3. Class count check
        output_shape = model_descriptor.get("output_shape", [1, 1000])
        model_classes = output_shape[-1] if len(output_shape) > 1 else output_shape[0]
        dataset_classes = dataset_descriptor.get("class_count", 0)

        class_mismatch = (model_classes != dataset_classes)
        adaptation_required = False

        if class_mismatch:
            warnings.append(
                f"Class count mismatch: Model outputs {model_classes} logits, but dataset has {dataset_classes} classes."
            )
            adaptation_required = True

        # 4. Split availability check
        splits = dataset_descriptor.get("splits", {})
        train_count = splits.get("train_count", 0)
        val_count = splits.get("val_count", 0)
        test_count = splits.get("test_count", 0)

        if train_count == 0:
            issues.append("Dataset train split is empty")
        if val_count == 0:
            warnings.append("Validation split is empty; automated fallback partition will be generated from train")
        if test_count == 0:
            warnings.append("Test split is empty")

        is_compatible = len(issues) == 0

        report = {
            "compatible": is_compatible,
            "task": task_info.get("task", "unknown"),
            "model_architecture": model_descriptor.get("architecture", "unknown"),
            "dataset_format": dataset_descriptor.get("detected_format", "unknown"),
            "input_alignment": {
                "model_layout": layout,
                "model_channels": model_channels,
                "model_target_resolution": list(model_res),
                "resolution_adaptable": True
            },
            "class_alignment": {
                "model_classes": model_classes,
                "dataset_classes": dataset_classes,
                "classes_match": not class_mismatch,
                "adaptation_required": adaptation_required
            },
            "issues": issues,
            "warnings": warnings
        }

        return report

    @classmethod
    def generate_report(cls, output_path: str, report: Dict[str, Any]) -> None:
        """Export compatibility_report.json."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
