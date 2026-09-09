"""Task Detection Service for UAQE.

Confidently infers machine learning task type from model tensor shapes and dataset metadata.
"""

from typing import Dict, List, Tuple, Optional, Any


class TaskDetector:
    """Infers task type (e.g. image_classification) from model and dataset metadata."""

    SUPPORTED_TASKS = ["image_classification"]

    @classmethod
    def detect_task(
        cls,
        model_descriptor: Dict[str, Any],
        dataset_descriptor: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect task type, capabilities, and confidence from model and dataset metadata."""
        input_shape = model_descriptor.get("input_shape", [])
        output_shape = model_descriptor.get("output_shape", [])
        class_count = dataset_descriptor.get("class_count", 0)
        detected_format = dataset_descriptor.get("detected_format", "unknown")
        model_task = model_descriptor.get("task", "")

        # 1. Inspect input tensor geometry: 4D tensor (N, C, H, W) or (N, H, W, C)
        is_4d_image = len(input_shape) == 4
        # Validate channel count for standard images (1=grayscale, 3=RGB, 4=RGBA)
        has_image_channels = False
        if is_4d_image:
            has_image_channels = (input_shape[1] in [1, 3, 4]) or (input_shape[3] in [1, 3, 4])

        # 2. Inspect output tensor geometry: 1D or 2D classification output (N, C) or (C,)
        is_classification_out = len(output_shape) in [1, 2]
        model_classes = output_shape[-1] if len(output_shape) > 1 else (output_shape[0] if output_shape else 0)

        # 3. Capability-driven evaluation
        if is_4d_image and is_classification_out:
            # Model architecture and tensor flow unequivocally represent image classification
            is_dataset_image = detected_format in ["image_folder", "cifar10_pickle", "csv_labeled_images"]
            
            if class_count > 1 and is_dataset_image:
                confidence = "high"
                warnings = []
            else:
                confidence = "medium"
                warnings = [
                    f"Model graph exhibits 4D image input {input_shape} and {model_classes}-class classification logits, "
                    f"but dataset reports class_count={class_count}. Class alignment and adaptation will be audited in compatibility phase."
                ]

            return {
                "task": "image_classification",
                "detected_candidates": ["image_classification"],
                "confidence": confidence,
                "input_modality": "2D_image",
                "target_representation": "discrete_categorical_classes",
                "supported": True,
                "model_classes": model_classes,
                "dataset_classes": class_count,
                "warnings": warnings
            }

        # 4. If not supported vision classification, provide structured actionable explanation
        return {
            "task": "unknown",
            "detected_candidates": [],
            "confidence": "low",
            "input_modality": "2D_image" if is_4d_image else "unknown",
            "target_representation": "discrete_categorical_classes" if is_classification_out else "unknown",
            "supported": False,
            "exact_reason": (
                f"Model input shape {input_shape} (rank {len(input_shape)}) and output shape {output_shape} (rank {len(output_shape)}) "
                f"do not match supported image classification patterns (requires 4D input tensor and 1D/2D output logits)."
            ),
            "actionable_explanation": (
                "Please ensure the provided model accepts 4D image inputs [batch, channels, height, width] and "
                "produces 2D categorical logits [batch, num_classes]."
            ),
            "error": f"Cannot confidently infer supported task for input_shape {input_shape} and output_shape {output_shape}"
        }
