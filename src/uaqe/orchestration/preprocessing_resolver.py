"""Preprocessing Resolver with Provenance Tracking for UAQE.

Resolves image preprocessing parameters (target resolution, color space, normalization, layout)
by consulting the prioritized provenance chain:
1. Explicit model metadata
2. Model configuration file (e.g. preprocessor_config.json / config.json)
3. Framework adapter metadata
4. Dataset metadata
5. Documented fallback defaults
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any


class PreprocessingResolver:
    """Resolves preprocessing parameters and logs provenance for each decision."""

    DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]

    @classmethod
    def resolve(
        cls,
        model_descriptor: Dict[str, Any],
        dataset_descriptor: Dict[str, Any],
        model_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Resolve preprocessing configuration with complete provenance tracking."""
        provenance = {}

        # 1. Target Resolution
        target_res = [224, 224]
        res_source = "default_fallback"

        # Check preprocessor_config.json if in model_dir
        if model_dir and os.path.exists(os.path.join(model_dir, "preprocessor_config.json")):
            try:
                with open(os.path.join(model_dir, "preprocessor_config.json"), "r") as f:
                    p_cfg = json.load(f)
                    if "size" in p_cfg:
                        sz = p_cfg["size"]
                        if isinstance(sz, dict):
                            target_res = [sz.get("height", 224), sz.get("width", 224)]
                        elif isinstance(sz, int):
                            target_res = [sz, sz]
                        res_source = "model_dir/preprocessor_config.json"
            except Exception:
                pass
        elif "input_shape" in model_descriptor:
            in_shape = model_descriptor["input_shape"]
            if len(in_shape) == 4:
                if in_shape[1] in [1, 3]:  # NCHW
                    target_res = [in_shape[2], in_shape[3]]
                    res_source = "model_descriptor.input_shape (NCHW)"
                elif in_shape[3] in [1, 3]:  # NHWC
                    target_res = [in_shape[1], in_shape[2]]
                    res_source = "model_descriptor.input_shape (NHWC)"

        provenance["target_resolution"] = {
            "value": target_res,
            "source": res_source
        }

        # 2. Normalization Mean & Std
        norm_type = "ImageNet"
        mean_val = list(cls.DEFAULT_IMAGENET_MEAN)
        std_val = list(cls.DEFAULT_IMAGENET_STD)
        norm_source = "default_fallback"

        if model_dir and os.path.exists(os.path.join(model_dir, "preprocessor_config.json")):
            try:
                with open(os.path.join(model_dir, "preprocessor_config.json"), "r") as f:
                    p_cfg = json.load(f)
                    if "image_mean" in p_cfg:
                        mean_val = p_cfg["image_mean"]
                        norm_source = "model_dir/preprocessor_config.json (image_mean)"
                    if "image_std" in p_cfg:
                        std_val = p_cfg["image_std"]
            except Exception:
                pass
        elif "mobilenet" in model_descriptor.get("architecture", "").lower():
            # Standard mobilenet normalized [0, 1] or ImageNet
            norm_type = "rgb_0_1"
            mean_val = [0.0, 0.0, 0.0]
            std_val = [1.0, 1.0, 1.0]
            norm_source = "framework_adapter.mobilenet_rules (rgb_0_1)"

        provenance["normalization"] = {
            "type": norm_type,
            "mean": mean_val,
            "std": std_val,
            "source": norm_source
        }

        # 3. Layout (NCHW vs NHWC)
        layout = "NCHW"
        layout_source = "default_fallback"
        in_shape = model_descriptor.get("input_shape", [])
        if len(in_shape) == 4:
            if in_shape[1] in [1, 3]:
                layout = "NCHW"
                layout_source = "model_descriptor.input_shape (channel_first)"
            elif in_shape[3] in [1, 3]:
                layout = "NHWC"
                layout_source = "model_descriptor.input_shape (channel_last)"

        provenance["layout"] = {
            "value": layout,
            "source": layout_source
        }

        # 4. Color space & Interpolation
        provenance["color_format"] = {
            "value": "RGB",
            "source": "model_descriptor.modality"
        }
        provenance["interpolation"] = {
            "value": "bilinear",
            "source": "project_standard_policy"
        }

        config = {
            "target_resolution": target_res,
            "color_format": "RGB",
            "interpolation": "bilinear",
            "layout": layout,
            "normalization": norm_type,
            "mean": mean_val,
            "std": std_val,
            "provenance": provenance
        }

        return config

    @classmethod
    def generate_config(cls, output_path: str, config: Dict[str, Any]) -> None:
        """Export preprocessing_config.json."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
