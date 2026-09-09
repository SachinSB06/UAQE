"""Precision policy abstraction and sensitivity-guided policy builders.

Defines `PrecisionPolicy` to configure default model precision and selective
per-layer, per-block, or per-category precision overrides (e.g. FP16, FP32)
derived programmatically from Phase A.2 quantization sensitivity reports.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from uaqe.common.types import Precision


@dataclass
class PrecisionPolicy:
    """Configurable machine-readable policy defining layer-level precision routing."""

    name: str = "default_policy"
    default_precision: Precision = Precision.INT8
    layer_overrides: Dict[str, Precision] = field(default_factory=dict)
    block_overrides: Dict[str, Precision] = field(default_factory=dict)
    category_overrides: Dict[str, Precision] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve_precision(
        self,
        layer_name: str,
        block_name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Precision:
        """Resolve effective precision for a given layer.

        Priority order:
        1. Exact layer_name match
        2. Partial layer_name substring match
        3. Category override
        4. Block override
        5. Default precision
        """
        # 1. Exact layer match
        if layer_name in self.layer_overrides:
            return self.layer_overrides[layer_name]

        # 2. Substring match for canonical names (e.g. "features.11/block.2/fc2/Conv")
        norm_layer = layer_name.strip("/").replace("/", ".")
        for k, p in self.layer_overrides.items():
            norm_k = k.strip("/").replace("/", ".")
            if norm_k in norm_layer or norm_layer in norm_k:
                return p

        # 3. Category override
        if category and category in self.category_overrides:
            return self.category_overrides[category]

        # 4. Block override
        if block_name and block_name in self.block_overrides:
            return self.block_overrides[block_name]

        return self.default_precision

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "default_precision": self.default_precision.value,
            "layer_overrides": {k: v.value for k, v in self.layer_overrides.items()},
            "block_overrides": {k: v.value for k, v in self.block_overrides.items()},
            "category_overrides": {k: v.value for k, v in self.category_overrides.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PrecisionPolicy:
        """Construct policy from dictionary."""
        return cls(
            name=data.get("name", "custom_policy"),
            default_precision=Precision(data.get("default_precision", "INT8")),
            layer_overrides={
                k: Precision(v) for k, v in data.get("layer_overrides", {}).items()
            },
            block_overrides={
                k: Precision(v) for k, v in data.get("block_overrides", {}).items()
            },
            category_overrides={
                k: Precision(v) for k, v in data.get("category_overrides", {}).items()
            },
            metadata=data.get("metadata", {}),
        )

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize policy to JSON string or save to file."""
        serialized = json.dumps(self.to_dict(), indent=2)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(serialized)
        return serialized

    @classmethod
    def from_json(cls, path_or_json: str) -> PrecisionPolicy:
        """Deserialize policy from JSON string or file path."""
        if os.path.exists(path_or_json):
            with open(path_or_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(path_or_json)
        return cls.from_dict(data)

    # -------------------------------------------------------------------------
    # Factory Methods consuming Quantization Sensitivity Report
    # -------------------------------------------------------------------------

    @classmethod
    def from_sensitivity_report(
        cls,
        report_path: str,
        top_k: int = 3,
        target_precision: Precision = Precision.FP16,
        policy_name: Optional[str] = None,
    ) -> PrecisionPolicy:
        """Build policy selecting top-K most sensitive layers from sensitivity report."""
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        metrics = report.get("layer_metrics", [])
        # Already ranked by sensitivity_score descending
        selected = metrics[:top_k]
        overrides = {m["layer_name"]: target_precision for m in selected}

        return cls(
            name=policy_name or f"top_{top_k}_sensitive_{target_precision.value.lower()}",
            default_precision=Precision.INT8,
            layer_overrides=overrides,
            metadata={
                "source_report": report_path,
                "strategy": "top_k",
                "k": top_k,
                "target_precision": target_precision.value,
                "selected_layers": [m["layer_name"] for m in selected],
                "selected_scores": {m["layer_name"]: m.get("sensitivity_score") for m in selected},
            },
        )

    @classmethod
    def from_categories(
        cls,
        report_path: str,
        categories: Sequence[str],
        target_precision: Precision = Precision.FP16,
        policy_name: Optional[str] = None,
    ) -> PrecisionPolicy:
        """Build policy selecting all layers matching specified semantic categories."""
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        metrics = report.get("layer_metrics", [])
        target_cats = set(categories)
        selected = [m for m in metrics if m.get("semantic_category") in target_cats]
        overrides = {m["layer_name"]: target_precision for m in selected}

        return cls(
            name=policy_name or f"categories_{'_'.join(categories)}_{target_precision.value.lower()}",
            default_precision=Precision.INT8,
            layer_overrides=overrides,
            category_overrides={cat: target_precision for cat in target_cats},
            metadata={
                "source_report": report_path,
                "strategy": "categories",
                "categories": list(categories),
                "target_precision": target_precision.value,
                "selected_layers": [m["layer_name"] for m in selected],
                "selected_count": len(selected),
            },
        )

    @classmethod
    def from_sensitivity_threshold(
        cls,
        report_path: str,
        min_sensitivity: float = 0.70,
        target_precision: Precision = Precision.FP16,
        policy_name: Optional[str] = None,
    ) -> PrecisionPolicy:
        """Build policy selecting all layers with sensitivity_score >= min_sensitivity."""
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        metrics = report.get("layer_metrics", [])
        selected = [m for m in metrics if m.get("sensitivity_score", 0.0) >= min_sensitivity]
        overrides = {m["layer_name"]: target_precision for m in selected}

        return cls(
            name=policy_name or f"threshold_{min_sensitivity:.2f}_{target_precision.value.lower()}",
            default_precision=Precision.INT8,
            layer_overrides=overrides,
            metadata={
                "source_report": report_path,
                "strategy": "threshold",
                "min_sensitivity": min_sensitivity,
                "target_precision": target_precision.value,
                "selected_layers": [m["layer_name"] for m in selected],
                "selected_count": len(selected),
            },
        )

    @classmethod
    def from_se_and_depthwise(
        cls,
        report_path: str,
        target_precision: Precision = Precision.FP16,
        policy_name: Optional[str] = None,
    ) -> PrecisionPolicy:
        """Build policy selecting sensitive SE and Depthwise convolution operations."""
        return cls.from_categories(
            report_path=report_path,
            categories=["se_fc2", "depthwise_conv"],
            target_precision=target_precision,
            policy_name=policy_name or f"se_depthwise_{target_precision.value.lower()}",
        )
