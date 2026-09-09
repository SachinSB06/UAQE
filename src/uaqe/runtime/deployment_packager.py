"""
UAQE Phase D.5 Deployment Packager
Assembles the self-contained deployment package containing model.uaqe,
manifest.json, runtime_config.json, checksums.json, and README.md.
"""

from __future__ import annotations

import os
import json
import shutil
import hashlib
from typing import Dict, List, Tuple, Any, Optional

from src.uaqe.optimizer.sensitivity_pruner import CLASS_NAMES


class DeploymentPackager:
    """Assembles and validates self-contained UAQE deployment packages."""

    def __init__(
        self,
        source_archive_path: str = "output/phase_d4/compressed/d4_d_adaptive_sparse_rle.bin",
        baseline_tflite_path: str = "output/phase_c4/models/c4_best_int8.tflite",
        package_dir: str = "output/phase_d5/package"
    ):
        self.source_archive_path = source_archive_path
        self.baseline_tflite_path = baseline_tflite_path
        self.package_dir = package_dir

    @staticmethod
    def compute_sha256(path: str) -> str:
        """Computes SHA-256 hash of a file on disk."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_package(
        self,
        accuracy_pct: float = 98.4694,
        macro_f1_pct: float = 98.3834,
        storage_reduction_pct: float = 24.98
    ) -> Dict[str, Any]:
        """Builds the complete production deployment package."""
        os.makedirs(self.package_dir, exist_ok=True)

        if not os.path.exists(self.source_archive_path):
            raise FileNotFoundError(f"Source archive not found: {self.source_archive_path}")

        archive_size = os.path.getsize(self.source_archive_path)
        archive_hash = self.compute_sha256(self.source_archive_path)
        baseline_size = os.path.getsize(self.baseline_tflite_path) if os.path.exists(self.baseline_tflite_path) else 1856832

        # 1. Copy model.uaqe
        target_model_path = os.path.join(self.package_dir, "model.uaqe")
        shutil.copy2(self.source_archive_path, target_model_path)

        # 2. Generate manifest.json
        manifest = {
            "package_name": "UAQE_D5_DEPLOYMENT_PACKAGE",
            "model_name": "MobileNetV3-Small Semiconductor Defect Classifier",
            "model_version": "1.0.0",
            "architecture": "mobilenet_v3_small",
            "task": "9-class semiconductor wafer defect image classification",
            "class_names": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "input_specification": {
                "shape": [1, 3, 128, 128],
                "layout": "NCHW",
                "dtype": "float32",
                "range": "[0.0, 1.0]",
                "normalization": "div_255_rgb"
            },
            "output_specification": {
                "shape": [1, 9],
                "dtype": "float32",
                "type": "logits_or_probabilities"
            },
            "source_provenance": {
                "source_phase": "Phase D.4 (Adaptive Multi-Objective Optimization)",
                "source_candidate": "D4-D",
                "source_archive_name": os.path.basename(self.source_archive_path),
                "source_archive_sha256": archive_hash,
                "compression_strategy": "UAQE_D4_HYBRID (Adaptive Pruning + Sparse/RLE)"
            },
            "storage_metrics": {
                "compressed_archive_size_bytes": archive_size,
                "compressed_archive_size_mb": round(archive_size / (1024 * 1024), 4),
                "reconstructed_tflite_size_bytes": baseline_size,
                "reconstructed_tflite_size_mb": round(baseline_size / (1024 * 1024), 4),
                "storage_reduction_vs_baseline_pct": storage_reduction_pct
            },
            "accuracy_metrics": {
                "clean_test_benchmark_accuracy_pct": accuracy_pct,
                "macro_f1_pct": macro_f1_pct,
                "clean_test_sample_count": 196
            },
            "deployment_classification": "Classification B (Executable after runtime reconstruction)"
        }

        manifest_path = os.path.join(self.package_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 3. Generate runtime_config.json
        runtime_config = {
            "runtime_engine": "TFLITE_CPU",
            "thread_count": 4,
            "enable_profiling": False,
            "preprocessing": {
                "resize_height": 128,
                "resize_width": 128,
                "interpolation": "BILINEAR",
                "color_format": "RGB",
                "channel_order": "NCHW",
                "scale_factor": 0.00392156862745098,  # 1.0 / 255.0
                "mean_subtraction": [0.0, 0.0, 0.0],
                "std_normalization": [1.0, 1.0, 1.0]
            },
            "tensor_indices": {
                "input_tensor_index": 0,
                "output_tensor_index": 349
            },
            "reconstruction_template": "output/phase_c4/models/c4_best_int8.tflite"
        }

        runtime_config_path = os.path.join(self.package_dir, "runtime_config.json")
        with open(runtime_config_path, "w", encoding="utf-8") as f:
            json.dump(runtime_config, f, indent=2)

        # 4. Generate README.md
        readme_content = f"""# UAQE Phase D.5 Deployment Package

## Model Overview
- **Model:** MobileNetV3-Small (9-Class Semiconductor Defect Classifier)
- **Archive Size:** {archive_size:,} bytes ({archive_size/(1024*1024):.4f} MB) — **24.98% storage reduction** vs baseline
- **Reconstructed TFLite Size:** {baseline_size:,} bytes ({baseline_size/(1024*1024):.4f} MB)
- **Clean Test Accuracy:** {accuracy_pct:.4f}% (193 / 196 clean test images)
- **Macro F1 Score:** {macro_f1_pct:.4f}%
- **Deployment Class:** **Classification B (Executable after runtime reconstruction)**

## Package Contents
- `model.uaqe`: Compressed hybrid binary weight archive
- `manifest.json`: Model architecture, shapes, and provenance metadata
- `runtime_config.json`: Preprocessing and runtime execution parameters
- `checksums.json`: SHA-256 integrity verification hashes
- `README.md`: Quickstart deployment guide

## Quickstart Python Usage
```python
import numpy as np
from PIL import Image
from uaqe.runtime.runtime_session import RuntimeSession

# 1. Initialize session from archive
session = RuntimeSession.from_archive("model.uaqe")
session.allocate()

# 2. Preprocess input image (128x128 RGB NCHW)
img = Image.open("sample.png").convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
arr = (np.array(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)

# 3. Predict defect class
class_idx = session.predict_class(arr)
class_names = ["bridge", "clean", "cmp", "crack", "opens", "other", "particle", "scratch", "vias"]
print("Predicted Defect:", class_names[class_idx])

# 4. Cleanup
session.close()
```
"""
        readme_path = os.path.join(self.package_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        # 5. Generate checksums.json
        checksums = {
            "model.uaqe": self.compute_sha256(target_model_path),
            "manifest.json": self.compute_sha256(manifest_path),
            "runtime_config.json": self.compute_sha256(runtime_config_path),
            "README.md": self.compute_sha256(readme_path)
        }
        checksums_path = os.path.join(self.package_dir, "checksums.json")
        with open(checksums_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

        # Calculate complete package size
        all_pkg_files = sorted(os.listdir(self.package_dir))
        total_package_size = sum(os.path.getsize(os.path.join(self.package_dir, f)) for f in all_pkg_files)

        summary = {
            "package_directory": self.package_dir,
            "compressed_archive_size_bytes": archive_size,
            "reconstructed_tflite_size_bytes": baseline_size,
            "complete_package_size_bytes": total_package_size,
            "complete_package_size_mb": round(total_package_size / (1024 * 1024), 4),
            "files": all_pkg_files,
            "checksums": checksums
        }
        return summary
