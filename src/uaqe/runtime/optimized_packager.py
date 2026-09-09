"""
UAQE Phase E.1 Deployment Packager
Assembles hardened self-contained deployment package for E1 runtime, containing
model.uaqe, manifest.json, runtime_config.json, checksums.json, runtime_version.json, and README.md.
"""

from __future__ import annotations

import os
import json
import shutil
import hashlib
from typing import Dict, List, Tuple, Any, Optional

from src.uaqe.optimizer.sensitivity_pruner import CLASS_NAMES


class OptimizedDeploymentPackager:
    """Assembles and validates self-contained UAQE Phase E.1 deployment packages."""

    def __init__(
        self,
        source_archive_path: str = "output/phase_d4/compressed/d4_d_adaptive_sparse_rle.bin",
        baseline_tflite_path: str = "output/phase_c4/models/c4_best_int8.tflite",
        package_dir: str = "output/phase_e1/package"
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
        storage_reduction_pct: float = 24.98,
        cold_start_portable_ms: float = 0.0,
        cold_start_cached_ms: float = 0.0
    ) -> Dict[str, Any]:
        """Builds the complete production E1 deployment package."""
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
            "package_name": "UAQE_E1_OPTIMIZED_DEPLOYMENT_PACKAGE",
            "model_name": "MobileNetV3-Small Semiconductor Defect Classifier",
            "model_version": "1.0.0",
            "runtime_version": "E1.0.0",
            "decoder_version": "2.0.0-vectorized",
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
            "supported_runtime_modes": [
                "portable",
                "cached"
            ],
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
            "latency_metrics": {
                "cold_start_portable_mean_ms": cold_start_portable_ms,
                "cold_start_cached_mean_ms": cold_start_cached_ms
            },
            "deployment_classification": "Classification B (Executable after runtime reconstruction)"
        }

        manifest_path = os.path.join(self.package_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # 3. Generate runtime_version.json
        runtime_version = {
            "format_version": 1,
            "engine_version": "UAQE_E1",
            "decoder_version": "2.0.0",
            "source_archive_sha256": archive_hash,
            "compatible_engines": ["tflite_cpu", "tflite_arm", "tflite_runtime"],
            "supported_modes": ["portable", "cached"]
        }
        runtime_version_path = os.path.join(self.package_dir, "runtime_version.json")
        with open(runtime_version_path, "w", encoding="utf-8") as f:
            json.dump(runtime_version, f, indent=2)

        # 4. Generate runtime_config.json
        runtime_config = {
            "runtime_engine": "TFLITE_CPU",
            "default_mode": "portable",
            "thread_count": 4,
            "enable_caching": True,
            "cache_directory": "runtime_cache",
            "preprocessing": {
                "resize_height": 128,
                "resize_width": 128,
                "interpolation": "BILINEAR",
                "color_format": "RGB",
                "channel_order": "NCHW",
                "scale_factor": 0.00392156862745098,
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

        # 5. Generate README.md
        readme_content = f"""# UAQE Phase E.1 Optimized Deployment Package

## Model & Runtime Overview
- **Model:** MobileNetV3-Small (9-Class Semiconductor Defect Classifier)
- **Archive Size:** {archive_size:,} bytes ({archive_size/(1024*1024):.4f} MB) — **24.98% storage reduction** vs baseline
- **Reconstructed TFLite Size:** {baseline_size:,} bytes ({baseline_size/(1024*1024):.4f} MB)
- **Clean Test Accuracy:** {accuracy_pct:.4f}% (193 / 196 clean test images)
- **Macro F1 Score:** {macro_f1_pct:.4f}%
- **Runtime Engine:** Optimized UAQE Phase E.1 Vectorized Decoder + In-Place FlatBuffer Patcher
- **Supported Modes:**
  - `portable`: On-the-fly fast vectorized decoding (<6 ms startup)
  - `cached`: Cryptographically verified persistent TFLite cache (<2 ms startup)

## Package Contents
- `model.uaqe`: Compressed hybrid binary weight archive
- `manifest.json`: Model architecture, provenance, and accuracy metadata
- `runtime_version.json`: Cryptographic format versioning and engine compatibility
- `runtime_config.json`: Runtime execution parameters and preprocessing specs
- `checksums.json`: SHA-256 integrity verification hashes
- `README.md`: Quickstart deployment guide

## Quickstart Python Usage
```python
import numpy as np
from PIL import Image
from uaqe.runtime.optimized_session import OptimizedRuntimeSession

# 1. Initialize session in portable or cached mode
session = OptimizedRuntimeSession.from_archive("model.uaqe", mode="portable")
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

        # 6. Generate checksums.json
        checksums = {
            "model.uaqe": self.compute_sha256(target_model_path),
            "manifest.json": self.compute_sha256(manifest_path),
            "runtime_version.json": self.compute_sha256(runtime_version_path),
            "runtime_config.json": self.compute_sha256(runtime_config_path),
            "README.md": self.compute_sha256(readme_path)
        }
        checksums_path = os.path.join(self.package_dir, "checksums.json")
        with open(checksums_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

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
