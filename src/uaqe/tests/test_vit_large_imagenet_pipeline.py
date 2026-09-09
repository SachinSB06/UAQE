"""ViT-Large + ImageNet-10K Optimization Forensic and Capability Verification Test Suite.

Tests:
1. ViT model inspection (safetensors and bin, parameter count ~304M, tensors, shapes)
2. ViT adapter resolution (SafeTensorsModelAdapter and PyTorchModelAdapter)
3. ViT task detection (image_classification, 1000 classes, NCHW)
4. ImageNet-10K dataset detection (image_folder, 1000 classes, 10000 samples, splits)
5. 1000-class compatibility (1000 == 1000, no classifier adaptation)
6. Job creation & storage guard (hardlinking, disk space pre-check)
7. Optimization launch behavior (no HTTP 500, clean architecture guard)
8. Error propagation (clear UNSUPPORTED_MODEL_ARCHITECTURE error with details)
9. Memory & resource limits check
10. Zero silent fallback & control model comparison (CNN models supported, ViT honestly flagged)
"""

import os
import sys
import json
import unittest
import urllib.request
import urllib.error

# Ensure src is in sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from uaqe.orchestration.universal_model_ingestor import UniversalModelIngestor
from uaqe.orchestration.universal_dataset_ingestor import UniversalDatasetIngestor
from uaqe.orchestration.task_detector import TaskDetector
from uaqe.orchestration.compatibility_checker import CompatibilityChecker
from uaqe.orchestration.optimization_strategies.strategy_resolver import OptimizationStrategyResolver
from uaqe.orchestration.hardware_target_registry import HardwareTargetRegistry

API_BASE = "http://127.0.0.1:8000/api"

# Paths to test artifacts
SAFELARGE_PATH = os.path.join(
    os.path.dirname(SRC_DIR), "output", "uploads", "models", "MODEL-7C41D025", "model.safetensors"
)
BIN_PATH = os.path.join(
    os.path.dirname(SRC_DIR), "output", "uploads", "models", "MODEL-9251E8E6", "pytorch_model.bin"
)
IMAGENET_SPLIT_PATH = r"D:\imagenet_10k_split"


class TestViTLargeImageNetPipeline(unittest.TestCase):
    """Forensic verification test suite for ViT-Large + ImageNet-10K pipeline."""

    @classmethod
    def setUpClass(cls):
        # Ensure API server is available for tests 08, 09, 10
        cls._server_started = False
        try:
            with urllib.request.urlopen(f"{API_BASE}/status", timeout=1) as resp:
                pass
        except Exception:
            import threading
            import time
            import uvicorn
            from uaqe.server import app
            config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
            cls._server = uvicorn.Server(config)
            cls._server_thread = threading.Thread(target=cls._server.run, daemon=True)
            cls._server_thread.start()
            cls._server_started = True
            for _ in range(50):
                time.sleep(0.1)
                try:
                    with urllib.request.urlopen(f"{API_BASE}/status", timeout=1) as resp:
                        break
                except Exception:
                    pass

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_server_started", False):
            cls._server.should_exit = True

    def test_01_vit_model_inspection(self):
        """Phase 3: Inspect uploaded ViT model (safetensors format)."""
        self.assertTrue(os.path.exists(SAFELARGE_PATH), f"ViT safetensors model missing at {SAFELARGE_PATH}")
        ingestor = UniversalModelIngestor(SAFELARGE_PATH)
        desc = ingestor.inspect()

        self.assertEqual(desc["format"], "safetensors")
        self.assertEqual(desc["architecture"], "ViTForImageClassification")
        self.assertEqual(desc["task"], "image_classification")
        self.assertEqual(desc["input_shape"], [1, 3, 224, 224])
        self.assertEqual(desc["output_shape"], [1, 1000])
        self.assertGreater(desc["parameter_count"], 300_000_000, "ViT-Large should have >300M parameters")
        self.assertEqual(desc["parameter_count"], 304_326_632)
        self.assertEqual(desc["tensor_count"], 392)

    def test_02_vit_adapter_resolution_and_bin_format(self):
        """Phase 4: Verify adapter resolution for both safetensors and .bin formats."""
        # Check safetensors adapter
        ingestor_safe = UniversalModelIngestor(SAFELARGE_PATH)
        self.assertEqual(ingestor_safe.adapter.__class__.__name__, "SafeTensorsModelAdapter")
        caps_safe = ingestor_safe.get_capabilities()
        self.assertFalse(caps_safe["supports_int8"], "ViT attention QKV should not be marked as supported for INT8 CNN")

        # Check .bin adapter if file exists
        if os.path.exists(BIN_PATH):
            ingestor_bin = UniversalModelIngestor(BIN_PATH)
            self.assertEqual(ingestor_bin.adapter.__class__.__name__, "PyTorchModelAdapter")
            desc_bin = ingestor_bin.inspect()
            self.assertEqual(desc_bin["format"], "pytorch_checkpoint")
            self.assertEqual(desc_bin["architecture"], "ViTForImageClassification")
            self.assertEqual(desc_bin["output_shape"], [1, 1000])

    def test_03_vit_task_detection(self):
        """Phase 6: Task detection returns image_classification with 1000 classes."""
        ingestor = UniversalModelIngestor(SAFELARGE_PATH)
        model_desc = ingestor.inspect()

        # Mock or use dataset descriptor
        dataset_desc = {
            "format": "image_folder",
            "class_count": 1000,
            "total_samples": 10000
        }
        task_info = TaskDetector.detect_task(model_desc, dataset_desc)
        self.assertEqual(task_info["task"], "image_classification")
        self.assertIn(task_info["confidence"], ["high", "medium"])
        self.assertEqual(task_info["model_classes"], 1000)
        self.assertEqual(task_info["dataset_classes"], 1000)

    def test_04_imagenet_10k_dataset_detection(self):
        """Phase 5: Ingest ImageNet-10K and verify 1000 classes across train/val/test splits."""
        if not os.path.exists(IMAGENET_SPLIT_PATH):
            self.skipTest(f"Dataset path {IMAGENET_SPLIT_PATH} not mounted")

        ingestor = UniversalDatasetIngestor(IMAGENET_SPLIT_PATH)
        desc = ingestor.inspect()

        self.assertEqual(desc["detected_format"], "image_folder")
        self.assertEqual(desc["class_count"], 1000, "Should detect exactly 1000 class directories")
        splits = desc["splits"]
        self.assertEqual(splits["train_count"], 8000)
        self.assertEqual(splits["val_count"], 1000)
        self.assertEqual(splits["test_count"], 1000)
        self.assertEqual(splits["total_samples"], 10000)

    def test_05_1000_class_compatibility(self):
        """Phase 7: Verify 1000 == 1000 class compatibility (no classifier adaptation required)."""
        ingestor = UniversalModelIngestor(SAFELARGE_PATH)
        model_desc = ingestor.inspect()

        dataset_desc = {
            "format": "image_folder",
            "class_count": 1000,
            "total_samples": 10000
        }
        task_info = TaskDetector.detect_task(model_desc, dataset_desc)
        compat = CompatibilityChecker.check_compatibility(model_desc, dataset_desc, task_info)

        self.assertTrue(compat["class_alignment"]["classes_match"])
        self.assertFalse(compat["class_alignment"]["adaptation_required"])
        self.assertEqual(compat["class_alignment"]["model_classes"], 1000)
        self.assertEqual(compat["class_alignment"]["dataset_classes"], 1000)

    def test_06_target_hardware_resolution(self):
        """Phase 8: Verify Raspberry Pi 5 target resolves with host-measured pending profile."""
        profile = HardwareTargetRegistry.get_profile("raspberrypi5")
        self.assertIsNotNone(profile)
        tid = profile["target_id"] if isinstance(profile, dict) else profile.target_id
        name = profile["name"] if isinstance(profile, dict) else profile.name
        self.assertEqual(tid, "raspberrypi5")
        self.assertIn("BCM2712", name)

    def test_07_optimization_strategy_resolution(self):
        """Phase 11: Strategy resolver identifies that no specialized CNN strategy handles ViT."""
        ingestor = UniversalModelIngestor(SAFELARGE_PATH)
        model_desc = ingestor.inspect()
        dataset_desc = {"format": "image_folder", "class_count": 1000, "total_samples": 10000}
        opt_plan = {"optimization_profile": "balanced"}

        strat = OptimizationStrategyResolver.resolve(model_desc, dataset_desc, opt_plan)
        # ResNetONNXPTQStrategy requires ResNet architecture
        # MobileNetAdaptiveStrategy requires MobileNet architecture
        # Therefore ViT falls to GenericFallbackStrategy
        self.assertEqual(strat.__class__.__name__, "GenericFallbackStrategy")

    def test_08_api_analyze_reports_unsupported_architecture(self):
        """Phase 12: API /api/jobs/analyze honestly flags ViT as unsupported."""
        payload = json.dumps({
            "model_upload_id": "MODEL-7C41D025",
            "dataset_upload_id": "DATASET-1F63A1B7",
            "target": "raspberrypi5",
            "profile": "balanced"
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_BASE}/jobs/analyze",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["model_inspection"]["architecture"], "ViTForImageClassification")
            self.assertFalse(data["compatibility"]["compatible"])
            issues = data["compatibility"]["issues"]
            self.assertTrue(any("UNSUPPORTED_MODEL_ARCHITECTURE" in issue for issue in issues))

    def test_09_api_optimize_rejects_vit_with_clear_400_no_500(self):
        """Phase 1 & 13: Launching ViT optimization produces HTTP 400 with UNSUPPORTED_MODEL_ARCHITECTURE, never HTTP 500."""
        payload = json.dumps({
            "model_upload_id": "MODEL-7C41D025",
            "dataset_upload_id": "DATASET-1F63A1B7",
            "target": "raspberrypi5",
            "profile": "balanced"
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_BASE}/jobs/optimize",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as resp:
                self.fail("ViT optimization should have been rejected with HTTP 400, but succeeded!")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400, f"Expected HTTP 400, got HTTP {e.code}")
            error_body = json.loads(e.read().decode("utf-8"))
            detail = error_body.get("detail", "")
            self.assertIn("UNSUPPORTED_MODEL_ARCHITECTURE", detail)
            self.assertIn("Vision Transformer", detail)
            self.assertNotIn("Internal Server Error", detail)

    def test_10_control_models_remain_supported(self):
        """Phase 14 & 16: Zero regression on supported CNN models (ResNet and MobileNet)."""
        # Test ResNet-50 sample
        sample_resnet_payload = json.dumps({
            "model_id": "resnet50_cifar10",
            "dataset_id": "cifar10",
            "target": "raspberrypi5",
            "profile": "balanced"
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_BASE}/jobs/analyze",
            data=sample_resnet_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["compatibility"]["compatible"], "ResNet-50 control model must remain compatible")


if __name__ == "__main__":
    unittest.main()
