"""Unit and Integration Tests for UAQE Phase E.3: Universal Orchestration."""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from uaqe.orchestration.universal_model_ingestor import UniversalModelIngestor
from uaqe.orchestration.universal_dataset_ingestor import UniversalDatasetIngestor
from uaqe.orchestration.task_detector import TaskDetector
from uaqe.orchestration.compatibility_checker import CompatibilityChecker
from uaqe.orchestration.preprocessing_resolver import PreprocessingResolver
from uaqe.orchestration.hardware_target_registry import HardwareTargetRegistry
from uaqe.orchestration.model_adaptation_service import ModelAdaptationService
from uaqe.orchestration.optimization_planner import OptimizationPlanner
from uaqe.orchestration.optimization_orchestrator import OptimizationOrchestrator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MOBILENET_ONNX = os.path.join(PROJECT_ROOT, "src", "models", "mobilenetv3_sem.onnx")
RESNET_SAFETENSORS = os.path.join(PROJECT_ROOT, "src", "models", "resnet50", "model.safetensors")
CIFAR10_DIR = r"D:\uaqe_datasets\cifar10\cifar-10-batches-py"
SEMICONDUCTOR_DIR = r"D:\semiconductor_dataset\dataset"


class TestPhaseE3UniversalOrchestration(unittest.TestCase):
    """Test suite for Phase E.3 Universal Model & Dataset Orchestration."""

    def test_model_format_detection_and_adapters(self):
        """Verify model format detection across supported file formats."""
        self.assertEqual(UniversalModelIngestor.detect_format(MOBILENET_ONNX), "onnx")
        self.assertEqual(UniversalModelIngestor.detect_format(RESNET_SAFETENSORS), "safetensors")
        self.assertEqual(UniversalModelIngestor.detect_format("nonexistent.random"), "unknown")

        # Ingest ONNX
        ingestor_onnx = UniversalModelIngestor(MOBILENET_ONNX)
        desc_onnx = ingestor_onnx.inspect()
        self.assertEqual(desc_onnx["format"], "onnx")
        self.assertEqual(desc_onnx["architecture"], "MobileNetV3-Small")
        self.assertTrue(desc_onnx["parameter_count"] > 900000)

        # Ingest SafeTensors
        ingestor_st = UniversalModelIngestor(RESNET_SAFETENSORS)
        desc_st = ingestor_st.inspect()
        self.assertEqual(desc_st["format"], "safetensors")
        self.assertEqual(desc_st["architecture"], "ResNetForImageClassification")
        self.assertTrue(desc_st["parameter_count"] > 20000000)

    def test_unsupported_model_format_handling(self):
        """Verify clear error when unsupported format is submitted."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Not a model")
            dummy_path = f.name
        try:
            with self.assertRaises(ValueError):
                UniversalModelIngestor(dummy_path)
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def test_dataset_format_detection_and_adapters(self):
        """Verify dataset format detection for CIFAR-10 and ImageFolder."""
        self.assertEqual(UniversalDatasetIngestor.detect_format(CIFAR10_DIR), "cifar10_pickle")
        self.assertEqual(UniversalDatasetIngestor.detect_format(SEMICONDUCTOR_DIR), "image_folder")
        self.assertEqual(UniversalDatasetIngestor.detect_format("C:/nonexistent_path"), "unknown")

        ingestor_cifar = UniversalDatasetIngestor(CIFAR10_DIR)
        splits = ingestor_cifar.load()
        desc_cifar = ingestor_cifar.get_descriptor()
        self.assertEqual(desc_cifar["class_count"], 10)
        self.assertEqual(desc_cifar["splits"]["train_count"], 45000)

    def test_task_detection(self):
        """Verify automated task detection logic."""
        m_desc = {"input_shape": [1, 3, 224, 224], "output_shape": [1, 10]}
        d_desc = {"class_count": 10}
        task = TaskDetector.detect_task(m_desc, d_desc)
        self.assertTrue(task["supported"])
        self.assertEqual(task["task"], "image_classification")

        # Invalid shape
        bad_desc = {"input_shape": [10], "output_shape": [10]}
        bad_task = TaskDetector.detect_task(bad_desc, d_desc)
        self.assertFalse(bad_task["supported"])

    def test_compatibility_checking(self):
        """Verify compatibility auditing between model and dataset."""
        m_desc = {"input_shape": [1, 3, 224, 224], "output_shape": [1, 1000], "architecture": "ResNetForImageClassification"}
        d_desc = {"class_count": 10, "detected_format": "cifar10_pickle", "splits": {"train_count": 45000, "val_count": 5000, "test_count": 10000}}
        t_info = {"task": "image_classification", "supported": True}

        report = CompatibilityChecker.check_compatibility(m_desc, d_desc, t_info)
        self.assertTrue(report["compatible"])
        self.assertTrue(report["class_alignment"]["adaptation_required"])
        self.assertEqual(report["class_alignment"]["model_classes"], 1000)
        self.assertEqual(report["class_alignment"]["dataset_classes"], 10)

    def test_preprocessing_resolver_provenance(self):
        """Verify preprocessing resolver records exact provenance for all parameters."""
        m_desc = {"input_shape": [1, 3, 224, 224], "architecture": "ResNetForImageClassification"}
        d_desc = {"class_count": 10}
        model_dir = os.path.join(PROJECT_ROOT, "src", "models", "resnet50")

        cfg = PreprocessingResolver.resolve(m_desc, d_desc, model_dir=model_dir)
        self.assertEqual(cfg["target_resolution"], [224, 224])
        self.assertIn("provenance", cfg)
        self.assertIn("target_resolution", cfg["provenance"])
        self.assertIn("source", cfg["provenance"]["target_resolution"])
        self.assertIn("normalization", cfg["provenance"])

    def test_hardware_target_registry(self):
        """Verify target hardware profiles and constraints."""
        targets = HardwareTargetRegistry.list_targets()
        self.assertIn("esp32", targets)
        self.assertIn("stm32", targets)
        self.assertIn("artix7", targets)
        self.assertIn("zynq7000", targets)
        self.assertIn("raspberrypi4", targets)
        self.assertIn("raspberrypi5", targets)

        rpi5 = HardwareTargetRegistry.get_profile("raspberrypi5")
        self.assertEqual(rpi5["hardware_class"], "SINGLE_BOARD_COMPUTER")
        self.assertIn("INT8", rpi5["supported_precisions"])
        self.assertFalse(rpi5["is_physical_measurement"])

    def test_model_adaptation_service(self):
        """Verify model adaptation service inspects and adapts 1000 -> 10 class head."""
        service = ModelAdaptationService()
        m_desc = {"output_shape": [1, 1000], "architecture": "ResNetForImageClassification", "parameter_count": 25557032}
        
        info = service.inspect_adaptation(m_desc, target_classes=10)
        self.assertTrue(info["adaptation_required"])
        self.assertTrue(info["adaptation_supported"])
        self.assertEqual(info["new_output_classes"], 10)

    def test_optimization_planner_dry_run(self):
        """Verify optimization plan generation in read-only mode."""
        ingestor = UniversalModelIngestor(MOBILENET_ONNX)
        m_desc = ingestor.inspect()
        caps = ingestor.get_capabilities()
        d_desc = {"dataset_name": "Semiconductor", "detected_format": "image_folder", "class_count": 9, "splits": {"train_count": 288, "val_count": 36, "test_count": 36}}
        compat = {"class_alignment": {"adaptation_required": False, "model_classes": 9, "dataset_classes": 9}}
        hw = HardwareTargetRegistry.get_profile("raspberrypi5")

        plan = OptimizationPlanner.create_plan(m_desc, caps, d_desc, compat, hw, profile_name="balanced")
        self.assertEqual(plan["optimization_profile"], "balanced")
        self.assertEqual(plan["quantization_plan"]["selected_precision"], "INT8")
        self.assertFalse(plan["quantization_plan"]["fallback_used"])

    def test_orchestrator_plan_only_and_approval_gate(self):
        """Verify orchestrator respects plan_only=True and does not modify original files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = OptimizationOrchestrator(output_root=tmpdir)
            job_config = {
                "model_path": MOBILENET_ONNX,
                "dataset_path": SEMICONDUCTOR_DIR,
                "target_hardware": "raspberrypi5",
                "optimization_profile": "balanced",
                "plan_only": True
            }
            res = orchestrator.run(job_config)
            self.assertEqual(res["status"], "PLAN_GENERATED")
            self.assertTrue(os.path.exists(res["optimization_plan_json"]))
            self.assertTrue(os.path.exists(res["optimization_plan_md"]))


if __name__ == "__main__":
    unittest.main()
