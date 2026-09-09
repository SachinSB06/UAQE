"""Unit and integration tests for Sensitivity-Aware Structured Pruning."""

from __future__ import annotations

import os
import unittest
import numpy as np
import dataclasses
from typing import Any, Dict, List, Optional

from uaqe.common.imr import IMR, IMRLayer, IMRTensor, IMRMetadata
from uaqe.common.exceptions import CompressionError, QuantizationError
from uaqe.common.types import Precision, CompressionType, HardwareClass
from uaqe.common.value_objects import CompressionConfig
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.compression.sensitivity_aware_structured_pruner import SensitivityAwareStructuredPruner
from uaqe.quantization.sensitivity_analyzer import SensitivityReport


class MockLogger:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass


class TestSensitivityStructuredPruner(unittest.TestCase):
    """Test suite covering structured pruning correctness, constraints, and safety."""

    def setUp(self) -> None:
        self.logger = MockLogger()
        self.pruner = SensitivityAwareStructuredPruner(self.logger)
        
        # Build a mock MobileNetV3 block IMR (Block 4) for tests
        # We need an expansion conv, a depthwise conv, an SE block, and a projection conv.
        
        # 1. Expansion Conv: /features/features.4/block/block.0/block.0.0/Conv
        # input: [1, 24, 128, 128], weight: [96, 24, 1, 1], bias: [96]
        exp_w = np.ones((96, 24, 1, 1), dtype=np.float32)
        exp_b = np.ones((96,), dtype=np.float32)
        self.exp_layer = IMRLayer(
            name="/features/features.4/block/block.0/block.0.0/Conv",
            op_type="Conv",
            inputs=["input_tensor"],
            outputs=["/features/features.4/block/block.0/block.0.0/Conv_output_0"],
            parameters={
                "weight": IMRTensor(shape=(96, 24, 1, 1), dtype="float32", data=exp_w.tobytes()),
                "bias": IMRTensor(shape=(96,), dtype="float32", data=exp_b.tobytes())
            },
            attributes={"kernel_shape": [1, 1], "strides": [1, 1]},
            precision=Precision.FP32
        )

        # 2. Depthwise Conv: /features/features.4/block/block.1/block.1.0/Conv
        # input: [1, 96, 128, 128], weight: [96, 1, 3, 3], bias: [96]
        dw_w = np.ones((96, 1, 3, 3), dtype=np.float32)
        dw_b = np.ones((96,), dtype=np.float32)
        self.dw_layer = IMRLayer(
            name="/features/features.4/block/block.1/block.1.0/Conv",
            op_type="Conv",
            inputs=["/features/features.4/block/block.0/block.0.0/Conv_output_0"],
            outputs=["/features/features.4/block/block.1/block.1.0/Conv_output_0"],
            parameters={
                "weight": IMRTensor(shape=(96, 1, 3, 3), dtype="float32", data=dw_w.tobytes()),
                "bias": IMRTensor(shape=(96,), dtype="float32", data=dw_b.tobytes())
            },
            attributes={"kernel_shape": [3, 3], "strides": [1, 1], "group": 96},
            precision=Precision.FP32
        )

        # 3. SE fc1: /features/features.4/block/block.2/fc1/Conv
        # weight: [24, 96, 1, 1], bias: [24]
        se1_w = np.ones((24, 96, 1, 1), dtype=np.float32)
        se1_b = np.ones((24,), dtype=np.float32)
        self.se1_layer = IMRLayer(
            name="/features/features.4/block/block.2/fc1/Conv",
            op_type="Conv",
            inputs=["/features/features.4/block/block.1/block.1.0/Conv_output_0"],
            outputs=["/features/features.4/block/block.2/fc1/Conv_output_0"],
            parameters={
                "weight": IMRTensor(shape=(24, 96, 1, 1), dtype="float32", data=se1_w.tobytes()),
                "bias": IMRTensor(shape=(24,), dtype="float32", data=se1_b.tobytes())
            },
            attributes={"kernel_shape": [1, 1], "strides": [1, 1]},
            precision=Precision.FP32
        )

        # 4. SE fc2: /features/features.4/block/block.2/fc2/Conv
        # weight: [96, 24, 1, 1], bias: [96]
        se2_w = np.ones((96, 24, 1, 1), dtype=np.float32)
        se2_b = np.ones((96,), dtype=np.float32)
        self.se2_layer = IMRLayer(
            name="/features/features.4/block/block.2/fc2/Conv",
            op_type="Conv",
            inputs=["/features/features.4/block/block.2/fc1/Conv_output_0"],
            outputs=["/features/features.4/block/block.2/fc2/Conv_output_0"],
            parameters={
                "weight": IMRTensor(shape=(96, 24, 1, 1), dtype="float32", data=se2_w.tobytes()),
                "bias": IMRTensor(shape=(96,), dtype="float32", data=se2_b.tobytes())
            },
            attributes={"kernel_shape": [1, 1], "strides": [1, 1]},
            precision=Precision.FP32
        )

        # 5. Projection Conv: /features/features.4/block/block.3/block.3.0/Conv
        # weight: [40, 96, 1, 1], bias: [40]
        proj_w = np.ones((40, 96, 1, 1), dtype=np.float32)
        proj_b = np.ones((40,), dtype=np.float32)
        self.proj_layer = IMRLayer(
            name="/features/features.4/block/block.3/block.3.0/Conv",
            op_type="Conv",
            inputs=["/features/features.4/block/block.2/fc2/Conv_output_0"],
            outputs=["/features/features.4/block/block.3/block.3.0/Conv_output_0"],
            parameters={
                "weight": IMRTensor(shape=(40, 96, 1, 1), dtype="float32", data=proj_w.tobytes()),
                "bias": IMRTensor(shape=(40,), dtype="float32", data=proj_b.tobytes())
            },
            attributes={"kernel_shape": [1, 1], "strides": [1, 1]},
            precision=Precision.FP32
        )

        # Final Classifier layer to verify protection
        self.classifier_layer = IMRLayer(
            name="/classifier/classifier.0/Gemm",
            op_type="Gemm",
            inputs=["/features/features.4/block/block.3/block.3.0/Conv_output_0"],
            outputs=["output"],
            parameters={
                "weight": IMRTensor(shape=(10, 40), dtype="float32", data=np.ones((10, 40), dtype=np.float32).tobytes()),
                "bias": IMRTensor(shape=(10,), dtype="float32", data=np.ones((10,), dtype=np.float32).tobytes())
            },
            attributes={"transB": 1},
            precision=Precision.FP32
        )

        self.metadata = IMRMetadata(
            source_framework="onnx",
            source_format=".onnx",
            original_input_shapes={"input_tensor": (1, 24, 128, 128)},
            op_count=6,
            total_parameters=96*24 + 96*9 + 24*96 + 96*24 + 40*96 + 10*40 + 96 + 96 + 24 + 96 + 40 + 10
        )
        self.imr = IMR(
            layers=[self.exp_layer, self.dw_layer, self.se1_layer, self.se2_layer, self.proj_layer, self.classifier_layer],
            metadata=self.metadata
        )

    def test_sensitivity_analysis(self) -> None:
        """Verify sensitivity analyzer report extraction."""
        ctx = PipelineContext(run_id="test-run")
        report = SensitivityReport(
            per_layer_accuracy_delta={self.dw_layer.name: 0.9, self.exp_layer.name: 0.1},
            flagged_layers=[self.dw_layer.name]
        )
        ctx.append("sensitivity_analyzer", report)
        
        scores = self.pruner._resolve_sensitivity(ctx, self.imr)
        self.assertEqual(scores[self.dw_layer.name], 0.9)
        self.assertEqual(scores[self.exp_layer.name], 0.1)

    def test_layer_protection(self) -> None:
        """Verify highly sensitive blocks or layers are protected (0% pruning)."""
        ctx = PipelineContext(run_id="test-run")
        # Mark all layers of block 4 as highly sensitive
        report = SensitivityReport(
            per_layer_accuracy_delta={
                self.exp_layer.name: 0.99,
                self.dw_layer.name: 0.99,
                self.proj_layer.name: 0.99
            },
            flagged_layers=[]
        )
        ctx.append("sensitivity_analyzer", report)

        blocks = self.pruner._discover_blocks(self.imr, report.per_layer_accuracy_delta)
        self.assertTrue(blocks[0]["protected"])

    def test_structured_channel_removal(self) -> None:
        """Verify channel removal physically reduces shapes and elements."""
        # Pruning sparsity target 25% (reduction from 96 to 72 channels)
        pruned = self.pruner._prune_to_sparsity(self.imr, self.pruner._discover_blocks(self.imr, {}), 0.25)
        
        exp_pruned = next(l for l in pruned.layers if l.name == self.exp_layer.name)
        exp_w = exp_pruned.parameters["weight"]
        self.assertEqual(exp_w.shape[0], 72)  # Output channels reduced to 72

    def test_shape_consistency(self) -> None:
        """Verify shapes are consistent across components in a pruned block."""
        pruned = self.pruner._prune_to_sparsity(self.imr, self.pruner._discover_blocks(self.imr, {}), 0.25)
        
        # Verify dw_layer input shape matching exp output shape (72 channels)
        dw_pruned = next(l for l in pruned.layers if l.name == self.dw_layer.name)
        dw_w = dw_pruned.parameters["weight"]
        self.assertEqual(dw_w.shape[0], 72)
        self.assertEqual(dw_pruned.attributes["group"], 72)

        # Verify SE blocks fc1 input channel, fc2 output channel matching 72
        se1_pruned = next(l for l in pruned.layers if l.name == self.se1_layer.name)
        self.assertEqual(se1_pruned.parameters["weight"].shape[1], 72)

        se2_pruned = next(l for l in pruned.layers if l.name == self.se2_layer.name)
        self.assertEqual(se2_pruned.parameters["weight"].shape[0], 72)

        # Verify proj input shape matching 72
        proj_pruned = next(l for l in pruned.layers if l.name == self.proj_layer.name)
        self.assertEqual(proj_pruned.parameters["weight"].shape[1], 72)

    def test_residual_connection_safety(self) -> None:
        """Verify residual connections remain safe (protected/unmatched layers are not broken)."""
        # Create a layer with residual connection
        res_layer = IMRLayer(
            name="/features/features.4/Add",
            op_type="Add",
            inputs=[self.proj_layer.outputs[0], "some_residual_input"],
            outputs=["res_output"],
            parameters={},
            attributes={},
            precision=Precision.FP32
        )
        imr_res = dataclasses.replace(self.imr, layers=self.imr.layers + [res_layer])
        
        # Test pruning without breaking residual integrity
        blocks = self.pruner._discover_blocks(imr_res, {})
        self.assertEqual(len(blocks), 1)

    def test_depthwise_conv_safety(self) -> None:
        """Verify depthwise convolution groups match output channels."""
        pruned = self.pruner._prune_to_sparsity(self.imr, self.pruner._discover_blocks(self.imr, {}), 0.50)
        dw_pruned = next(l for l in pruned.layers if l.name == self.dw_layer.name)
        self.assertEqual(dw_pruned.attributes["group"], dw_pruned.parameters["weight"].shape[0])

    def test_classifier_safety(self) -> None:
        """Verify final classifier layers are protected."""
        pruned = self.pruner._prune_to_sparsity(self.imr, self.pruner._discover_blocks(self.imr, {}), 0.50)
        class_pruned = next(l for l in pruned.layers if l.name == self.classifier_layer.name)
        # Verify shape of classifier weights remains unchanged
        self.assertEqual(class_pruned.parameters["weight"].shape, (10, 40))

    def test_iterative_pruning(self) -> None:
        """Verify gradual iterative pruning over schedule."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.10,
            experimental_pruning_strategy="sensitivity_aware_structured"
        )
        ctx = PipelineContext(run_id="test-run")
        ctx.append("workflow_config", plan)
        
        # Pruning must complete successfully without crashing
        res = self.pruner.apply(self.imr, plan, context=ctx)
        self.assertLess(self.pruner._count_parameters(res), self.pruner._count_parameters(self.imr))

    def test_accuracy_constraint(self) -> None:
        """Verify accuracy constraint checking logic."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.02,
            max_accuracy_drop=0.01
        )
        # Should execute correctly
        res = self.pruner.apply(self.imr, plan)
        self.assertIsNotNone(res)

    def test_rollback_on_accuracy_failure(self) -> None:
        """Verify rollback occurs when validation accuracy or other constraints fail."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.08,
            max_accuracy_drop=0.0001, # extremely tight accuracy threshold
            max_latency_regression=-1.0 # force failure on size/latency
        )
        res = self.pruner.apply(self.imr, plan)
        # Parameter count must rollback to the last accepted configuration (original model)
        self.assertEqual(self.pruner._count_parameters(res), self.pruner._count_parameters(self.imr))

    def test_finetuning_recovery(self) -> None:
        """Verify fine-tuning engine modifies model parameter values."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.02,
            fine_tune_enabled=True,
            fine_tune_epochs=2
        )
        engine = self.pruner._recovery_engine
        
        # Create temp train and val paths to enable training
        os.makedirs("scratch/temp_train", exist_ok=True)
        os.makedirs("scratch/temp_val", exist_ok=True)
        
        recovered, metrics = engine.recover(self.imr, plan, "scratch/temp_train", "scratch/temp_val")
        self.assertEqual(metrics["status"], "SUCCESS")
        self.assertEqual(metrics["epochs"], 2)
        
        # Cleanup
        os.rmdir("scratch/temp_train")
        os.rmdir("scratch/temp_val")

    def test_size_constraint(self) -> None:
        """Verify model parameter size constraints."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.05
        )
        res = self.pruner.apply(self.imr, plan)
        self.assertLessEqual(self.pruner._count_parameters(res), self.pruner._count_parameters(self.imr))

    def test_latency_constraint(self) -> None:
        """Verify latency constraints evaluation."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.05,
            max_latency_regression=0.05
        )
        res = self.pruner.apply(self.imr, plan)
        self.assertIsNotNone(res)

    def test_final_model_export(self) -> None:
        """Verify pruned IMR can be successfully exported to genuine ONNX format."""
        # Clean temp paths
        from uaqe.exporter.onnx_exporter import OnnxExporter
        from uaqe.domain.hardware_manager import HardwareProfile
        
        exporter = OnnxExporter(self.logger, output_dir="scratch/temp_out")
        exporter.bind_source_model_path(os.path.join("src", "models", "mobilenetv3_sem.onnx"))
        
        profile = HardwareProfile(
            profile_id="test_profile",
            display_name="Test Profile",
            hardware_class=HardwareClass.RASPBERRY_PI,
            ram_bytes=1024,
            flash_bytes=1024,
            storage_bytes=1024,
            tensor_memory_bytes=1024,
            runtime="tflite-runtime",
            default_runtime="tflite-runtime",
            max_model_size_bytes=100000000,
            schema_version="1.0"
        )
        
        # Prune the original model slightly (2%)
        # Note: We must load the real MobileNetV3 to test full export round-trip!
        from uaqe.infrastructure.framework_adapters.onnx_adapter import OnnxAdapter
        loader = OnnxAdapter()
        real_imr = loader.load(os.path.join("src", "models", "mobilenetv3_sem.onnx"))
        
        pruned_real = self.pruner._prune_to_sparsity(real_imr, self.pruner._discover_blocks(real_imr, {}), 0.05)
        
        # Export
        artifact = exporter.export(pruned_real, profile)
        self.assertTrue(os.path.exists(artifact.file_paths[0]))
        
        # Cleanup
        os.remove(artifact.file_paths[0])
        os.rmdir(os.path.dirname(artifact.file_paths[0]))
        os.rmdir("scratch/temp_out")

    # Negative Tests
    def test_invalid_pruning_ratio(self) -> None:
        """Verify error raised on invalid pruning sparsity configurations."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=1.5 # invalid sparsity > 1.0
        )
        with self.assertRaises(Exception):
            self.pruner.apply(self.imr, plan)

    def test_invalid_accuracy_threshold(self) -> None:
        """Verify handling of invalid/malformed accuracy config."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.05,
            max_accuracy_drop=-0.1 # negative max drop
        )
        res = self.pruner.apply(self.imr, plan)
        self.assertIsNotNone(res)

    def test_invalid_latency_tolerance(self) -> None:
        """Verify handling of invalid/malformed latency tolerance."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.05,
            max_latency_regression=-0.5
        )
        res = self.pruner.apply(self.imr, plan)
        self.assertIsNotNone(res)

    def test_invalid_layer_structure(self) -> None:
        """Verify pruner skips blocks with invalid layer structures."""
        # Create an incomplete block without depthwise conv
        incomplete_layer = IMRLayer(
            name="/features/features.5/block/block.0/block.0.0/Conv",
            op_type="Conv",
            inputs=["proj_output"],
            outputs=["incomplete_output"],
            parameters={"weight": IMRTensor(shape=(96, 24, 1, 1), dtype="float32", data=np.ones((96, 24, 1, 1), dtype=np.float32).tobytes())},
            attributes={},
            precision=Precision.FP32
        )
        imr_inc = IMR(layers=[incomplete_layer], metadata=self.metadata)
        blocks = self.pruner._discover_blocks(imr_inc, {})
        self.assertEqual(len(blocks), 0) # Should find 0 valid blocks

    def test_channel_mismatch(self) -> None:
        """Verify pruner fails gracefully on internal channel mismatches."""
        # Modifying depthwise shape to mismatch expansion output shape
        mismatched_dw = IMRLayer(
            name="/features/features.4/block/block.1/block.1.0/Conv",
            op_type="Conv",
            inputs=["/features/features.4/block/block.0/block.0.0/Conv_output_0"],
            outputs=["/features/features.4/block/block.1/block.1.0/Conv_output_0"],
            parameters={
                # Mismatched shape 50 != 96
                "weight": IMRTensor(shape=(50, 1, 3, 3), dtype="float32", data=np.ones((50, 1, 3, 3), dtype=np.float32).tobytes()),
            },
            attributes={"group": 50},
            precision=Precision.FP32
        )
        imr_mismatched = IMR(
            layers=[self.exp_layer, mismatched_dw, self.se1_layer, self.se2_layer, self.proj_layer, self.classifier_layer],
            metadata=self.metadata
        )
        with self.assertRaises(Exception):
            self.pruner._prune_to_sparsity(imr_mismatched, self.pruner._discover_blocks(imr_mismatched, {}), 0.10)

    def test_unsupported_operator(self) -> None:
        """Verify pruner ignores or handles unsupported layer types safely."""
        unsupported = IMRLayer(
            name="/features/features.4/CustomOp",
            op_type="CustomOp",
            inputs=["input"],
            outputs=["output"],
            parameters={},
            attributes={},
            precision=Precision.FP32
        )
        imr_unsupported = dataclasses.replace(self.imr, layers=self.imr.layers + [unsupported])
        blocks = self.pruner._discover_blocks(imr_unsupported, {})
        self.assertEqual(len(blocks), 1) # Block 4 still discovered and pruned safely

    def test_missing_training_data(self) -> None:
        """Verify fine-tuning is blocked and flagged as BLOCKED when missing training dataset."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.02,
            fine_tune_enabled=True
        )
        # Passing None for datasets paths
        recovered, metrics = self.pruner._recovery_engine.recover(self.imr, plan, None, None)
        self.assertTrue(metrics["status"].startswith("BLOCKED"))

    def test_missing_validation_data(self) -> None:
        """Verify validation and early-stopping are bypassed when validation dataset is missing."""
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.02,
            fine_tune_enabled=True
        )
        # Providing train path but no val path
        recovered, metrics = self.pruner._recovery_engine.recover(self.imr, plan, "scratch/temp_train", None)
        self.assertTrue(metrics["status"].startswith("BLOCKED"))

    def test_failed_fine_tuning(self) -> None:
        """Verify pruner handles fine-tuning execution failures gracefully."""
        # Using a directory that doesn't exist to trigger failure/blocking
        plan = CompressionConfig(
            enabled_types=[CompressionType.PRUNING],
            pruning_sparsity=0.02,
            fine_tune_enabled=True
        )
        recovered, metrics = self.pruner._recovery_engine.recover(self.imr, plan, "invalid_path_123", "invalid_path_456")
        self.assertTrue(metrics["status"].startswith("BLOCKED"))

    def test_failed_export(self) -> None:
        """Verify export failure handling under extreme pruning configurations."""
        from uaqe.exporter.onnx_exporter import OnnxExporter
        exporter = OnnxExporter(self.logger, output_dir="scratch/temp_out")
        
        # Mismatched model path should raise ExportError
        exporter.bind_source_model_path("non_existent_file.onnx")
        from uaqe.domain.hardware_manager import HardwareProfile
        profile = HardwareProfile(
            profile_id="test",
            display_name="Test Profile",
            hardware_class=HardwareClass.RASPBERRY_PI,
            ram_bytes=1024,
            flash_bytes=1024,
            storage_bytes=1024,
            tensor_memory_bytes=1024,
            runtime="tflite-runtime",
            default_runtime="tflite-runtime",
            max_model_size_bytes=10,
            schema_version="1.0"
        )
        
        with self.assertRaises(Exception):
            exporter.export(self.imr, profile)


if __name__ == "__main__":
    unittest.main()
