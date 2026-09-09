"""Shared enumerations for the Universal AI Quantization Engine.

This module defines the framework-agnostic enumeration types consumed
throughout ``uaqe``. It has no dependencies on any other ``uaqe`` package
and no third-party dependencies, per the layering rules in
``09_Architecture_Lock.md`` §8.

Locked contract: ``03_API_Specification.md`` §1.2.
Locked members: ``09_Architecture_Lock.md`` §9 (via cross-reference to
``03_API_Specification.md``).
"""

from enum import Enum


class Precision(Enum):
    """Numeric precision applied to a model, layer, or tensor.

    Used by ``QuantizationConfig`` (per-model and per-layer overrides) and
    by ``IMRLayer.precision`` to record the precision a layer currently
    holds within the Internal Model Representation (IMR).
    """

    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"
    MIXED = "MIXED"


class HardwareClass(Enum):
    """Broad category of deployment target hardware.

    Used by ``HardwareConfig`` to select the target class and by
    ``PipelineBuilder.build()`` to select the concrete pipeline stage
    sequence and backend implementations appropriate for that class.
    """

    FPGA = "FPGA"
    EMBEDDED = "EMBEDDED"
    RASPBERRY_PI = "RASPBERRY_PI"


class CompressionType(Enum):
    """Model compression technique.

    Used by ``CompressionConfig.enabled_types`` to select which
    compression techniques a ``CompressionEngine`` should apply.
    """

    PRUNING = "PRUNING"
    WEIGHT_CLUSTERING = "WEIGHT_CLUSTERING"
    HUFFMAN = "HUFFMAN"
    RLE = "RLE"
    NONE = "NONE"


class ExportFormat(Enum):
    """Output artifact encoding produced by an exporter backend."""

    MEM = "MEM"
    HEX = "HEX"
    BIN = "BIN"
    TFLITE = "TFLITE"
    ONNX = "ONNX"
    H_HEADER = "H_HEADER"


class PipelineErrorPolicy(Enum):
    """Policy governing how ``PipelineOrchestrator`` reacts to a stage
    failure.

    Used by ``ExecutionConfig.on_error`` and consumed by
    ``PipelineOrchestrator.run()``/``resume()``.
    """

    ABORT = "ABORT"
    SKIP_STAGE = "SKIP_STAGE"
    BEST_EFFORT = "BEST_EFFORT"
