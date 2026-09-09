"""``ExportFormat.ONNX`` exporter backend.

Scope note (see :mod:`~uaqe.exporter.tflite_exporter`'s module
docstring for the identical reasoning applied to that format): a
byte-compatible ONNX ``ModelProto`` requires the ``onnx`` protobuf
schema that ``uaqe.infrastructure.framework_adapters.onnx_adapter``
depends on for *loading* ``.onnx`` models. This backend instead writes
the same self-contained, versioned **UAQE interchange container**
:mod:`~uaqe.exporter.tflite_exporter` defines — magic header,
length-prefixed JSON graph description, raw parameter blob — under the
``.onnx`` extension, so a downstream conversion step that already links
``onnx``'s protobuf bindings can build the official ``ModelProto``
without re-deriving the graph from the IMR.
"""

from __future__ import annotations

import os
from typing import List, Sequence

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import ExportFormat
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.exporter.binary_exporter import DEFAULT_OUTPUT_DIR, write_artifact_file
from uaqe.exporter.exporter import DeploymentArtifact
from uaqe.exporter.tflite_exporter import build_container

#: Magic bytes identifying this backend's container format and its
#: version. Distinct from
#: :data:`~uaqe.exporter.tflite_exporter.MAGIC` purely so a downstream
#: reader can tell, without inspecting the file extension, which
#: conversion step a given container was produced for; the container
#: *structure* itself (see :func:`~uaqe.exporter.tflite_exporter.
#: build_container`) is otherwise identical.
MAGIC = b"UAQEONX1"


def consumer_supports_fp16(node: onnx.NodeProto) -> bool:
    """Explicit, deterministic compatibility helper.
    Currently, standard computational CPU ops like Conv, Gemm, MatMul, Add, Mul, BatchNormalization
    require homogenous input types. If activation inputs are float32 (which is the default in MobileNetV3),
    they cannot consume float16 directly.
    Only operators specifically whitelisted/designed to natively process mixed-precision can bypass casting.
    """
    # For now, CPU execution provider does not support mixed-precision on standard nodes natively,
    # so we return False. Later, it can be extended for other hardware or runtimes.
    return False


class OnnxExporter(IExporterBackend):
    """``ExportFormat.ONNX`` backend — exports a genuine ONNX ModelProto."""

    EXPORT_FORMAT = ExportFormat.ONNX

    def __init__(
        self,
        logger: ILogger,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        supported_target_profile_ids: Sequence[str] = (),
    ) -> None:
        """Initialize the ``OnnxExporter``.

        Args:
            logger: Structured logging sink.
            output_dir: Base directory artifact files are written under.
            supported_target_profile_ids: See
                :class:`~uaqe.exporter.binary_exporter.BinaryExporter`'s
                constructor docstring for how this is wired.
        """
        self._logger = logger
        self._output_dir = output_dir
        self._supported_target_profile_ids: List[str] = list(
            supported_target_profile_ids
        )
        self._source_model_path: Optional[str] = None

    def bind_source_model_path(self, source_model_path: Optional[str]) -> None:
        """Bind the original source model path for reference."""
        self._source_model_path = source_model_path

    def export(self, imr: IMR, target: HardwareProfile) -> DeploymentArtifact:
        """Serialize ``imr`` to a genuine ONNX model for ``target``.

        Args:
            imr: The final, fully-optimized model to export.
            target: The resolved hardware profile to export for.

        Returns:
            A ``DeploymentArtifact`` naming the single written file.

        Raises:
            ExportError: If the file cannot be written or check_model fails.
        """
        import onnx
        from onnx import numpy_helper
        import numpy as np
        from uaqe.common.exceptions import ExportError

        if not self._source_model_path or not os.path.exists(self._source_model_path):
            raise ExportError(
                f"Source ONNX model path '{self._source_model_path}' is invalid or does not exist. "
                "A genuine ONNX export requires the original ModelProto file.",
                code="EXPORT_SOURCE_MODEL_MISSING"
            )

        try:
            model = onnx.load(self._source_model_path)
        except Exception as e:
            raise ExportError(
                f"Failed to load original ONNX model from '{self._source_model_path}': {e}",
                code="EXPORT_SOURCE_MODEL_LOAD_FAILED"
            ) from e

        # Build a lookup dict of updated parameters from final IMR
        updated_parameters = {}
        for layer in imr.layers:
            for param_name, tensor in layer.parameters.items():
                updated_parameters[param_name] = (layer, tensor)

        # Track which initializers are converted to FLOAT16 or INT8
        converted_fp16_initializers = set()
        converted_int8_initializers = set()
        extra_initializers = []

        # Match ONNX initializers with IMR parameters
        new_initializers = []
        for init in model.graph.initializer:
            if init.name in updated_parameters:
                layer, imr_tensor = updated_parameters[init.name]

                # Determine optimized representation and source precision
                is_fp16 = False
                is_int8 = False
                source_precision_key = f"{init.name}_source_precision"
                source_precision = layer.attributes.get(source_precision_key, "")

                if imr_tensor.dtype == "float16":
                    is_fp16 = True
                elif imr_tensor.dtype == "int8":
                    is_int8 = True
                elif imr_tensor.dtype == "clustered_uint8":
                    if source_precision in ("float16", "FP16"):
                        is_fp16 = True
                    elif source_precision in ("int8", "INT8"):
                        is_int8 = True

                # Reconstruct numpy array values from IMR data
                if imr_tensor.dtype == "float16":
                    arr = np.frombuffer(imr_tensor.data, dtype=np.float16)
                elif imr_tensor.dtype == "int8":
                    arr = np.frombuffer(imr_tensor.data, dtype=np.int8)
                elif imr_tensor.dtype == "clustered_uint8":
                    # Reconstruct clustered_uint8 using codebook
                    codebook_key = f"{init.name}_codebook"
                    if codebook_key not in layer.attributes:
                        raise ExportError(
                            f"Missing codebook for clustered tensor '{init.name}' in layer '{layer.name}'.",
                            code="EXPORT_CLUSTERING_CODEBOOK_MISSING"
                        )
                    codebook = layer.attributes[codebook_key]
                    indices = np.frombuffer(imr_tensor.data, dtype=np.uint8)
                    centroids = np.array(codebook.centroids, dtype=np.float32)
                    reconstructed = centroids[indices]

                    if is_int8:
                        # NO DOUBLE QUANTIZATION RULE
                        if source_precision in ("int8", "INT8"):
                            arr = np.clip(np.round(reconstructed), -128, 127).astype(np.int8)
                        else:
                            scale_key = f"{init.name}_scale"
                            scale = layer.attributes.get(scale_key)
                            if scale is None or scale <= 0.0:
                                scale = 1.0
                            arr = np.clip(np.round(reconstructed / scale), -128, 127).astype(np.int8)
                    else:
                        arr = reconstructed
                else:
                    arr = np.frombuffer(imr_tensor.data, dtype=np.float32)

                # Reshape array to original shape to preserve shapes/element counts
                arr = arr.reshape(imr_tensor.shape)

                if is_fp16:
                    arr_fp16 = arr.astype(np.float16)
                    # Create genuine FLOAT16 initializer
                    new_init = numpy_helper.from_array(arr_fp16, name=init.name)
                    
                    # Verify shape and dims are unchanged
                    if list(new_init.dims) != list(imr_tensor.shape):
                        raise ExportError(f"Shape mismatch on FP16 export for {init.name}", code="EXPORT_SHAPE_MISMATCH")
                    if arr_fp16.size != np.prod(imr_tensor.shape):
                        raise ExportError(f"Element count mismatch for {init.name}", code="EXPORT_ELEMENT_COUNT_MISMATCH")
                    
                    # Verify 2 bytes/element in storage
                    serialized_bytes = len(new_init.raw_data) if new_init.raw_data else arr_fp16.nbytes
                    if serialized_bytes != 2 * arr_fp16.size:
                        raise ExportError(f"FLOAT16 storage byte mismatch for {init.name}: expected {2 * arr_fp16.size}, got {serialized_bytes}", code="EXPORT_STORAGE_MISMATCH")
                    
                    new_initializers.append(new_init)
                    converted_fp16_initializers.add(init.name)
                elif is_int8:
                    scale_key = f"{init.name}_scale"
                    zp_key = f"{init.name}_zero_point"
                    
                    scale = layer.attributes.get(scale_key)
                    zero_point = layer.attributes.get(zp_key)

                    # Validate scale + zero-point existence and constraints
                    if scale is None:
                        raise ExportError(f"Missing scale for INT8 parameter '{init.name}' in layer '{layer.name}'.", code="EXPORT_METADATA_MISSING")
                    if zero_point is None:
                        raise ExportError(f"Missing zero_point for INT8 parameter '{init.name}' in layer '{layer.name}'.", code="EXPORT_METADATA_MISSING")
                    # Reject per-channel explicitly
                    if hasattr(scale, "__len__") and not isinstance(scale, (str, bytes)):
                        raise ExportError(f"Per-channel INT8 quantization is not yet supported. Parameter '{init.name}' in layer '{layer.name}' uses per-channel scale.", code="UNSUPPORTED_QUANTIZATION")
                    if hasattr(zero_point, "__len__") and not isinstance(zero_point, (str, bytes)):
                        raise ExportError(f"Per-channel INT8 quantization is not yet supported. Parameter '{init.name}' in layer '{layer.name}' uses per-channel zero_point.", code="UNSUPPORTED_QUANTIZATION")

                    if not np.isfinite(scale) or scale <= 0.0:
                        raise ExportError(f"Invalid scale for INT8 parameter '{init.name}' in layer '{layer.name}': scale must be finite and > 0.", code="EXPORT_INVALID_METADATA")
                    if not isinstance(zero_point, (int, np.integer)):
                        raise ExportError(f"Invalid zero-point type for '{init.name}': expected integer, got {type(zero_point).__name__}.", code="EXPORT_INVALID_METADATA")
                    if not (-128 <= zero_point <= 127):
                        raise ExportError(f"Invalid zero-point value for '{init.name}': zero-point must be in [-128, 127], got {zero_point}.", code="EXPORT_INVALID_METADATA")

                    arr_int8 = np.asarray(arr, dtype=np.int8)
                    new_init = numpy_helper.from_array(arr_int8, name=init.name)

                    # Verify shape and dims are unchanged
                    if list(new_init.dims) != list(imr_tensor.shape):
                        raise ExportError(f"Shape mismatch on INT8 export for {init.name}", code="EXPORT_SHAPE_MISMATCH")
                    if arr_int8.size != np.prod(imr_tensor.shape):
                        raise ExportError(f"Element count mismatch for {init.name}", code="EXPORT_ELEMENT_COUNT_MISMATCH")
                    
                    # Verify 1 byte/element in storage
                    serialized_bytes = len(new_init.raw_data) if new_init.raw_data else arr_int8.nbytes
                    if serialized_bytes != arr_int8.size:
                        raise ExportError(f"INT8 storage byte mismatch for {init.name}: expected {arr_int8.size}, got {serialized_bytes}", code="EXPORT_STORAGE_MISMATCH")

                    new_initializers.append(new_init)
                    converted_int8_initializers.add(init.name)

                    # Create scale and zero-point initializers as required by DequantizeLinear
                    scale_init = numpy_helper.from_array(np.array(scale, dtype=np.float32), name=f"{init.name}__scale")
                    zp_init = numpy_helper.from_array(np.array(zero_point, dtype=np.int8), name=f"{init.name}__zero_point")
                    extra_initializers.extend([scale_init, zp_init])
                else:
                    # Match target datatype of original initializer to preserve types
                    if init.data_type == onnx.TensorProto.INT8:
                        arr = np.clip(np.round(arr), -128, 127).astype(np.int8)
                    else:
                        arr = arr.astype(np.float32)
                    new_init = numpy_helper.from_array(arr, name=init.name)
                    new_initializers.append(new_init)
            else:
                new_initializers.append(init)

        # Update initializers list with extra scale/zp parameters
        del model.graph.initializer[:]
        model.graph.initializer.extend(new_initializers + extra_initializers)

        # Insert Cast/DequantizeLinear nodes selectively per consumer compatibility
        new_nodes = []

        # 1. FP16 Cast node handling
        for init_name in converted_fp16_initializers:
            consumers = []
            for node in model.graph.node:
                if init_name in node.input:
                    consumers.append(node)
            if not consumers:
                continue

            needs_fp32_cast = False
            for consumer in consumers:
                if not consumer_supports_fp16(consumer):
                    needs_fp32_cast = True
                    break

            if needs_fp32_cast:
                cast_node_name = f"__fp16_cast__{init_name}"
                cast_output_name = f"{init_name}__fp32"
                
                existing_names = set()
                for node in model.graph.node:
                    existing_names.update(node.output)
                if cast_output_name in existing_names:
                    suffix = 0
                    while f"{cast_output_name}_{suffix}" in existing_names:
                        suffix += 1
                    cast_output_name = f"{cast_output_name}_{suffix}"

                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[init_name],
                    outputs=[cast_output_name],
                    name=cast_node_name,
                    to=onnx.TensorProto.FLOAT
                )
                new_nodes.append(cast_node)

                for consumer in consumers:
                    if not consumer_supports_fp16(consumer):
                        for idx, inp in enumerate(consumer.input):
                            if inp == init_name:
                                consumer.input[idx] = cast_output_name

        # 2. INT8 DequantizeLinear handling (primary Q/DQ path)
        for init_name in converted_int8_initializers:
            consumers = []
            for node in model.graph.node:
                if init_name in node.input:
                    consumers.append(node)
            if not consumers:
                continue

            # Naming conventions
            dequant_node_name = f"__int8_dequant__{init_name}"
            dequant_output_name = f"{init_name}__dequantized"

            existing_names = set()
            for node in model.graph.node:
                existing_names.update(node.output)
            if dequant_output_name in existing_names:
                suffix = 0
                while f"{dequant_output_name}_{suffix}" in existing_names:
                    suffix += 1
                dequant_output_name = f"{dequant_output_name}_{suffix}"

            # Create DequantizeLinear node
            dequant_node = onnx.helper.make_node(
                "DequantizeLinear",
                inputs=[init_name, f"{init_name}__scale", f"{init_name}__zero_point"],
                outputs=[dequant_output_name],
                name=dequant_node_name
            )
            new_nodes.append(dequant_node)

            # Update consumers to use the dequantized output name (reused across nodes)
            for consumer in consumers:
                for idx, inp in enumerate(consumer.input):
                    if inp == init_name:
                        consumer.input[idx] = dequant_output_name

        # Insert new nodes at the beginning of the graph (topologically valid since they depend on initializers)
        for node in reversed(new_nodes):
            model.graph.node.insert(0, node)

        # Update node attributes from IMR (e.g. group attribute for depthwise conv)
        for node in model.graph.node:
            imr_layer = next((l for l in imr.layers if l.name == node.name or (node.output and node.output[0] in l.outputs)), None)
            if imr_layer and "group" in imr_layer.attributes:
                for attr in node.attribute:
                    if attr.name == "group":
                        attr.i = imr_layer.attributes["group"]

        # Run shape inference to update graph output shapes and value_info shapes
        try:
            import onnx.shape_inference
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            self._logger.warning(f"ONNX shape inference failed: {e}")

        # Structural check
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            raise ExportError(
                f"Generated ONNX model failed structural check_model: {e}",
                code="EXPORT_ONNX_CHECK_FAILED"
            ) from e

        path = os.path.join(self._output_dir, target.profile_id, "model.onnx")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            onnx.save(model, path)
        except Exception as e:
            raise ExportError(
                f"Failed to save genuine ONNX model to '{path}': {e}",
                code="EXPORT_SAVE_FAILED"
            ) from e

        # Verify with onnxruntime InferenceSession when available
        try:
            import onnxruntime as ort
            ort.InferenceSession(path)
            self._logger.info("ONNX Runtime load verification passed.", path=path)
        except Exception as e:
            self._logger.warning("ONNX Runtime verification loading skipped or failed.", error=str(e))

        size_bytes = os.path.getsize(path)
        self._logger.info(
            "Exported genuine ONNX model.",
            target_profile_id=target.profile_id,
            path=path,
            size_bytes=size_bytes,
        )
        return DeploymentArtifact(
            file_paths=[path],
            export_format=self.EXPORT_FORMAT,
            target_profile_id=target.profile_id,
            size_bytes=size_bytes,
        )

    def supported_targets(self) -> List[str]:
        """Return the ``HardwareProfile.profile_id`` values this backend
        supports.
        """
        return list(self._supported_target_profile_ids)
