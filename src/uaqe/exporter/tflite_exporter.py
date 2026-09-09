"""``ExportFormat.TFLITE`` exporter backend.

Scope note (read before relying on this for a real TFLite Micro
deployment): building a byte-compatible TensorFlow Lite FlatBuffer
requires the FlatBuffer schema compiler and the ``tflite``/
``tensorflow.lite`` schema bindings that
``uaqe.infrastructure.framework_adapters.tflite_adapter`` already
depends on for *loading* ``.tflite`` models. Round-tripping through
that same schema for writing is a materially larger undertaking (a
full FlatBuffer builder pass) than this package's scope covers, so this
backend instead writes a self-contained, versioned **UAQE interchange
container**: a small magic header, a length-prefixed JSON graph
description, and the same raw parameter buffer
:mod:`~uaqe.exporter.binary_exporter` produces. A downstream conversion
step (outside this package) that already links the FlatBuffer schema
can read this container and re-emit an official ``.tflite`` FlatBuffer
without needing to re-derive the graph from the IMR itself. The file
still carries the ``.tflite`` extension so existing deployment tooling
that dispatches on file extension continues to route it correctly to
that conversion step.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, List, Sequence

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import ExportFormat
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.exporter.binary_exporter import (
    DEFAULT_OUTPUT_DIR,
    serialize_parameters,
    write_artifact_file,
)
from uaqe.exporter.exporter import DeploymentArtifact

#: Magic bytes identifying this backend's container format and its
#: version, checked by any downstream reader before parsing further.
MAGIC = b"UAQETFL1"


def _build_graph_metadata(imr: IMR) -> Dict[str, Any]:
    """Build the JSON-serializable graph description embedded in the
    container, sufficient for a downstream FlatBuffer conversion step
    to reconstruct tensor offsets without re-parsing the parameter blob
    byte-by-byte.

    Args:
        imr: The model being exported.

    Returns:
        A plain ``dict`` describing every layer, its op type, shape and
        precision metadata, and (for parameters) the byte offset/length
        of its data within the accompanying parameter blob.
    """
    layers_meta: List[Dict[str, Any]] = []
    running_offset = 0
    for layer in imr.topological_order():
        parameters_meta: Dict[str, Any] = {}
        for name in sorted(layer.parameters):
            tensor = layer.parameters[name]
            parameters_meta[name] = {
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
                "offset": running_offset,
                "length": len(tensor.data),
            }
            running_offset += len(tensor.data)
        layers_meta.append(
            {
                "name": layer.name,
                "op_type": layer.op_type,
                "inputs": list(layer.inputs),
                "outputs": list(layer.outputs),
                "precision": layer.precision.value,
                "attributes": layer.attributes,
                "parameters": parameters_meta,
            }
        )
    return {
        "source_framework": imr.metadata.source_framework,
        "source_format": imr.metadata.source_format,
        "op_count": imr.metadata.op_count,
        "total_parameters": imr.metadata.total_parameters,
        "layers": layers_meta,
    }


def _serialize_value(val):
    if hasattr(val, "centroids"):
        return {"centroids": list(val.centroids)}
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, tuple):
        return [_serialize_value(v) for v in val]
    return val


def build_container(imr: IMR) -> bytes:
    """Build this backend's full container payload for ``imr``.

    Args:
        imr: The model to encode.

    Returns:
        ``MAGIC`` followed by a 4-byte big-endian JSON-metadata length,
        the UTF-8 JSON metadata itself, and finally the raw parameter
        blob (offsets in the metadata are relative to the start of this
        trailing blob).
    """
    metadata = _build_graph_metadata(imr)
    metadata_serializable = _serialize_value(metadata)
    metadata_bytes = json.dumps(metadata_serializable, separators=(",", ":")).encode("utf-8")
    parameter_blob = serialize_parameters(imr)
    return (
        MAGIC
        + struct.pack(">I", len(metadata_bytes))
        + metadata_bytes
        + parameter_blob
    )


import onnx
from onnx import numpy_helper
import numpy as np
import tensorflow as tf

class ONNXToTFModel(tf.keras.Model):
    def __init__(self, onnx_model):
        super().__init__()
        self.graph = onnx_model.graph
        self.initializers = {}
        for init in self.graph.initializer:
            self.initializers[init.name] = tf.constant(numpy_helper.to_array(init))

    def call(self, inputs):
        tensors = {self.graph.input[0].name: inputs}
        # Convert input NCHW to NHWC if it is a 4D spatial tensor
        if len(inputs.shape) == 4:
            x = tf.transpose(inputs, perm=[0, 2, 3, 1])
        else:
            x = inputs
        tensors[self.graph.input[0].name] = x

        for node in self.graph.node:
            op_type = node.op_type
            
            def get_input(name):
                if name in tensors:
                    return tensors[name]
                if name in self.initializers:
                    return self.initializers[name]
                raise ValueError(f"Tensor/Initializer '{name}' not found.")

            if op_type == "Cast":
                val = get_input(node.input[0])
                to_type = None
                for attr in node.attribute:
                    if attr.name == "to":
                        to_type = attr.i
                if to_type == onnx.TensorProto.FLOAT:
                    tensors[node.output[0]] = tf.cast(val, tf.float32)
                elif to_type == onnx.TensorProto.FLOAT16:
                    tensors[node.output[0]] = tf.cast(val, tf.float16)
                else:
                    tensors[node.output[0]] = val

            elif op_type == "DequantizeLinear":
                val = get_input(node.input[0])
                scale = get_input(node.input[1])
                zp = get_input(node.input[2])

                zp_f32 = tf.cast(zp, tf.float32)
                scale_f32 = tf.cast(scale, tf.float32)
                val_f32 = tf.cast(val, tf.float32)

                tensors[node.output[0]] = scale_f32 * (val_f32 - zp_f32)

            elif op_type == "Conv":
                X_name = node.input[0]
                W_name = node.input[1]
                B_name = node.input[2] if len(node.input) > 2 else None

                X_tf = tensors[X_name]
                W_tf = get_input(W_name)

                attributes = {attr.name: attr for attr in node.attribute}
                strides = attributes["strides"].ints if "strides" in attributes else [1, 1]
                pads = attributes["pads"].ints if "pads" in attributes else [0, 0, 0, 0]
                dilations = attributes["dilations"].ints if "dilations" in attributes else [1, 1]
                group = attributes["group"].i if "group" in attributes else 1

                if group > 1:
                    W_tf = tf.transpose(W_tf, perm=[2, 3, 0, 1])
                    if pads == [1, 1, 1, 1]:
                        X_tf = tf.pad(X_tf, [[0, 0], [1, 1], [1, 1], [0, 0]])
                        padding_mode = "valid"
                    elif pads == [2, 2, 2, 2]:
                        X_tf = tf.pad(X_tf, [[0, 0], [2, 2], [2, 2], [0, 0]])
                        padding_mode = "valid"
                    else:
                        padding_mode = "same" if pads[0] > 0 else "valid"

                    conv_out = tf.nn.depthwise_conv2d(
                        X_tf, W_tf,
                        strides=[1, strides[0], strides[1], 1],
                        padding=padding_mode.upper(),
                        dilations=dilations
                    )
                else:
                    W_tf = tf.transpose(W_tf, perm=[2, 3, 1, 0])
                    if pads == [1, 1, 1, 1]:
                        X_tf = tf.pad(X_tf, [[0, 0], [1, 1], [1, 1], [0, 0]])
                        padding_mode = "valid"
                    elif pads == [2, 2, 2, 2]:
                        X_tf = tf.pad(X_tf, [[0, 0], [2, 2], [2, 2], [0, 0]])
                        padding_mode = "valid"
                    else:
                        padding_mode = "same" if pads[0] > 0 else "valid"

                    conv_out = tf.nn.conv2d(
                        X_tf, W_tf,
                        strides=[1, strides[0], strides[1], 1],
                        padding=padding_mode.upper(),
                        dilations=dilations
                    )

                if B_name:
                    bias = get_input(B_name)
                    conv_out = tf.nn.bias_add(conv_out, bias)

                tensors[node.output[0]] = conv_out

            elif op_type == "Relu":
                X_tf = tensors[node.input[0]]
                tensors[node.output[0]] = tf.nn.relu(X_tf)

            elif op_type == "HardSigmoid":
                X_tf = tensors[node.input[0]]
                alpha = 0.2
                beta = 0.5
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha = attr.f
                    elif attr.name == "beta":
                        beta = attr.f
                tensors[node.output[0]] = tf.clip_by_value(alpha * X_tf + beta, 0.0, 1.0)

            elif op_type == "Mul":
                A_tf = get_input(node.input[0])
                B_tf = get_input(node.input[1])
                tensors[node.output[0]] = tf.multiply(A_tf, B_tf)

            elif op_type == "Add":
                A_tf = get_input(node.input[0])
                B_tf = get_input(node.input[1])
                tensors[node.output[0]] = tf.add(A_tf, B_tf)

            elif op_type == "GlobalAveragePool":
                X_tf = tensors[node.input[0]]
                tensors[node.output[0]] = tf.reduce_mean(X_tf, axis=[1, 2], keepdims=True)

            elif op_type == "Flatten":
                X_tf = tensors[node.input[0]]
                tensors[node.output[0]] = tf.keras.layers.Flatten()(X_tf)

            elif op_type == "Gemm":
                A_tf = get_input(node.input[0])
                B_tf = get_input(node.input[1])
                C_tf = get_input(node.input[2]) if len(node.input) > 2 else None

                transB = 0
                alpha = 1.0
                beta = 1.0
                for attr in node.attribute:
                    if attr.name == "transB":
                        transB = attr.i
                    elif attr.name == "alpha":
                        alpha = attr.f
                    elif attr.name == "beta":
                        beta = attr.f

                if transB == 1:
                    B_tf = tf.transpose(B_tf, perm=[1, 0])

                gemm_out = alpha * tf.matmul(A_tf, B_tf)
                if C_tf is not None:
                    gemm_out = gemm_out + beta * C_tf

                tensors[node.output[0]] = gemm_out

            elif op_type == "Identity":
                tensors[node.output[0]] = get_input(node.input[0])

            else:
                raise NotImplementedError(f"Unsupported node type: {op_type}")

        final_output = tensors[self.graph.output[0].name]
        return final_output


class TFLiteExporter(IExporterBackend):
    """``ExportFormat.TFLITE`` backend — see the module docstring for
    this container's relationship to the official FlatBuffer format.

    Attributes:
        EXPORT_FORMAT: This backend's ``ExportFormat``.
    """

    EXPORT_FORMAT = ExportFormat.TFLITE

    def __init__(
        self,
        logger: ILogger,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        supported_target_profile_ids: Sequence[str] = (),
    ) -> None:
        """Initialize the ``TFLiteExporter``.

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
        self._calibration_dataset_path: Optional[str] = None

    def bind_source_model_path(self, path: Optional[str]) -> None:
        """Bind the source model path from the orchestration stage."""
        self._source_model_path = path

    def bind_calibration_dataset_path(self, path: Optional[str]) -> None:
        """Bind the calibration dataset path for representative dataset creation."""
        self._calibration_dataset_path = path

    def _build_representative_dataset_generator(
        self, calibration_path: Optional[str], input_shape: List[int]
    ):
        """Create a generator yielding representative samples for TFLiteConverter."""
        if not calibration_path or not os.path.exists(calibration_path):
            return None

        # Load samples from manifest or directory
        samples = []
        if calibration_path.endswith(".json") and os.path.isfile(calibration_path):
            with open(calibration_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if "images" in manifest:
                from uaqe.evaluation.real_dataset_adapter import RealDatasetAdapter

                cal_root = manifest.get("calibration_root", os.path.dirname(calibration_path))
                prep_ref = manifest.get("preprocessing_reference", {})
                class_mapping = prep_ref.get("class_mapping", {})
                prep_mode = prep_ref.get("preprocessing", "rgb_0_1")
                shape_tuple = tuple(prep_ref.get("input_shape", input_shape))

                adapter = RealDatasetAdapter(
                    dataset_path=cal_root if os.path.isabs(cal_root) else os.path.abspath(cal_root),
                    class_mapping=class_mapping,
                    preprocessing_mode=prep_mode,
                    input_shape=shape_tuple,
                )
                cur = os.path.dirname(os.path.abspath(calibration_path))
                project_root = cur
                while project_root and not os.path.exists(os.path.join(project_root, "datasets")):
                    parent = os.path.dirname(project_root)
                    if parent == project_root:
                        break
                    project_root = parent

                img_list = []
                for img in manifest["images"]:
                    rel_p = img["relative_path"]
                    abs_p = os.path.abspath(rel_p)
                    if not os.path.exists(abs_p):
                        candidate = os.path.join(project_root, rel_p)
                        if os.path.exists(candidate):
                            abs_p = candidate
                        else:
                            abs_p = os.path.join(
                                os.path.dirname(os.path.abspath(calibration_path)),
                                os.path.basename(rel_p),
                            )
                    conf_class = img.get("mapped_configured_class", img.get("class", ""))
                    class_idx = class_mapping.get(conf_class, 0)
                    img_list.append((abs_p, class_idx, img.get("class", "")))
                adapter.samples = img_list

                def rep_gen():
                    for i in range(len(adapter)):
                        tensor = adapter[i]["tensor"]
                        if len(tensor.shape) == 3:
                            tensor = np.expand_dims(tensor, axis=0)
                        yield [tensor.astype(np.float32)]

                return rep_gen

        elif os.path.isdir(calibration_path):
            from uaqe.quantization.calibration_dataset import CalibrationDataset
            try:
                ds = CalibrationDataset(calibration_path, batch_size=1)
                samples = []
                for batch in ds:
                    for name, vals in batch.inputs.items():
                        arr = np.array(vals, dtype=np.float32).reshape(input_shape)
                        samples.append(arr)
                if not samples:
                    return None
                def rep_gen():
                    for s in samples:
                        yield [s]
                return rep_gen
            except Exception:
                return None

        return None

    def export(self, imr: IMR, target: HardwareProfile) -> DeploymentArtifact:
        """Serialize ``imr`` to a genuine TFLite model for ``target``."""
        from uaqe.common.exceptions import ExportError
        from uaqe.exporter.onnx_exporter import OnnxExporter

        if not getattr(self, "_source_model_path", None) or not os.path.exists(self._source_model_path):
            raise ExportError(
                f"Runtime 'tflite-runtime' requires a genuine TFLite FlatBuffer. Source model path '{getattr(self, '_source_model_path', None)}' is invalid or missing.",
                code="EXPORT_SOURCE_MODEL_MISSING"
            )

        # 1. Programmatically export optimized ONNX model
        self._logger.info("Programmatically generating optimized ONNX model first...")
        onnx_exporter = OnnxExporter(logger=self._logger, output_dir=self._output_dir)
        onnx_exporter.bind_source_model_path(self._source_model_path)
        onnx_artifact = onnx_exporter.export(imr, target)
        opt_onnx_path = onnx_artifact.file_paths[0]

        # 2. Check for representative dataset calibration
        cal_path = getattr(self, "_calibration_dataset_path", None)
        # Determine model for translation: when calibrating with representative dataset,
        # translating the clean source model allows TFLiteConverter to perform clean INT8
        # quantization across all ops without zero-point conflicts from ONNX DequantizeLinear.
        model_to_translate = (
            self._source_model_path
            if (cal_path and os.path.exists(cal_path) and os.path.exists(self._source_model_path))
            else opt_onnx_path
        )

        try:
            onnx_model = onnx.load(model_to_translate)
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            raise ExportError(
                f"ONNX model check failed for '{model_to_translate}': {e}",
                code="EXPORT_ONNX_CHECK_FAILED"
            ) from e

        # 3. Instantiate and trace custom Keras model
        try:
            tf_model = ONNXToTFModel(onnx_model)
            
            # Dynamically extract input shape from ONNX model graph to avoid mismatches
            onnx_input = onnx_model.graph.input[0]
            input_shape = [dim.dim_value for dim in onnx_input.type.tensor_type.shape.dim]
            input_shape = [1 if (dim is None or dim <= 0) else dim for dim in input_shape]
            
            # Call on dummy input once to build internal variables
            dummy_input = tf.random.normal(input_shape)
            _ = tf_model(dummy_input)
        except Exception as e:
            raise ExportError(
                f"Failed to translate ONNX graph to Keras/TF model: {e}",
                code="TFLITE_CONVERSION_TRANSLATION_FAILED"
            ) from e

        rep_gen = self._build_representative_dataset_generator(cal_path, input_shape)

        # 4. Run TFLite Converter
        try:
            self._logger.info("Running TensorFlow Lite Converter...")
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if rep_gen is not None:
                self._logger.info("Wiring representative dataset into TensorFlow Lite Converter...")
                converter.representative_dataset = rep_gen
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                ]
            tflite_model_bytes = converter.convert()
        except Exception as e:
            raise ExportError(
                f"TensorFlow Lite Converter failed: {e}",
                code="TFLITE_CONVERSION_FAILED"
            ) from e

        if not tflite_model_bytes or len(tflite_model_bytes) == 0:
            raise ExportError(
                "Generated TFLite model FlatBuffer is empty.",
                code="TFLITE_EMPTY_FLATBUFFER"
            )

        # 5. Save .tflite file
        path = os.path.join(self._output_dir, target.profile_id, "model.tflite")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                f.write(tflite_model_bytes)
        except Exception as e:
            raise ExportError(
                f"Failed to save TFLite FlatBuffer to '{path}': {e}",
                code="EXPORT_SAVE_FAILED"
            ) from e

        # 6. Verify loading with TFLite Interpreter
        try:
            interpreter = tf.lite.Interpreter(path)
            interpreter.allocate_tensors()
            self._logger.info("TFLite Interpreter load and tensor allocation passed.", path=path)
        except Exception as e:
            raise ExportError(
                f"Generated TFLite model failed load/allocation check: {e}",
                code="TFLITE_INTERPRETER_LOAD_FAILED"
            ) from e

        size_bytes = os.path.getsize(path)
        self._logger.info(
            "Exported genuine TFLite model.",
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
