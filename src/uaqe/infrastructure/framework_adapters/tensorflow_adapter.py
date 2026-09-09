"""``.pb`` model loader: the ``IFrameworkAdapter`` implementation for
TensorFlow frozen graphs.

Parses a TensorFlow frozen ``GraphDef`` into the framework-independent
``IMR``, per ``03_API_Specification.md`` §15. This is the only module
in ``uaqe`` permitted to import the ``tensorflow`` third-party library
for frozen-graph parsing (``07_Coding_Standards.md`` §14 rule 4);
``keras_adapter.py`` and ``tflite_adapter.py`` each depend on
``tensorflow`` independently for their own, unrelated file formats.

Locked contract: ``03_API_Specification.md`` §15.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from google.protobuf.message import DecodeError
from tensorflow.core.framework import types_pb2

from uaqe.common.exceptions import ModelLoadError
from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.interfaces.i_framework_adapter import IFrameworkAdapter
from uaqe.common.types import Precision

_SUPPORTED_EXTENSIONS = frozenset({".pb"})

_CONST_OP = "Const"
_PLACEHOLDER_OP = "Placeholder"

# tf.DType name -> Precision. Any dtype not listed here (e.g. int32
# shape tensors, bool masks) does not by itself determine a layer's
# Precision; see _infer_layer_precision below.
_DTYPE_TO_PRECISION: Dict[str, Precision] = {
    "float32": Precision.FP32,
    "float16": Precision.FP16,
    "int8": Precision.INT8,
    "uint8": Precision.INT8,
    "qint8": Precision.INT8,
    "quint8": Precision.INT8,
}


class TensorFlowAdapter(IFrameworkAdapter):
    """Loads ``.pb`` frozen-graph files into the framework-independent ``IMR``."""

    def load(self, path: str) -> IMR:
        """Load the TensorFlow frozen graph at ``path`` and return its ``IMR``.

        Args:
            path: Filesystem path to the ``.pb`` file.

        Returns:
            The loaded model as an ``IMR``.

        Raises:
            ModelLoadError: If ``path`` does not exist, is corrupt, or
                is not a valid frozen ``GraphDef``.
        """
        graph_def = self._load_graph_def(path)

        const_arrays: Dict[str, np.ndarray] = {
            node.name: self._const_node_to_array(node)
            for node in graph_def.node
            if node.op == _CONST_OP
        }

        layers: List[IMRLayer] = []
        for node in graph_def.node:
            if node.op == _CONST_OP:
                # Constants are folded into consuming layers'
                # `parameters`, not represented as their own IMR layer,
                # mirroring how the ONNX adapter treats initializers.
                continue

            parameters: Dict[str, IMRTensor] = {}
            inputs: List[str] = []
            for input_name in node.input:
                clean_name = input_name[1:] if input_name.startswith("^") else input_name
                base_name = clean_name.split(":", 1)[0]
                if base_name in const_arrays:
                    parameters[base_name] = self._to_imr_tensor(const_arrays[base_name])
                else:
                    inputs.append(clean_name)

            attributes: Dict[str, Any] = {
                key: self._attr_value_to_python(value) for key, value in node.attr.items()
            }
            layers.append(
                IMRLayer(
                    name=node.name,
                    op_type=node.op,
                    inputs=inputs,
                    outputs=[node.name],
                    parameters=parameters,
                    attributes=attributes,
                    precision=self._infer_layer_precision(parameters, node),
                )
            )

        metadata = IMRMetadata(
            source_framework="tensorflow",
            source_format=".pb",
            original_input_shapes=self._extract_input_shapes(graph_def),
            op_count=len(layers),
            total_parameters=sum(
                int(np.prod(array.shape)) if array.shape else 1 for array in const_arrays.values()
            ),
        )
        return IMR(layers=layers, metadata=metadata)

    def supports(self, extension: str) -> bool:
        """Report whether this adapter can load files with ``extension``.

        Args:
            extension: A file extension, including the leading dot.

        Returns:
            ``True`` if ``extension`` (case-insensitive) is ``.pb``.
        """
        return extension.lower() in _SUPPORTED_EXTENSIONS

    def _load_graph_def(self, path: str) -> "tf.compat.v1.GraphDef":
        """Load and parse a frozen ``GraphDef`` from disk.

        Args:
            path: Filesystem path to the ``.pb`` file.

        Returns:
            The parsed ``GraphDef``.

        Raises:
            ModelLoadError: If the file cannot be read or is not a
                valid serialized ``GraphDef``.
        """
        try:
            with tf.io.gfile.GFile(path, "rb") as handle:
                raw_bytes = handle.read()
        except (FileNotFoundError, tf.errors.NotFoundError) as exc:
            raise ModelLoadError(
                f"TensorFlow model file not found: {path!r}.",
                code="TENSORFLOW_FILE_NOT_FOUND",
                remediation_hint="Verify the model path is correct.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - tf.io raises varied I/O errors
            raise ModelLoadError(
                f"Cannot read {path!r} as a TensorFlow model file.",
                code="TENSORFLOW_READ_FAILED",
                remediation_hint="Verify the file is readable and not corrupted.",
            ) from exc

        graph_def = tf.compat.v1.GraphDef()
        try:
            graph_def.ParseFromString(raw_bytes)
        except DecodeError as exc:
            raise ModelLoadError(
                f"Cannot parse {path!r} as a TensorFlow frozen GraphDef.",
                code="TENSORFLOW_PARSE_FAILED",
                remediation_hint="Verify the file is a frozen GraphDef protobuf (not a SavedModel directory).",
            ) from exc

        if len(graph_def.node) == 0:
            raise ModelLoadError(
                f"TensorFlow model {path!r} contains no graph nodes.",
                code="TENSORFLOW_EMPTY_GRAPH",
                remediation_hint="Verify the frozen graph was exported correctly.",
            )
        return graph_def

    def _const_node_to_array(self, node: Any) -> np.ndarray:
        """Decode a ``Const`` node's embedded tensor into a numpy array.

        Args:
            node: A ``NodeDef`` with ``op == "Const"``.

        Returns:
            The constant's value as a numpy array.
        """
        tensor_proto = node.attr["value"].tensor
        return tf.make_ndarray(tensor_proto)

    def _to_imr_tensor(self, array: np.ndarray) -> IMRTensor:
        """Convert a numpy array (from a ``Const`` node) into an ``IMRTensor``.

        Args:
            array: The constant's decoded value.

        Returns:
            The equivalent ``IMRTensor``.
        """
        return IMRTensor(
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            data=array.tobytes(),
        )

    def _attr_value_to_python(self, attr_value: Any) -> Any:
        """Decode an ``AttrValue`` proto field into a plain Python value.

        Args:
            attr_value: A single node ``attr`` map value.

        Returns:
            The plain-Python representation of whichever ``oneof``
            field is set (e.g. ``int``, ``float``, ``bool``, ``str``,
            a shape tuple, or a dtype name); the raw proto tensor
            payload for ``Const`` nodes is handled separately via
            :meth:`_const_node_to_array` and is not decoded here.
        """
        kind = attr_value.WhichOneof("value")
        if kind == "i":
            return attr_value.i
        if kind == "f":
            return attr_value.f
        if kind == "b":
            return attr_value.b
        if kind == "s":
            return attr_value.s.decode("utf-8", errors="replace")
        if kind == "type":
            return types_pb2.DataType.Name(attr_value.type)
        if kind == "shape":
            return tuple(
                dim.size if dim.size >= 0 else -1 for dim in attr_value.shape.dim
            )
        if kind == "list":
            return list(attr_value.list.i) or list(attr_value.list.f) or list(attr_value.list.s)
        return None

    def _infer_layer_precision(self, parameters: Dict[str, IMRTensor], node: Any) -> Precision:
        """Infer a layer's ``Precision`` from its parameters or declared dtype.

        Args:
            parameters: The layer's extracted parameter tensors.
            node: The originating ``NodeDef``, consulted for a ``"T"``
                or ``"dtype"`` attribute when no parameter tensor is
                present (e.g. a parameter-free activation node).

        Returns:
            The ``Precision`` matching the first recognized parameter
            dtype, falling back to the node's own declared dtype
            attribute, or ``Precision.FP32`` if neither is recognized.
        """
        for tensor in parameters.values():
            precision = _DTYPE_TO_PRECISION.get(tensor.dtype)
            if precision is not None:
                return precision

        for attr_name in ("T", "dtype"):
            if attr_name in node.attr:
                type_name = types_pb2.DataType.Name(node.attr[attr_name].type).replace("DT_", "").lower()
                precision = _DTYPE_TO_PRECISION.get(type_name)
                if precision is not None:
                    return precision
        return Precision.FP32

    def _extract_input_shapes(self, graph_def: "tf.compat.v1.GraphDef") -> Dict[str, Tuple[int, ...]]:
        """Extract the model's declared input shapes from its ``Placeholder`` nodes.

        Args:
            graph_def: The parsed frozen graph.

        Returns:
            A mapping of input tensor name to shape. Symbolic (dynamic)
            dimensions, e.g. a batch dimension, are represented as
            ``-1``.
        """
        shapes: Dict[str, Tuple[int, ...]] = {}
        for node in graph_def.node:
            if node.op != _PLACEHOLDER_OP:
                continue
            shape_attr = node.attr.get("shape")
            if shape_attr is None:
                continue
            shapes[node.name] = tuple(
                dim.size if dim.size >= 0 else -1 for dim in shape_attr.shape.dim
            )
        return shapes
