"""``.tflite`` model loader: the ``IFrameworkAdapter`` implementation
for TensorFlow Lite FlatBuffer models.

Parses a ``.tflite`` FlatBuffer schema graph into the
framework-independent ``IMR``, per ``03_API_Specification.md`` §15.
This is the only module in ``uaqe`` permitted to import
``tensorflow.lite`` (or the standalone ``tflite`` FlatBuffer schema
bindings) for parsing this format (``07_Coding_Standards.md`` §14
rule 4); ``keras_adapter.py`` and ``tensorflow_adapter.py`` each
depend on ``tensorflow`` independently for their own, unrelated file
formats.

Locked contract: ``03_API_Specification.md`` §15.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from tensorflow.lite.python import schema_py_generated as tflite_schema

from uaqe.common.exceptions import ModelLoadError
from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.interfaces.i_framework_adapter import IFrameworkAdapter
from uaqe.common.types import Precision

_SUPPORTED_EXTENSIONS = frozenset({".tflite"})

# TFLite TensorType enum value -> Precision. Any TensorType not listed
# here (e.g. INT32/INT64 shape or index tensors, BOOL masks, STRING)
# does not by itself determine a layer's Precision; see
# _infer_layer_precision below.
_TENSOR_TYPE_TO_PRECISION: Dict[int, Precision] = {
    tflite_schema.TensorType.FLOAT32: Precision.FP32,
    tflite_schema.TensorType.FLOAT16: Precision.FP16,
    tflite_schema.TensorType.INT8: Precision.INT8,
    tflite_schema.TensorType.UINT8: Precision.INT8,
}

# Human-readable dtype names for IMRTensor.dtype, keyed the same as
# _TENSOR_TYPE_TO_PRECISION above.
_TENSOR_TYPE_TO_DTYPE_NAME: Dict[int, str] = {
    tflite_schema.TensorType.FLOAT32: "float32",
    tflite_schema.TensorType.FLOAT16: "float16",
    tflite_schema.TensorType.INT8: "int8",
    tflite_schema.TensorType.UINT8: "uint8",
    tflite_schema.TensorType.INT32: "int32",
    tflite_schema.TensorType.INT64: "int64",
    tflite_schema.TensorType.BOOL: "bool",
    tflite_schema.TensorType.INT16: "int16",
}


class TFLiteAdapter(IFrameworkAdapter):
    """Loads ``.tflite`` files into the framework-independent ``IMR``."""

    def load(self, path: str) -> IMR:
        """Load the TFLite model at ``path`` and return its ``IMR``.

        Args:
            path: Filesystem path to the ``.tflite`` file.

        Returns:
            The loaded model as an ``IMR``.

        Raises:
            ModelLoadError: If ``path`` does not exist, is corrupt, or
                cannot be parsed as a TFLite FlatBuffer model.
        """
        model = self._load_model(path)
        graph = model.Subgraphs(0)
        op_codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]

        tensor_names = [
            graph.Tensors(i).Name().decode("utf-8") if graph.Tensors(i).Name() else f"tensor_{i}"
            for i in range(graph.TensorsLength())
        ]
        constant_tensors = self._extract_constant_tensors(model, graph)

        layers: List[IMRLayer] = []
        used_names: Dict[str, int] = {}
        for index in range(graph.OperatorsLength()):
            operator = graph.Operators(index)
            op_type = self._resolve_op_type(operator, op_codes)
            layer_name = self._resolve_layer_name(op_type, index, used_names)

            input_indices = [operator.Inputs(i) for i in range(operator.InputsLength())]
            output_indices = [operator.Outputs(i) for i in range(operator.OutputsLength())]

            parameters: Dict[str, IMRTensor] = {}
            inputs: List[str] = []
            for tensor_index in input_indices:
                if tensor_index < 0:
                    # A negative index denotes an optional, absent
                    # input (e.g. no bias tensor); skip it entirely.
                    continue
                name = tensor_names[tensor_index]
                if tensor_index in constant_tensors:
                    parameters[name] = constant_tensors[tensor_index]
                else:
                    inputs.append(name)

            outputs = [tensor_names[tensor_index] for tensor_index in output_indices]

            layers.append(
                IMRLayer(
                    name=layer_name,
                    op_type=op_type,
                    inputs=inputs,
                    outputs=outputs,
                    parameters=parameters,
                    attributes=self._extract_attributes(operator),
                    precision=self._infer_layer_precision(parameters, graph, output_indices),
                )
            )

        metadata = IMRMetadata(
            source_framework="tflite",
            source_format=".tflite",
            original_input_shapes=self._extract_input_shapes(graph, tensor_names),
            op_count=len(layers),
            total_parameters=sum(
                int(np.prod(tensor.shape)) if tensor.shape else 1
                for tensor in constant_tensors.values()
            ),
        )
        return IMR(layers=layers, metadata=metadata)

    def supports(self, extension: str) -> bool:
        """Report whether this adapter can load files with ``extension``.

        Args:
            extension: A file extension, including the leading dot.

        Returns:
            ``True`` if ``extension`` (case-insensitive) is ``.tflite``.
        """
        return extension.lower() in _SUPPORTED_EXTENSIONS

    def _load_model(self, path: str) -> "tflite_schema.Model":
        """Load and parse a TFLite FlatBuffer model from disk.

        Args:
            path: Filesystem path to the ``.tflite`` file.

        Returns:
            The parsed FlatBuffer ``Model``, with exactly one subgraph
            assumed (per TFLite's single-subgraph convention for
            inference-only exports).

        Raises:
            ModelLoadError: If the file cannot be read, is not a valid
                TFLite FlatBuffer, or contains zero subgraphs.
        """
        try:
            with open(path, "rb") as handle:
                raw_bytes = handle.read()
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"TFLite model file not found: {path!r}.",
                code="TFLITE_FILE_NOT_FOUND",
                remediation_hint="Verify the model path is correct.",
            ) from exc
        except OSError as exc:
            raise ModelLoadError(
                f"Cannot read {path!r} as a TFLite model file.",
                code="TFLITE_READ_FAILED",
                remediation_hint="Verify the file is readable and not corrupted.",
            ) from exc

        try:
            model = tflite_schema.Model.GetRootAsModel(raw_bytes, 0)
        except Exception as exc:  # noqa: BLE001 - flatbuffers raises varied decode errors
            raise ModelLoadError(
                f"Cannot parse {path!r} as a TFLite FlatBuffer model.",
                code="TFLITE_PARSE_FAILED",
                remediation_hint="Verify the file is a valid .tflite FlatBuffer export.",
            ) from exc

        if model.SubgraphsLength() == 0:
            raise ModelLoadError(
                f"TFLite model {path!r} contains no subgraphs.",
                code="TFLITE_EMPTY_GRAPH",
                remediation_hint="Verify the model was exported correctly.",
            )
        return model

    def _resolve_op_type(self, operator: Any, op_codes: List[Any]) -> str:
        """Resolve an operator's human-readable op-type name.

        Args:
            operator: The FlatBuffer ``Operator`` table.
            op_codes: The model's ``OperatorCode`` table, indexed by
                ``operator.OpcodeIndex()``.

        Returns:
            The builtin op name (e.g. ``"CONV_2D"``), or the custom
            op's registered string name when the opcode is
            ``CUSTOM``.
        """
        op_code = op_codes[operator.OpcodeIndex()]
        builtin_code = op_code.BuiltinCode()
        if builtin_code == tflite_schema.BuiltinOperator.CUSTOM:
            custom_name = op_code.CustomCode()
            return custom_name.decode("utf-8") if custom_name else "CUSTOM"
        return tflite_schema.BuiltinOperator.Name(builtin_code)

    def _resolve_layer_name(self, op_type: str, index: int, used_names: Dict[str, int]) -> str:
        """Resolve a unique, deterministic layer name.

        TFLite operators carry no name of their own (only tensors do),
        so a synthetic name is generated from the op type and its
        position in the subgraph, disambiguated on collision.

        Args:
            op_type: The operator's resolved op-type name.
            index: The operator's position in ``graph.Operators``.
            used_names: A mapping tracking how many times each base
                name has been generated so far, mutated in place.

        Returns:
            A unique layer name.
        """
        base_name = f"{op_type}_{index}"
        occurrence = used_names.get(base_name, 0)
        used_names[base_name] = occurrence + 1
        return base_name if occurrence == 0 else f"{base_name}_{occurrence}"

    def _extract_constant_tensors(
        self, model: "tflite_schema.Model", graph: "tflite_schema.SubGraph"
    ) -> Dict[int, IMRTensor]:
        """Decode every tensor in ``graph`` backed by model-buffer data.

        A TFLite tensor is a constant (weight/bias) rather than an
        activation if and only if its ``Buffer`` index points at a
        non-empty buffer in the model's buffer pool.

        Args:
            model: The parsed FlatBuffer model, providing the shared
                buffer pool.
            graph: The subgraph whose tensors are inspected.

        Returns:
            A mapping of tensor index (within ``graph.Tensors``) to
            its decoded ``IMRTensor``, for constant tensors only.
        """
        constants: Dict[int, IMRTensor] = {}
        for tensor_index in range(graph.TensorsLength()):
            tensor = graph.Tensors(tensor_index)
            buffer = model.Buffers(tensor.Buffer())
            data = buffer.DataAsNumpy()
            if not isinstance(data, np.ndarray) or data.size == 0:
                # An empty/absent buffer means this tensor is a
                # runtime activation, not a constant.
                continue
            shape = tuple(int(tensor.Shape(i)) for i in range(tensor.ShapeLength()))
            dtype_name = _TENSOR_TYPE_TO_DTYPE_NAME.get(tensor.Type(), "uint8")
            array = data.view(np.dtype(dtype_name)).reshape(shape) if shape else data.view(np.dtype(dtype_name))
            constants[tensor_index] = IMRTensor(
                shape=shape,
                dtype=dtype_name,
                data=array.tobytes(),
            )
        return constants

    def _extract_attributes(self, operator: Any) -> Dict[str, Any]:
        """Extract an operator's builtin option fields as attributes.

        Args:
            operator: The FlatBuffer ``Operator`` table.

        Returns:
            A mapping of option field name to value, drawn from the
            operator's ``BuiltinOptions`` table; empty if the operator
            carries no builtin options table (e.g. some custom ops).
        """
        options = operator.BuiltinOptions()
        if options is None:
            return {}
        attributes: Dict[str, Any] = {}
        for field_name in dir(options):
            if field_name.startswith("_") or field_name in ("Init", "Offset"):
                continue
            try:
                value = getattr(options, field_name)()
            except TypeError:
                # Fields requiring an index argument (vector accessors)
                # are not scalar hyperparameters; skip them.
                continue
            if isinstance(value, (int, float, bool, bytes, str)) or value is None:
                attributes[field_name] = value.decode("utf-8") if isinstance(value, bytes) else value
        return attributes

    def _infer_layer_precision(
        self,
        parameters: Dict[str, IMRTensor],
        graph: "tflite_schema.SubGraph",
        output_indices: List[int],
    ) -> Precision:
        """Infer a layer's ``Precision`` from its weights or output tensor type.

        Args:
            parameters: The layer's extracted constant/weight tensors.
            graph: The subgraph, consulted for output tensor types
                when no weight tensor is present (e.g. a
                parameter-free activation op).
            output_indices: The operator's output tensor indices.

        Returns:
            The ``Precision`` matching the first recognized weight
            dtype, falling back to the first recognized output
            tensor's declared type, or ``Precision.FP32`` if neither
            is recognized.
        """
        for tensor in parameters.values():
            precision = _TENSOR_TYPE_TO_PRECISION.get(
                {v: k for k, v in _TENSOR_TYPE_TO_DTYPE_NAME.items()}.get(tensor.dtype, -1)
            )
            if precision is not None:
                return precision

        for tensor_index in output_indices:
            tensor_type = graph.Tensors(tensor_index).Type()
            precision = _TENSOR_TYPE_TO_PRECISION.get(tensor_type)
            if precision is not None:
                return precision
        return Precision.FP32

    def _extract_input_shapes(
        self, graph: "tflite_schema.SubGraph", tensor_names: List[str]
    ) -> Dict[str, Tuple[int, ...]]:
        """Extract the model's declared input shapes.

        Args:
            graph: The subgraph, whose ``Inputs()`` list gives the
                true model input tensor indices.
            tensor_names: All tensor names in the subgraph, indexed by
                tensor index.

        Returns:
            A mapping of input tensor name to shape. Symbolic (e.g.
            batch) dimensions are represented as ``-1``.
        """
        shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(graph.InputsLength()):
            tensor_index = graph.Inputs(i)
            tensor = graph.Tensors(tensor_index)
            shape = tuple(
                int(tensor.Shape(dim)) if tensor.Shape(dim) > 0 else -1
                for dim in range(tensor.ShapeLength())
            )
            shapes[tensor_names[tensor_index]] = shape
        return shapes