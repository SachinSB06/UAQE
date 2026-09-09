"""``.onnx`` model loader: the ``IFrameworkAdapter`` implementation for
the ONNX framework.

Parses an ONNX ``ModelProto`` graph into the framework-independent
``IMR``, per ``03_API_Specification.md`` §15. This is the only module
in ``uaqe`` permitted to import the ``onnx`` third-party library
(``07_Coding_Standards.md`` §14 rule 4).

Locked contract: ``03_API_Specification.md`` §15.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from uaqe.common.exceptions import ModelLoadError
from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.interfaces.i_framework_adapter import IFrameworkAdapter
from uaqe.common.types import Precision

_SUPPORTED_EXTENSIONS = frozenset({".onnx"})

# ONNX/numpy dtype name -> Precision. Any dtype not listed here (e.g.
# int32/int64 shape/index tensors, bool masks) does not by itself
# determine a layer's Precision; see _infer_layer_precision below.
_DTYPE_TO_PRECISION: Dict[str, Precision] = {
    "float32": Precision.FP32,
    "float16": Precision.FP16,
    "int8": Precision.INT8,
    "uint8": Precision.INT8,
}


class OnnxAdapter(IFrameworkAdapter):
    """Loads ``.onnx`` files into the framework-independent ``IMR``."""

    def load(self, path: str) -> IMR:
        """Load the ONNX model at ``path`` and return its ``IMR``.

        Args:
            path: Filesystem path to the ``.onnx`` file.

        Returns:
            The loaded model as an ``IMR``.

        Raises:
            ModelLoadError: If ``path`` does not exist, is corrupt, or
                fails ONNX's own structural checker.
        """
        model = self._load_model_proto(path)
        graph = model.graph

        initializer_arrays: Dict[str, np.ndarray] = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in graph.initializer
        }

        layers: List[IMRLayer] = []
        used_names: Dict[str, int] = {}
        for index, node in enumerate(graph.node):
            layer_name = self._resolve_layer_name(node, index, used_names)
            parameters = {
                input_name: self._to_imr_tensor(initializer_arrays[input_name])
                for input_name in node.input
                if input_name in initializer_arrays
            }
            attributes: Dict[str, Any] = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            layers.append(
                IMRLayer(
                    name=layer_name,
                    op_type=node.op_type,
                    inputs=list(node.input),
                    outputs=list(node.output),
                    parameters=parameters,
                    attributes=attributes,
                    precision=self._infer_layer_precision(parameters),
                )
            )

        original_input_shapes = self._extract_input_shapes(graph, initializer_arrays)
        total_parameters = sum(
            int(np.prod(array.shape)) if array.shape else 1 for array in initializer_arrays.values()
        )

        metadata = IMRMetadata(
            source_framework="onnx",
            source_format=".onnx",
            original_input_shapes=original_input_shapes,
            op_count=len(layers),
            total_parameters=total_parameters,
        )
        return IMR(layers=layers, metadata=metadata)

    def supports(self, extension: str) -> bool:
        """Report whether this adapter can load files with ``extension``.

        Args:
            extension: A file extension, including the leading dot.

        Returns:
            ``True`` if ``extension`` (case-insensitive) is ``.onnx``.
        """
        return extension.lower() in _SUPPORTED_EXTENSIONS

    def _load_model_proto(self, path: str) -> "onnx.ModelProto":
        """Load and structurally check an ONNX ``ModelProto`` from disk.

        Args:
            path: Filesystem path to the ``.onnx`` file.

        Returns:
            The parsed, checker-validated ``ModelProto``.

        Raises:
            ModelLoadError: If the file cannot be read, is not valid
                protobuf, or fails ``onnx.checker.check_model``.
        """
        try:
            model = onnx.load(path)
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"ONNX model file not found: {path!r}.",
                code="ONNX_FILE_NOT_FOUND",
                remediation_hint="Verify the model path is correct.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - onnx raises varied decode errors
            raise ModelLoadError(
                f"Cannot parse {path!r} as an ONNX model.",
                code="ONNX_PARSE_FAILED",
                remediation_hint="Verify the file is a valid ONNX protobuf model.",
            ) from exc

        try:
            onnx.checker.check_model(model)
        except Exception as exc:  # noqa: BLE001 - onnx raises its own ValidationError
            raise ModelLoadError(
                f"ONNX model {path!r} failed structural validation.",
                code="ONNX_MODEL_INVALID",
                remediation_hint="Verify the model was exported correctly (e.g. via torch.onnx.export).",
            ) from exc

        return model

    def _resolve_layer_name(
        self, node: "onnx.NodeProto", index: int, used_names: Dict[str, int]
    ) -> str:
        """Resolve a unique, non-empty layer name for ``node``.

        ONNX node names are optional and frequently blank; a synthetic,
        deterministic name is generated when needed, and disambiguated
        if a collision would otherwise occur.

        Args:
            node: The ONNX node to name.
            index: The node's position in ``graph.node``, used to
                generate a synthetic name.
            used_names: A mapping tracking how many times each base
                name has been generated so far, mutated in place.

        Returns:
            A unique layer name.
        """
        base_name = node.name if node.name else f"{node.op_type}_{index}"
        occurrence = used_names.get(base_name, 0)
        used_names[base_name] = occurrence + 1
        return base_name if occurrence == 0 else f"{base_name}_{occurrence}"

    def _to_imr_tensor(self, array: np.ndarray) -> IMRTensor:
        """Convert a numpy array (from an ONNX initializer) into an ``IMRTensor``.

        Args:
            array: The initializer's decoded value.

        Returns:
            The equivalent ``IMRTensor``.
        """
        return IMRTensor(
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            data=array.tobytes(),
        )

    def _infer_layer_precision(self, parameters: Dict[str, IMRTensor]) -> Precision:
        """Infer a layer's ``Precision`` from its parameter tensor dtypes.

        Args:
            parameters: The layer's extracted parameter tensors.

        Returns:
            The ``Precision`` matching the first recognized parameter
            dtype, or ``Precision.FP32`` if ``parameters`` is empty or
            no parameter dtype is recognized (the common case for an
            un-quantized ONNX export, and for parameter-free layers
            such as activations).
        """
        for tensor in parameters.values():
            precision = _DTYPE_TO_PRECISION.get(tensor.dtype)
            if precision is not None:
                return precision
        return Precision.FP32

    def _extract_input_shapes(
        self, graph: "onnx.GraphProto", initializer_arrays: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[int, ...]]:
        """Extract the model's declared input shapes, excluding initializers.

        Args:
            graph: The ONNX graph.
            initializer_arrays: Every initializer, by name; ``graph.input``
                redundantly lists initializers under some ONNX opset
                versions, and these are not true model inputs.

        Returns:
            A mapping of true input tensor name to shape. Symbolic
            (dynamic) dimensions, e.g. a ``"batch"`` dimension param,
            are represented as ``-1``.
        """
        shapes: Dict[str, Tuple[int, ...]] = {}
        for value_info in graph.input:
            if value_info.name in initializer_arrays:
                continue
            dims = value_info.type.tensor_type.shape.dim
            shapes[value_info.name] = tuple(
                dim.dim_value if dim.HasField("dim_value") else -1 for dim in dims
            )
        return shapes
