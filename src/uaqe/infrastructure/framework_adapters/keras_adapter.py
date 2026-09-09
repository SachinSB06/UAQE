"""``.h5``/``.keras`` model loader: the ``IFrameworkAdapter``
implementation for Keras.

Parses a saved Keras model (HDF5 ``.h5`` or the native ``.keras``
zip archive) into the framework-independent ``IMR``, per
``03_API_Specification.md`` §15. This is the only module in ``uaqe``
permitted to import ``tensorflow.keras`` for loading these two saved-model
formats (``07_Coding_Standards.md`` §14 rule 4); ``tensorflow_adapter.py``
and ``tflite_adapter.py`` each depend on ``tensorflow`` independently
for their own, unrelated file formats.

Locked contract: ``03_API_Specification.md`` §15.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from uaqe.common.exceptions import ModelLoadError
from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.interfaces.i_framework_adapter import IFrameworkAdapter
from uaqe.common.types import Precision

_SUPPORTED_EXTENSIONS = frozenset({".h5", ".keras"})

# numpy dtype name -> Precision. Any dtype not listed here does not by
# itself determine a layer's Precision; see _infer_layer_precision
# below.
_DTYPE_TO_PRECISION: Dict[str, Precision] = {
    "float32": Precision.FP32,
    "float16": Precision.FP16,
    "int8": Precision.INT8,
    "uint8": Precision.INT8,
}

# get_config() keys that are themselves nested layer/config
# sub-structures rather than scalar hyperparameters; excluded from
# `attributes` since they duplicate information already captured by
# IMR's own layer/parameter graph and are not always JSON-plain.
_EXCLUDED_CONFIG_KEYS = frozenset({"layers", "layer", "cell", "cells", "build_input_shape"})


class KerasAdapter(IFrameworkAdapter):
    """Loads ``.h5``/``.keras`` files into the framework-independent ``IMR``."""

    def load(self, path: str) -> IMR:
        """Load the Keras model at ``path`` and return its ``IMR``.

        Args:
            path: Filesystem path to the ``.h5`` or ``.keras`` file.

        Returns:
            The loaded model as an ``IMR``.

        Raises:
            ModelLoadError: If ``path`` does not exist, is corrupt, or
                cannot be deserialized into a Keras model.
        """
        model = self._load_model(path)

        layers: List[IMRLayer] = []
        for layer in model.layers:
            parameters = {
                self._weight_name(layer, weight): self._to_imr_tensor(weight.numpy())
                for weight in layer.weights
            }
            layers.append(
                IMRLayer(
                    name=layer.name,
                    op_type=type(layer).__name__,
                    inputs=self._tensor_names(getattr(layer, "input", None)),
                    outputs=self._tensor_names(getattr(layer, "output", None)),
                    parameters=parameters,
                    attributes=self._extract_attributes(layer),
                    precision=self._infer_layer_precision(parameters),
                )
            )

        metadata = IMRMetadata(
            source_framework="keras",
            source_format=self._extension_of(path),
            original_input_shapes=self._extract_input_shapes(model),
            op_count=len(layers),
            total_parameters=int(model.count_params()),
        )
        return IMR(layers=layers, metadata=metadata)

    def supports(self, extension: str) -> bool:
        """Report whether this adapter can load files with ``extension``.

        Args:
            extension: A file extension, including the leading dot.

        Returns:
            ``True`` if ``extension`` (case-insensitive) is ``.h5`` or
            ``.keras``.
        """
        return extension.lower() in _SUPPORTED_EXTENSIONS

    def _load_model(self, path: str) -> "keras.Model":
        """Load and deserialize a Keras model from disk.

        Args:
            path: Filesystem path to the ``.h5`` or ``.keras`` file.

        Returns:
            The deserialized ``keras.Model``.

        Raises:
            ModelLoadError: If the file cannot be read or is not a
                valid Keras saved-model archive.
        """
        try:
            return keras.models.load_model(path, compile=False)
        except (FileNotFoundError, OSError, tf.errors.NotFoundError) as exc:
            raise ModelLoadError(
                f"Keras model file not found: {path!r}.",
                code="KERAS_FILE_NOT_FOUND",
                remediation_hint="Verify the model path is correct.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - keras raises varied deserialization errors
            raise ModelLoadError(
                f"Cannot parse {path!r} as a Keras model.",
                code="KERAS_PARSE_FAILED",
                remediation_hint="Verify the file is a valid .h5 or .keras saved model.",
            ) from exc

    def _weight_name(self, layer: "keras.layers.Layer", weight: "tf.Variable") -> str:
        """Resolve a stable, layer-scoped name for a weight variable.

        Args:
            layer: The owning layer.
            weight: The weight variable belonging to ``layer``.

        Returns:
            The weight's own variable name, or a synthetic
            ``"<layer_name>/weight_<index>"`` name if the variable
            carries no name of its own.
        """
        name = getattr(weight, "name", None)
        if name:
            return name
        return f"{layer.name}/weight_{layer.weights.index(weight)}"

    def _to_imr_tensor(self, array: np.ndarray) -> IMRTensor:
        """Convert a numpy array (from a layer weight) into an ``IMRTensor``.

        Args:
            array: The weight's current value.

        Returns:
            The equivalent ``IMRTensor``.
        """
        return IMRTensor(
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            data=array.tobytes(),
        )

    def _tensor_names(self, tensor_or_tensors: Any) -> List[str]:
        """Resolve the tensor name(s) of a layer's symbolic input or output.

        Args:
            tensor_or_tensors: The value of ``layer.input`` or
                ``layer.output``, which may be a single ``KerasTensor``,
                a list of them (multi-input/output layers), or ``None``
                (e.g. layers not yet connected in a functional graph).

        Returns:
            The resolved tensor name(s); an empty list if
            ``tensor_or_tensors`` is ``None``.
        """
        if tensor_or_tensors is None:
            return []
        if isinstance(tensor_or_tensors, (list, tuple)):
            return [getattr(tensor, "name", str(tensor)) for tensor in tensor_or_tensors]
        return [getattr(tensor_or_tensors, "name", str(tensor_or_tensors))]

    def _extract_attributes(self, layer: "keras.layers.Layer") -> Dict[str, Any]:
        """Extract a layer's operator-specific hyperparameters.

        Args:
            layer: The Keras layer to inspect.

        Returns:
            The layer's ``get_config()`` mapping, with nested
            layer/config sub-structures removed (see
            ``_EXCLUDED_CONFIG_KEYS``); empty if ``layer`` does not
            support configuration introspection.
        """
        try:
            config = layer.get_config()
        except NotImplementedError:
            return {}
        return {key: value for key, value in config.items() if key not in _EXCLUDED_CONFIG_KEYS}

    def _infer_layer_precision(self, parameters: Dict[str, IMRTensor]) -> Precision:
        """Infer a layer's ``Precision`` from its weight tensor dtypes.

        Args:
            parameters: The layer's extracted weight tensors.

        Returns:
            The ``Precision`` matching the first recognized weight
            dtype, or ``Precision.FP32`` if ``parameters`` is empty or
            no weight dtype is recognized (the common case for an
            un-quantized Keras model, and for weight-free layers such
            as activations).
        """
        for tensor in parameters.values():
            precision = _DTYPE_TO_PRECISION.get(tensor.dtype)
            if precision is not None:
                return precision
        return Precision.FP32

    def _extract_input_shapes(self, model: "keras.Model") -> Dict[str, Tuple[int, ...]]:
        """Extract the model's declared input shapes.

        Args:
            model: The loaded Keras model.

        Returns:
            A mapping of input tensor name to shape. Symbolic (e.g.
            batch) dimensions are represented as ``-1``.
        """
        shapes: Dict[str, Tuple[int, ...]] = {}
        for input_tensor in model.inputs:
            shapes[input_tensor.name] = tuple(
                dim if dim is not None else -1 for dim in input_tensor.shape
            )
        return shapes

    def _extension_of(self, path: str) -> str:
        """Return the lowercase file extension (including the dot) of ``path``.

        Args:
            path: Filesystem path to the source model file.

        Returns:
            The lowercase extension, e.g. ``".keras"``.
        """
        index = path.rfind(".")
        return path[index:].lower() if index != -1 else ""
