"""``.pth``/``.pt`` model loader: the ``IFrameworkAdapter`` implementation
for PyTorch.

Parses a PyTorch ``nn.Module`` (eager checkpoint or ``TorchScript``
module) into the framework-independent ``IMR``, per
``03_API_Specification.md`` §15. This is the only module in ``uaqe``
permitted to import the ``torch`` third-party library
(``07_Coding_Standards.md`` §14 rule 4).

Locked contract: ``03_API_Specification.md`` §15.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from uaqe.common.exceptions import ModelLoadError
from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.interfaces.i_framework_adapter import IFrameworkAdapter
from uaqe.common.types import Precision

_SUPPORTED_EXTENSIONS = frozenset({".pth", ".pt"})

# torch dtype name (as printed by ``str(tensor.dtype)``, minus the
# ``torch.`` prefix) -> Precision. Any dtype not listed here (e.g.
# int32/int64/bool index or mask buffers) does not by itself determine
# a layer's Precision; see _infer_layer_precision below.
_DTYPE_TO_PRECISION: Dict[str, Precision] = {
    "float32": Precision.FP32,
    "float": Precision.FP32,
    "float16": Precision.FP16,
    "half": Precision.FP16,
    "bfloat16": Precision.FP16,
    "int8": Precision.INT8,
    "quint8": Precision.INT8,
    "qint8": Precision.INT8,
    "qint32": Precision.INT8,
}

# Well-known hyperparameter attribute names surfaced by common
# ``torch.nn`` layer types (Conv*d, Linear, BatchNorm*d, pooling,
# RNN/LSTM/GRU, etc.). Reading only these fixed names (rather than every
# public attribute) keeps ``attributes`` limited to genuine
# operator configuration.
_KNOWN_ATTRIBUTE_NAMES: Tuple[str, ...] = (
    "in_channels",
    "out_channels",
    "in_features",
    "out_features",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "bias",
    "eps",
    "momentum",
    "num_features",
    "affine",
    "num_groups",
    "hidden_size",
    "input_size",
    "num_layers",
    "batch_first",
    "p",
    "inplace",
)


class TorchAdapter(IFrameworkAdapter):
    """Loads ``.pth``/``.pt`` files into the framework-independent ``IMR``."""

    def load(self, path: str) -> IMR:
        """Load the PyTorch model at ``path`` and return its ``IMR``.

        Args:
            path: Filesystem path to the ``.pth``/``.pt`` file.

        Returns:
            The loaded model as an ``IMR``.

        Raises:
            ModelLoadError: If ``path`` does not exist, is corrupt, is
                a bare ``state_dict`` with no attached architecture, or
                otherwise cannot be resolved into an ``nn.Module``.
        """
        model = self._load_module(path)
        model.eval()

        layers: List[IMRLayer] = []
        previous_output: List[str] = []
        total_parameters = 0
        for name, module in model.named_modules():
            if name == "" or next(module.children(), None) is not None:
                # Skip the root module and any container/composite
                # module; only leaf modules (the actual operators) are
                # represented as IMR layers.
                continue

            parameters = self._extract_parameters(module)
            total_parameters += sum(int(np.prod(t.shape)) if t.shape else 1 for t in parameters.values())

            layer_name = name
            output_name = f"{layer_name}::output"
            layers.append(
                IMRLayer(
                    name=layer_name,
                    op_type=type(module).__name__,
                    inputs=list(previous_output),
                    outputs=[output_name],
                    parameters=parameters,
                    attributes=self._extract_attributes(module),
                    precision=self._infer_layer_precision(parameters),
                )
            )
            previous_output = [output_name]

        metadata = IMRMetadata(
            source_framework="pytorch",
            source_format=self._extension_of(path),
            original_input_shapes=self._extract_input_shapes(model),
            op_count=len(layers),
            total_parameters=total_parameters,
        )
        return IMR(layers=layers, metadata=metadata)

    def supports(self, extension: str) -> bool:
        """Report whether this adapter can load files with ``extension``.

        Args:
            extension: A file extension, including the leading dot.

        Returns:
            ``True`` if ``extension`` (case-insensitive) is ``.pth`` or
            ``.pt``.
        """
        return extension.lower() in _SUPPORTED_EXTENSIONS

    def _load_module(self, path: str) -> "torch.nn.Module":
        """Load ``path`` into a ``torch.nn.Module``.

        ``TorchScript`` archives (``torch.jit.save`` output) are tried
        first since ``torch.jit.load`` always yields a module that
        exposes both structure (``named_modules``) and weights
        (``named_parameters``); eagerly pickled checkpoints
        (``torch.save`` of a full ``nn.Module``) are tried next.

        Args:
            path: Filesystem path to the ``.pth``/``.pt`` file.

        Returns:
            The loaded, structure-bearing module.

        Raises:
            ModelLoadError: If ``path`` does not exist, is not a valid
                PyTorch archive, or resolves to a bare ``state_dict``
                (a plain tensor mapping with no attached module
                structure) rather than a full model.
        """
        try:
            return torch.jit.load(path, map_location="cpu")
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"PyTorch model file not found: {path!r}.",
                code="TORCH_FILE_NOT_FOUND",
                remediation_hint="Verify the model path is correct.",
            ) from exc
        except Exception:  # noqa: BLE001 - fall through to eager load below
            pass

        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"PyTorch model file not found: {path!r}.",
                code="TORCH_FILE_NOT_FOUND",
                remediation_hint="Verify the model path is correct.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - torch raises varied unpickling errors
            raise ModelLoadError(
                f"Cannot parse {path!r} as a PyTorch model.",
                code="TORCH_PARSE_FAILED",
                remediation_hint="Verify the file is a valid TorchScript or pickled nn.Module archive.",
            ) from exc

        if not isinstance(obj, torch.nn.Module):
            raise ModelLoadError(
                f"{path!r} contains a bare state_dict, not a full model.",
                code="TORCH_STATE_DICT_ONLY",
                remediation_hint=(
                    "Save the complete model (e.g. via torch.jit.script/trace, "
                    "or torch.save(model)) rather than only its state_dict."
                ),
            )
        return obj

    def _extract_parameters(self, module: "torch.nn.Module") -> Dict[str, IMRTensor]:
        """Extract a leaf module's own parameters and buffers.

        Args:
            module: A leaf ``nn.Module`` (no child modules).

        Returns:
            A mapping of parameter/buffer name to ``IMRTensor``,
            restricted to tensors owned directly by ``module`` (not
            recursed into children, since ``module`` has none).
        """
        parameters: Dict[str, IMRTensor] = {}
        for param_name, tensor in module.named_parameters(recurse=False):
            parameters[param_name] = self._to_imr_tensor(tensor)
        for buffer_name, tensor in module.named_buffers(recurse=False):
            parameters[buffer_name] = self._to_imr_tensor(tensor)
        return parameters

    def _to_imr_tensor(self, tensor: "torch.Tensor") -> IMRTensor:
        """Convert a ``torch.Tensor`` (parameter or buffer) into an ``IMRTensor``.

        Args:
            tensor: The parameter or buffer tensor to convert.

        Returns:
            The equivalent ``IMRTensor``, detached and moved to CPU
            before its bytes are read.
        """
        detached = tensor.detach().cpu().contiguous()
        dtype_name = str(detached.dtype).replace("torch.", "")
        return IMRTensor(
            shape=tuple(int(dim) for dim in detached.shape),
            dtype=dtype_name,
            data=detached.numpy().tobytes() if detached.dtype != torch.bfloat16 else detached.view(torch.int16).numpy().tobytes(),
        )

    def _extract_attributes(self, module: "torch.nn.Module") -> Dict[str, Any]:
        """Extract a leaf module's operator-specific hyperparameters.

        Args:
            module: A leaf ``nn.Module``.

        Returns:
            A mapping of hyperparameter name to value, limited to the
            fixed set of well-known ``torch.nn`` attribute names in
            ``_KNOWN_ATTRIBUTE_NAMES`` that are actually present on
            ``module``.
        """
        attributes: Dict[str, Any] = {}
        for attribute_name in _KNOWN_ATTRIBUTE_NAMES:
            if hasattr(module, attribute_name):
                value = getattr(module, attribute_name)
                if isinstance(value, torch.Tensor):
                    continue
                attributes[attribute_name] = value
        return attributes

    def _infer_layer_precision(self, parameters: Dict[str, IMRTensor]) -> Precision:
        """Infer a layer's ``Precision`` from its parameter tensor dtypes.

        Args:
            parameters: The layer's extracted parameter/buffer tensors.

        Returns:
            The ``Precision`` matching the first recognized parameter
            dtype, or ``Precision.FP32`` if ``parameters`` is empty or
            no parameter dtype is recognized (the common case for an
            un-quantized eager/TorchScript model, and for
            parameter-free layers such as activations).
        """
        for tensor in parameters.values():
            precision = _DTYPE_TO_PRECISION.get(tensor.dtype)
            if precision is not None:
                return precision
        return Precision.FP32

    def _extract_input_shapes(self, model: "torch.nn.Module") -> Dict[str, Tuple[int, ...]]:
        """Extract the model's declared input shapes, when available.

        Args:
            model: The loaded module.

        Returns:
            A mapping of input name to shape, derived from the
            ``TorchScript`` graph's declared input types when ``model``
            is a scripted/traced module. Returns an empty mapping for
            eager modules, which carry no serialized input-shape
            metadata of their own.
        """
        shapes: Dict[str, Tuple[int, ...]] = {}
        graph = getattr(model, "graph", None)
        if graph is None:
            return shapes
        for graph_input in list(graph.inputs())[1:]:  # skip the implicit `self`
            tensor_type = graph_input.type()
            sizes = getattr(tensor_type, "sizes", None)
            dims = sizes() if callable(sizes) else None
            if dims is not None:
                shapes[graph_input.debugName()] = tuple(dim if dim is not None else -1 for dim in dims)
        return shapes

    def _extension_of(self, path: str) -> str:
        """Return the lowercase file extension (including the dot) of ``path``.

        Args:
            path: Filesystem path to the source model file.

        Returns:
            The lowercase extension, e.g. ``".pt"``.
        """
        index = path.rfind(".")
        return path[index:].lower() if index != -1 else ""
