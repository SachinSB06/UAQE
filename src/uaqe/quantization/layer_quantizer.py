"""Stateless per-tensor and per-layer numeric quantization math shared by
every concrete ``IQuantizationStrategy`` in ``uaqe.quantization``.

``LayerQuantizer`` owns the one piece of arithmetic every precision
strategy in this package needs (uniform-affine scale/zero-point
computation, quantize/dequantize round-tripping, and INT4 nibble
packing) so that :class:`~uaqe.quantization.int8_quantizer.Int8Quantizer`,
:class:`~uaqe.quantization.int4_quantizer.Int4Quantizer`, and
:class:`~uaqe.quantization.mixed_precision_quantizer.MixedPrecisionQuantizer`
share exactly one implementation of that arithmetic rather than three
subtly different ones. It has no knowledge of *which* layers should be
quantized to *which* precision — that decision belongs to
``uaqe.quantization.precision_recommender`` and
``uaqe.quantization.quantization_planner``; this module only knows how
to quantize one already-selected tensor or layer once asked.

Only ``float32``-backed ``IMRTensor`` instances (``IMRTensor.dtype ==
"float32"``) are quantizable; every other dtype is passed through
unchanged by :meth:`LayerQuantizer.quantize_layer`, since a tensor that
is not floating-point (e.g. an already-quantized parameter re-entering
the pipeline) has no meaningful affine range to compute.
"""

from __future__ import annotations

import array
import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, Tuple

from uaqe.common.exceptions import QuantizationError
from uaqe.common.imr import IMRLayer, IMRTensor
from uaqe.common.types import Precision

#: ``IMRTensor.dtype`` value this module knows how to read/write via the
#: stdlib ``array`` module. Any other floating-point encoding is treated
#: as unquantizable by this package.
_FLOAT32_DTYPE = "float32"

#: ``array`` module type code for 32-bit IEEE-754 floats.
_FLOAT32_TYPECODE = "f"

#: ``array`` module type code used for signed 8-bit quantized storage.
_INT8_TYPECODE = "b"

#: Dtype string this module writes onto an ``IMRTensor`` produced by
#: :meth:`LayerQuantizer.quantize_tensor` at ``bit_width=8``.
INT8_DTYPE = "int8"

#: Dtype string this module writes onto an ``IMRTensor`` produced by
#: :meth:`LayerQuantizer.quantize_tensor` at ``bit_width=4``. Two INT4
#: elements are packed per byte (low nibble first); see
#: :meth:`LayerQuantizer._pack_int4`/:meth:`LayerQuantizer._unpack_int4`.
INT4_DTYPE = "int4_packed"

#: Supported quantized bit widths, keyed to their signed integer range.
_BIT_WIDTH_RANGES: Dict[int, Tuple[int, int]] = {
    8: (-128, 127),
    4: (-8, 7),
}


@dataclass(frozen=True)
class QuantizationParams:
    """The affine mapping between a real-valued tensor and its quantized
    integer representation, ``real_value = scale * (q - zero_point)``.

    Attributes:
        scale: The quantization step size, in real-valued units per
            integer step. Always positive.
        zero_point: The integer value representing real value ``0.0``.
            ``0`` for the symmetric quantization this package uses
            exclusively (per-tensor symmetric, no asymmetric/affine
            zero-point shift), retained as an explicit field so a future
            asymmetric strategy can populate it without a signature
            change.
        bit_width: The signed integer bit width these parameters were
            computed for (``8`` or ``4``).
        symmetric: Whether the mapping is zero-centered (``zero_point ==
            0``). Always ``True`` for the strategies shipped in this
            package.
    """

    scale: float
    zero_point: int
    bit_width: int
    symmetric: bool = True


class LayerQuantizer:
    """Stateless quantize/dequantize arithmetic for one tensor or layer
    at a time.

    Holds no instance state and no configuration beyond what is passed
    to each call, so a single instance may be safely shared (and reused
    across threads) by every strategy in this package.
    """

    def compute_params(
        self, values: array.array, bit_width: int
    ) -> QuantizationParams:
        """Compute the symmetric affine mapping for a set of real values.

        Args:
            values: The real-valued (``float32``) samples to quantize,
                typically every element of one ``IMRTensor``.
            bit_width: The signed integer bit width to quantize to
                (``8`` or ``4``).

        Returns:
            The computed :class:`QuantizationParams`.

        Raises:
            QuantizationError: If ``bit_width`` is not supported, or if
                ``values`` is empty.
        """
        if bit_width not in _BIT_WIDTH_RANGES:
            raise QuantizationError(
                f"Unsupported quantization bit width: {bit_width}.",
                code="QUANT_UNSUPPORTED_BIT_WIDTH",
                remediation_hint="Use bit_width=8 or bit_width=4.",
            )
        if len(values) == 0:
            raise QuantizationError(
                "Cannot compute quantization parameters for an empty tensor.",
                code="QUANT_EMPTY_TENSOR",
            )

        max_abs = max(abs(min(values)), abs(max(values)))
        _, qmax = _BIT_WIDTH_RANGES[bit_width]
        if max_abs == 0.0:
            # A degenerate all-zero tensor still needs a positive scale
            # so that dequantization does not divide by (or multiply by)
            # zero downstream.
            scale = 1.0
        else:
            scale = max_abs / float(qmax)
        return QuantizationParams(scale=scale, zero_point=0, bit_width=bit_width)

    def quantize_tensor(
        self, tensor: IMRTensor, bit_width: int
    ) -> Tuple[IMRTensor, QuantizationParams]:
        """Quantize one ``float32`` ``IMRTensor`` to the given bit width.

        Args:
            tensor: The tensor to quantize. Must have ``dtype ==
                "float32"``.
            bit_width: The signed integer bit width to quantize to
                (``8`` or ``4``).

        Returns:
            A tuple of the new, quantized ``IMRTensor`` (same ``shape``,
            new ``dtype``/``data``) and the :class:`QuantizationParams`
            used, which downstream consumers (e.g. an exporter) need to
            dequantize or requantize this tensor later.

        Raises:
            QuantizationError: If ``tensor.dtype`` is not ``"float32"``,
                if the tensor's byte length does not match its declared
                ``shape``, or if ``bit_width`` is unsupported.
        """
        if tensor.dtype != _FLOAT32_DTYPE:
            raise QuantizationError(
                f"Cannot quantize tensor of dtype {tensor.dtype!r}; "
                f"only {_FLOAT32_DTYPE!r} tensors are supported.",
                code="QUANT_UNSUPPORTED_DTYPE",
            )

        values = array.array(_FLOAT32_TYPECODE)
        try:
            values.frombytes(tensor.data)
        except ValueError as err:
            raise QuantizationError(
                "Tensor byte buffer length is not a whole number of "
                "float32 elements.",
                code="QUANT_MALFORMED_BUFFER",
            ) from err

        expected_elements = math.prod(tensor.shape) if tensor.shape else 1
        if len(values) != expected_elements:
            raise QuantizationError(
                f"Tensor declares shape {tensor.shape!r} "
                f"({expected_elements} elements) but its buffer holds "
                f"{len(values)} float32 elements.",
                code="QUANT_SHAPE_MISMATCH",
            )

        params = self.compute_params(values, bit_width)
        qmin, qmax = _BIT_WIDTH_RANGES[bit_width]
        quantized = [
            _clamp(round(value / params.scale), qmin, qmax) for value in values
        ]

        if bit_width == 8:
            packed = array.array(_INT8_TYPECODE, quantized).tobytes()
            new_dtype = INT8_DTYPE
        else:
            packed = self._pack_int4(quantized)
            new_dtype = INT4_DTYPE

        new_tensor = IMRTensor(shape=tensor.shape, dtype=new_dtype, data=packed)
        return new_tensor, params

    def dequantize_tensor(
        self, tensor: IMRTensor, params: QuantizationParams | None = None
    ) -> IMRTensor:
        """Reconstruct an approximate ``float32`` tensor from a
        quantized one.

        Used by evaluation/benchmarking stages (and by tests validating
        this package's round-trip error) rather than by the quantization
        strategies themselves, which only move forward from ``float32``
        to a quantized dtype.

        Args:
            tensor: A tensor previously produced by
                :meth:`quantize_tensor` (``dtype`` is
                :data:`INT8_DTYPE` or :data:`INT4_DTYPE` or ``"float16"``).
            params: The :class:`QuantizationParams` returned alongside
                ``tensor`` by the original :meth:`quantize_tensor` call.
                Can be ``None`` for FP16 tensors.

        Returns:
            A new ``float32`` ``IMRTensor`` approximating the original
            real-valued tensor.

        Raises:
            QuantizationError: If ``tensor.dtype`` is not one this
                module produced.
        """
        if tensor.dtype == "float16":
            import numpy as np
            f32_arr = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
            real_values = array.array(_FLOAT32_TYPECODE, f32_arr)
            return IMRTensor(
                shape=tensor.shape, dtype=_FLOAT32_DTYPE, data=real_values.tobytes()
            )

        if tensor.dtype == INT8_DTYPE:
            if params is None:
                raise QuantizationError("Missing quantization parameters for INT8 tensor.")
            quantized = array.array(_INT8_TYPECODE)
            quantized.frombytes(tensor.data)
        elif tensor.dtype == INT4_DTYPE:
            if params is None:
                raise QuantizationError("Missing quantization parameters for INT4 tensor.")
            element_count = math.prod(tensor.shape) if tensor.shape else 1
            quantized = self._unpack_int4(tensor.data, element_count)
        else:
            raise QuantizationError(
                f"Cannot dequantize tensor of dtype {tensor.dtype!r}; "
                f"expected {INT8_DTYPE!r}, {INT4_DTYPE!r}, or 'float16'.",
                code="QUANT_UNSUPPORTED_DTYPE",
            )

        real_values = array.array(
            _FLOAT32_TYPECODE, (value * params.scale for value in quantized)
        )
        return IMRTensor(
            shape=tensor.shape, dtype=_FLOAT32_DTYPE, data=real_values.tobytes()
        )

    def quantize_layer(
        self, layer: IMRLayer, bit_width: int, precision: Precision
    ) -> Tuple[IMRLayer, Dict[str, QuantizationParams]]:
        """Quantize every ``float32`` parameter tensor on one layer.

        Non-``float32`` parameters (if any) pass through unchanged; a
        layer with no parameters at all (e.g. an activation-only op such
        as ``ReLU``) simply has its ``precision`` field updated.

        Args:
            layer: The layer to quantize.
            bit_width: The signed integer bit width to quantize
                parameters to (``8`` or ``4``).
            precision: The ``Precision`` value to record on the
                returned layer (``Precision.INT8`` or ``Precision.INT4``
                for the strategies in this package).

        Returns:
            A tuple of the new ``IMRLayer`` (new ``parameters`` mapping,
            new ``precision``) and the :class:`QuantizationParams` used
            for each quantized parameter, keyed by parameter name.

        Raises:
            QuantizationError: If any ``float32`` parameter cannot be
                quantized (see :meth:`quantize_tensor`).
        """
        new_parameters = {}
        new_attributes = dict(layer.attributes)
        params_by_name: Dict[str, QuantizationParams] = {}
        for param_name, tensor in layer.parameters.items():
            if tensor.dtype != _FLOAT32_DTYPE:
                new_parameters[param_name] = tensor
                continue
            quantized_tensor, params = self.quantize_tensor(tensor, bit_width)
            new_parameters[param_name] = quantized_tensor
            params_by_name[param_name] = params
            new_attributes[f"{param_name}_scale"] = params.scale
            new_attributes[f"{param_name}_zero_point"] = params.zero_point

        new_layer = dataclasses.replace(
            layer, parameters=new_parameters, attributes=new_attributes, precision=precision
        )
        return new_layer, params_by_name

    @staticmethod
    def _pack_int4(values) -> bytes:
        """Pack signed 4-bit integers two-per-byte, low nibble first."""
        packed = bytearray((len(values) + 1) // 2)
        for index, value in enumerate(values):
            nibble = value & 0x0F
            byte_index = index // 2
            if index % 2 == 0:
                packed[byte_index] = nibble
            else:
                packed[byte_index] |= nibble << 4
        return bytes(packed)

    @staticmethod
    def _unpack_int4(data: bytes, element_count: int) -> array.array:
        """Unpack signed 4-bit integers previously packed by
        :meth:`_pack_int4`, sign-extending each nibble back to ``int``.
        """
        values = array.array("i")
        for index in range(element_count):
            byte = data[index // 2]
            nibble = byte & 0x0F if index % 2 == 0 else (byte >> 4) & 0x0F
            if nibble >= 8:
                nibble -= 16
            values.append(nibble)
        return values


def _clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp ``value`` to the inclusive ``[minimum, maximum]`` range."""
    return max(minimum, min(maximum, value))
