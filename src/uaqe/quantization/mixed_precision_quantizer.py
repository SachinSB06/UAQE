"""Mixed-precision ``IQuantizationStrategy`` implementation.

``MixedPrecisionQuantizer`` routes each layer of an ``IMR`` to the
quantization treatment appropriate for that layer's *own* resolved
target precision, rather than applying one precision uniformly like
:class:`~uaqe.quantization.int8_quantizer.Int8Quantizer` or
:class:`~uaqe.quantization.int4_quantizer.Int4Quantizer` do. It is the
strategy selected whenever a run's final per-layer precision mapping
(as built by :class:`~uaqe.quantization.precision_recommender.
PrecisionRecommender`) is not uniform — i.e. whenever
``QuantizationPlan.per_layer_precision`` contains more than one distinct
``Precision`` value.

This class reuses the exact same
:class:`~uaqe.quantization.layer_quantizer.LayerQuantizer` arithmetic
``Int8Quantizer``/``Int4Quantizer`` use, so a layer quantized to INT8
via this strategy is byte-for-byte identical to one quantized to INT8
via ``Int8Quantizer`` directly — the three strategies differ only in
*how many* precisions they route across a model, never in the
arithmetic applied once a precision is chosen for a given layer.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from uaqe.common.exceptions import QuantizationError
from uaqe.common.imr import IMR, IMRLayer, IMRTensor
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.interfaces.i_quantization_strategy import IQuantizationStrategy
from uaqe.common.types import Precision
from uaqe.common.value_objects import QuantizationConfig
from uaqe.quantization.layer_quantizer import LayerQuantizer

#: This strategy's unique registration name, returned by
#: :meth:`MixedPrecisionQuantizer.name`.
STRATEGY_NAME = "mixed_precision"

#: The signed integer bit width backing each executable ``Precision``
#: this strategy can route a layer to.
_BIT_WIDTH_BY_PRECISION = {
    Precision.INT8: 8,
    Precision.INT4: 4,
}

#: Precisions this strategy passes a layer through unquantized, since
#: they require no integer conversion.
_PASSTHROUGH_PRECISIONS = frozenset({Precision.FP32})

#: The precision a layer falls back to when the model-wide
#: ``default_precision`` is itself ``Precision.MIXED`` and the layer has
#: no more specific ``per_layer_overrides`` entry — INT8 is the
#: conservative middle ground between full precision and INT4.
_MIXED_FALLBACK_PRECISION = Precision.INT8


class MixedPrecisionQuantizer(IQuantizationStrategy):
    """Routes each layer of an ``IMR`` to its own target precision.

    Attributes:
        _logger: Structured logging sink.
        _layer_quantizer: Shared per-tensor/per-layer quantization
            arithmetic.
    """

    def __init__(
        self, logger: ILogger, layer_quantizer: Optional[LayerQuantizer] = None
    ) -> None:
        """Initialize the strategy.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            layer_quantizer: Overrides the default
                :class:`~uaqe.quantization.layer_quantizer.LayerQuantizer`;
                ``LayerQuantizer`` is stateless, so the default instance
                is sufficient for ordinary use and this parameter exists
                mainly to ease test-double substitution.
        """
        self._logger = logger
        self._layer_quantizer = layer_quantizer or LayerQuantizer()

    def apply(self, imr: IMR, plan: QuantizationConfig) -> IMR:
        """Quantize each layer of ``imr`` to its own resolved precision.

        Each layer's target precision is
        ``plan.per_layer_overrides.get(layer.name, plan.default_precision)``.
        If that resolves to ``Precision.MIXED`` (i.e. the model-wide
        default itself is ``MIXED`` and this layer has no more specific
        override), the layer falls back to
        :data:`_MIXED_FALLBACK_PRECISION` and a warning is logged, since
        ``MIXED`` is a routing signal for this strategy, not itself an
        executable per-layer precision.

        Args:
            imr: The model to quantize.
            plan: The quantization configuration supplying
                ``default_precision`` and ``per_layer_overrides``.

        Returns:
            A new ``IMR`` with every layer quantized (or passed through,
            for ``FP32``/``FP16`` layers) per its resolved precision.
            The input ``imr`` is not modified.

        Raises:
            QuantizationError: If a layer's parameters cannot be
                quantized (see
                :meth:`~uaqe.quantization.layer_quantizer.
                LayerQuantizer.quantize_layer`).
        """
        new_layers = []
        precision_counts = {}
        for layer in imr.layers:
            target = plan.per_layer_overrides.get(layer.name, plan.default_precision)
            if target == Precision.MIXED:
                self._logger.warning(
                    f"Layer {layer.name!r} resolved to Precision.MIXED with "
                    f"no more specific override; falling back to "
                    f"{_MIXED_FALLBACK_PRECISION.value}.",
                    strategy=self.name(),
                )
                target = _MIXED_FALLBACK_PRECISION

            new_layers.append(self._quantize_one(layer, target))
            precision_counts[target] = precision_counts.get(target, 0) + 1

        self._logger.info(
            "Applied mixed-precision quantization.",
            strategy=self.name(),
            precision_distribution={
                precision.value: count for precision, count in precision_counts.items()
            },
        )
        return dataclasses.replace(imr, layers=new_layers)

    def name(self) -> str:
        """Return this strategy's unique registration name."""
        return STRATEGY_NAME

    def _quantize_one(self, layer: IMRLayer, target: Precision) -> IMRLayer:
        """Quantize (or pass through) a single layer per ``target``.

        Args:
            layer: The layer to process.
            target: The layer's resolved, already-``MIXED``-resolved
                target precision.

        Returns:
            The new (or, for a passthrough precision, unchanged) layer.

        Raises:
            QuantizationError: If ``target`` is not a recognized
                passthrough or quantizable precision, or if
                quantization of the layer's parameters fails.
        """
        if target in _PASSTHROUGH_PRECISIONS:
            return dataclasses.replace(layer, precision=target)

        if target == Precision.FP16:
            import numpy as np
            new_parameters = {}
            for param_name, tensor in layer.parameters.items():
                if tensor.dtype == "float32":
                    f32_arr = np.frombuffer(tensor.data, dtype=np.float32)
                    f16_arr = f32_arr.astype(np.float16)
                    new_tensor = IMRTensor(shape=tensor.shape, dtype="float16", data=f16_arr.tobytes())
                    new_parameters[param_name] = new_tensor
                else:
                    new_parameters[param_name] = tensor
            return dataclasses.replace(layer, parameters=new_parameters, precision=Precision.FP16)

        bit_width = _BIT_WIDTH_BY_PRECISION.get(target)
        if bit_width is None:
            raise QuantizationError(
                f"MixedPrecisionQuantizer cannot route layer {layer.name!r} "
                f"to unrecognized precision {target.value}.",
                code="QUANT_UNSUPPORTED_PRECISION",
            )

        new_layer, _ = self._layer_quantizer.quantize_layer(layer, bit_width, target)
        return new_layer
