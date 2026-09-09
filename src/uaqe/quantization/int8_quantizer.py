"""Uniform INT8 ``IQuantizationStrategy`` implementation.

``Int8Quantizer`` quantizes every ``float32`` layer parameter in an
``IMR`` to signed 8-bit integers using symmetric per-tensor scaling,
via the shared arithmetic in
:class:`~uaqe.quantization.layer_quantizer.LayerQuantizer`. It is
registered with the plugin mechanism the same way any third-party
``IQuantizationStrategy`` would be
(``10_Module_Development_Guide.md`` §18), so
:class:`~uaqe.quantization.quantization_planner.QuantizationPlanner`
depends on it only through the ``IQuantizationStrategy`` port, never by
importing this class directly outside of composition-root wiring.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

from uaqe.common.exceptions import QuantizationError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.interfaces.i_quantization_strategy import IQuantizationStrategy
from uaqe.common.types import Precision
from uaqe.common.value_objects import QuantizationConfig
from uaqe.quantization.layer_quantizer import LayerQuantizer

#: This strategy's unique registration name, returned by :meth:`Int8Quantizer.name`.
STRATEGY_NAME = "int8"

#: The signed integer bit width this strategy quantizes to.
_BIT_WIDTH = 8


class Int8Quantizer(IQuantizationStrategy):
    """Quantizes an ``IMR`` to uniform INT8 precision.

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
        """Quantize every layer of ``imr`` to INT8.

        A layer named in ``plan.per_layer_overrides`` with
        ``Precision.FP32`` is left unquantized (an explicit escape
        hatch for layers the caller wants to keep at full precision even
        under an otherwise-uniform INT8 pass); any other per-layer
        override value is a contract violation for this
        single-precision strategy.

        Args:
            imr: The model to quantize.
            plan: The quantization configuration. Only
                ``per_layer_overrides`` entries of ``Precision.FP32`` or
                ``Precision.INT8`` are honored; ``default_precision`` is
                not consulted, since this strategy's entire purpose is
                to quantize to INT8 (per-layer routing across multiple
                precisions is
                :class:`~uaqe.quantization.mixed_precision_quantizer.
                MixedPrecisionQuantizer`'s responsibility, not this
                strategy's).

        Returns:
            A new ``IMR`` with every eligible layer quantized to INT8.
            The input ``imr`` is not modified.

        Raises:
            QuantizationError: If a ``per_layer_overrides`` entry
                requests a precision other than ``Precision.FP32`` or
                ``Precision.INT8``, or if a layer's parameters cannot be
                quantized (see
                :meth:`~uaqe.quantization.layer_quantizer.
                LayerQuantizer.quantize_layer`).
        """
        new_layers = []
        quantized_count = 0
        for layer in imr.layers:
            target = plan.per_layer_overrides.get(layer.name, Precision.INT8)
            if target == Precision.FP32:
                new_layers.append(layer)
                continue
            if target != Precision.INT8:
                raise QuantizationError(
                    f"Int8Quantizer cannot honor per-layer override "
                    f"{target.value} for layer {layer.name!r}; only "
                    f"Precision.FP32 (pass-through) or Precision.INT8 are "
                    f"valid overrides for this single-precision strategy.",
                    code="QUANT_INVALID_OVERRIDE",
                )
            new_layer, _ = self._layer_quantizer.quantize_layer(
                layer, _BIT_WIDTH, Precision.INT8
            )
            new_layers.append(new_layer)
            quantized_count += 1

        self._logger.info(
            "Applied INT8 quantization.",
            strategy=self.name(),
            quantized_layer_count=quantized_count,
            total_layer_count=len(imr.layers),
        )
        return dataclasses.replace(imr, layers=new_layers)

    def name(self) -> str:
        """Return this strategy's unique registration name."""
        return STRATEGY_NAME
