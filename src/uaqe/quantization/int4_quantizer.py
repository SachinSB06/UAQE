"""Uniform INT4 ``IQuantizationStrategy`` implementation.

``Int4Quantizer`` quantizes every ``float32`` layer parameter in an
``IMR`` to signed 4-bit integers (two elements packed per byte) using
symmetric per-tensor scaling, via the shared arithmetic in
:class:`~uaqe.quantization.layer_quantizer.LayerQuantizer`. INT4 trades
additional quantization error for roughly half the parameter storage of
INT8, making it most appropriate for flash/RAM-constrained
``HardwareClass.EMBEDDED`` targets whose ``HardwareProfile.
supported_precisions`` include ``Precision.INT4`` — a constraint this
strategy's caller (:class:`~uaqe.quantization.quantization_planner.
QuantizationPlanner`, via
:class:`~uaqe.quantization.precision_recommender.PrecisionRecommender`)
enforces before this strategy is ever invoked, not this strategy itself.
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

#: This strategy's unique registration name, returned by :meth:`Int4Quantizer.name`.
STRATEGY_NAME = "int4"

#: The signed integer bit width this strategy quantizes to.
_BIT_WIDTH = 4


class Int4Quantizer(IQuantizationStrategy):
    """Quantizes an ``IMR`` to uniform INT4 precision.

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
        """Quantize every layer of ``imr`` to INT4.

        A layer named in ``plan.per_layer_overrides`` with
        ``Precision.FP32`` is left unquantized (an explicit escape
        hatch for layers the caller wants to keep at full precision even
        under an otherwise-uniform INT4 pass); any other per-layer
        override value is a contract violation for this
        single-precision strategy.

        Args:
            imr: The model to quantize.
            plan: The quantization configuration. Only
                ``per_layer_overrides`` entries of ``Precision.FP32`` or
                ``Precision.INT4`` are honored; ``default_precision`` is
                not consulted, for the same reason documented on
                :meth:`~uaqe.quantization.int8_quantizer.Int8Quantizer.
                apply`.

        Returns:
            A new ``IMR`` with every eligible layer quantized to INT4.
            The input ``imr`` is not modified.

        Raises:
            QuantizationError: If a ``per_layer_overrides`` entry
                requests a precision other than ``Precision.FP32`` or
                ``Precision.INT4``, or if a layer's parameters cannot be
                quantized (see
                :meth:`~uaqe.quantization.layer_quantizer.
                LayerQuantizer.quantize_layer`).
        """
        new_layers = []
        quantized_count = 0
        for layer in imr.layers:
            target = plan.per_layer_overrides.get(layer.name, Precision.INT4)
            if target == Precision.FP32:
                new_layers.append(layer)
                continue
            if target != Precision.INT4:
                raise QuantizationError(
                    f"Int4Quantizer cannot honor per-layer override "
                    f"{target.value} for layer {layer.name!r}; only "
                    f"Precision.FP32 (pass-through) or Precision.INT4 are "
                    f"valid overrides for this single-precision strategy.",
                    code="QUANT_INVALID_OVERRIDE",
                )
            new_layer, _ = self._layer_quantizer.quantize_layer(
                layer, _BIT_WIDTH, Precision.INT4
            )
            new_layers.append(new_layer)
            quantized_count += 1

        self._logger.info(
            "Applied INT4 quantization.",
            strategy=self.name(),
            quantized_layer_count=quantized_count,
            total_layer_count=len(imr.layers),
        )
        return dataclasses.replace(imr, layers=new_layers)

    def name(self) -> str:
        """Return this strategy's unique registration name."""
        return STRATEGY_NAME
