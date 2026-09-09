"""Estimates per-layer and total inference latency for an ``IMR`` on a
``HardwareProfile``.

This is not part of the locked ``05_Hardware_Profile_Spec.md`` /
``03_API_Specification.md`` surface — see the scope note in
:mod:`uaqe.hardware.hardware_manager`. It exists to give a caller an
order-of-magnitude latency figure before a real
``uaqe.domain.benchmark.Benchmarker`` run against physical or simulated
hardware is available; it is deliberately conservative about precision
and is never a substitute for ``Benchmarker.run_benchmark``'s measured
``BenchmarkResult``.

Estimation model: for each layer, an approximate multiply-accumulate
(MAC) count is derived from its parameter volume, then converted to a
duration via a hardware-class-specific throughput figure, scaled by a
precision-dependent slowdown factor. When ``HardwareProfile.clock_speed_hz``
is populated (only FPGA profiles in the v1 database — see
``05_Hardware_Profile_Spec.md`` §4), a cycles-per-MAC model is used
instead, which is more precise than the throughput fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from uaqe.common.imr import IMR, IMRLayer
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import HardwareClass, Precision

if TYPE_CHECKING:
    # Deferred to break the hardware_manager <-> latency_estimator
    # import cycle; see the matching note in compatibility_checker.py.
    from uaqe.hardware.hardware_manager import HardwareProfile

#: Approximate cycles consumed per MAC operation at INT8, for hardware
#: classes whose ``clock_speed_hz`` is populated. Only ``FPGA`` profiles
#: currently carry a non-null ``clock_speed_hz`` (``05_Hardware_Profile_
#: Spec.md`` §3-§4), so this table is consulted only in that case today,
#: but is kept general for when embedded ``clock_speed_hz`` values are
#: populated in a later schema revision (``05_Hardware_Profile_Spec.md``
#: §7.3).
CYCLES_PER_MAC_INT8: Dict[HardwareClass, float] = {
    HardwareClass.FPGA: 0.05,  # deeply pipelined/parallel MAC array
    HardwareClass.EMBEDDED: 2.0,  # scalar MCU core, no dedicated MAC array
    HardwareClass.RASPBERRY_PI: 0.25,  # NEON-accelerated CPU core
}

#: Fallback reference throughput, in MACs/second at INT8, used when
#: ``clock_speed_hz`` is ``None`` (the common case for v1 embedded
#: profiles). Order-of-magnitude figures for a representative device in
#: each class, not per-board measurements.
FALLBACK_MACS_PER_SECOND_INT8: Dict[HardwareClass, float] = {
    HardwareClass.FPGA: 5.0e10,
    HardwareClass.EMBEDDED: 2.0e7,
    HardwareClass.RASPBERRY_PI: 1.0e9,
}

#: Multiplicative slowdown relative to INT8 for other precisions —
#: lower-bit-width ops are cheaper per MAC, floating point is costlier.
PRECISION_SLOWDOWN: Dict[Precision, float] = {
    Precision.INT4: 0.6,
    Precision.INT8: 1.0,
    Precision.FP16: 2.0,
    Precision.FP32: 4.0,
    Precision.MIXED: 1.5,
}

#: Op types with no learned parameters, modeled as a small fixed
#: per-invocation cost rather than a MAC-derived one.
FIXED_COST_OP_TYPES = frozenset(
    {"ReLU", "Sigmoid", "Tanh", "Softmax", "BatchNorm", "Add", "Concat", "Reshape"}
)

#: The fixed per-invocation cost, in milliseconds, applied to
#: :data:`FIXED_COST_OP_TYPES`, before the precision slowdown factor.
FIXED_OP_COST_MS: float = 0.001


@dataclass
class LatencyEstimate:
    """The outcome of latency estimation for one ``IMR``/``HardwareProfile``
    pair.

    Attributes:
        total_latency_ms: The sum of every layer's estimated latency.
        per_layer_latency_ms: Each layer's estimated latency, in
            milliseconds, keyed by ``IMRLayer.name``.
        used_clock_model: Whether the cycles-per-MAC model
            (``clock_speed_hz``-based) was used, as opposed to the
            fallback reference-throughput model.
        assumptions: Human-readable notes on the estimation model used
            and its known limitations.
    """

    total_latency_ms: float
    per_layer_latency_ms: Dict[str, float] = field(default_factory=dict)
    used_clock_model: bool = False
    assumptions: List[str] = field(default_factory=list)


class LatencyEstimator:
    """Estimates inference latency for an ``IMR`` on a ``HardwareProfile``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            estimator operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``LatencyEstimator``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def estimate(self, imr: IMR, profile: HardwareProfile) -> LatencyEstimate:
        """Estimate per-layer and total inference latency for ``imr`` on
        ``profile``.

        Args:
            imr: The model to estimate.
            profile: The candidate deployment target.

        Returns:
            The resulting ``LatencyEstimate``.
        """
        assumptions: List[str] = [
            "Latency figures are an order-of-magnitude estimate derived "
            "from parameter-volume MAC proxies, not a measured benchmark; "
            "use uaqe.domain.benchmark.Benchmarker for a measured result."
        ]
        used_clock_model = profile.clock_speed_hz is not None
        if used_clock_model:
            assumptions.append(
                f"Used cycles-per-MAC model at clock_speed_hz="
                f"{profile.clock_speed_hz}."
            )
        else:
            assumptions.append(
                "clock_speed_hz is null for this profile; used the "
                "fallback reference-throughput model instead."
            )

        per_layer_latency_ms: Dict[str, float] = {}
        for layer in imr.layers:
            per_layer_latency_ms[layer.name] = self._estimate_layer_latency_ms(
                layer, profile, used_clock_model
            )

        total_latency_ms = sum(per_layer_latency_ms.values())

        if self.logger is not None:
            self.logger.info(
                "Estimated inference latency.",
                profile_id=profile.profile_id,
                total_latency_ms=total_latency_ms,
                used_clock_model=used_clock_model,
            )

        return LatencyEstimate(
            total_latency_ms=total_latency_ms,
            per_layer_latency_ms=per_layer_latency_ms,
            used_clock_model=used_clock_model,
            assumptions=assumptions,
        )

    def _estimate_layer_latency_ms(
        self, layer: IMRLayer, profile: HardwareProfile, used_clock_model: bool
    ) -> float:
        """Estimate one layer's latency, in milliseconds.

        Args:
            layer: The layer to estimate.
            profile: The candidate deployment target.
            used_clock_model: Whether to use the cycles-per-MAC model
                (``True``) or the fallback throughput model (``False``).

        Returns:
            The estimated latency for ``layer``, in milliseconds.
        """
        slowdown = PRECISION_SLOWDOWN.get(layer.precision, 1.0)

        if layer.op_type in FIXED_COST_OP_TYPES and not layer.parameters:
            return FIXED_OP_COST_MS * slowdown

        mac_count = self._estimate_mac_count(layer)
        if mac_count == 0:
            return FIXED_OP_COST_MS * slowdown

        if used_clock_model and profile.clock_speed_hz:
            cycles_per_mac = CYCLES_PER_MAC_INT8.get(profile.hardware_class, 1.0)
            total_cycles = mac_count * cycles_per_mac * slowdown
            seconds = total_cycles / profile.clock_speed_hz
        else:
            macs_per_second = FALLBACK_MACS_PER_SECOND_INT8.get(
                profile.hardware_class, 1.0e7
            )
            seconds = (mac_count * slowdown) / macs_per_second

        return seconds * 1000.0

    def _estimate_mac_count(self, layer: IMRLayer) -> int:
        """Estimate a layer's multiply-accumulate operation count.

        Uses total learned-parameter element count as a proxy for MAC
        count — exact for a single-use dense/matmul layer, an
        under-estimate for a convolution reused across spatial
        positions (whose true MAC count also scales with output
        spatial size, which the ``IMR`` does not record). This keeps
        the estimate conservative (a lower bound) rather than
        overstating latency.

        Args:
            layer: The layer to estimate.

        Returns:
            The estimated MAC count for ``layer``.
        """
        total_elements = 0
        for tensor in layer.parameters.values():
            element_count = 1
            for dim in tensor.shape:
                element_count *= int(dim)
            total_elements += element_count
        return total_elements
