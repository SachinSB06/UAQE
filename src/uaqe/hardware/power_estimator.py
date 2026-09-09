"""Estimates power draw and per-inference energy for a ``HardwareProfile``.

Like :mod:`uaqe.hardware.latency_estimator`, this is not part of the
locked ``05_Hardware_Profile_Spec.md`` surface (see the scope note in
:mod:`uaqe.hardware.hardware_manager`); it derives an order-of-magnitude
figure from a reference active-mode power table plus a
``LatencyEstimate``, for early-stage hardware comparison before a
measured on-device power reading is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import HardwareClass, Precision
from uaqe.hardware.latency_estimator import LatencyEstimate

if TYPE_CHECKING:
    # Deferred to break the hardware_manager <-> power_estimator import
    # cycle; see the matching note in compatibility_checker.py.
    from uaqe.hardware.hardware_manager import HardwareProfile

#: Reference active-mode (inference-time) power draw, in milliwatts, at
#: INT8, per hardware class. Order-of-magnitude figures representative
#: of a typical board in each class, not per-profile measured values —
#: an FPGA's true draw in particular varies enormously with the
#: synthesized design, so this is a coarse placeholder pending a real
#: ``uaqe.domain.benchmark.Benchmarker`` power reading.
ACTIVE_POWER_MW: Dict[HardwareClass, float] = {
    HardwareClass.FPGA: 900.0,
    HardwareClass.EMBEDDED: 150.0,
    HardwareClass.RASPBERRY_PI: 3000.0,
}

#: Idle/quiescent power draw, in milliwatts, per hardware class — the
#: floor :meth:`PowerEstimator.estimate` reports if it were ever asked
#: for a zero-latency estimate.
IDLE_POWER_MW: Dict[HardwareClass, float] = {
    HardwareClass.FPGA: 300.0,
    HardwareClass.EMBEDDED: 20.0,
    HardwareClass.RASPBERRY_PI: 700.0,
}

#: Multiplicative power scaling relative to INT8 for other precisions —
#: wider datapaths draw more current per operation.
PRECISION_POWER_MULTIPLIER: Dict[Precision, float] = {
    Precision.INT4: 0.7,
    Precision.INT8: 1.0,
    Precision.FP16: 1.3,
    Precision.FP32: 1.6,
    Precision.MIXED: 1.15,
}

#: Ratio of peak to average active power, applied as a fixed headroom
#: factor rather than derived from any per-layer signal.
PEAK_TO_AVERAGE_RATIO: float = 1.2


@dataclass
class PowerEstimate:
    """The outcome of power estimation for one ``HardwareProfile``.

    Attributes:
        average_power_mw: The estimated average active-mode power draw,
            in milliwatts, for the dominant precision supplied.
        peak_power_mw: The estimated peak power draw, in milliwatts.
        energy_per_inference_mj: The estimated energy consumed by one
            inference, in millijoules, derived from
            ``average_power_mw`` and the supplied ``LatencyEstimate``.
        assumptions: Human-readable notes on the estimation model used
            and its known limitations.
    """

    average_power_mw: float
    peak_power_mw: float
    energy_per_inference_mj: float
    assumptions: List[str] = field(default_factory=list)


class PowerEstimator:
    """Estimates power draw and per-inference energy for a
    ``HardwareProfile``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            estimator operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``PowerEstimator``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def estimate(
        self,
        profile: HardwareProfile,
        latency: LatencyEstimate,
        dominant_precision: Precision = Precision.INT8,
    ) -> PowerEstimate:
        """Estimate power draw and per-inference energy for ``profile``.

        Args:
            profile: The candidate deployment target.
            latency: A previously computed ``LatencyEstimate`` for the
                same model/profile pair, used to derive energy.
            dominant_precision: The precision most of the model executes
                in, used to scale the reference power figure. Callers
                with a per-layer precision mix should pass the
                highest-share precision (e.g. from a ``QuantizationPlan``).

        Returns:
            The resulting ``PowerEstimate``.
        """
        reference_power_mw = ACTIVE_POWER_MW.get(profile.hardware_class, 500.0)
        multiplier = PRECISION_POWER_MULTIPLIER.get(dominant_precision, 1.0)
        average_power_mw = reference_power_mw * multiplier
        peak_power_mw = average_power_mw * PEAK_TO_AVERAGE_RATIO

        # power_mw * time_ms = 1e-3 W * 1e-3 s = 1e-6 J = 1e-3 mJ.
        energy_per_inference_mj = (average_power_mw * latency.total_latency_ms) / 1000.0

        assumptions = [
            "Power figures are a reference-table estimate for a typical "
            f"{profile.hardware_class.value} board, not a per-profile "
            "measured value; use uaqe.domain.benchmark.Benchmarker for a "
            "measured result once available.",
            f"Scaled by dominant_precision={dominant_precision.value} "
            f"(multiplier={multiplier}).",
            f"peak_power_mw uses a fixed {PEAK_TO_AVERAGE_RATIO}x "
            "peak-to-average headroom factor.",
        ]
        if not latency.used_clock_model:
            assumptions.append(
                "energy_per_inference_mj is derived from a latency figure "
                "that itself used the fallback throughput model (no "
                "clock_speed_hz on this profile); treat as order-of-"
                "magnitude only."
            )

        if self.logger is not None:
            self.logger.info(
                "Estimated power draw.",
                profile_id=profile.profile_id,
                average_power_mw=average_power_mw,
                energy_per_inference_mj=energy_per_inference_mj,
            )

        return PowerEstimate(
            average_power_mw=average_power_mw,
            peak_power_mw=peak_power_mw,
            energy_per_inference_mj=energy_per_inference_mj,
            assumptions=assumptions,
        )
