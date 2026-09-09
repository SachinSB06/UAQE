"""Per-layer numeric-precision recommendation.

``PrecisionRecommender`` turns a ``QuantizationConfig`` (a model-wide
default plus optional per-layer overrides), an optional
:class:`~uaqe.quantization.sensitivity_analyzer.SensitivityReport`, and
the run's ``HardwareProfile`` into a concrete per-layer precision
mapping. It is a stateless, non-``PipelineStage`` collaborator consumed
by :class:`~uaqe.quantization.quantization_planner.QuantizationPlanner`
— unlike ``Calibrator``/``SensitivityAnalyzer``, its output is not
itself a distinct pipeline stage result; it is an intermediate step the
planner uses while building one ``QuantizationPlan``.

Per ``05_Hardware_Profile_Spec.md`` §6 rule 2 (also stated in
``10_Module_Development_Guide.md`` §6's Integration Checklist for this
package's advisor-equivalent class), a recommended precision that the
target ``HardwareProfile`` does not list in ``supported_precisions`` is
a hard constraint violation, never an advisory suggestion: this class
always overrides an unsupported precision and always logs a warning
when it does so, even when the override came directly from user-supplied
``QuantizationConfig.per_layer_overrides``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import Precision
from uaqe.common.value_objects import QuantizationConfig
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.quantization.sensitivity_analyzer import SensitivityReport

#: Precision search order, most- to least-permissive, used both to pick
#: a fallback when a requested precision is unsupported and to "step up"
#: a flagged-sensitive layer to a safer precision.
_PRECISION_ORDER: List[Precision] = [
    Precision.FP32,
    Precision.FP16,
    Precision.INT8,
    Precision.INT4,
]


@dataclass(frozen=True)
class PrecisionRecommendation:
    """The outcome of per-layer precision recommendation.

    Attributes:
        per_layer_precision: The recommended ``Precision`` for every
            layer in the analyzed ``IMR``, keyed by ``IMRLayer.name``.
            Every value is guaranteed to be a member of the target
            ``HardwareProfile.supported_precisions``.
        rationale: A short, human-readable explanation of why each
            layer received its recommended precision, keyed by the same
            layer names as ``per_layer_precision`` — surfaced verbatim
            in ``uaqe.quantization.quantization_report``.
        overridden_layers: Names of layers whose originally requested
            precision (from ``QuantizationConfig``) was not supported by
            the target hardware and was therefore overridden.
        stepped_up_layers: Names of layers flagged as
            quantization-sensitive by the ``SensitivityReport`` whose
            resolved precision was stepped up to a more permissive,
            supported precision as a result.
    """

    per_layer_precision: Dict[str, Precision] = field(default_factory=dict)
    rationale: Dict[str, str] = field(default_factory=dict)
    overridden_layers: List[str] = field(default_factory=list)
    stepped_up_layers: List[str] = field(default_factory=list)


class PrecisionRecommender:
    """Recommends a per-layer ``Precision`` mapping, hard-filtered
    against hardware support.

    Attributes:
        _logger: Structured logging sink.
    """

    def __init__(self, logger: ILogger) -> None:
        """Initialize the recommender.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
        """
        self._logger = logger

    def recommend(
        self,
        imr: IMR,
        profile: HardwareProfile,
        config: QuantizationConfig,
        sensitivity: Optional[SensitivityReport] = None,
    ) -> PrecisionRecommendation:
        """Recommend a per-layer precision mapping for ``imr``.

        Args:
            imr: The model to recommend precisions for.
            profile: The resolved deployment target; recommendations
                are always filtered against
                ``profile.supported_precisions``.
            config: The run's ``QuantizationConfig``, supplying the
                model-wide ``default_precision`` and any
                ``per_layer_overrides``.
            sensitivity: An optional
                :class:`~uaqe.quantization.sensitivity_analyzer.
                SensitivityReport`; when supplied, every
                ``flagged_layers`` entry is stepped up to the next more
                permissive supported precision, if one exists.

        Returns:
            The computed :class:`PrecisionRecommendation`, guaranteed to
            map every layer to a precision in
            ``profile.supported_precisions``.
        """
        supported = set(profile.supported_precisions)
        _, default_overridden = self._resolve(config.default_precision, supported)
        if default_overridden:
            self._logger.warning(
                f"Default precision {config.default_precision.value} is not "
                f"supported by hardware profile {profile.profile_id!r}; each "
                f"layer using the model default will be overridden "
                f"individually below.",
                hardware_profile_id=profile.profile_id,
            )

        flagged = set(sensitivity.flagged_layers) if sensitivity is not None else set()

        per_layer_precision: Dict[str, Precision] = {}
        rationale: Dict[str, str] = {}
        overridden_layers: List[str] = []
        stepped_up_layers: List[str] = []

        for layer in imr.layers:
            requested = config.per_layer_overrides.get(
                layer.name, config.default_precision
            )
            resolved, was_overridden = self._resolve(requested, supported)
            reason = (
                f"per-layer override ({requested.value})"
                if layer.name in config.per_layer_overrides
                else f"model default ({config.default_precision.value})"
            )
            if was_overridden:
                overridden_layers.append(layer.name)
                reason += (
                    f", overridden to {resolved.value} — {requested.value} is "
                    f"unsupported by hardware profile {profile.profile_id!r}"
                )
                self._logger.warning(
                    f"Layer {layer.name!r} requested precision "
                    f"{requested.value}, unsupported by hardware profile "
                    f"{profile.profile_id!r}; overriding to {resolved.value}.",
                    hardware_profile_id=profile.profile_id,
                )
            elif layer.name in flagged:
                stepped_up = self._step_up(resolved, supported)
                if stepped_up != resolved:
                    reason += (
                        f", stepped up to {stepped_up.value} due to flagged "
                        f"quantization sensitivity"
                    )
                    resolved = stepped_up
                    stepped_up_layers.append(layer.name)
                else:
                    reason += (
                        ", flagged as quantization-sensitive but no more "
                        "permissive supported precision is available"
                    )

            per_layer_precision[layer.name] = resolved
            rationale[layer.name] = reason

        return PrecisionRecommendation(
            per_layer_precision=per_layer_precision,
            rationale=rationale,
            overridden_layers=overridden_layers,
            stepped_up_layers=stepped_up_layers,
        )

    @staticmethod
    def _resolve(requested: Precision, supported: set) -> Tuple[Precision, bool]:
        """Return ``requested`` unchanged if supported, else the
        nearest fallback and a flag indicating an override occurred.

        ``Precision.MIXED`` is treated as always "supported" here: it
        is a per-model routing signal consumed by
        ``uaqe.quantization.mixed_precision_quantizer.
        MixedPrecisionQuantizer``, not an executable numeric precision a
        ``HardwareProfile`` itself lists.
        """
        if requested == Precision.MIXED or requested in supported:
            return requested, False
        for candidate in _PRECISION_ORDER:
            if candidate in supported:
                return candidate, True
        # No precision at all is supported by this profile: this is a
        # hardware-profile authoring defect, not a per-run condition
        # this recommender can resolve, so it surfaces the requested
        # value unchanged and lets downstream hardware-compatibility
        # checking (not this package) raise the appropriate error.
        return requested, True

    @staticmethod
    def _step_up(current: Precision, supported: set) -> Precision:
        """Return the next more-permissive supported precision after
        ``current``, or ``current`` unchanged if none exists.
        """
        if current not in _PRECISION_ORDER:
            return current
        current_index = _PRECISION_ORDER.index(current)
        for candidate in reversed(_PRECISION_ORDER[:current_index]):
            if candidate in supported:
                return candidate
        return current
