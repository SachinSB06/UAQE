"""Checks whether an ``IMR`` can be deployed to a given ``HardwareProfile``.

Implements the enforcement contract in ``05_Hardware_Profile_Spec.md``
§6: layer/op-type support, precision support, and total model size are
checked in code; ``HardwareProfile.constraints`` free-text notes are
surfaced verbatim rather than parsed (§6 rule 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional

from uaqe.common.exceptions import HardwareIncompatibilityError
from uaqe.common.imr import IMR, IMRLayer
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import CompatibilityReport
from uaqe.common.types import HardwareClass

if TYPE_CHECKING:
    # Deferred to break the hardware_manager <-> compatibility_checker
    # import cycle: HardwareManager composes a CompatibilityChecker at
    # runtime, while CompatibilityChecker only needs HardwareProfile as
    # a type hint here, so this import never needs to execute.
    from uaqe.hardware.hardware_manager import HardwareProfile

#: Op types with no implementation in the v1 bare-metal FPGA HDL backend
#: (``05_Hardware_Profile_Spec.md`` §3 constraint note: "Recurrent
#: layers (LSTM/GRU) unsupported in v1 HDL backend").
FPGA_UNSUPPORTED_OP_TYPES: FrozenSet[str] = frozenset({"LSTM", "GRU", "RNN"})

#: Op types requiring native floating-point execution; FPGA profiles in
#: this database have no native FPU, so any layer of these types not
#: already fully quantized is a violation on that class
#: (``05_Hardware_Profile_Spec.md`` §3 constraint note).
FLOATING_POINT_ONLY_OP_TYPES: FrozenSet[str] = frozenset({"FusedBatchNormTraining"})


class CompatibilityChecker:
    """Checks an ``IMR`` for layer, precision, and size compatibility
    with a ``HardwareProfile``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            checker operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``CompatibilityChecker``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def check(
        self, imr: IMR, profile: HardwareProfile, strict: bool = False
    ) -> CompatibilityReport:
        """Check ``imr`` against ``profile`` and produce a full report.

        Args:
            imr: The model to check.
            profile: The candidate deployment target.
            strict: If ``True``, raise ``HardwareIncompatibilityError``
                when ``imr`` is not fully compatible instead of
                returning a report with ``compatible=False``.

        Returns:
            The resulting ``CompatibilityReport``.

        Raises:
            HardwareIncompatibilityError: If ``strict`` is ``True`` and
                ``imr`` is not fully compatible with ``profile``.
        """
        unsupported_layers = self.find_unsupported_layers(imr, profile)
        constraint_violations = self._check_size_constraints(imr, profile)
        # HardwareProfile.constraints is surfaced verbatim as advisory
        # context alongside any triggered violation, per
        # 05_Hardware_Profile_Spec.md §6 rule 1 — it is never itself
        # parsed to decide compatible/incompatible.
        compatible = not unsupported_layers and not constraint_violations

        if self.logger is not None:
            self.logger.info(
                "Checked hardware compatibility.",
                profile_id=profile.profile_id,
                compatible=compatible,
                unsupported_layer_count=len(unsupported_layers),
            )

        report = CompatibilityReport(
            compatible=compatible,
            unsupported_layers=unsupported_layers,
            constraint_violations=constraint_violations,
        )

        if strict and not compatible:
            raise HardwareIncompatibilityError(
                f"IMR is incompatible with hardware profile "
                f"{profile.profile_id!r}.",
                code="HARDWARE_INCOMPATIBLE",
                remediation_hint=(
                    "Inspect CompatibilityReport.unsupported_layers and "
                    "constraint_violations; adjust the quantization/"
                    "compression plan or select a different profile."
                ),
            )
        return report

    def find_unsupported_layers(self, imr: IMR, profile: HardwareProfile) -> List[str]:
        """Return the names of every layer that cannot run on ``profile``.

        A layer is unsupported if its ``op_type`` is not implemented for
        ``profile.hardware_class`` (per :data:`FPGA_UNSUPPORTED_OP_TYPES`
        and :data:`FLOATING_POINT_ONLY_OP_TYPES`), or if its ``precision``
        is absent from ``profile.supported_precisions``.

        Args:
            imr: The model to check.
            profile: The candidate deployment target.

        Returns:
            The unsupported layers' ``IMRLayer.name`` values, in the
            order they appear in ``imr.layers``.
        """
        unsupported: List[str] = []
        for layer in imr.layers:
            if self._is_op_type_unsupported(layer, profile) or self._is_precision_unsupported(
                layer, profile
            ):
                unsupported.append(layer.name)
        return unsupported

    def _is_op_type_unsupported(self, layer: IMRLayer, profile: HardwareProfile) -> bool:
        """Report whether ``layer.op_type`` has no implementation for
        ``profile.hardware_class``.

        Args:
            layer: The layer to check.
            profile: The candidate deployment target.

        Returns:
            ``True`` if ``layer`` cannot execute on ``profile``'s
            hardware class due to its operator type.
        """
        if profile.hardware_class != HardwareClass.FPGA:
            return False
        return (
            layer.op_type in FPGA_UNSUPPORTED_OP_TYPES
            or layer.op_type in FLOATING_POINT_ONLY_OP_TYPES
        )

    def _is_precision_unsupported(self, layer: IMRLayer, profile: HardwareProfile) -> bool:
        """Report whether ``layer.precision`` is unsupported by ``profile``.

        Args:
            layer: The layer to check.
            profile: The candidate deployment target.

        Returns:
            ``True`` if ``layer.precision`` is absent from
            ``profile.supported_precisions``.
        """
        return layer.precision not in profile.supported_precisions

    def _check_size_constraints(self, imr: IMR, profile: HardwareProfile) -> List[str]:
        """Check whole-model size constraints that are not layer-specific.

        Args:
            imr: The model to check.
            profile: The candidate deployment target.

        Returns:
            Human-readable descriptions of every triggered size
            constraint, empty if none were triggered.
        """
        violations: List[str] = []
        estimated_bytes = self._estimate_serialized_size(imr)
        if estimated_bytes > profile.max_model_size_bytes:
            violations.append(
                f"Estimated serialized model size ({estimated_bytes} bytes) "
                f"exceeds max_model_size_bytes ({profile.max_model_size_bytes})."
            )
        if profile.hardware_class == HardwareClass.FPGA and profile.fpga_resources is None:
            violations.append(
                f"Profile {profile.profile_id!r} is HardwareClass.FPGA but "
                "declares no fpga_resources."
            )
        return violations

    def _estimate_serialized_size(self, imr: IMR) -> int:
        """Estimate an ``IMR``'s total serialized size, in bytes.

        This sums every parameter tensor's raw byte length across every
        layer. It is a lower-bound estimate: it does not account for
        runtime/container overhead (e.g. a ``.tflite`` flatbuffer's own
        framing), so an ``Exporter`` backend's own size check against
        the final artifact remains authoritative
        (``05_Hardware_Profile_Spec.md`` §6 rule 3).

        Args:
            imr: The model to estimate.

        Returns:
            The estimated total parameter payload size, in bytes.
        """
        total = 0
        for layer in imr.layers:
            for tensor in layer.parameters.values():
                total += len(tensor.data)
        return total
