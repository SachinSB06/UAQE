"""Pre-export validation gate.

``ExportValidator`` runs immediately before
:class:`~uaqe.exporter.exporter.Exporter` invokes a backend's
``export()``. It is deliberately narrow and re-checks only what has
changed meaning by the time a run reaches this, the pipeline's last
IMR-adjacent stage (``09_Architecture_Lock.md`` §12:
``... -> memory_optimizer -> exporter -> ...``):

- ``LayerCompatibilityChecker`` (``uaqe.domain.compatibility``) already
  ran earlier, before quantization/compression could still change
  layer op types — its result is not re-derived here.
- ``MemoryOptimizer`` already validated activation-arena usage against
  ``HardwareProfile.tensor_memory_bytes`` strictly (see that class's
  docstring).

What has *not* yet been checked against the final, fully quantized and
compressed ``IMR`` is: (1) that every layer's final ``Precision`` is
still one the target actually executes, since quantization/compression
are the stages that could have left a layer at an unsupported
precision, and (2) that the final serialized parameter size fits
``HardwareProfile.max_model_size_bytes`` — a check that is only
meaningful once compression has already run. This module owns exactly
those two checks.
"""

from __future__ import annotations

from uaqe.common.exceptions import ExportError
from uaqe.common.imr import IMR
from uaqe.common.types import ExportFormat
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.exporter.binary_exporter import serialize_parameters


class ExportValidator:
    """Validates that a final ``IMR`` may be exported for a given
    ``HardwareProfile`` and ``ExportFormat``.
    """

    def validate(
        self, imr: IMR, target: HardwareProfile, export_format: ExportFormat
    ) -> None:
        """Validate ``imr`` against ``target`` immediately before export.

        Args:
            imr: The final, fully-optimized model to export.
            target: The resolved deployment target.
            export_format: The ``ExportFormat`` selected by
                :class:`~uaqe.exporter.export_planner.ExportPlanner`.

        Raises:
            ExportError: If ``imr`` has no layers, if any layer's
                ``precision`` is not in ``target.supported_precisions``,
                or if the serialized parameter size exceeds
                ``target.max_model_size_bytes``.
        """
        del export_format  # Reserved for a future per-format size/shape rule.

        if not imr.layers:
            raise ExportError(
                "Cannot export an IMR with zero layers.",
                code="EXPORT_EMPTY_MODEL",
            )

        unsupported = sorted(
            {
                layer.name
                for layer in imr.layers
                if layer.precision not in target.supported_precisions
            }
        )
        if unsupported:
            raise ExportError(
                f"{len(unsupported)} layer(s) hold a precision unsupported "
                f"by target {target.profile_id!r} "
                f"(supported: {[p.value for p in target.supported_precisions]}): "
                f"{unsupported}.",
                code="EXPORT_UNSUPPORTED_PRECISION",
            )

        serialized_size = len(serialize_parameters(imr))
        if serialized_size > target.max_model_size_bytes:
            raise ExportError(
                f"Serialized model size {serialized_size:,} bytes exceeds "
                f"target {target.profile_id!r}'s max_model_size_bytes "
                f"({target.max_model_size_bytes:,}).",
                code="EXPORT_MODEL_TOO_LARGE",
            )
