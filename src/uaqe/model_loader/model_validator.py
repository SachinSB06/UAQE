"""Structural sanity checking of a loaded ``IMR``, prior to analysis.

Per ``09_Architecture_Lock.md`` §12, ``model_validator`` always runs
immediately after ``model_loader`` and before ``model_analyzer`` in the
locked 19-stage pipeline order; per ``10_Module_Development_Guide.md``
§3 Integration Checklist, this ordering must hold for every
``HardwareClass``.

Locked contract: ``03_API_Specification.md`` §3.2.
Locked payload mapping: ``09_Architecture_Lock.md`` §9
(``model_validator`` -> ``None``).
"""

from __future__ import annotations

import time
from typing import List

from uaqe.common.exceptions import ModelValidationError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage


class ModelValidator(PipelineStage):
    """Runs structural sanity checks on the ``IMR`` produced by ``ModelLoader``.

    Per ``04_Data_Flow.md`` §3, validation is a gate rather than a data
    source: downstream stages never read this stage's ``payload``
    (always ``None``); they instead rely on ``context.has("model_validator")``
    and the recorded ``StageResult.success`` as a precondition check
    before reading ``model_loader``'s ``IMR``.

    Attributes:
        _logger: The logger this stage reports validation events to.
    """

    def __init__(self, logger: ILogger) -> None:
        """Initialize the validator.

        Args:
            logger: The logger to report validation start/end and any
                accumulated warnings to.
        """
        self._logger: ILogger = logger

    def execute(self, context: PipelineContext) -> StageResult:
        """Validate the ``IMR`` produced by the ``model_loader`` stage.

        Args:
            context: The run's pipeline context; the ``IMR`` to
                validate is read from ``context.get("model_loader").payload``.

        Returns:
            A ``StageResult`` with ``payload=None`` and ``success=True``
            if validation passes; any non-fatal observations are
            recorded in ``warnings``.

        Raises:
            ModelValidationError: If the ``IMR`` fails any structural
                check in :meth:`validate`.
        """
        started_at = time.monotonic()
        self._logger.info("model_validator: starting validation", stage="model_validator")

        imr: IMR = context.get("model_loader").payload
        warnings: List[str] = []

        try:
            self.validate(imr)
        except ModelValidationError as exc:
            log_context = exc.to_log_context()
            log_context["stage"] = self.name()
            self._logger.error(
                "model_validator: validation failed",
                **log_context,
            )
            raise

        if imr.metadata.op_count != len(imr.layers):
            # validate() already raises on this mismatch; unreachable
            # in practice, but guards against future refactors that
            # might loosen that check without updating this comment.
            warnings.append(
                "IMRMetadata.op_count does not match len(imr.layers); "
                "downstream stage counts may be unreliable."
            )

        duration_ms = (time.monotonic() - started_at) * 1000.0
        self._logger.info(
            "model_validator: validation passed",
            stage="model_validator",
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=None,
            warnings=warnings,
            duration_ms=duration_ms,
        )

    def validate(self, imr: IMR) -> None:
        """Run every structural sanity check against ``imr``.

        Per ``04_Data_Flow.md`` §3 and
        ``10_Module_Development_Guide.md`` §3, the checks are:

        - ``imr.layers`` is non-empty.
        - Every ``IMRLayer.inputs`` name resolves to either a tensor
          produced by another layer's ``outputs``, a name declared in
          ``IMRMetadata.original_input_shapes``, or a name present in
          that same layer's own ``parameters`` (an initializer/weight
          tensor, which per the ONNX spec is a valid tensor source
          with no producing node) (i.e. no dangling edges).
        - The layer graph contains no cycle (delegated to
          ``IMR.topological_order()``, which itself raises
          ``ModelValidationError`` on a cycle per
          ``uaqe.common.imr``).
        - ``IMRMetadata.op_count`` matches ``len(imr.layers)``.
        - Every parameter tensor's declared shape has strictly positive
          dimensions (a valid tensor shape).

        Args:
            imr: The loaded model representation to validate.

        Raises:
            ModelValidationError: If any of the checks above fails.
        """
        if len(imr.layers) == 0:
            raise ModelValidationError(
                "IMR contains no layers.",
                code="IMR_EMPTY",
                remediation_hint="Verify the source model was exported with its full graph intact.",
            )

        if imr.metadata.op_count != len(imr.layers):
            raise ModelValidationError(
                f"IMRMetadata.op_count ({imr.metadata.op_count}) does not match "
                f"len(imr.layers) ({len(imr.layers)}).",
                code="IMR_OP_COUNT_MISMATCH",
                remediation_hint="Verify the source adapter populated op_count from the final layer list.",
            )

        known_tensor_names = set(imr.metadata.original_input_shapes.keys())
        for layer in imr.layers:
            known_tensor_names.update(layer.outputs)
            # A layer's own initializer/parameter tensors (weights,
            # biases, etc.) are valid tensor sources too: the adapter
            # keeps their names in both ``layer.inputs`` (per ONNX,
            # where a node's ``input`` list may reference a graph
            # initializer directly, with no producing node) and
            # ``layer.parameters``. Without this, every layer that
            # consumes a weight/bias would be flagged as dangling.
            known_tensor_names.update(layer.parameters.keys())

        for layer in imr.layers:
            for input_name in layer.inputs:
                if input_name not in known_tensor_names:
                    raise ModelValidationError(
                        f"Layer {layer.name!r} has a dangling input "
                        f"{input_name!r} produced by no layer and declared "
                        f"in no input shape.",
                        code="IMR_DANGLING_INPUT",
                        remediation_hint="Verify the source model graph is fully connected.",
                    )
            if len(layer.outputs) == 0:
                raise ModelValidationError(
                    f"Layer {layer.name!r} declares no outputs.",
                    code="IMR_LAYER_NO_OUTPUTS",
                    remediation_hint="Every layer must produce at least one output tensor.",
                )

        for layer in imr.layers:
            for parameter_name, tensor in layer.parameters.items():
                if any(dimension <= 0 for dimension in tensor.shape):
                    raise ModelValidationError(
                        f"Layer {layer.name!r} parameter {parameter_name!r} has "
                        f"an invalid shape {tensor.shape!r}.",
                        code="IMR_INVALID_TENSOR_SHAPE",
                        remediation_hint="Every tensor dimension must be a positive integer.",
                    )

        # Delegated cycle check: raises ModelValidationError("...cycle...")
        # if the graph cannot be topologically ordered.
        imr.topological_order()
