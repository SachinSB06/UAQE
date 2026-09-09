"""``WorkflowConfig`` — the single input value object a caller supplies
to start a run of the Universal AI Quantization Engine.

This plays exactly the role ``03_API_Specification.md`` §14.1 and
``11_Implementation_Rules.md`` §6.6's factory-interaction diagram
(``RunRequest -> WorkflowController -> PipelineBuilder``) locks for
``RunRequest``: it is the DTO an ``uaqe.interface`` entry point (CLI,
REST, batch) constructs from user input and passes to
:class:`~uaqe.application.workflow_controller.WorkflowController`. It
is named ``WorkflowConfig`` rather than ``RunRequest`` in this
package's decomposition (mirroring, e.g., ``uaqe.optimizer``'s
documented precedent of diverging file/class names from
``02_Folder_Structure.md`` while preserving the locked *role* — see
``uaqe.optimizer``'s own ``__init__.py`` docstring), so that every
class in this ten-file package shares the ``workflow_``/``stage``/
``pipeline_`` naming family end to end.

Per ``06_Config_Spec.md`` §8's locked precedence order,
``config_overrides`` on this object outranks every subsystem
``advisor`` block and every ``config/*.json`` file default; it is the
one place a caller may override a single field of an otherwise
repository-sourced ``*Config`` value object for a single run, without
mutating the (frozen, read-only-at-runtime) source config files
themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from uaqe.application.application_errors import WorkflowConfigurationError


@dataclass(frozen=True)
class WorkflowConfig:
    """Everything needed to start one run of the pipeline.

    Attributes:
        model_path: Filesystem path to the source model file to load.
            Handed to :class:`~uaqe.model_loader.model_loader.
            ModelLoader` via its ``bind_source_path`` seam by
            :class:`~uaqe.application.stage_factory.StageFactory`
            (``uaqe.model_loader.model_loader`` module docstring) —
            the one piece of the doc-locked ``ModelIngestionService``'s
            responsibility this package's decomposition folds into
            stage assembly rather than a separate service file.
        hardware_profile_id: The ``HardwareProfile.profile_id`` to
            resolve for this run, e.g. ``"artix7"``, ``"esp32"``,
            ``"raspberrypi4"``. Handed to
            :class:`~uaqe.hardware.hardware_manager.HardwareManager`'s
            constructor by ``StageFactory`` — folding in the doc-locked
            ``HardwareSelectionService``'s one responsibility the same
            way ``model_path`` folds in ``ModelIngestionService``'s.
        calibration_dataset_path: Optional filesystem path to a
            calibration dataset, consumed by
            :class:`~uaqe.quantization.calibrator.Calibrator`.
            ``None`` means calibration runs from layer weights alone
            (``uaqe.quantization.calibrator`` module docstring).
        evaluation_dataset_path: Optional filesystem path to a labeled
            evaluation dataset, consumed by
            :class:`~uaqe.evaluation.evaluator.Evaluator`'s accuracy
            scoring. ``None`` disables accuracy-delta evaluation for
            this run.
        config_overrides: Per-run overrides layered on top of every
            subsystem config file's defaults, per ``06_Config_Spec.md``
            §8's locked precedence order (this field outranks every
            other source). Keys are dotted subsystem field paths, e.g.
            ``{"quantization.default_precision": "INT8"}``.
        run_id_prefix: Optional override for ``config.json``'s
            ``default_run_id_prefix``, used by
            :class:`~uaqe.application.workflow_context.WorkflowContext`
            when it mints a fresh ``run_id`` for this run. ``None``
            uses the process-wide default sourced from
            ``IConfigRepository`` at ``CompositionRoot`` wiring time.
    """

    model_path: str
    hardware_profile_id: str
    runtime: Optional[str] = None
    calibration_dataset_path: Optional[str] = None
    evaluation_dataset_path: Optional[str] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    run_id_prefix: Optional[str] = None

    def validate(self) -> None:
        """Validate this configuration's own fields, independent of any
        repository or filesystem lookup.

        Deliberately shallow: this checks only what can be known from
        the object's own fields (non-empty required strings, a
        well-formed ``config_overrides`` mapping). It does not confirm
        ``model_path`` exists on disk or that ``hardware_profile_id``
        resolves to a known profile — those are, respectively,
        :class:`~uaqe.model_loader.model_loader.ModelLoader` (a
        ``ModelLoadError``) and
        :class:`~uaqe.hardware.hardware_manager.HardwareManager`'s
        (a ``ConfigurationError``) responsibilities once the pipeline
        actually runs, per ``11_Implementation_Rules.md`` §4.1's rule
        that a stage validates its own preconditions rather than
        having them pre-checked by an upstream caller on its behalf.

        Raises:
            WorkflowConfigurationError: If ``model_path`` or
                ``hardware_profile_id`` is empty/whitespace-only, or if
                ``config_overrides`` is not a ``str``-keyed mapping.
        """
        if not self.model_path or not self.model_path.strip():
            raise WorkflowConfigurationError(
                "WorkflowConfig.model_path must be a non-empty path.",
                code="WORKFLOW_CONFIG_MISSING_MODEL_PATH",
                remediation_hint=(
                    "Supply the filesystem path to the source model file."
                ),
            )
        if not self.hardware_profile_id or not self.hardware_profile_id.strip():
            raise WorkflowConfigurationError(
                "WorkflowConfig.hardware_profile_id must be a non-empty "
                "identifier.",
                code="WORKFLOW_CONFIG_MISSING_HARDWARE_PROFILE_ID",
                remediation_hint=(
                    "Supply a HardwareProfile.profile_id, e.g. 'esp32' or "
                    "'artix7'."
                ),
            )
        if not isinstance(self.config_overrides, dict) or not all(
            isinstance(key, str) for key in self.config_overrides
        ):
            raise WorkflowConfigurationError(
                "WorkflowConfig.config_overrides must be a Dict[str, Any].",
                code="WORKFLOW_CONFIG_INVALID_OVERRIDES",
                remediation_hint=(
                    "Use dotted subsystem field paths as keys, e.g. "
                    "{'quantization.default_precision': 'INT8'}."
                ),
            )

    def override(self, dotted_key: str, default: Any) -> Any:
        """Resolve one subsystem field, honoring this config's override
        precedence.

        Args:
            dotted_key: The dotted subsystem field path to look up,
                e.g. ``"quantization.default_precision"``.
            default: The value to return when ``dotted_key`` is not
                present in :attr:`config_overrides` — normally the
                value already loaded from the relevant ``config/*.json``
                file (or its ``advisor`` block) by ``IConfigRepository``.

        Returns:
            ``config_overrides[dotted_key]`` if present, else
            ``default`` — implementing the top tier of
            ``06_Config_Spec.md`` §8's locked precedence order.
        """
        return self.config_overrides.get(dotted_key, default)
