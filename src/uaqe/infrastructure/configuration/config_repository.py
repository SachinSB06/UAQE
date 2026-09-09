"""Sole first-party implementation of ``IConfigRepository``.

Loads every top-level configuration value object from the files under
``config/`` (JSON) and ``settings.yaml`` (YAML), per
``06_Config_Spec.md``. Per ``11_Implementation_Rules.md`` §5.1, each
file is parsed and validated once per process lifetime and cached;
this class exposes one typed ``load_*_config()`` method per subsystem
rather than a generic ``get(key)`` that would bypass type validation.

Locked contract: ``03_API_Specification.md`` §17.1.
Locked location: ``09_Architecture_Lock.md`` §8, module ownership table
(``uaqe.infrastructure.repositories``: ``ConfigRepository``,
``HardwareProfileRepository``, ``FilesystemRepository``).

Note on ``optimization.json``: ``06_Config_Spec.md`` documents a
dedicated JSON file for every other subsystem config
(``compression.json``, ``quantization.json``, ``benchmark.json``,
``reports.json``, ``hardware.json``) but does not document one for
``OptimizationConfig``, even though ``load_optimization_config()`` is
part of the locked ``IConfigRepository`` contract
(``09_Architecture_Lock.md`` §7). This is treated as a documentation
gap, not a license to omit the method: ``optimization.json`` is read
following the exact structural convention every sibling config file
already uses (top-level ``schema_version`` plus a direct field-for-
field mapping onto the value object), so that the file continues to
follow the same schema-validation contract as the rest of ``config/``.

Thread safety (``11_Implementation_Rules.md`` §5.7): all public
``load_*_config()`` methods are safe for concurrent read access from
multiple stages running under ``execution.parallelism``
(``06_Config_Spec.md`` §2). A single :class:`threading.Lock` guards
first population of each cache slot only — once populated, reads never
block on the lock. No method ever returns a mutable reference into this
repository's internal cache: every returned ``*Config`` instance is a
defensive copy of the cached value, so a caller mutating a returned
``per_layer_overrides``/``enabled_types``/``parallelism``/
``objective_weights`` collection can never corrupt the shared cache.

Raw file I/O and ``schema_version`` validation are delegated to three
internal collaborators colocated in this package —
:class:`~uaqe.infrastructure.repositories.json_loader.JsonConfigLoader`,
:class:`~uaqe.infrastructure.repositories.yaml_loader.YamlConfigLoader`,
and
:class:`~uaqe.infrastructure.repositories.config_validator.ConfigSchemaValidator`
— none of which are part of the locked ``uaqe.infrastructure.repositories``
class inventory (``09_Architecture_Lock.md`` §8) or ever used outside
this class; see their module docstrings for why they exist as separate
modules.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from uaqe.common.exceptions import ConfigurationError
from uaqe.common.interfaces.i_config_repository import IConfigRepository
from uaqe.common.types import CompressionType, PipelineErrorPolicy, Precision
from uaqe.common.value_objects import (
    CompressionConfig,
    ExecutionConfig,
    OptimizationConfig,
    QuantizationConfig,
)
from uaqe.infrastructure.configuration.config_validator import ConfigSchemaValidator
from uaqe.infrastructure.configuration.json_loader import JsonConfigLoader
from uaqe.infrastructure.configuration.yaml_loader import YamlConfigLoader


class ConfigRepository(IConfigRepository):
    """Loads and caches every ``*Config`` value object from ``config/``.

    Attributes:
        base_path: The directory containing ``config.json``,
            ``settings.yaml``, ``compression.json``,
            ``quantization.json``, and ``optimization.json``.
    """

    def __init__(self, base_path: str) -> None:
        """Initialize a ``ConfigRepository`` rooted at ``base_path``.

        Args:
            base_path: The directory containing every ``config/*``
                file this repository loads.
        """
        self.base_path: str = base_path
        self._json_loader: JsonConfigLoader = JsonConfigLoader()
        self._yaml_loader: YamlConfigLoader = YamlConfigLoader()
        self._schema_validator: ConfigSchemaValidator = ConfigSchemaValidator()
        self._json_cache: Dict[str, Dict[str, Any]] = {}
        self._yaml_cache: Dict[str, Dict[str, Any]] = {}
        self._quantization_config: QuantizationConfig | None = None
        self._compression_config: CompressionConfig | None = None
        self._execution_config: ExecutionConfig | None = None
        self._optimization_config: OptimizationConfig | None = None
        self._file_lock: threading.Lock = threading.Lock()
        self._quantization_lock: threading.Lock = threading.Lock()
        self._compression_lock: threading.Lock = threading.Lock()
        self._execution_lock: threading.Lock = threading.Lock()
        self._optimization_lock: threading.Lock = threading.Lock()

    def load_quantization_config(self) -> QuantizationConfig:
        """Load ``quantization.json`` into a ``QuantizationConfig``.

        Returns:
            A defensive copy of the parsed, validated
            ``QuantizationConfig``. Mutating the returned instance's
            ``per_layer_overrides`` dict never affects this
            repository's cache.

        Raises:
            ConfigurationError: If ``quantization.json`` is missing,
                malformed, or fails enum validation.
        """
        if self._quantization_config is None:
            with self._quantization_lock:
                if self._quantization_config is None:
                    data = self._load_json("quantization.json")
                    try:
                        default_precision = Precision(data["default_precision"])
                        per_layer_overrides = {
                            layer_name: Precision(value)
                            for layer_name, value in data.get(
                                "per_layer_overrides", {}
                            ).items()
                        }
                        self._quantization_config = QuantizationConfig(
                            default_precision=default_precision,
                            per_layer_overrides=per_layer_overrides,
                            calibration_batch_size=int(data["calibration_batch_size"]),
                            sensitivity_threshold=float(data["sensitivity_threshold"]),
                        )
                    except (KeyError, ValueError) as exc:
                        raise ConfigurationError(
                            "quantization.json is missing a required field or has "
                            "an invalid enum value.",
                            code="INVALID_QUANTIZATION_CONFIG",
                            remediation_hint=(
                                "Validate quantization.json against 06_Config_Spec.md §5."
                            ),
                        ) from exc

        return QuantizationConfig(
            default_precision=self._quantization_config.default_precision,
            per_layer_overrides=deepcopy(self._quantization_config.per_layer_overrides),
            calibration_batch_size=self._quantization_config.calibration_batch_size,
            sensitivity_threshold=self._quantization_config.sensitivity_threshold,
        )

    def load_compression_config(self) -> CompressionConfig:
        """Load ``compression.json`` into a ``CompressionConfig``.

        Returns:
            A defensive copy of the parsed, validated
            ``CompressionConfig``. Mutating the returned instance's
            ``enabled_types`` list never affects this repository's
            cache.

        Raises:
            ConfigurationError: If ``compression.json`` is missing,
                malformed, or fails enum validation.
        """
        if self._compression_config is None:
            with self._compression_lock:
                if self._compression_config is None:
                    data = self._load_json("compression.json")
                    try:
                        enabled_types = [
                            CompressionType(entry) for entry in data["enabled_types"]
                        ]
                        self._compression_config = CompressionConfig(
                            enabled_types=enabled_types,
                            target_ratio=float(data["target_ratio"]),
                            pruning_sparsity=float(data["pruning_sparsity"]),
                            experimental_pruning_strategy=str(data.get("experimental_pruning_strategy", "magnitude_pruning")),
                            max_accuracy_drop=float(data.get("max_accuracy_drop", 0.01)),
                            max_latency_regression=float(data.get("max_latency_regression", 0.05)),
                            fine_tune_enabled=bool(data.get("fine_tune_enabled", False)),
                            fine_tune_epochs=int(data.get("fine_tune_epochs", 5)),
                            learning_rate=float(data.get("learning_rate", 1e-4)),
                            optimizer=str(data.get("optimizer", "Adam")),
                            scheduler=str(data.get("scheduler", "CosineAnnealing")),
                            batch_size=int(data.get("batch_size", 16)),
                            weight_decay=float(data.get("weight_decay", 1e-4)),
                            early_stopping=bool(data.get("early_stopping", True)),
                        )
                    except (KeyError, ValueError) as exc:
                        raise ConfigurationError(
                            "compression.json is missing a required field or has "
                            "an invalid enum value.",
                            code="INVALID_COMPRESSION_CONFIG",
                            remediation_hint=(
                                "Validate compression.json against 06_Config_Spec.md §4."
                            ),
                        ) from exc

        return CompressionConfig(
            enabled_types=list(self._compression_config.enabled_types),
            target_ratio=self._compression_config.target_ratio,
            pruning_sparsity=self._compression_config.pruning_sparsity,
            experimental_pruning_strategy=self._compression_config.experimental_pruning_strategy,
            max_accuracy_drop=self._compression_config.max_accuracy_drop,
            max_latency_regression=self._compression_config.max_latency_regression,
            fine_tune_enabled=self._compression_config.fine_tune_enabled,
            fine_tune_epochs=self._compression_config.fine_tune_epochs,
            learning_rate=self._compression_config.learning_rate,
            optimizer=self._compression_config.optimizer,
            scheduler=self._compression_config.scheduler,
            batch_size=self._compression_config.batch_size,
            weight_decay=self._compression_config.weight_decay,
            early_stopping=self._compression_config.early_stopping,
        )

    def load_execution_config(self) -> ExecutionConfig:
        """Load ``config.json`` and ``settings.yaml`` into an ``ExecutionConfig``.

        Returns:
            A defensive copy of the parsed, validated
            ``ExecutionConfig``. Mutating the returned instance's
            ``parallelism`` dict never affects this repository's
            cache.

        Raises:
            ConfigurationError: If either source file is missing,
                malformed, or fails enum validation.
        """
        if self._execution_config is None:
            with self._execution_lock:
                if self._execution_config is None:
                    config_data = self._load_json("config.json")
                    settings_data = self._load_yaml("settings.yaml")
                    try:
                        on_error = PipelineErrorPolicy(config_data["on_error"])
                        parallelism = dict(
                            settings_data.get("execution", {}).get("parallelism", {})
                        )
                        self._execution_config = ExecutionConfig(
                            parallelism=parallelism,
                            large_model_threshold_mb=int(
                                config_data["large_model_threshold_mb"]
                            ),
                            on_error=on_error,
                        )
                    except (KeyError, ValueError) as exc:
                        raise ConfigurationError(
                            "config.json/settings.yaml is missing a required field "
                            "or has an invalid enum value.",
                            code="INVALID_EXECUTION_CONFIG",
                            remediation_hint=(
                                "Validate config.json against 06_Config_Spec.md §1 "
                                "and settings.yaml against §2."
                            ),
                        ) from exc

        return ExecutionConfig(
            parallelism=dict(self._execution_config.parallelism),
            large_model_threshold_mb=self._execution_config.large_model_threshold_mb,
            on_error=self._execution_config.on_error,
        )

    def load_optimization_config(self) -> OptimizationConfig:
        """Load ``optimization.json`` into an ``OptimizationConfig``.

        See the module docstring for why this file's schema is
        inferred from sibling config files rather than sourced from
        ``06_Config_Spec.md`` directly.

        Returns:
            A defensive copy of the parsed, validated
            ``OptimizationConfig``. Mutating the returned instance's
            ``objectives`` list or ``objective_weights`` dict never
            affects this repository's cache.

        Raises:
            ConfigurationError: If ``optimization.json`` is missing or
                malformed.
        """
        if self._optimization_config is None:
            with self._optimization_lock:
                if self._optimization_config is None:
                    data = self._load_json("optimization.json")
                    try:
                        self._optimization_config = OptimizationConfig(
                            objectives=list(data["objectives"]),
                            objective_weights={
                                str(name): float(weight)
                                for name, weight in data.get(
                                    "objective_weights", {}
                                ).items()
                            },
                            max_search_iterations=int(data["max_search_iterations"]),
                        )
                    except (KeyError, ValueError) as exc:
                        raise ConfigurationError(
                            "optimization.json is missing a required field or has "
                            "an invalid value.",
                            code="INVALID_OPTIMIZATION_CONFIG",
                            remediation_hint=(
                                "Validate optimization.json's objectives, "
                                "objective_weights, and max_search_iterations fields."
                            ),
                        ) from exc

        return OptimizationConfig(
            objectives=list(self._optimization_config.objectives),
            objective_weights=dict(self._optimization_config.objective_weights),
            max_search_iterations=self._optimization_config.max_search_iterations,
        )

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Read, parse, cache, and schema-validate a JSON file under ``base_path``.

        Args:
            filename: The JSON file name, relative to ``base_path``.

        Returns:
            The parsed JSON content as a mapping.

        Raises:
            ConfigurationError: If the file is missing, is not valid
                JSON, or declares an unrecognized ``schema_version``.
        """
        with self._file_lock:
            if filename in self._json_cache:
                return self._json_cache[filename]

            path = Path(self.base_path) / filename
            data = self._json_loader.load(path)
            self._schema_validator.validate_schema_version(data, filename)
            self._json_cache[filename] = data
            return data

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Read, parse, cache, and schema-validate a YAML file under ``base_path``.

        Args:
            filename: The YAML file name, relative to ``base_path``.

        Returns:
            The parsed YAML content as a mapping.

        Raises:
            ConfigurationError: If the file is missing, is not valid
                YAML, or declares an unrecognized ``schema_version``.
        """
        with self._file_lock:
            if filename in self._yaml_cache:
                return self._yaml_cache[filename]

            path = Path(self.base_path) / filename
            data = self._yaml_loader.load(path)
            self._schema_validator.validate_schema_version(data, filename)
            self._yaml_cache[filename] = data
            return data
