"""Raw YAML file reading for the Configuration module.

``YamlConfigLoader`` is an internal collaborator of
:class:`uaqe.infrastructure.repositories.config_repository.ConfigRepository`,
used only for ``settings.yaml`` (``06_Config_Spec.md`` §2). It is
**not** part of the locked ``uaqe.infrastructure.repositories`` class
inventory (``09_Architecture_Lock.md`` §8) — see the module docstring
of :mod:`uaqe.infrastructure.repositories.json_loader` for the
rationale, which applies identically here. It is never imported
outside this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from uaqe.common.exceptions import ConfigurationError


class YamlConfigLoader:
    """Reads and parses a single YAML config file into a mapping."""

    def load(self, path: Path) -> Dict[str, Any]:
        """Read and parse ``path`` as YAML.

        Args:
            path: The path to the YAML file to read.

        Returns:
            The parsed YAML content as a mapping, or an empty mapping
            if the file parses to ``None`` (e.g. an empty file).

        Raises:
            ConfigurationError: If the file is missing or unreadable
                (``CONFIG_FILE_MISSING``), or its content is not valid
                YAML (``CONFIG_FILE_MALFORMED``).
        """
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Required config file {path.name!r} is missing or unreadable.",
                code="CONFIG_FILE_MISSING",
                remediation_hint=f"Create {path.name} under {path.parent}.",
            ) from exc

        try:
            data: Dict[str, Any] = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Config file {path.name!r} is not valid YAML.",
                code="CONFIG_FILE_MALFORMED",
                remediation_hint=f"Validate {path.name}'s YAML syntax.",
            ) from exc

        return data
