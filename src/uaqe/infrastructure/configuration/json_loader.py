"""Raw JSON file reading for the Configuration module.

``JsonConfigLoader`` is an internal collaborator of
:class:`uaqe.infrastructure.repositories.config_repository.ConfigRepository`.
It is **not** part of the locked ``uaqe.infrastructure.repositories``
class inventory (``09_Architecture_Lock.md`` §8, which names exactly
``ConfigRepository``, ``HardwareProfileRepository``,
``FilesystemRepository``) — it exists purely to separate "how a JSON
file is read off disk" from "how ``ConfigRepository`` caches, validates,
and maps that content onto a locked ``*Config`` value object". It is
never imported outside this package.

This loader deliberately knows nothing about caching, ``schema_version``
validation, or any specific config file's expected shape — see
:class:`uaqe.infrastructure.repositories.config_validator.ConfigSchemaValidator`
for schema-version validation, which ``ConfigRepository`` applies to
this loader's output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from uaqe.common.exceptions import ConfigurationError


class JsonConfigLoader:
    """Reads and parses a single JSON config file into a mapping."""

    def load(self, path: Path) -> Dict[str, Any]:
        """Read and parse ``path`` as JSON.

        Args:
            path: The path to the JSON file to read.

        Returns:
            The parsed JSON content as a mapping.

        Raises:
            ConfigurationError: If the file is missing or unreadable
                (``CONFIG_FILE_MISSING``), or its content is not valid
                JSON (``CONFIG_FILE_MALFORMED``).
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
            data: Dict[str, Any] = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Config file {path.name!r} is not valid JSON.",
                code="CONFIG_FILE_MALFORMED",
                remediation_hint=f"Validate {path.name}'s JSON syntax.",
            ) from exc

        return data
