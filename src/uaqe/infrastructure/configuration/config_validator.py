"""``schema_version`` validation for the Configuration module.

``ConfigSchemaValidator`` is an internal collaborator of
:class:`uaqe.infrastructure.repositories.config_repository.ConfigRepository`.
It is **not** part of the locked ``uaqe.infrastructure.repositories``
class inventory (``09_Architecture_Lock.md`` §8) — see the module
docstring of :mod:`uaqe.infrastructure.repositories.json_loader` for
the rationale, which applies identically here. It is never imported
outside this package.

Implements ``06_Config_Spec.md`` §9 rule 1: every ``config/*.json``
file (and ``settings.yaml``) must declare ``schema_version``;
``ConfigRepository`` raises ``ConfigurationError`` if it is absent or
if its major version component is unrecognized (the same rule
``HardwareProfileRepository`` applies to hardware profile files, per
``05_Hardware_Profile_Spec.md`` §7).
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet

from uaqe.common.exceptions import ConfigurationError

#: Major ``schema_version`` components this build of ``uaqe`` accepts,
#: per ``06_Config_Spec.md`` §9 rule 1.
DEFAULT_SUPPORTED_SCHEMA_MAJOR_VERSIONS: FrozenSet[str] = frozenset({"1"})


class ConfigSchemaValidator:
    """Validates the ``schema_version`` field common to every config file.

    Attributes:
        supported_major_versions: The set of major ``schema_version``
            components (e.g. ``"1"`` for ``"1.0"``) this validator
            accepts.
    """

    def __init__(
        self,
        supported_major_versions: FrozenSet[
            str
        ] = DEFAULT_SUPPORTED_SCHEMA_MAJOR_VERSIONS,
    ) -> None:
        """Initialize a ``ConfigSchemaValidator``.

        Args:
            supported_major_versions: The set of major
                ``schema_version`` components this validator accepts.
                Defaults to :data:`DEFAULT_SUPPORTED_SCHEMA_MAJOR_VERSIONS`.
        """
        self.supported_major_versions: FrozenSet[str] = supported_major_versions

    def validate_schema_version(self, data: Dict[str, Any], filename: str) -> None:
        """Reject a config mapping with a missing or unrecognized ``schema_version``.

        Args:
            data: The parsed config mapping to validate.
            filename: The source file name, used for error context.

        Raises:
            ConfigurationError: If ``schema_version`` is absent, is not
                of the form ``"<major>.<minor>"``, or its major version
                component is not in :attr:`supported_major_versions`.
        """
        schema_version = data.get("schema_version")
        if not schema_version or "." not in str(schema_version):
            raise ConfigurationError(
                f"Config file {filename!r} is missing a valid schema_version.",
                code="UNRECOGNIZED_SCHEMA_VERSION",
                remediation_hint=f"Add a schema_version field to {filename}.",
            )
        major_version = str(schema_version).split(".")[0]
        if major_version not in self.supported_major_versions:
            raise ConfigurationError(
                f"Config file {filename!r} declares unsupported schema_version "
                f"{schema_version!r}.",
                code="UNRECOGNIZED_SCHEMA_VERSION",
                remediation_hint=(
                    f"Supported major versions: {sorted(self.supported_major_versions)}."
                ),
            )
