"""Exporter-backend port consumed by ``uaqe.domain.export.Exporter``.

Concrete backends under ``uaqe.infrastructure.exporter_backends``
implement this interface, one per supported board/target, so that
``Exporter`` can select and invoke the right backend for a resolved
``HardwareProfile`` without depending on any backend's concrete
encoding logic.

Note: ``HardwareProfile`` and ``DeploymentArtifact`` are defined in
``uaqe.domain`` (``domain.hardware`` and ``domain.export``
respectively). Per ``09_Architecture_Lock.md`` §8, ``uaqe.common`` has
no dependency on ``uaqe.domain``, so both types are referenced here as
``TYPE_CHECKING``-only forward references rather than runtime imports —
mirroring the ``"PipelineContext"`` forward reference already used for
``IReportRenderer`` in ``03_API_Specification.md`` §1.11.

Locked contract: ``03_API_Specification.md`` §1.10.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from uaqe.common.imr import IMR

if TYPE_CHECKING:
    from uaqe.domain.export.exporter import DeploymentArtifact
    from uaqe.domain.hardware.hardware_manager import HardwareProfile


class IExporterBackend(ABC):
    """Abstract port for exporting an ``IMR`` to a deployment artifact
    for a specific hardware target.
    """

    @abstractmethod
    def export(self, imr: IMR, target: "HardwareProfile") -> "DeploymentArtifact":
        """Export ``imr`` as a deployment artifact for ``target``.

        Args:
            imr: The final, fully-optimized model to export.
            target: The resolved hardware profile to export for.

        Returns:
            The produced ``DeploymentArtifact``.

        Raises:
            ExportError: If export cannot be completed for ``target``.
        """
        raise NotImplementedError

    @abstractmethod
    def supported_targets(self) -> List[str]:
        """Return the ``HardwareProfile.profile_id`` values this backend
        supports.
        """
        raise NotImplementedError
