"""Hardware-profile repository port consumed by
``uaqe.domain.hardware.HardwareManager`` and
``uaqe.application.hardware_selection_service.HardwareSelectionService``.

``uaqe.infrastructure.repositories.HardwareProfileRepository`` is the
sole first-party implementation, injected via ``CompositionRoot``.

Note: ``HardwareProfile`` is defined in ``uaqe.domain.hardware``. Per
``09_Architecture_Lock.md`` §8, ``uaqe.common`` has no dependency on
``uaqe.domain``, so it is referenced here as a ``TYPE_CHECKING``-only
forward reference, matching the quoted-annotation pattern established
by ``IReportRenderer`` in ``03_API_Specification.md`` §1.11.

Locked contract: ``03_API_Specification.md`` §1.12.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from uaqe.domain.hardware_manager import HardwareProfile


class IHardwareProfileRepository(ABC):
    """Abstract port for resolving ``HardwareProfile`` instances by id."""

    @abstractmethod
    def get(self, profile_id: str) -> "HardwareProfile":
        """Resolve the ``HardwareProfile`` identified by ``profile_id``.

        Args:
            profile_id: The unique identifier of the hardware profile.

        Returns:
            The resolved ``HardwareProfile``.

        Raises:
            ConfigurationError: If ``profile_id`` cannot be resolved.
        """
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> List["HardwareProfile"]:
        """Return every ``HardwareProfile`` known to this repository."""
        raise NotImplementedError
