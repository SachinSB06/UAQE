"""Report-renderer port consumed by
``uaqe.domain.reporting.report_generator.ReportGenerator``.

Concrete renderers implement this interface so that ``ReportGenerator``
can produce one ``ReportDocument`` per registered renderer from a
completed run's ``PipelineContext`` without needing to know the specific
output format (e.g. HTML, Markdown, JSON) any renderer produces.

Note: ``PipelineContext`` is defined in
``uaqe.domain.pipeline.pipeline_context`` and ``ReportDocument`` in
``uaqe.domain.reporting.report_generator``. Per ``09_Architecture_Lock.md``
§8, ``uaqe.common`` has no dependency on ``uaqe.domain``, so both types
are referenced here as ``TYPE_CHECKING``-only forward references — the
same quoted-annotation pattern already used for ``"HardwareProfile"`` in
``i_hardware_profile_repository.py`` and ``i_exporter_backend.py``.

Locked contract: ``03_API_Specification.md`` §1.11.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uaqe.domain.pipeline_context import PipelineContext
    from uaqe.domain.reporting.report_generator import ReportDocument


class IReportRenderer(ABC):
    """Abstract port for rendering a completed run's ``PipelineContext``
    into a single ``ReportDocument``.
    """

    @abstractmethod
    def render(self, context: "PipelineContext") -> "ReportDocument":
        """Render ``context`` into a ``ReportDocument``.

        Args:
            context: The ``PipelineContext`` of the run being reported
                on, carrying every stage's ``StageResult`` and any
                accumulated errors.

        Returns:
            The rendered ``ReportDocument``.
        """
        raise NotImplementedError

    @abstractmethod
    def report_type(self) -> str:
        """Return this renderer's unique registration name (e.g.
        ``"html"``, ``"markdown"``, ``"json"``).
        """
        raise NotImplementedError
