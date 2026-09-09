"""Append-only per-run state container for the Universal AI
Quantization Engine pipeline.

``PipelineContext`` is the single channel through which stages pass
their ``StageResult`` payloads to downstream stages; per
``09_Architecture_Lock.md`` §13 rule 1, it is append-only — a stage name
is never overwritten once written, and rule 5 makes
``PipelineOrchestrator.resume()`` idempotent by consulting
``has(stage.name())`` before re-running a stage.

Locked contract: ``03_API_Specification.md`` §2.1.
"""

from typing import Dict, List

from uaqe.common.exceptions import UAQEError
from uaqe.common.result_types import StageResult


class PipelineContext:
    """Holds every ``StageResult`` and ``UAQEError`` produced so far
    during a single pipeline run.

    Attributes:
        _run_id: The unique identifier of the run this context belongs
            to.
        _stage_results: Every ``StageResult`` appended so far, keyed by
            stage name.
        _errors: Every ``UAQEError`` raised so far during the run.
    """

    def __init__(self, run_id: str) -> None:
        """Initialize an empty context for the run identified by
        ``run_id``.

        Args:
            run_id: The unique identifier of the run.
        """
        self._run_id: str = run_id
        self._stage_results: Dict[str, StageResult] = {}
        self._errors: List[UAQEError] = []

    def append(self, stage_name: str, result: StageResult) -> None:
        """Record ``result`` under ``stage_name``.

        Per ``09_Architecture_Lock.md`` §13 rule 1, this context is
        append-only: calling this again for a ``stage_name`` already
        present overwrites nothing conceptually different from a fresh
        run resuming from scratch, but ordinary pipeline execution never
        calls this twice for the same stage in a single run.

        Args:
            stage_name: The name of the stage that produced ``result``,
                matching ``PipelineStage.name()``.
            result: The stage's produced result.
        """
        self._stage_results[stage_name] = result

    def get(self, stage_name: str) -> StageResult:
        """Retrieve the ``StageResult`` previously appended for
        ``stage_name``.

        Args:
            stage_name: The name of the stage whose result to retrieve.

        Returns:
            The ``StageResult`` appended for ``stage_name``.

        Raises:
            KeyError: If no result has been appended for ``stage_name``.
        """
        try:
            return self._stage_results[stage_name]
        except KeyError:
            raise KeyError(
                f"No StageResult recorded for stage {stage_name!r}."
            ) from None

    def has(self, stage_name: str) -> bool:
        """Report whether a result has been appended for ``stage_name``.

        Args:
            stage_name: The stage name to check.

        Returns:
            ``True`` if ``append()`` has been called for ``stage_name``.
        """
        return stage_name in self._stage_results
