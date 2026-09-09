"""Abstract base class for every stage in the 19-stage pipeline defined
in ``09_Architecture_Lock.md`` §12.

Every concrete stage across ``uaqe.domain.{model,compatibility,hardware,
quantization,compression,optimization,export,evaluation,benchmark,
advisory,reporting}`` subclasses ``PipelineStage``. Per
``03_API_Specification.md`` §22 rule 3, ``execute()`` reads its inputs
only via ``PipelineContext.get(...)`` and always returns a
``StageResult`` — never ``None``, never a raw domain object.

Locked contract: ``03_API_Specification.md`` §2.2.
"""

from abc import ABC, abstractmethod

from uaqe.common.result_types import StageResult
from uaqe.domain.pipeline_context import PipelineContext


class PipelineStage(ABC):
    """Abstract base class for a single pipeline stage."""

    @abstractmethod
    def execute(self, context: PipelineContext) -> StageResult:
        """Execute this stage's work and return its result.

        Args:
            context: The current run's pipeline context, from which
                this stage reads any prior stage's ``StageResult`` it
                depends on via ``context.get(...)``.

        Returns:
            This stage's ``StageResult``. Implementations do not append
            it to ``context`` themselves — ``PipelineOrchestrator`` owns
            that responsibility after a successful call.
        """
        raise NotImplementedError

    def name(self) -> str:
        """Return this stage's registration/context-key name.

        The default implementation derives the name from the concrete
        stage's context key by looking it up in the locked mapping in
        ``09_Architecture_Lock.md`` §9 (``snake_case`` of the class,
        e.g. ``ModelLoader`` -> ``"model_loader"``). Concrete stages may
        override this if their class name does not already follow that
        convention.

        Returns:
            The stage's context-key name.
        """
        class_name = type(self).__name__
        result_chars = []
        for index, char in enumerate(class_name):
            if char.isupper() and index > 0:
                result_chars.append("_")
            result_chars.append(char.lower())
        return "".join(result_chars)
