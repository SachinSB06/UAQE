"""Framework-adapter port consumed by ``uaqe.domain.model.ModelLoader``.

Every source-framework loader under
``uaqe.infrastructure.framework_adapters`` implements this interface so
that ``ModelLoader`` never needs to know whether a model originated from
PyTorch, TensorFlow, Keras, ONNX, or TFLite.

Locked contract: ``03_API_Specification.md`` §1.7.
"""

from abc import ABC, abstractmethod

from uaqe.common.imr import IMR


class IFrameworkAdapter(ABC):
    """Abstract port for loading a source-framework model file into the
    framework-independent ``IMR``.
    """

    @abstractmethod
    def load(self, path: str) -> IMR:
        """Load the model at ``path`` and return its ``IMR``.

        Args:
            path: Filesystem path to the source model file.

        Returns:
            The loaded model as an ``IMR``.

        Raises:
            ModelLoadError: If ``path`` does not exist, is corrupt, or
                is not in a format this adapter supports.
        """
        raise NotImplementedError

    @abstractmethod
    def supports(self, extension: str) -> bool:
        """Report whether this adapter can load files with ``extension``.

        Args:
            extension: A file extension, including the leading dot (e.g.
                ``".onnx"``).

        Returns:
            ``True`` if this adapter supports ``extension``.
        """
        raise NotImplementedError
