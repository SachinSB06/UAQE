"""``ModelRepository`` — read-only resolution of raw input model files
by id, for hand-off to ``ModelLoader``/``IFrameworkAdapter``.

.. important::
    **This class is not part of the locked architecture.**
    ``09_Architecture_Lock.md`` §8's module ownership table for
    ``uaqe.infrastructure.repositories`` names exactly
    ``ConfigRepository``, ``HardwareProfileRepository``,
    ``FilesystemRepository`` — no ``ModelRepository``. Nothing in
    ``02_Folder_Structure.md`` or ``03_API_Specification.md`` §3.1
    defines this class; ``ModelLoader`` (the locked pipeline stage)
    takes a raw ``path: str`` directly — via
    ``PipelineContext``/CLI input — and delegates to
    ``IFrameworkAdapter.load()``, with no repository in between. This
    file was generated on explicit request despite that gap; it should
    be treated as a proposal, and formalized via the RFC process (or
    removed) before being wired into ``CompositionRoot`` or
    ``ModelLoader``.

Rationale for the shape below: no folder in ``02_Folder_Structure.md``
is designated as the owned root for arbitrary user-supplied input
model files (only ``tests/sample_models/`` exists, as CI fixtures), so
this is *not* one of the roots ``FilesystemRepository`` is exclusively
responsible for (``workdir/``, ``outputs/``, ``reports/``, ``logs/``
per ``11_Implementation_Rules.md`` §5.3). ``ModelRepository`` gives
``ModelLoader`` (or a future caller resolving a ``model_id`` rather
than an ad hoc path) a single, path-contained place to look up an
input model's bytes and detect its file extension — mirroring the
same containment discipline ``FilesystemRepository`` applies to its
own roots, applied here to a configured input-models directory
instead. Per ``10_Module_Development_Guide.md`` §0 rule 7's spirit
(config/data access stays inside a dedicated repository), this keeps
``ModelLoader`` itself free of any ``os.path``/``pathlib`` calls
against this directory, exactly as it already is against every other
root.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from uaqe.common.exceptions import ModelLoadError


class ModelRepository:
    """Path-contained, read-only access to input model files by id.

    Attributes:
        models_path: The absolute, resolved root directory every
            ``model_id`` is resolved against. No path this repository
            reads may resolve outside of it.
    """

    def __init__(self, models_path: str) -> None:
        """Initialize a ``ModelRepository`` rooted at ``models_path``.

        Args:
            models_path: The root directory containing every input
                model file this repository resolves.

        Raises:
            ModelLoadError: If ``models_path`` does not exist or is
                not a directory.
        """
        resolved = Path(models_path).resolve()
        if not resolved.is_dir():
            raise ModelLoadError(
                f"Model repository root {models_path!r} does not exist or is "
                "not a directory.",
                code="MODEL_REPOSITORY_ROOT_UNAVAILABLE",
                remediation_hint=(
                    f"Create the directory {models_path} before constructing "
                    "ModelRepository."
                ),
            )
        self.models_path: Path = resolved

    def get_model_path(self, model_id: str) -> str:
        """Resolve ``model_id`` to an absolute path of an existing file.

        Args:
            model_id: A path relative to ``models_path`` identifying
                one model file, e.g. ``"resnet18.onnx"`` or
                ``"vision/mnist.pth"``.

        Returns:
            The resolved absolute filesystem path, as a string.

        Raises:
            ModelLoadError: If ``model_id`` would resolve outside
                ``models_path`` (path traversal attempt), or if no
                file exists at the resolved path.
        """
        resolved = self._resolve(model_id)
        if not resolved.is_file():
            raise ModelLoadError(
                f"No model file found for model_id {model_id!r} under "
                f"{self.models_path}.",
                code="MODEL_NOT_FOUND",
                remediation_hint=(
                    f"Verify {model_id} exists under {self.models_path}."
                ),
            )
        return str(resolved)

    def read_model_bytes(self, model_id: str) -> bytes:
        """Read and return the raw bytes of the model identified by ``model_id``.

        Args:
            model_id: A path relative to ``models_path`` identifying
                one model file.

        Returns:
            The file's raw byte content, ready to be handed to an
            ``IFrameworkAdapter``.

        Raises:
            ModelLoadError: If ``model_id`` would resolve outside
                ``models_path``, if no file exists at the resolved
                path, or if the read fails for any other reason (e.g.
                permission denied).
        """
        resolved = Path(self.get_model_path(model_id))
        try:
            return resolved.read_bytes()
        except OSError as exc:
            raise ModelLoadError(
                f"Cannot read model file for model_id {model_id!r}.",
                code="MODEL_READ_FAILED",
                remediation_hint="Verify the source file is readable.",
            ) from exc

    def detect_extension(self, model_id: str) -> str:
        """Return the lowercase file extension of ``model_id``, without the dot.

        A thin convenience for callers (e.g. ``ModelLoader.detect_framework()``)
        that need the raw extension without duplicating path parsing;
        this makes no claim about which framework the extension maps
        to — that mapping is ``ModelLoader``'s responsibility.

        Args:
            model_id: A path relative to ``models_path`` identifying
                one model file.

        Returns:
            The file extension in lowercase, without the leading dot
            (e.g. ``"onnx"``), or an empty string if there is none.

        Raises:
            ModelLoadError: If ``model_id`` would resolve outside
                ``models_path``, or if no file exists at the resolved
                path.
        """
        resolved = Path(self.get_model_path(model_id))
        return resolved.suffix.lower().lstrip(".")

    def list_models(self) -> List[str]:
        """Return every model file's id, relative to ``models_path``.

        Returns:
            Every regular file under ``models_path`` (recursively), as
            ``model_id`` strings usable with :meth:`get_model_path`,
            :meth:`read_model_bytes`, and :meth:`detect_extension`.
        """
        return sorted(
            str(path.relative_to(self.models_path))
            for path in self.models_path.rglob("*")
            if path.is_file()
        )

    def _resolve(self, model_id: str) -> Path:
        """Resolve ``model_id`` against ``models_path``, enforcing containment.

        Args:
            model_id: The caller-supplied relative path.

        Returns:
            The resolved absolute ``Path``, guaranteed to lie inside
            ``models_path``.

        Raises:
            ModelLoadError: If the resolved path would lie outside
                ``models_path`` (path traversal attempt).
        """
        candidate = (self.models_path / model_id).resolve()
        if not candidate.is_relative_to(self.models_path):
            raise ModelLoadError(
                f"model_id {model_id!r} resolves outside the repository root "
                f"{self.models_path}.",
                code="PATH_TRAVERSAL_REJECTED",
                remediation_hint="Use a model_id that stays within the configured root.",
            )
        return candidate
