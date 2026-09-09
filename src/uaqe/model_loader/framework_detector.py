"""Extension-to-framework resolution backing
``ModelLoader.detect_framework()``.

Per ``01_Project_Architecture.md`` §5 step 2, "Framework Detection" is a
distinct workflow step from adapter selection (step 3 onward, handled by
:mod:`uaqe.domain.model.loader_factory`). ``FrameworkDetector`` isolates
that one responsibility — mapping a source file's extension to the
canonical framework name — so ``ModelLoader`` itself stays a thin
coordinator, per ``07_Coding_Standards.md`` §9's one-responsibility-per-
module rule.

``FrameworkDetector`` is an internal implementation detail of
``uaqe.domain.model``, not a class in the locked inventory of
``09_Architecture_Lock.md`` §3 (``uaqe.domain.model``: ``ModelLoader``,
``ModelValidator``, ``AnalysisResult``, ``ModelAnalyzer``). It exists to
satisfy ``09_Architecture_Lock.md`` §11's "Internal API" carve-out
(helper classes not listed in ``03_API_Specification.md`` may be added
during implementation, provided no locked public signature changes) and
backs the already-locked ``ModelLoader.detect_framework(path: str) ->
str`` method (``03_API_Specification.md`` §3.1) without altering it.
It performs extension recognition only; it never imports a concrete
``IFrameworkAdapter`` or any ``uaqe.infrastructure`` module
(``10_Module_Development_Guide.md`` §3 Integration Checklist).

Locked contract backed: ``03_API_Specification.md`` §3.1.
"""

from __future__ import annotations

from typing import Dict

from uaqe.common.exceptions import ModelLoadError

# Extension -> canonical framework name, per 02_Folder_Structure.md §5
# and 11_Implementation_Rules.md §6.1's Framework Factory mapping:
# .pth/.pt -> torch, .onnx -> onnx, .pb -> tensorflow, .h5/.keras -> keras,
# .tflite -> tflite.
_EXTENSION_TO_FRAMEWORK: Dict[str, str] = {
    ".pth": "pytorch",
    ".pt": "pytorch",
    ".onnx": "onnx",
    ".pb": "tensorflow",
    ".h5": "keras",
    ".keras": "keras",
    ".tflite": "tflite",
}


class FrameworkDetector:
    """Resolves the canonical source-framework name for a model file.

    Stateless: holds no adapter references and no I/O handles, so a
    single instance may be shared and reused across every run.
    """

    def detect(self, path: str) -> str:
        """Resolve the canonical framework name for the file at ``path``.

        Args:
            path: Filesystem path to the source model file.

        Returns:
            The canonical framework name (``"pytorch"``, ``"onnx"``,
            ``"tensorflow"``, ``"keras"``, or ``"tflite"``) matching
            ``path``'s extension.

        Raises:
            ModelLoadError: If ``path`` has no extension, or its
                extension does not match any known source-framework
                format.
        """
        extension = self.extension_of(path)
        framework = _EXTENSION_TO_FRAMEWORK.get(extension)
        if framework is None:
            raise ModelLoadError(
                f"Unrecognized model file extension {extension!r} for {path!r}.",
                code="MODEL_FRAMEWORK_UNRECOGNIZED",
                remediation_hint=(
                    "Supported extensions are .pth, .pt, .onnx, .pb, .h5, "
                    ".keras, and .tflite."
                ),
            )
        return framework

    def extension_of(self, path: str) -> str:
        """Return the lowercase file extension (including the dot) of ``path``.

        Args:
            path: Filesystem path to inspect.

        Returns:
            The lowercase extension, e.g. ``".onnx"``.

        Raises:
            ModelLoadError: If ``path`` has no extension to resolve.
        """
        index = path.rfind(".")
        if index == -1:
            raise ModelLoadError(
                f"Cannot determine a file extension for {path!r}.",
                code="MODEL_EXTENSION_MISSING",
                remediation_hint="Provide a model file with a recognized extension.",
            )
        return path[index:].lower()
