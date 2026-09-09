"""Adapter resolution backing ``ModelLoader``'s adapter-selection step.

Implements the "Framework Factory" pattern described in
``11_Implementation_Rules.md`` §6.1: resolves an ``IFrameworkAdapter``
from the constructor-injected ``adapters`` list by extension, so that
adding a new input format never requires modifying ``ModelLoader`` —
per ``10_Module_Development_Guide.md`` §3's Future Extensions note,
only a new adapter needs to be registered with the factory.

``LoaderFactory`` is an internal implementation detail of
``uaqe.domain.model``, not a class in the locked inventory of
``09_Architecture_Lock.md`` §3 — see the same "Internal API" carve-out
(§11) discussed in ``framework_detector.py``. It never imports a
concrete ``IFrameworkAdapter`` implementation; it only operates on
instances handed to it through ``ModelLoader``'s constructor injection
(``10_Module_Development_Guide.md`` §3, Dependencies).

Locked contract backed: ``03_API_Specification.md`` §1.7 (``IFrameworkAdapter``),
``04_Data_Flow.md`` §2 (adapter-selection step of the Input Flow).
"""

from __future__ import annotations

from typing import List

from uaqe.common.exceptions import ModelLoadError
from uaqe.common.interfaces.i_framework_adapter import IFrameworkAdapter


class LoaderFactory:
    """Resolves the ``IFrameworkAdapter`` that supports a given extension.

    Attributes:
        _adapters: The full, constructor-injected set of adapters this
            factory may resolve from, in the order they were provided.
    """

    def __init__(self, adapters: List[IFrameworkAdapter]) -> None:
        """Initialize the factory with the run's available adapters.

        Args:
            adapters: The ``IFrameworkAdapter`` instances to resolve
                from, as injected into ``ModelLoader`` by
                ``CompositionRoot``.
        """
        self._adapters: List[IFrameworkAdapter] = adapters

    def resolve(self, extension: str) -> IFrameworkAdapter:
        """Return the first adapter that supports ``extension``.

        Per ``04_Data_Flow.md`` §2, selection takes the first adapter
        in ``self._adapters`` for which ``adapter.supports(extension)``
        is ``True`` — adapter order is therefore significant when more
        than one adapter could plausibly claim the same extension.

        Args:
            extension: A file extension, including the leading dot
                (e.g. ``".onnx"``), as resolved by
                :class:`~uaqe.domain.model.framework_detector.FrameworkDetector`.

        Returns:
            The first matching ``IFrameworkAdapter``.

        Raises:
            ModelLoadError: If no injected adapter supports
                ``extension``.
        """
        for adapter in self._adapters:
            if adapter.supports(extension):
                return adapter
        raise ModelLoadError(
            f"No registered IFrameworkAdapter supports extension {extension!r}.",
            code="MODEL_ADAPTER_NOT_FOUND",
            remediation_hint=(
                "Register an IFrameworkAdapter for this extension via "
                "CompositionRoot, or via PluginRegistry for an external plugin."
            ),
        )
