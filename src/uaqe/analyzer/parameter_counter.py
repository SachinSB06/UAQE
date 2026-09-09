"""Per-layer and total learnable-parameter counting for an analyzed ``IMR``.

Computes the counts consumed by ``ModelAnalyzer.analyze()`` to populate
``AnalysisResult.parameter_count`` (``03_API_Specification.md`` §3.3),
by summing the element count of every ``IMRTensor`` held in each
``IMRLayer.parameters`` mapping.

Internal helper for ``uaqe.analyzer`` (``09_Architecture_Lock.md`` §11,
"Internal API") — not part of the locked class inventory; the only
locked classes in this analysis area are ``AnalysisResult`` and
``ModelAnalyzer`` (``model_analyzer.py``).
"""

from __future__ import annotations

from math import prod
from typing import Dict

from uaqe.common.imr import IMR, IMRLayer


class ParameterCounter:
    """Counts learnable-parameter elements per layer and for a whole ``IMR``."""

    def count_layer(self, layer: IMRLayer) -> int:
        """Count the total parameter elements held by a single layer.

        Args:
            layer: The layer whose ``parameters`` tensors to count.

        Returns:
            The sum, across every tensor in ``layer.parameters``, of
            that tensor's element count (the product of its ``shape``;
            a rank-0/scalar tensor counts as one element).
        """
        total = 0
        for tensor in layer.parameters.values():
            total += prod(tensor.shape) if tensor.shape else 1
        return total

    def count_total(self, imr: IMR) -> int:
        """Count total parameter elements across every layer of ``imr``.

        Args:
            imr: The model representation to count parameters for.

        Returns:
            The sum of :meth:`count_layer` over every layer in
            ``imr.layers``.
        """
        return sum(self.count_layer(layer) for layer in imr.layers)

    def count_per_layer(self, imr: IMR) -> Dict[str, int]:
        """Build a per-layer parameter-count breakdown.

        Args:
            imr: The model representation to break down.

        Returns:
            A mapping of ``IMRLayer.name`` to that layer's parameter
            element count, in ``imr.layers`` order.
        """
        return {layer.name: self.count_layer(layer) for layer in imr.layers}
