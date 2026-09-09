"""Layer-graph structural summarization for an analyzed ``IMR``.

Feeds ``AnalysisResult.layer_graph_summary`` and
``AnalysisResult.op_type_histogram`` (``03_API_Specification.md``
§3.3). Internal helper for ``uaqe.analyzer``; not part of the locked
class inventory (``09_Architecture_Lock.md`` §11).
"""

from __future__ import annotations

from math import prod
from typing import Any, Dict, List

from uaqe.common.imr import IMR, IMRLayer, IMRTensor


def _element_count(tensor: IMRTensor) -> int:
    """Return ``tensor``'s element count, treating a scalar shape as one element."""
    return prod(tensor.shape) if tensor.shape else 1


class LayerAnalyzer:
    """Summarizes an ``IMR``'s layer graph: op histogram, per-layer
    descriptors, and structural (depth) metrics."""

    def op_type_histogram(self, imr: IMR) -> Dict[str, int]:
        """Count how many layers of each ``op_type`` appear in ``imr``.

        Args:
            imr: The model representation to summarize.

        Returns:
            A mapping of ``op_type`` to the number of layers with that
            type, in first-seen order.
        """
        histogram: Dict[str, int] = {}
        for layer in imr.layers:
            histogram[layer.op_type] = histogram.get(layer.op_type, 0) + 1
        return histogram

    def per_layer_summary(self, imr: IMR) -> List[Dict[str, Any]]:
        """Build a compact per-layer descriptor list, in topological order.

        Args:
            imr: The model representation to summarize.

        Returns:
            One dict per layer (``name``, ``op_type``, ``precision``,
            ``input_count``, ``output_count``, ``parameter_count``),
            ordered by ``IMR.topological_order``.
        """
        summaries: List[Dict[str, Any]] = []
        for layer in imr.topological_order():
            summaries.append(
                {
                    "name": layer.name,
                    "op_type": layer.op_type,
                    "precision": layer.precision.value,
                    "input_count": len(layer.inputs),
                    "output_count": len(layer.outputs),
                    "parameter_count": sum(
                        _element_count(tensor) for tensor in layer.parameters.values()
                    ),
                }
            )
        return summaries

    def graph_depth(self, imr: IMR) -> int:
        """Compute the longest producer/consumer dependency chain in ``imr``.

        Args:
            imr: The model representation to measure.

        Returns:
            The number of layers along the longest dependency chain
            (``1`` for a single-layer or fully-parallel graph, ``0``
            for an empty graph).
        """
        order = imr.topological_order()
        if not order:
            return 0

        producer_of: Dict[str, str] = {}
        for layer in order:
            for output_name in layer.outputs:
                producer_of[output_name] = layer.name

        depth_by_name: Dict[str, int] = {}
        for layer in order:
            producer_names = {
                producer_of[input_name]
                for input_name in layer.inputs
                if input_name in producer_of
            }
            depth_by_name[layer.name] = 1 + max(
                (depth_by_name[name] for name in producer_names), default=0
            )
        return max(depth_by_name.values())

    def build_summary(self, imr: IMR) -> Dict[str, Any]:
        """Build the full ``layer_graph_summary`` payload.

        Args:
            imr: The model representation to summarize.

        Returns:
            A dict with keys ``layer_count``, ``op_type_histogram``,
            ``graph_depth``, and ``layers`` (the
            :meth:`per_layer_summary` list), matching the shape
            consumed by ``AnalysisResult.layer_graph_summary``
            (``03_API_Specification.md`` §3.3).
        """
        return {
            "layer_count": len(imr.layers),
            "op_type_histogram": self.op_type_histogram(imr),
            "graph_depth": self.graph_depth(imr),
            "layers": self.per_layer_summary(imr),
        }
