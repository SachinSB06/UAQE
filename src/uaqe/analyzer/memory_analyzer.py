"""Memory-footprint estimation for an analyzed ``IMR``.

Feeds ``AnalysisResult.estimated_memory_bytes``
(``03_API_Specification.md`` §3.3). Internal helper for
``uaqe.analyzer``; not part of the locked class inventory
(``09_Architecture_Lock.md`` §11).

Weight memory is computed exactly from ``IMRTensor.shape``/``dtype``.
Activation (working-buffer) memory cannot be computed exactly for the
same reason ``FlopsEstimator`` cannot compute exact spatially-extended
FLOPs (see ``flops_estimator.py``): the IMR carries no output
activation shapes. This module instead reports an activation-memory
*estimate* using the conservative, widely-used double-buffering rule
of thumb — one buffer for a layer's input activation, one for its
output, sized off the single largest layer's weight footprint — and is
documented here as an estimate rather than an exact figure.
"""

from __future__ import annotations

from math import prod
from typing import Dict

from uaqe.common.imr import IMR, IMRLayer

# numpy/array dtype name (as emitted by ``str(array.dtype)`` across
# every first-party framework adapter, see
# ``infrastructure/framework_adapters/*``) -> element size in bytes.
# Any dtype not listed falls back to ``_DEFAULT_DTYPE_SIZE_BYTES`` via
# :func:`_dtype_size_bytes`.
_DTYPE_SIZE_BYTES: Dict[str, int] = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint64": 8,
    "uint32": 4,
    "uint16": 2,
    "uint8": 1,
    "bool": 1,
}
_DEFAULT_DTYPE_SIZE_BYTES = 4


def _dtype_size_bytes(dtype: str) -> int:
    """Resolve the element size, in bytes, of an ``IMRTensor.dtype`` string.

    Args:
        dtype: The tensor's dtype string, e.g. ``"float32"``.

    Returns:
        The known element size for ``dtype``, or
        ``_DEFAULT_DTYPE_SIZE_BYTES`` (4, matching FP32) if ``dtype``
        is not recognized.
    """
    return _DTYPE_SIZE_BYTES.get(dtype.lower(), _DEFAULT_DTYPE_SIZE_BYTES)


class MemoryAnalyzer:
    """Estimates weight, activation, and total memory footprint for an ``IMR``."""

    def weight_bytes_for_layer(self, layer: IMRLayer) -> int:
        """Compute the exact weight-tensor memory footprint of one layer.

        Args:
            layer: The layer whose ``parameters`` tensors to size.

        Returns:
            The sum, across every tensor in ``layer.parameters``, of
            ``element_count * dtype_size_bytes``.
        """
        total = 0
        for tensor in layer.parameters.values():
            element_count = prod(tensor.shape) if tensor.shape else 1
            total += element_count * _dtype_size_bytes(tensor.dtype)
        return total

    def total_weight_bytes(self, imr: IMR) -> int:
        """Sum exact weight memory across every layer of ``imr``.

        Args:
            imr: The model representation to size.

        Returns:
            The sum of :meth:`weight_bytes_for_layer` over every layer.
        """
        return sum(self.weight_bytes_for_layer(layer) for layer in imr.layers)

    def per_layer_weight_bytes(self, imr: IMR) -> Dict[str, int]:
        """Build a per-layer weight-memory breakdown.

        Args:
            imr: The model representation to break down.

        Returns:
            A mapping of ``IMRLayer.name`` to that layer's exact
            weight-memory footprint, in bytes.
        """
        return {layer.name: self.weight_bytes_for_layer(layer) for layer in imr.layers}

    def estimated_activation_bytes(self, imr: IMR) -> int:
        """Estimate peak activation (working-buffer) memory for ``imr``.

        See the module docstring for why this is an estimate rather
        than an exact figure.

        Args:
            imr: The model representation to estimate.

        Returns:
            Twice the largest single layer's weight-memory footprint
            (a double-buffered input/output activation estimate), or
            ``0`` if ``imr`` has no layers.
        """
        per_layer = self.per_layer_weight_bytes(imr)
        if not per_layer:
            return 0
        return 2 * max(per_layer.values())

    def estimate_total(self, imr: IMR) -> int:
        """Estimate total memory footprint (weights + activations) for ``imr``.

        Args:
            imr: The model representation to estimate.

        Returns:
            ``total_weight_bytes(imr) + estimated_activation_bytes(imr)``.
        """
        return self.total_weight_bytes(imr) + self.estimated_activation_bytes(imr)
