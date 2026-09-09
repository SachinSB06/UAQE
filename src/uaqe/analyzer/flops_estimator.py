"""Best-effort per-layer FLOPs (floating-point / multiply-accumulate
operation) estimation for an analyzed ``IMR``.

Feeds ``AnalysisResult.estimated_flops`` (``03_API_Specification.md``
§3.3). Internal helper for ``uaqe.analyzer``; not part of the locked
class inventory (``09_Architecture_Lock.md`` §11, "Internal API").

**Documented limitation:** ``IMRLayer`` carries no output *activation*
shape (only tensor *names* in ``inputs``/``outputs``,
``03_API_Specification.md`` §1.1), and no first-party framework
adapter currently populates one via shape inference/propagation. Exact
FLOPs for spatially-extended ops (e.g. ``Conv2d``) therefore cannot be
computed from the IMR alone — the true value depends on the number of
output positions, itself a function of the (unknown) input spatial
size, stride, and padding. This estimator instead reports the standard
**per-output-position** multiply-accumulate cost (doubled for the
multiply+add pair) for weight-bearing ops:

- *Exact* for ops whose output has exactly one "position" per weight
  application (``Linear``/``Dense``/``MatMul``/``Gemm``-family ops):
  this equals ``2 * in_features * out_features``, i.e.
  ``2 * weight_parameter_count``.
- A **lower bound** for spatially-extended ops (``Conv*``-family): the
  true value is this estimate multiplied by the (unknown) number of
  output spatial positions. Every layer flagged this way is surfaced
  via :meth:`FlopsEstimator.lower_bound_warnings` so callers
  (``ModelAnalyzer``, downstream ``ReportGenerator``) can present the
  caveat rather than the figure alone.

Non-weight-bearing ops (activations, pooling, normalization, etc.)
have a per-*activation-element* cost, not a per-*parameter* cost, and
are reported as ``0`` FLOPs rather than guessed, for the same
shape-unavailability reason.
"""

from __future__ import annotations

from math import prod
from typing import Dict, FrozenSet, List, Tuple

from uaqe.common.imr import IMR, IMRLayer

# Op-type name (as emitted verbatim by any first-party framework
# adapter — see ``infrastructure/framework_adapters/*`` — since each
# adapter emits its own native operator naming convention rather than
# a normalized vocabulary: PyTorch class names, Keras class names,
# ONNX ``op_type``, raw TensorFlow node ``op``, TFLite opcode names).
_DENSE_OPS: FrozenSet[str] = frozenset(
    {
        "Linear", "Dense", "MatMul", "Gemm", "BatchMatMul", "BatchMatMulV2",
        "FullyConnected",
    }
)
_CONV_OPS: FrozenSet[str] = frozenset(
    {
        "Conv1d", "Conv2d", "Conv3d", "Conv2D", "Conv3D", "Conv",
        "DepthwiseConv2d", "DepthwiseConv2D", "ConvTranspose2d",
        "Conv2DTranspose", "SeparableConv2D",
    }
)


class FlopsEstimator:
    """Estimates per-layer and total FLOPs for an ``IMR``.

    See the module docstring for the exactness caveats this estimator
    is subject to given the current IMR schema.
    """

    def estimate_layer(self, layer: IMRLayer) -> Tuple[int, bool]:
        """Estimate FLOPs for a single layer.

        Args:
            layer: The layer to estimate.

        Returns:
            A ``(flops, is_lower_bound)`` pair: ``flops`` is the
            estimated FLOPs (``0`` for ops with no known weight-bearing
            cost model), and ``is_lower_bound`` is ``True`` when the
            true value may exceed ``flops`` because it depends on an
            output spatial size not available in the IMR (see the
            module docstring).
        """
        weight_params = sum(
            prod(tensor.shape) if tensor.shape else 1
            for tensor in layer.parameters.values()
        )
        if layer.op_type in _DENSE_OPS:
            return 2 * weight_params, False
        if layer.op_type in _CONV_OPS:
            return 2 * weight_params, True
        return 0, False

    def estimate_total(self, imr: IMR) -> int:
        """Sum :meth:`estimate_layer` FLOPs across every layer of ``imr``.

        Args:
            imr: The model representation to estimate.

        Returns:
            The sum of every layer's estimated FLOPs.
        """
        return sum(self.estimate_layer(layer)[0] for layer in imr.layers)

    def estimate_per_layer(self, imr: IMR) -> Dict[str, int]:
        """Build a per-layer FLOPs breakdown.

        Args:
            imr: The model representation to break down.

        Returns:
            A mapping of ``IMRLayer.name`` to that layer's estimated
            FLOPs.
        """
        return {layer.name: self.estimate_layer(layer)[0] for layer in imr.layers}

    def lower_bound_warnings(self, imr: IMR) -> List[str]:
        """Collect a warning per layer whose FLOPs estimate is a lower bound.

        Args:
            imr: The model representation to inspect.

        Returns:
            One human-readable warning string per layer flagged by
            :meth:`estimate_layer` as ``is_lower_bound``, suitable for
            ``StageResult.warnings``.
        """
        warnings: List[str] = []
        for layer in imr.layers:
            flops, is_lower_bound = self.estimate_layer(layer)
            if is_lower_bound:
                warnings.append(
                    f"Layer {layer.name!r} ({layer.op_type}): FLOPs estimate "
                    f"({flops}) is a per-output-position lower bound; the IMR "
                    "does not carry output spatial dimensions needed for an "
                    "exact figure."
                )
        return warnings
