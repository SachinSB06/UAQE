"""Framework-op-vocabulary compatibility scanning for an analyzed ``IMR``.

Flags layers whose ``op_type`` is outside the operator vocabulary
UAQE's own domain-layer engines (``QuantizationEngine``,
``CompressionEngine``) currently know how to reason about, independent
of any particular deployment target.

**Scope boundary:** this is deliberately narrower than, and conceptually
precedes, ``uaqe.domain.compatibility.layer_compatibility_checker
.LayerCompatibilityChecker`` (``03_API_Specification.md`` §4.1), which
performs the locked, hardware-*specific* compatibility check ("Step 5.
Unsupported Layer Detection" against a chosen ``HardwareProfile``,
``01_Project_Architecture.md`` §5). This module answers the earlier,
hardware-agnostic question "does UAQE recognize this operator at all?"
so ``ModelAnalyzer`` can surface it as an analysis-time warning;
``LayerCompatibilityChecker`` separately answers "can *this* hardware
target run it?" using the resolved ``HardwareProfile``. Internal helper
for ``uaqe.analyzer``; not part of the locked class inventory
(``09_Architecture_Lock.md`` §11).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

from uaqe.common.imr import IMR, IMRLayer

# The operator vocabulary UAQE's domain-layer engines recognize,
# spanning the native naming conventions emitted by every first-party
# IFrameworkAdapter (PyTorch class names, Keras class names, ONNX
# op_type, raw TensorFlow node op, TFLite opcode names — see
# infrastructure/framework_adapters/*). Additive: registering a newly
# supported op is a pure addition to this set
# (``09_Architecture_Lock.md`` §0 rule 2), never a modification of an
# existing entry.
KNOWN_OP_TYPES: FrozenSet[str] = frozenset(
    {
        # Dense / matmul family
        "Linear", "Dense", "MatMul", "Gemm", "BatchMatMul", "BatchMatMulV2",
        "FullyConnected",
        # Convolution family
        "Conv1d", "Conv2d", "Conv3d", "Conv2D", "Conv3D", "Conv",
        "DepthwiseConv2d", "DepthwiseConv2D", "ConvTranspose2d",
        "Conv2DTranspose", "SeparableConv2D",
        # Normalization
        "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "BatchNormalization",
        "LayerNormalization", "GroupNorm",
        # Activation
        "ReLU", "Relu", "Relu6", "ReLU6", "Sigmoid", "Tanh", "GELU", "Gelu",
        "LeakyReLU", "LeakyRelu", "Softmax", "ELU", "PReLU", "HardSigmoid",
        # Pooling
        "MaxPool2d", "MaxPool2D", "AvgPool2d", "AveragePooling2D",
        "GlobalAveragePooling2D", "AdaptiveAvgPool2d", "GlobalAveragePool",
        # Structural / elementwise
        "Flatten", "Reshape", "Dropout", "Identity", "Concat", "Concatenate",
        "Add", "Mul", "Pad", "Transpose", "Squeeze", "Unsqueeze",
        # Recurrent
        "LSTM", "GRU", "RNN",
        # Embedding / lookup
        "Embedding", "Gather",
    }
)


class UnsupportedLayerDetector:
    """Flags ``IMRLayer`` instances whose ``op_type`` is outside ``KNOWN_OP_TYPES``."""

    def is_supported(self, layer: IMRLayer) -> bool:
        """Report whether ``layer``'s ``op_type`` is a recognized operator.

        Args:
            layer: The layer to check.

        Returns:
            ``True`` if ``layer.op_type`` is a member of
            ``KNOWN_OP_TYPES``.
        """
        return layer.op_type in KNOWN_OP_TYPES

    def find_unsupported(self, imr: IMR) -> List[str]:
        """Find every layer in ``imr`` with an unrecognized ``op_type``.

        Args:
            imr: The model representation to scan.

        Returns:
            The ``name`` of every layer for which :meth:`is_supported`
            is ``False``, in ``imr.layers`` order.
        """
        return [layer.name for layer in imr.layers if not self.is_supported(layer)]

    def build_warnings(self, imr: IMR) -> List[str]:
        """Build one human-readable warning per unsupported layer in ``imr``.

        Args:
            imr: The model representation to scan.

        Returns:
            One warning string per layer flagged by
            :meth:`find_unsupported`, suitable for
            ``StageResult.warnings``.
        """
        by_name: Dict[str, IMRLayer] = {layer.name: layer for layer in imr.layers}
        return [
            f"Layer {name!r} has unrecognized op_type "
            f"{by_name[name].op_type!r}; quantization/compression "
            "candidacy cannot be determined for it."
            for name in self.find_unsupported(imr)
        ]
