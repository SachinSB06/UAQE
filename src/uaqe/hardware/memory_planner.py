"""Plans activation-arena memory usage for an ``IMR`` against a
``HardwareProfile``'s ``tensor_memory_bytes`` ceiling.

Per ``05_Hardware_Profile_Spec.md`` §6 rule 4, ``tensor_memory_bytes``
is a hard ceiling for activation-arena planning, distinct from
``ram_bytes``/``flash_bytes`` (which bound total footprint including
code and static buffers). This module plans only the activation arena;
static weight footprint is reported separately for context.

Note on activation sizing: the ``IMR`` (``uaqe.common.imr``) does not
carry a shape/dtype registry for intermediate activation tensors — only
learned parameter tensors (``IMRLayer.parameters``) carry a concrete
``shape``. Where a layer's ``attributes`` mapping supplies an
``"output_shape"`` entry (a ``Tuple[int, ...]``, as a framework adapter
may attach when it is known), that shape is used directly. Otherwise
this planner falls back to a documented heuristic (see
:meth:`MemoryPlanner._estimate_activation_bytes`) and records the
fallback in the returned plan so a caller can see where a number is
exact versus estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from uaqe.common.exceptions import HardwareIncompatibilityError
from uaqe.common.imr import IMR, IMRLayer
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import Precision

if TYPE_CHECKING:
    # Deferred to break the hardware_manager <-> memory_planner import
    # cycle; see the matching note in compatibility_checker.py.
    from uaqe.hardware.hardware_manager import HardwareProfile

#: Bytes occupied by one tensor element at a given precision. ``MIXED``
#: has no single element width; a conservative FP16-equivalent width is
#: used as a stand-in when a layer's precision is ``MIXED``.
BYTES_PER_ELEMENT: Dict[Precision, float] = {
    Precision.FP32: 4.0,
    Precision.FP16: 2.0,
    Precision.INT8: 1.0,
    Precision.INT4: 0.5,
    Precision.MIXED: 2.0,
}


@dataclass
class MemoryPlan:
    """The outcome of activation-arena memory planning for one
    ``IMR``/``HardwareProfile`` pair.

    Attributes:
        buffer_arena_bytes: The size of the double-buffered activation
            arena this plan allocates — the number of bytes a runtime
            should reserve up front for activations.
        tensor_reuse_map: For each layer's output tensor name, the
            arena buffer slot (``"buffer_a"`` or ``"buffer_b"``) it is
            assigned to under the ping-pong reuse strategy.
        peak_memory_bytes: The highest simultaneous activation-arena
            usage observed across the plan — the value compared against
            ``HardwareProfile.tensor_memory_bytes``.
        static_weight_bytes: The total size of every parameter tensor,
            reported for context; not itself checked against
            ``tensor_memory_bytes`` (weights are bounded by
            ``flash_bytes``/``max_model_size_bytes`` instead, per
            ``05_Hardware_Profile_Spec.md`` §6 rule 4).
        per_layer_activation_bytes: Each layer's estimated output
            activation size, in bytes.
        assumptions: Human-readable notes on which figures are exact
            (from a declared ``output_shape``) versus heuristic
            estimates, and any other planning caveats.
    """

    buffer_arena_bytes: int
    tensor_reuse_map: Dict[str, str] = field(default_factory=dict)
    peak_memory_bytes: int = 0
    static_weight_bytes: int = 0
    per_layer_activation_bytes: Dict[str, int] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)


class MemoryPlanner:
    """Plans a two-buffer ("ping-pong") activation arena for an ``IMR``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            planner operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``MemoryPlanner``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def plan(self, imr: IMR, profile: HardwareProfile, strict: bool = True) -> MemoryPlan:
        """Plan activation-arena memory usage for ``imr`` on ``profile``.

        Layers are visited in topological order and assigned
        alternating arena slots (``"buffer_a"`` / ``"buffer_b"``), so
        that only two layers' worth of activations are ever live at
        once — a standard streaming-inference reuse pattern for memory-
        constrained runtimes (``tflite-micro``, ``bare-metal-hdl``).

        Args:
            imr: The model to plan for.
            profile: The candidate deployment target.
            strict: If ``True``, raise ``HardwareIncompatibilityError``
                when the planned peak exceeds
                ``profile.tensor_memory_bytes``.

        Returns:
            The resulting ``MemoryPlan``.

        Raises:
            HardwareIncompatibilityError: If ``strict`` is ``True`` and
                ``peak_memory_bytes`` exceeds
                ``profile.tensor_memory_bytes``.
        """
        ordered_layers = imr.topological_order()
        assumptions: List[str] = []
        per_layer_activation_bytes: Dict[str, int] = {}
        tensor_reuse_map: Dict[str, str] = {}

        slot_names = ("buffer_a", "buffer_b")
        peak_memory_bytes = 0
        previous_bytes = 0
        for index, layer in enumerate(ordered_layers):
            activation_bytes = self._estimate_activation_bytes(layer, assumptions)
            per_layer_activation_bytes[layer.name] = activation_bytes
            slot = slot_names[index % 2]
            for output_name in layer.outputs:
                tensor_reuse_map[output_name] = slot
            # Peak usage at this step is this layer's own output plus
            # the still-live previous layer's output occupying the
            # other slot, until the next layer overwrites it.
            live_bytes = activation_bytes + previous_bytes
            peak_memory_bytes = max(peak_memory_bytes, live_bytes)
            previous_bytes = activation_bytes

        static_weight_bytes = sum(
            len(tensor.data)
            for layer in imr.layers
            for tensor in layer.parameters.values()
        )
        buffer_arena_bytes = peak_memory_bytes

        if self.logger is not None:
            self.logger.info(
                "Planned activation-arena memory.",
                profile_id=profile.profile_id,
                peak_memory_bytes=peak_memory_bytes,
                tensor_memory_bytes=profile.tensor_memory_bytes,
            )

        plan = MemoryPlan(
            buffer_arena_bytes=buffer_arena_bytes,
            tensor_reuse_map=tensor_reuse_map,
            peak_memory_bytes=peak_memory_bytes,
            static_weight_bytes=static_weight_bytes,
            per_layer_activation_bytes=per_layer_activation_bytes,
            assumptions=assumptions,
        )

        if strict and peak_memory_bytes > profile.tensor_memory_bytes:
            raise HardwareIncompatibilityError(
                f"Planned peak activation memory ({peak_memory_bytes} bytes) "
                f"exceeds tensor_memory_bytes ({profile.tensor_memory_bytes}) "
                f"for hardware profile {profile.profile_id!r}.",
                code="ACTIVATION_MEMORY_OVERFLOW",
                remediation_hint=(
                    "Reduce activation footprint (e.g. via CompressionPlan "
                    "or a smaller batch size) or select a hardware profile "
                    "with more tensor_memory_bytes."
                ),
            )
        return plan

    def _estimate_activation_bytes(
        self, layer: IMRLayer, assumptions: List[str]
    ) -> int:
        """Estimate one layer's output activation footprint, in bytes.

        Uses ``layer.attributes["output_shape"]`` when present (an
        exact figure). Otherwise falls back to a heuristic: the layer's
        total parameter element count at its own precision's element
        width, which approximates output activation volume reasonably
        for the common case of a layer whose output width scales with
        its weight tensor (e.g. a dense/conv layer's channel count) —
        this is a documented order-of-magnitude estimate, not a
        measured figure, for layers with no declared ``output_shape``.

        Args:
            layer: The layer to estimate.
            assumptions: Mutated in place with a note identifying
                whether this layer's figure is exact or heuristic.

        Returns:
            The estimated activation size, in bytes.
        """
        output_shape = layer.attributes.get("output_shape")
        element_bytes = BYTES_PER_ELEMENT.get(layer.precision, 4.0)

        if isinstance(output_shape, tuple) and output_shape:
            element_count = 1
            for dim in output_shape:
                element_count *= int(dim)
            return int(element_count * element_bytes)

        assumptions.append(
            f"Layer {layer.name!r} ({layer.op_type}) has no declared "
            "output_shape attribute; activation size estimated from "
            "parameter element count as an order-of-magnitude proxy."
        )
        parameter_element_count = sum(
            self._tensor_element_count(tensor.shape)
            for tensor in layer.parameters.values()
        )
        if parameter_element_count == 0:
            # Parameter-free ops (e.g. ReLU, pooling) are assumed to
            # pass through their (unknown) input volume unchanged; a
            # small fixed placeholder keeps the arena from reporting
            # zero bytes for a real tensor.
            assumptions.append(
                f"Layer {layer.name!r} has no parameters; using a fixed "
                "1 KiB placeholder activation size."
            )
            return 1024
        return int(parameter_element_count * element_bytes)

    def _tensor_element_count(self, shape: Tuple[int, ...]) -> int:
        """Compute the number of elements described by ``shape``.

        Args:
            shape: A tensor's dimensions.

        Returns:
            The product of every dimension in ``shape``, or ``0`` for
            an empty shape.
        """
        if not shape:
            return 0
        count = 1
        for dim in shape:
            count *= int(dim)
        return count
