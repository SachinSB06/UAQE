"""Operator-fusion structural pass for an ``IMR`` graph.

``FusionOptimizer`` is the second structural pass
:class:`~uaqe.optimizer.optimization_planner.OptimizationPlanner` may
apply to a candidate ``IMR``, after
:class:`~uaqe.optimizer.graph_optimizer.GraphOptimizer`'s pass-through/
duplicate cleanup: it collapses a linear chain of layers matching a
known-fusible ``op_type`` sequence (e.g. ``Conv2D -> BatchNorm -> ReLU``)
into a single ``IMRLayer``, reducing both op count (fewer per-layer
dispatch/scheduling overheads, which
:class:`~uaqe.optimizer.latency_optimizer.LatencyOptimizer` scores as
lower total latency) and the number of intermediate activation buffers
:mod:`uaqe.hardware.memory_planner` must arena-plan for.

Like ``GraphOptimizer``, this is a plain collaborator (not a
``PipelineStage``) invoked by ``OptimizationPlanner``. It never mutates
the input ``IMR`` (frozen dataclasses throughout, per
``uaqe.common.imr``); it always returns a new ``IMR``.

Fusion model: a chain of layers ``[l0, l1, ..., lk]`` matching one of
:data:`_FUSION_PATTERNS` (checked longest-pattern-first at every
position so a full 3-op chain is preferred over its 2-op prefix) is
fused when every internal edge is *linear* — each ``li`` has exactly one
output, and that output has exactly one consumer, namely ``l(i+1)``.
This guarantees fusing the chain changes nothing observable to any
other layer in the graph: no other layer reads an intermediate tensor
that disappears. The fused layer keeps ``l0``'s inputs and ``lk``'s
outputs, so every layer *outside* the chain that already referenced
``lk``'s outputs needs no rewiring at all.

This module does not perform the numeric weight-folding a production
fusion pass would (e.g. folding ``BatchNorm`` scale/shift into a
preceding ``Conv2D``'s weight/bias tensors) — each original layer's
parameter tensors are carried into the fused layer verbatim, namespaced
by their originating layer name, so no data is lost and a downstream
exporter backend can still recover the pre-fusion arithmetic if needed.
Numeric folding is a natural follow-on enhancement once a specific
export backend's expected fused-kernel parameter layout is defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from uaqe.common.imr import IMR, IMRLayer, IMRTensor
from uaqe.common.interfaces.i_logger import ILogger

#: Linear ``op_type`` sequences eligible for fusion, longest first so
#: a full chain is preferred over one of its own prefixes at the same
#: starting position (see module docstring).
_FUSION_PATTERNS: List[Tuple[str, ...]] = sorted(
    [
        ("Conv2D", "BatchNorm", "ReLU"),
        ("MatMul", "Add", "ReLU"),
        ("Conv2D", "BatchNorm"),
        ("Conv2D", "ReLU"),
        ("MatMul", "Add"),
        ("Add", "ReLU"),
    ],
    key=len,
    reverse=True,
)


@dataclass
class FusionResult:
    """The outcome of one :meth:`FusionOptimizer.optimize` call.

    Attributes:
        optimized_imr: The ``IMR`` with every matched chain replaced by
            a single fused ``IMRLayer``.
        fused_groups: The original layer names fused together, one
            inner list per fused group, in fusion order.
        notes: Human-readable notes on which chains were fused into
            which layer, suitable for inclusion in
            :class:`~uaqe.optimizer.optimization_report.
            OptimizationReportDocument`.
    """

    optimized_imr: IMR
    fused_groups: List[List[str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class FusionOptimizer:
    """Fuses linear chains of known-fusible layers into single layers.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            optimizer operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``FusionOptimizer``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def optimize(self, imr: IMR) -> FusionResult:
        """Fuse every eligible linear chain in ``imr``.

        Args:
            imr: The model to fuse. Not modified.

        Returns:
            The resulting :class:`FusionResult`.
        """
        ordered_layers = imr.topological_order()
        consumer_count = self._consumer_counts(ordered_layers)

        output_layers: List[IMRLayer] = []
        fused_groups: List[List[str]] = []
        notes: List[str] = []
        skip_names = set()

        for index, layer in enumerate(ordered_layers):
            if layer.name in skip_names:
                continue
            chain = self._match_pattern(ordered_layers, index, consumer_count)
            if chain is None:
                output_layers.append(layer)
                continue

            fused_layer = self._fuse_chain(chain)
            output_layers.append(fused_layer)
            fused_groups.append([member.name for member in chain])
            notes.append(
                f"Fused {[member.op_type for member in chain]} chain "
                f"({[member.name for member in chain]}) into "
                f"{fused_layer.name!r}."
            )
            for member in chain[1:]:
                skip_names.add(member.name)

        optimized_imr = IMR(layers=output_layers, metadata=imr.metadata)

        if self.logger is not None:
            self.logger.info(
                "Fusion pass complete.",
                fused_group_count=len(fused_groups),
                remaining_layer_count=len(output_layers),
            )

        return FusionResult(
            optimized_imr=optimized_imr,
            fused_groups=fused_groups,
            notes=notes,
        )

    def _consumer_counts(self, layers: List[IMRLayer]) -> Dict[str, int]:
        """Count how many layers consume each tensor name as an input.

        Args:
            layers: The layers to scan.

        Returns:
            A mapping of tensor name to the number of layers whose
            ``inputs`` include it.
        """
        counts: Dict[str, int] = {}
        for layer in layers:
            for input_name in layer.inputs:
                counts[input_name] = counts.get(input_name, 0) + 1
        return counts

    def _match_pattern(
        self,
        layers: List[IMRLayer],
        start: int,
        consumer_count: Dict[str, int],
    ) -> Optional[List[IMRLayer]]:
        """Find the longest fusible chain starting at ``layers[start]``.

        Args:
            layers: The full topologically-ordered layer list.
            start: The index to attempt to start a chain at.
            consumer_count: Tensor-name consumer counts, from
                :meth:`_consumer_counts`.

        Returns:
            The matched chain of layers, or ``None`` if no pattern in
            :data:`_FUSION_PATTERNS` matches at ``start``.
        """
        for pattern in _FUSION_PATTERNS:
            end = start + len(pattern)
            if end > len(layers):
                continue
            candidate = layers[start:end]
            if tuple(member.op_type for member in candidate) != pattern:
                continue
            if self._is_linear_chain(candidate, consumer_count):
                return candidate
        return None

    def _is_linear_chain(
        self, chain: List[IMRLayer], consumer_count: Dict[str, int]
    ) -> bool:
        """Check that every internal edge of ``chain`` is linear.

        Args:
            chain: A candidate sequence of layers matching a fusion
                pattern's ``op_type`` sequence.
            consumer_count: Tensor-name consumer counts.

        Returns:
            ``True`` if every layer but the last has exactly one output
            and that output is consumed by exactly one layer overall
            (namely the next layer in ``chain``); ``False`` otherwise.
        """
        for upstream, downstream in zip(chain, chain[1:]):
            if len(upstream.outputs) != 1:
                return False
            if consumer_count.get(upstream.outputs[0], 0) != 1:
                return False
            if upstream.outputs[0] not in downstream.inputs:
                return False
        return True

    def _fuse_chain(self, chain: List[IMRLayer]) -> IMRLayer:
        """Collapse ``chain`` into a single ``IMRLayer``.

        Args:
            chain: A chain previously validated by
                :meth:`_is_linear_chain`.

        Returns:
            A new ``IMRLayer`` whose ``inputs`` are the first member's
            inputs, whose ``outputs`` are the last member's outputs
            (so nothing outside the chain needs rewiring), whose
            ``parameters`` are every member's parameters namespaced by
            originating layer name, and whose ``precision`` is the
            first member's (the precision the fused kernel's primary
            compute — e.g. the ``Conv2D``/``MatMul`` — already ran at).
        """
        first, last = chain[0], chain[-1]
        fused_name = "_".join(member.name for member in chain) + "_fused"
        fused_op_type = "Fused" + "".join(member.op_type for member in chain)

        parameters: Dict[str, IMRTensor] = {}
        attributes: Dict[str, Any] = {
            "fused_from": [member.name for member in chain],
            "fused_op_sequence": [member.op_type for member in chain],
        }
        for member in chain:
            for parameter_name, tensor in member.parameters.items():
                parameters[f"{member.name}.{parameter_name}"] = tensor
            for attribute_name, value in member.attributes.items():
                attributes.setdefault(f"{member.name}.{attribute_name}", value)

        return IMRLayer(
            name=fused_name,
            op_type=fused_op_type,
            inputs=list(first.inputs),
            outputs=list(last.outputs),
            parameters=parameters,
            attributes=attributes,
            precision=first.precision,
        )
