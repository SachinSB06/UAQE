"""Framework-independent structural cleanup of an ``IMR`` graph.

``GraphOptimizer`` is the first structural pass ``Optimizer``/
``OptimizationPlanner`` may apply to a candidate ``IMR`` before
:class:`~uaqe.optimizer.fusion_optimizer.FusionOptimizer`'s operator
fusion pass: it removes layers that provably do not change model output
(pass-through ops, exact duplicates) so that every later pass — fusion,
latency estimation, memory planning — operates on a smaller, simpler
graph. Per ``09_Architecture_Lock.md`` §13 rule 2, no stage assumes a
fixed "the IMR" context key; this class is a plain collaborator (not
itself a ``PipelineStage``) invoked by
:class:`~uaqe.optimizer.optimization_planner.OptimizationPlanner`, which
owns reading/writing the run's current IMR.

Two independent, order-preserving passes are applied:

1. **Identity/pass-through elimination** — a layer whose ``op_type`` is
   in :data:`_IDENTITY_OP_TYPES`, has exactly one input, exactly one
   output, and no learned parameters is removed, and every downstream
   layer that consumed its output is rewired to consume its input
   directly instead. ``Dropout`` is included because it is a no-op at
   inference time (``09_Architecture_Lock.md`` never contradicts this);
   ``Identity`` is a no-op by definition.
2. **Duplicate (common-subexpression) elimination** — two layers with
   the same ``op_type``, the same (already-rewired) inputs, the same
   attributes, and bit-identical parameter tensors are guaranteed to
   compute the same output; the later one is removed and its consumers
   rewired to the earlier (canonical) layer's output.

Both passes only ever remove or rewire — no layer's parameters or
attributes are ever mutated in place, and ``IMRLayer``/``IMR`` are
frozen dataclasses (``uaqe.common.imr``), so the input ``IMR`` passed to
:meth:`GraphOptimizer.optimize` is never modified; a new ``IMR`` is
always returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from uaqe.common.imr import IMR, IMRLayer
from uaqe.common.interfaces.i_logger import ILogger

#: Op types eliminated as pure pass-through whenever this optimizer
#: runs, since neither alters model output at inference time: ``Dropout``
#: is disabled (identity) outside of training, and ``Identity`` is
#: pass-through by definition.
_IDENTITY_OP_TYPES = frozenset({"Identity", "Dropout"})


@dataclass
class GraphOptimizationResult:
    """The outcome of one :meth:`GraphOptimizer.optimize` call.

    Attributes:
        optimized_imr: The cleaned-up ``IMR``, with every eliminated
            layer removed and every remaining layer's ``inputs``
            rewired past any removed layer.
        removed_layers: The names of every layer removed, in removal
            order (identity eliminations before duplicate eliminations).
        notes: Human-readable notes on what was removed and why,
            suitable for inclusion in
            :class:`~uaqe.optimizer.optimization_report.
            OptimizationReportDocument`.
    """

    optimized_imr: IMR
    removed_layers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class GraphOptimizer:
    """Removes pass-through and duplicate layers from an ``IMR``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            optimizer operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize a ``GraphOptimizer``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def optimize(self, imr: IMR) -> GraphOptimizationResult:
        """Run both cleanup passes over ``imr``.

        Args:
            imr: The model to clean up. Not modified.

        Returns:
            The resulting :class:`GraphOptimizationResult`.
        """
        layers, removed_identity, identity_notes = self._eliminate_identity_layers(imr)
        layers, removed_duplicates, duplicate_notes = self._eliminate_duplicate_layers(
            layers
        )
        removed_layers = removed_identity + removed_duplicates
        notes = identity_notes + duplicate_notes

        optimized_imr = IMR(layers=layers, metadata=imr.metadata)

        if self.logger is not None:
            self.logger.info(
                "Graph cleanup complete.",
                removed_layer_count=len(removed_layers),
                remaining_layer_count=len(layers),
            )

        return GraphOptimizationResult(
            optimized_imr=optimized_imr,
            removed_layers=removed_layers,
            notes=notes,
        )

    def _eliminate_identity_layers(
        self, imr: IMR
    ) -> Tuple[List[IMRLayer], List[str], List[str]]:
        """Remove pass-through layers and rewire their consumers.

        Args:
            imr: The model to clean up.

        Returns:
            A ``(kept_layers, removed_names, notes)`` tuple.
        """
        rename: Dict[str, str] = {}
        removed: List[str] = []
        notes: List[str] = []
        kept: List[IMRLayer] = []

        for layer in imr.layers:
            is_pass_through = (
                layer.op_type in _IDENTITY_OP_TYPES
                and len(layer.inputs) == 1
                and len(layer.outputs) == 1
                and not layer.parameters
            )
            if is_pass_through:
                rename[layer.outputs[0]] = layer.inputs[0]
                removed.append(layer.name)
                notes.append(
                    f"Removed pass-through layer {layer.name!r} "
                    f"({layer.op_type}); consumers rewired to "
                    f"{layer.inputs[0]!r}."
                )
            else:
                kept.append(layer)

        if not rename:
            return kept, removed, notes

        def resolve(tensor_name: str) -> str:
            visited = set()
            while tensor_name in rename and tensor_name not in visited:
                visited.add(tensor_name)
                tensor_name = rename[tensor_name]
            return tensor_name

        rewired: List[IMRLayer] = []
        for layer in kept:
            new_inputs = [resolve(name) for name in layer.inputs]
            if new_inputs != layer.inputs:
                layer = replace(layer, inputs=new_inputs)
            rewired.append(layer)
        return rewired, removed, notes

    def _eliminate_duplicate_layers(
        self, layers: List[IMRLayer]
    ) -> Tuple[List[IMRLayer], List[str], List[str]]:
        """Remove layers that are bit-identical to an earlier layer.

        Args:
            layers: Layers already passed through identity elimination.

        Returns:
            A ``(kept_layers, removed_names, notes)`` tuple.
        """
        canonical_output_by_signature: Dict[Tuple, str] = {}
        rename: Dict[str, str] = {}
        removed: List[str] = []
        notes: List[str] = []
        kept: List[IMRLayer] = []

        for layer in layers:
            signature = self._layer_signature(layer)
            canonical_output = canonical_output_by_signature.get(signature)
            if canonical_output is not None and len(layer.outputs) == 1:
                rename[layer.outputs[0]] = canonical_output
                removed.append(layer.name)
                notes.append(
                    f"Removed duplicate layer {layer.name!r} "
                    f"({layer.op_type}); consumers rewired to canonical "
                    f"output {canonical_output!r}."
                )
                continue
            kept.append(layer)
            if len(layer.outputs) == 1:
                canonical_output_by_signature[signature] = layer.outputs[0]

        if not rename:
            return kept, removed, notes

        rewired: List[IMRLayer] = []
        for layer in kept:
            new_inputs = [rename.get(name, name) for name in layer.inputs]
            if new_inputs != layer.inputs:
                layer = replace(layer, inputs=new_inputs)
            rewired.append(layer)
        return rewired, removed, notes

    def _to_hashable(self, value):
        """Recursively convert ``value`` into an equivalent hashable form.

        Operator attributes commonly hold plain ``list`` values (e.g.
        ``stride``, ``padding``, ``kernel_shape``), which are unhashable
        and therefore unusable as part of a ``dict`` key. This converts
        lists into tuples and dicts into a sorted tuple of
        ``(key, value)`` pairs, recursively, so nested structures are
        fully hashable while still comparing equal for two layers with
        identical attribute values.

        Args:
            value: An attribute value of arbitrary (possibly nested)
                type.

        Returns:
            A hashable equivalent of ``value``.
        """
        if isinstance(value, (list, tuple)):
            return tuple(self._to_hashable(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (key, self._to_hashable(item)) for key, item in value.items()
                )
            )
        return value

    def _layer_signature(self, layer: IMRLayer) -> Tuple:
        """Build a hashable signature identifying ``layer``'s computed
        value, independent of its name.

        Two layers with an identical signature are assumed to compute
        identical output given identical (already-rewired) inputs — a
        safe assumption since it also compares parameter tensor bytes,
        not just shape/dtype.

        Args:
            layer: The layer to fingerprint.

        Returns:
            A tuple usable as a ``dict`` key.

        Note:
            Every value in ``layer.attributes`` and every parameter's
            ``shape`` is passed through :meth:`_to_hashable` first, so
            list-valued attributes (e.g. ``stride``, ``padding``,
            ``kernel_shape``) and nested lists/dicts no longer raise
            ``TypeError: unhashable type: 'list'`` when the signature
            is used as a ``dict`` key. The ``try/except`` fallback is
            kept as a last-resort safety net for any value type this
            conversion does not anticipate.
        """
        try:
            attribute_signature = tuple(
                sorted(
                    (name, self._to_hashable(value))
                    for name, value in layer.attributes.items()
                )
            )
            parameter_signature = tuple(
                sorted(
                    (
                        name,
                        self._to_hashable(tensor.shape),
                        tensor.dtype,
                        tensor.data,
                    )
                    for name, tensor in layer.parameters.items()
                )
            )
        except TypeError:
            attribute_signature = (id(layer),)
            parameter_signature = ()
        return (
            layer.op_type,
            tuple(layer.inputs),
            attribute_signature,
            parameter_signature,
            layer.precision,
        )
