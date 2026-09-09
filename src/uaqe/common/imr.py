"""Internal Model Representation (IMR) for the Universal AI Quantization
Engine.

The IMR is the framework-independent graph representation that every
``IFrameworkAdapter`` produces and every domain-layer stage (quantization,
compression, optimization, export) consumes, so that no stage above
``uaqe.infrastructure.framework_adapters`` needs to know whether a model
originated from PyTorch, TensorFlow, Keras, ONNX, or TFLite.

This module has no dependencies on any other ``uaqe`` package beyond
``uaqe.common`` itself, per the layering rules in
``09_Architecture_Lock.md`` §8.

Locked contract: ``03_API_Specification.md`` §1.1.
"""

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from uaqe.common.exceptions import ModelValidationError
from uaqe.common.types import Precision


@dataclass(frozen=True)
class IMRTensor:
    """An immutable, framework-independent tensor (weight, bias, or
    activation buffer) within the IMR.

    Attributes:
        shape: The tensor's dimensions, e.g. ``(64, 3, 3, 3)``.
        dtype: The tensor's element data type, expressed as a
            framework-independent string (e.g. ``"float32"``).
        data: An opaque buffer handle holding the tensor's raw bytes.
            Reference-counted per ``01_Project_Architecture.md`` §17:
            multiple ``IMRTensor`` instances may share the same
            underlying buffer object (e.g. tied weights) rather than
            each holding an independent copy.
    """

    shape: Tuple[int, ...]
    dtype: str
    data: bytes

    def is_shared(self) -> bool:
        """Report whether this tensor's underlying buffer is shared.

        A buffer is considered shared when more references to the
        underlying ``bytes`` object exist than the implicit references
        created by this attribute access and the reference-count query
        itself, indicating another ``IMRTensor`` (or other holder) also
        points at the same buffer object rather than an independent
        copy.

        Returns:
            ``True`` if the underlying buffer object has additional
            external references beyond this tensor; ``False`` otherwise.
        """
        # sys.getrefcount's own parameter binding and the attribute
        # lookup above each contribute one transient reference; a
        # buffer held by exactly this tensor and nothing else reports
        # a baseline of 3 references under CPython.
        return sys.getrefcount(self.data) > 3


@dataclass(frozen=True)
class IMRLayer:
    """An immutable, framework-independent representation of a single
    model layer (operator) within the IMR graph.

    Attributes:
        name: The unique identifier of this layer within the IMR.
        op_type: The framework-independent operator type (e.g.
            ``"Conv2D"``, ``"MatMul"``, ``"ReLU"``).
        inputs: Names of the tensors consumed by this layer, used to
            resolve producer/consumer edges during
            :meth:`IMR.topological_order`.
        outputs: Names of the tensors produced by this layer.
        parameters: Learned parameters (weights, biases) keyed by
            parameter name.
        attributes: Operator-specific attributes (e.g. stride, padding)
            that do not themselves need a dedicated field.
        precision: The numeric precision this layer currently holds.
    """

    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    parameters: Dict[str, IMRTensor]
    attributes: Dict[str, Any]
    precision: Precision


@dataclass(frozen=True)
class IMRMetadata:
    """Immutable provenance and summary information for an ``IMR``.

    Attributes:
        source_framework: The framework the model was loaded from (e.g.
            ``"pytorch"``, ``"tensorflow"``, ``"keras"``, ``"onnx"``,
            ``"tflite"``).
        source_format: The specific source file format/extension the
            model was loaded from (e.g. ``".pth"``, ``".onnx"``).
        original_input_shapes: The model's declared input shapes prior
            to any IMR transformation, keyed by input tensor name.
        op_count: The total number of layers (operators) in the IMR.
        total_parameters: The total number of learnable parameter
            elements across all layers.
    """

    source_framework: str
    source_format: str
    original_input_shapes: Dict[str, Tuple[int, ...]]
    op_count: int
    total_parameters: int


@dataclass(frozen=True)
class IMR:
    """The Internal Model Representation: a framework-independent graph
    of layers plus provenance metadata.

    ``IMR`` is the single object type threaded through the pipeline's
    model-transforming stages (``model_loader`` → ``quantization_engine``
    → ``compression_engine`` → ``optimization_engine`` →
    ``memory_optimizer``); per ``09_Architecture_Lock.md`` §13 rule 2,
    there is no fixed "the IMR" context key — each stage produces a new
    ``IMR`` value representing the current state of the model.

    Attributes:
        layers: The ordered collection of layers comprising the model
            graph. Order of this list is not guaranteed to be a valid
            topological order; use :meth:`topological_order` when
            execution order matters.
        metadata: Provenance and summary information about this IMR.
    """

    layers: List[IMRLayer]
    metadata: IMRMetadata
    _layers_by_name: Dict[str, IMRLayer] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Build the internal name-to-layer lookup used by
        :meth:`get_layer`.

        Uses ``object.__setattr__`` because ``IMR`` is a frozen
        dataclass; this populates a derived index, not a new public
        field, so it does not alter the locked public attribute surface.
        """
        object.__setattr__(
            self, "_layers_by_name", {layer.name: layer for layer in self.layers}
        )

    def get_layer(self, name: str) -> IMRLayer:
        """Retrieve a layer by name.

        Args:
            name: The layer name to look up.

        Returns:
            The ``IMRLayer`` with the given name.

        Raises:
            KeyError: If no layer with the given name exists in this
                IMR. This mirrors the lookup-by-key convention already
                established for ``PipelineContext.get()`` in
                ``03_API_Specification.md`` §2.1.
        """
        try:
            return self._layers_by_name[name]
        except KeyError:
            raise KeyError(f"No layer named {name!r} in this IMR.") from None

    def topological_order(self) -> List[IMRLayer]:
        """Compute a valid execution order of this IMR's layers.

        Dependency edges are derived from tensor production/consumption:
        a layer depends on every other layer that produces one of its
        input tensor names. Layer inputs with no producer within this
        IMR (i.e. model inputs) impose no dependency.

        Returns:
            The layers of this IMR ordered such that every layer appears
            after all layers it depends on. Layers with no dependency
            relationship to one another are ordered by name for
            determinism.

        Raises:
            ModelValidationError: If the layer graph contains a cycle
                and therefore has no valid topological order.
        """
        producer_of: Dict[str, str] = {}
        for layer in self.layers:
            for output_name in layer.outputs:
                producer_of[output_name] = layer.name

        dependencies: Dict[str, set] = {layer.name: set() for layer in self.layers}
        dependents: Dict[str, set] = {layer.name: set() for layer in self.layers}
        for layer in self.layers:
            for input_name in layer.inputs:
                producer_name = producer_of.get(input_name)
                if producer_name is not None and producer_name != layer.name:
                    dependencies[layer.name].add(producer_name)
                    dependents[producer_name].add(layer.name)

        in_degree: Dict[str, int] = {
            name: len(deps) for name, deps in dependencies.items()
        }
        queue = deque(sorted(name for name, degree in in_degree.items() if degree == 0))
        ordered_names: List[str] = []
        while queue:
            current = queue.popleft()
            ordered_names.append(current)
            for dependent in sorted(dependents[current]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
            queue = deque(sorted(queue))

        if len(ordered_names) != len(self.layers):
            raise ModelValidationError(
                "IMR contains a cycle and cannot be topologically ordered.",
                code="IMR_CYCLE_DETECTED",
            )

        return [self._layers_by_name[name] for name in ordered_names]
