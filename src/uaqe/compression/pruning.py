"""Shared magnitude-scoring, mask, and application arithmetic for every
concrete pruning ``ICompressionStrategy`` in ``uaqe.compression``.

``PruningApplier`` owns the one piece of arithmetic
:class:`~uaqe.compression.magnitude_pruner.MagnitudePruner` and
:class:`~uaqe.compression.structured_pruner.StructuredPruner` both need
— scoring elements (or structured groups of elements) by magnitude,
deciding which survive a target sparsity, and turning that decision
into a sparse-encoded ``IMRLayer`` via
:class:`~uaqe.compression.sparse_encoder.SparseEncoder` — so the two
strategies share exactly one implementation of mask application rather
than two subtly different ones. It has no opinion on *how* a mask
should be scored at the top level (unstructured element-wise vs.
structured group-wise) beyond providing one method for each; strategy
selection between the two belongs to
:class:`~uaqe.compression.compression_planner.CompressionPlanner`.
"""

from __future__ import annotations

import array
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from uaqe.common.exceptions import CompressionError
from uaqe.common.imr import IMRLayer, IMRTensor
from uaqe.compression.sparse_encoder import SparseEncoder

#: ``IMRTensor.dtype`` value this module knows how to score and prune.
#: Any other dtype is passed through unpruned, since a tensor that is
#: not floating-point has no meaningful magnitude to rank.
_FLOAT32_DTYPE = "float32"

#: ``array`` module type code for 32-bit IEEE-754 floats.
_FLOAT32_TYPECODE = "f"


@dataclass(frozen=True)
class PruningMask:
    """The set of tensor-flat-index positions that survive pruning for
    one parameter tensor.

    Attributes:
        parameter_name: The ``IMRLayer.parameters`` key this mask
            applies to.
        kept_indices: The flat (row-major) indices of the tensor's
            elements to retain; every other element is pruned to zero.
        total_element_count: The total number of elements in the
            tensor this mask was computed for, used by :meth:`sparsity`.
    """

    parameter_name: str
    kept_indices: Tuple[int, ...]
    total_element_count: int

    def sparsity(self) -> float:
        """Return the fraction of elements this mask prunes.

        Returns:
            ``0.0`` for an empty tensor (nothing to prune); otherwise
            ``(total_element_count - len(kept_indices)) /
            total_element_count``.
        """
        if self.total_element_count == 0:
            return 0.0
        pruned_count = self.total_element_count - len(self.kept_indices)
        return pruned_count / self.total_element_count


@dataclass(frozen=True)
class PruningStatistics:
    """The outcome of applying one or more :class:`PruningMask`
    instances to a single ``IMRLayer``.

    Attributes:
        layer_name: The ``IMRLayer.name`` these statistics describe.
        per_parameter_sparsity: The achieved :meth:`PruningMask.sparsity`
            for every pruned parameter, keyed by parameter name.
        total_pruned_elements: The total element count pruned across
            every parameter on this layer.
        total_element_count: The total element count considered across
            every parameter on this layer (pruned and kept).
    """

    layer_name: str
    per_parameter_sparsity: Dict[str, float] = field(default_factory=dict)
    total_pruned_elements: int = 0
    total_element_count: int = 0

    def overall_sparsity(self) -> float:
        """Return this layer's aggregate sparsity across all parameters.

        Returns:
            ``0.0`` if ``total_element_count`` is ``0``; otherwise
            ``total_pruned_elements / total_element_count``.
        """
        if self.total_element_count == 0:
            return 0.0
        return self.total_pruned_elements / self.total_element_count


class PruningApplier:
    """Stateless mask-scoring and mask-application arithmetic for one
    layer at a time.

    Holds no instance state beyond its injected
    :class:`~uaqe.compression.sparse_encoder.SparseEncoder`, so a single
    instance may be safely shared (and reused across threads) by every
    pruning strategy in this package.

    Attributes:
        _sparse_encoder: Shared dense/sparse tensor conversion.
    """

    def __init__(self, sparse_encoder: Optional[SparseEncoder] = None) -> None:
        """Initialize the applier.

        Args:
            sparse_encoder: Overrides the default
                :class:`~uaqe.compression.sparse_encoder.SparseEncoder`;
                ``SparseEncoder`` is stateless, so the default instance
                is sufficient for ordinary use and this parameter exists
                mainly to ease test-double substitution.
        """
        self._sparse_encoder = sparse_encoder or SparseEncoder()

    def compute_magnitude_mask(
        self, tensor: IMRTensor, parameter_name: str, sparsity: float
    ) -> PruningMask:
        """Score every element of ``tensor`` by absolute magnitude and
        keep the ``(1 - sparsity)`` fraction with the largest magnitude.

        This is unstructured, element-wise pruning: surviving elements
        may be scattered anywhere within the tensor, which yields the
        best accuracy-per-pruned-weight trade-off but produces a
        storage layout (see :class:`~uaqe.compression.sparse_encoder.
        SparseEncoder`) that gains no benefit from hardware vector
        instructions the way whole-group removal
        (:meth:`compute_structured_mask`) does.

        Args:
            tensor: The tensor to score. Must have ``dtype ==
                "float32"``.
            parameter_name: The parameter name this mask will be
                recorded under.
            sparsity: The target fraction of elements to prune, in
                ``[0.0, 1.0)``.

        Returns:
            The computed :class:`PruningMask`.

        Raises:
            CompressionError: If ``sparsity`` is outside ``[0.0, 1.0)``,
                if ``tensor.dtype`` is not ``"float32"``, or if the
                tensor's byte length does not match its declared
                ``shape``.
        """
        values = self._read_values(tensor)
        _validate_sparsity(sparsity)

        element_count = len(values)
        keep_count = round(element_count * (1.0 - sparsity))
        ranked_indices = sorted(
            range(element_count), key=lambda index: abs(values[index]), reverse=True
        )
        kept_indices = tuple(sorted(ranked_indices[:keep_count]))
        return PruningMask(
            parameter_name=parameter_name,
            kept_indices=kept_indices,
            total_element_count=element_count,
        )

    def compute_structured_mask(
        self, tensor: IMRTensor, parameter_name: str, sparsity: float
    ) -> PruningMask:
        """Score whole leading-dimension groups (e.g. output
        channels/filters) of ``tensor`` by aggregate (L2-norm)
        magnitude and drop the weakest groups until at least
        ``sparsity`` of the tensor's elements are pruned.

        Structured pruning removes entire contiguous groups rather than
        scattered individual elements, so the surviving tensor can be
        physically reshaped smaller (dropping whole rows) instead of
        merely stored sparsely — the trade-off favored on hardware
        targets that cannot exploit unstructured sparsity, at the cost
        of typically needing a coarser (less accuracy-preserving)
        pruning decision than :meth:`compute_magnitude_mask`.

        Args:
            tensor: The tensor to score. Must have ``dtype ==
                "float32"`` and a non-empty ``shape`` (at least one
                dimension to group along).
            parameter_name: The parameter name this mask will be
                recorded under.
            sparsity: The minimum fraction of elements to prune, in
                ``[0.0, 1.0)``. Because whole groups are dropped
                together, the achieved sparsity may exceed this target
                slightly (see :meth:`PruningMask.sparsity` on the
                returned mask for the exact achieved value).

        Returns:
            The computed :class:`PruningMask`, whose ``kept_indices``
            is the union of the flat indices of every retained group.

        Raises:
            CompressionError: If ``sparsity`` is outside ``[0.0, 1.0)``,
                if ``tensor.dtype`` is not ``"float32"``, if the
                tensor's byte length does not match its declared
                ``shape``, or if ``tensor.shape`` is empty (a scalar has
                no leading dimension to group along).
        """
        values = self._read_values(tensor)
        _validate_sparsity(sparsity)
        if not tensor.shape:
            raise CompressionError(
                f"Cannot structurally prune parameter {parameter_name!r}: "
                f"a scalar (empty shape) tensor has no leading dimension "
                f"to group along.",
                code="COMPRESS_STRUCTURED_PRUNING_SCALAR",
            )

        group_count = tensor.shape[0]
        group_size = len(values) // group_count if group_count else 0
        group_norms = []
        for group_index in range(group_count):
            start = group_index * group_size
            end = start + group_size
            group_norms.append(
                (math.sqrt(sum(v * v for v in values[start:end])), group_index)
            )

        drop_group_count = round(group_count * sparsity)
        weakest_first = sorted(group_norms, key=lambda pair: pair[0])
        dropped_groups = {index for _, index in weakest_first[:drop_group_count]}

        kept_indices: List[int] = []
        for group_index in range(group_count):
            if group_index in dropped_groups:
                continue
            start = group_index * group_size
            kept_indices.extend(range(start, start + group_size))

        return PruningMask(
            parameter_name=parameter_name,
            kept_indices=tuple(kept_indices),
            total_element_count=len(values),
        )

    def apply_mask(
        self, layer: IMRLayer, masks: Dict[str, PruningMask]
    ) -> Tuple[IMRLayer, PruningStatistics]:
        """Apply every mask in ``masks`` to the matching parameter of
        ``layer``, sparse-encoding each pruned tensor.

        Parameters with no entry in ``masks`` are left unchanged.

        Args:
            layer: The layer to prune.
            masks: The :class:`PruningMask` to apply for each parameter
                that should be pruned, keyed by parameter name; must
                have been computed against that same parameter's
                current tensor (see :meth:`compute_magnitude_mask` /
                :meth:`compute_structured_mask`).

        Returns:
            A tuple of the new ``IMRLayer`` (sparse-encoded pruned
            parameters, other parameters unchanged) and the resulting
            :class:`PruningStatistics` for this layer. The input
            ``layer`` is not modified.

        Raises:
            CompressionError: If a mask's ``kept_indices`` cannot be
                encoded against its parameter's current tensor (see
                :meth:`~uaqe.compression.sparse_encoder.
                SparseEncoder.encode_tensor`).
        """
        new_parameters = {}
        per_parameter_sparsity: Dict[str, float] = {}
        total_pruned_elements = 0
        total_element_count = 0

        for param_name, tensor in layer.parameters.items():
            mask = masks.get(param_name)
            if mask is None:
                new_parameters[param_name] = tensor
                continue

            dense_size = self._sparse_encoder.dense_byte_size(tensor)
            sparse_tensor = self._sparse_encoder.encode_tensor(
                tensor, mask.kept_indices
            )
            sparse_size = len(sparse_tensor.data)

            if sparse_size > dense_size:
                # Dense fallback: keep original dtype, zero out pruned elements
                import numpy as np
                if tensor.dtype == "float32":
                    arr = np.frombuffer(tensor.data, dtype=np.float32).copy()
                elif tensor.dtype == "float16":
                    arr = np.frombuffer(tensor.data, dtype=np.float16).copy()
                elif tensor.dtype == "int8":
                    arr = np.frombuffer(tensor.data, dtype=np.int8).copy()
                else:
                    raise CompressionError(f"Unsupported dtype {tensor.dtype!r}")

                kept_set = set(mask.kept_indices)
                for index in range(len(arr)):
                    if index not in kept_set:
                        arr[index] = 0
                new_parameters[param_name] = IMRTensor(
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    data=arr.tobytes(),
                )
            else:
                new_parameters[param_name] = sparse_tensor

            per_parameter_sparsity[param_name] = mask.sparsity()
            total_pruned_elements += mask.total_element_count - len(mask.kept_indices)
            total_element_count += mask.total_element_count

        new_layer = dataclasses.replace(layer, parameters=new_parameters)
        stats = PruningStatistics(
            layer_name=layer.name,
            per_parameter_sparsity=per_parameter_sparsity,
            total_pruned_elements=total_pruned_elements,
            total_element_count=total_element_count,
        )
        return new_layer, stats

    def _read_values(self, tensor: IMRTensor) -> array.array:
        """Read ``tensor``'s raw buffer as an array of ``float32``
        values.

        Args:
            tensor: The tensor to read.

        Returns:
            The tensor's elements as a stdlib ``array.array``.

        Raises:
            CompressionError: If ``tensor.dtype`` is not supported,
                or if the buffer length does not match the declared
                ``shape``.
        """
        import numpy as np
        
        if tensor.dtype == "float32":
            f32_arr = np.frombuffer(tensor.data, dtype=np.float32)
        elif tensor.dtype == "float16":
            f32_arr = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
        elif tensor.dtype == "int8":
            f32_arr = np.frombuffer(tensor.data, dtype=np.int8).astype(np.float32)
        else:
            raise CompressionError(
                f"Cannot score tensor of dtype {tensor.dtype!r} for "
                f"pruning; only float32, float16, and int8 are supported.",
                code="COMPRESS_UNSUPPORTED_DTYPE",
            )

        expected_elements = math.prod(tensor.shape) if tensor.shape else 1
        if len(f32_arr) != expected_elements:
            raise CompressionError(
                f"Tensor declares shape {tensor.shape!r} "
                f"({expected_elements} elements) but its buffer holds "
                f"{len(f32_arr)} elements.",
                code="COMPRESS_SHAPE_MISMATCH",
            )
        return array.array(_FLOAT32_TYPECODE, f32_arr)


def _validate_sparsity(sparsity: float) -> None:
    """Validate that ``sparsity`` is a usable pruning target.

    Args:
        sparsity: The candidate sparsity value.

    Raises:
        CompressionError: If ``sparsity`` is outside ``[0.0, 1.0)``.
    """
    if not (0.0 <= sparsity < 1.0):
        raise CompressionError(
            f"Pruning sparsity must be in [0.0, 1.0), got {sparsity}.",
            code="COMPRESS_INVALID_SPARSITY",
        )
