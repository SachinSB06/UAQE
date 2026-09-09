"""Weight-clustering ``ICompressionStrategy`` implementation
(``CompressionType.WEIGHT_CLUSTERING``).

``WeightClusterCompressor`` replaces every ``float32`` layer parameter
in an ``IMR`` with a small, per-tensor codebook of representative
values (found via a 1-D Lloyd's-algorithm k-means) plus one byte per
element recording which codebook entry it maps to. This trades a
little accuracy for a large storage reduction: a tensor with
``num_clusters <= 256`` distinct values needs only one byte per
element plus the small codebook itself, versus four bytes per element
for a dense ``float32`` tensor.

The codebook is not stored inline in the tensor's own byte buffer (a
uniform-width byte-per-element buffer has no room for it) but is
attached to the owning ``IMRLayer.attributes`` under a per-parameter
key, so that a downstream consumer (an exporter or evaluator) can
dequantize a clustered tensor back to ``float32`` by indexing the
codebook with each stored byte.
"""

from __future__ import annotations

import array
import dataclasses
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from uaqe.common.exceptions import CompressionError
from uaqe.common.imr import IMR, IMRLayer, IMRTensor
from uaqe.common.interfaces.i_compression_strategy import ICompressionStrategy
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.value_objects import CompressionConfig

#: This strategy's unique registration name, returned by
#: :meth:`WeightClusterCompressor.name`.
STRATEGY_NAME = "weight_clustering"

#: ``IMRTensor.dtype`` values this strategy clusters; any other dtype on a
#: layer's parameter is passed through unclustered.
_CLUSTERABLE_DTYPES = {"float32", "float16", "int8"}

#: Dtype string this module writes onto a clustered ``IMRTensor``: one
#: unsigned byte per element, indexing that parameter's codebook.
CLUSTERED_DTYPE = "clustered_uint8"

#: ``array`` module type code for 32-bit IEEE-754 floats.
_FLOAT32_TYPECODE = "f"

#: ``array`` module type code for unsigned 8-bit codebook indices.
_UINT8_TYPECODE = "B"

#: Default codebook size: 256 distinct centroids fit exactly in one
#: unsigned byte per element.
_DEFAULT_NUM_CLUSTERS = 256

#: Default cap on Lloyd's-algorithm refinement iterations per tensor.
_DEFAULT_MAX_ITERATIONS = 10

#: Fixed seed for the k-means centroid initialization, so that
#: compression of a given model is reproducible run to run.
_KMEANS_RANDOM_SEED = 0

#: The ``IMRLayer.attributes`` key template each parameter's centroid
#: codebook is attached under, formatted with the parameter name.
_CODEBOOK_ATTRIBUTE_TEMPLATE = "{param_name}_codebook"


@dataclass(frozen=True)
class ClusteringCodebook:
    """The centroid values a clustered tensor's byte indices refer to.

    Attributes:
        centroids: The representative ``float32`` value for every
            codebook entry, indexed by the corresponding stored byte
            (``centroids[stored_byte]`` reconstructs the approximate
            original value).
    """

    centroids: Tuple[float, ...]


class WeightClusterCompressor(ICompressionStrategy):
    """Compresses an ``IMR`` by replacing tensors with a small
    per-tensor value codebook plus one index byte per element.

    Attributes:
        _logger: Structured logging sink.
        _num_clusters: The codebook size (number of distinct centroids)
            computed per tensor, capped at 256 by the one-byte-per-index
            storage format.
        _max_iterations: The maximum number of Lloyd's-algorithm
            refinement iterations run per tensor.
    """

    def __init__(
        self,
        logger: ILogger,
        num_clusters: int = _DEFAULT_NUM_CLUSTERS,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """Initialize the strategy.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            num_clusters: The codebook size to target for every
                clustered tensor. Must be in ``[1, 256]``, since each
                element's codebook index is stored in a single
                unsigned byte.
            max_iterations: The maximum number of Lloyd's-algorithm
                refinement iterations to run per tensor before
                returning the best centroids found so far.

        Raises:
            CompressionError: If ``num_clusters`` is outside
                ``[1, 256]``.
        """
        if not (1 <= num_clusters <= 256):
            raise CompressionError(
                f"num_clusters must be in [1, 256] to fit a single "
                f"index byte per element, got {num_clusters}.",
                code="COMPRESS_INVALID_CLUSTER_COUNT",
            )
        self._logger = logger
        self._num_clusters = num_clusters
        self._max_iterations = max_iterations

    def apply(self, imr: IMR, plan: CompressionConfig) -> IMR:
        """Cluster every eligible parameter tensor of ``imr``.

        Args:
            imr: The model to compress.
            plan: The compression configuration. Not consulted directly
                by this strategy (the codebook size is a constructor
                parameter, not a per-run config field), accepted only
                to satisfy the ``ICompressionStrategy`` contract.

        Returns:
            A new ``IMR`` with every eligible parameter
            tensor replaced by a :data:`CLUSTERED_DTYPE` tensor and its
            layer's ``attributes`` populated with the corresponding
            :class:`ClusteringCodebook`. The input ``imr`` is not
            modified.

        Raises:
            CompressionError: If a tensor's byte buffer does not match
                its declared ``shape``.
        """
        del plan  # Codebook size is fixed at construction, not per-run.
        new_layers = [self._cluster_layer(layer) for layer in imr.layers]
        self._logger.info(
            "Applied weight clustering.",
            strategy=self.name(),
            num_clusters=self._num_clusters,
        )
        return dataclasses.replace(imr, layers=new_layers)

    def name(self) -> str:
        """Return this strategy's unique registration name."""
        return STRATEGY_NAME

    def _cluster_layer(self, layer: IMRLayer) -> IMRLayer:
        """Cluster every eligible parameter of one layer.

        Args:
            layer: The layer to process.

        Returns:
            A new ``IMRLayer`` with eligible parameters clustered and
            their codebooks attached to ``attributes``; ineligible
            parameters and other attributes are left unchanged.

        Raises:
            CompressionError: If an eligible tensor's byte buffer does
                not match its declared ``shape``.
        """
        new_parameters = {}
        new_attributes = dict(layer.attributes)
        for param_name, tensor in layer.parameters.items():
            if tensor.dtype not in _CLUSTERABLE_DTYPES:
                new_parameters[param_name] = tensor
                continue
            res = self._cluster_tensor(tensor, param_name)
            if res is None:
                new_parameters[param_name] = tensor
            else:
                clustered_tensor, codebook = res
                new_parameters[param_name] = clustered_tensor
                new_attributes[_CODEBOOK_ATTRIBUTE_TEMPLATE.format(param_name=param_name)] = (
                    codebook
                )
                new_attributes[f"{param_name}_source_precision"] = tensor.dtype

        return dataclasses.replace(
            layer, parameters=new_parameters, attributes=new_attributes
        )

    def _cluster_tensor(
        self, tensor: IMRTensor, param_name: str
    ) -> Tuple[IMRTensor, ClusteringCodebook] | None:
        """Cluster a single tensor into a codebook and
        per-element index bytes.

        Args:
            tensor: The tensor to cluster.
            param_name: The owning parameter's name, used only in error
                messages.

        Returns:
            A tuple of the new :data:`CLUSTERED_DTYPE` ``IMRTensor`` and
            its :class:`ClusteringCodebook`, or None if clustering would
            not reduce the tensor's size.

        Raises:
            CompressionError: If the tensor's byte buffer does not
                match its declared ``shape``.
        """
        import numpy as np
        
        if tensor.dtype == "float32":
            values = np.frombuffer(tensor.data, dtype=np.float32)
            element_size = 4
        elif tensor.dtype == "float16":
            values = np.frombuffer(tensor.data, dtype=np.float16).astype(np.float32)
            element_size = 2
        elif tensor.dtype == "int8":
            values = np.frombuffer(tensor.data, dtype=np.int8).astype(np.float32)
            element_size = 1
        else:
            raise CompressionError(
                f"Unsupported dtype {tensor.dtype!r} for clustering",
                code="COMPRESS_UNSUPPORTED_DTYPE",
            )

        N = len(values)
        expected_elements = math.prod(tensor.shape) if tensor.shape else 1
        if N != expected_elements:
            raise CompressionError(
                f"Parameter {param_name!r} declares shape "
                f"{tensor.shape!r} ({expected_elements} elements) but "
                f"its buffer holds {N} elements.",
                code="COMPRESS_SHAPE_MISMATCH",
            )

        C = min(self._num_clusters, N) or 1
        dense_size = len(tensor.data)
        clustered_size = N + C * element_size
        
        if clustered_size >= dense_size:
            # Fall back: clustering does not reduce storage size
            return None

        import array
        val_arr = array.array(_FLOAT32_TYPECODE, values)
        centroids, assignments = self._kmeans_1d(val_arr, C)

        packed = array.array(_UINT8_TYPECODE, assignments).tobytes()
        clustered_tensor = IMRTensor(
            shape=tensor.shape, dtype=CLUSTERED_DTYPE, data=packed
        )
        return clustered_tensor, ClusteringCodebook(centroids=tuple(centroids))

    def _kmeans_1d(
        self, values: array.array, num_clusters: int
    ) -> Tuple[List[float], List[int]]:
        """Run 1-D Lloyd's-algorithm k-means over ``values``.

        Args:
            values: The real-valued samples to cluster.
            num_clusters: The number of centroids to fit; must be
                ``<= len(values)``.

        Returns:
            A tuple of the fitted centroid list (length
            ``num_clusters``) and the per-element centroid-index
            assignment list (length ``len(values)``).
        """
        import numpy as np
        val_np = np.array(values, dtype=np.float32)
        rng = random.Random(_KMEANS_RANDOM_SEED)
        distinct_sorted = sorted(set(values))
        if len(distinct_sorted) <= num_clusters:
            centroids = list(distinct_sorted) + [distinct_sorted[-1]] * (
                num_clusters - len(distinct_sorted)
            )
            centroids_np = np.array(centroids, dtype=np.float32)
        else:
            centroids = rng.sample(distinct_sorted, num_clusters)
            centroids_np = np.array(centroids, dtype=np.float32)

        # To avoid high memory consumption when broadcasting:
        # We compute assignments in chunks.
        chunk_size = 100000
        assignments = np.zeros(len(val_np), dtype=np.int32)
        
        for _ in range(self._max_iterations):
            changed = False
            old_assignments = assignments.copy()
            
            # Compute new assignments in chunks
            for start_idx in range(0, len(val_np), chunk_size):
                end_idx = min(start_idx + chunk_size, len(val_np))
                val_chunk = val_np[start_idx:end_idx]
                
                # Compute distance from each value in chunk to all centroids
                dists = np.abs(val_chunk[:, None] - centroids_np[None, :])
                assignments[start_idx:end_idx] = np.argmin(dists, axis=1)
                
            if not np.array_equal(old_assignments, assignments):
                changed = True
                
            # Update centroids
            new_centroids = np.copy(centroids_np)
            for c in range(num_clusters):
                mask = (assignments == c)
                if np.any(mask):
                    new_centroids[c] = np.mean(val_np[mask])
            centroids_np = new_centroids
            
            if not changed:
                break
                
        return centroids_np.tolist(), assignments.tolist()
