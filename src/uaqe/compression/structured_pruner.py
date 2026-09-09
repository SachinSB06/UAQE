"""Structured (channel/filter-group) ``ICompressionStrategy``
implementation.

``StructuredPruner`` prunes every ``float32`` layer parameter in an
``IMR`` by dropping whole leading-dimension groups (e.g. output
channels or filters) with the weakest aggregate (L2-norm) magnitude,
via the shared mask scoring/application logic in
:class:`~uaqe.compression.pruning.PruningApplier`. Unlike
:class:`~uaqe.compression.magnitude_pruner.MagnitudePruner`'s
scattered, element-wise pruning, this strategy's coarser, group-wise
decisions preserve dense (non-sparse) hardware execution on targets
that cannot exploit unstructured sparsity, at the cost of typically
needing to prune more conservatively for the same accuracy budget. It
is registered with the plugin mechanism the same way any third-party
``ICompressionStrategy`` would be
(``10_Module_Development_Guide.md`` §18), so
:class:`~uaqe.compression.compression_planner.CompressionPlanner`
depends on it only through the ``ICompressionStrategy`` port, never by
importing this class directly outside of composition-root wiring.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_compression_strategy import ICompressionStrategy
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.value_objects import CompressionConfig
from uaqe.compression.pruning import PruningApplier, PruningMask

#: This strategy's unique registration name, returned by
#: :meth:`StructuredPruner.name`.
STRATEGY_NAME = "structured_pruning"

#: ``IMRTensor.dtype`` values this strategy scores and prunes; any other dtype
#: on a layer's parameter is passed through unpruned.
_PRUNABLE_DTYPES = {"float32", "float16", "int8"}

#: Minimum leading-dimension size (group count) a parameter tensor must
#: have to be eligible for structured pruning; a tensor with only one
#: group (e.g. a bias vector reshaped to a single row) has nothing to
#: drop without removing the entire parameter.
_MIN_GROUP_COUNT = 2


class StructuredPruner(ICompressionStrategy):
    """Prunes an ``IMR`` by structured, whole-group magnitude scoring.

    Attributes:
        _logger: Structured logging sink.
        _pruning_applier: Shared mask scoring/application arithmetic.
    """

    def __init__(
        self, logger: ILogger, pruning_applier: Optional[PruningApplier] = None
    ) -> None:
        """Initialize the strategy.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            pruning_applier: Overrides the default
                :class:`~uaqe.compression.pruning.PruningApplier`;
                ``PruningApplier`` is stateless, so the default instance
                is sufficient for ordinary use and this parameter exists
                mainly to ease test-double substitution.
        """
        self._logger = logger
        self._pruning_applier = pruning_applier or PruningApplier()

    def apply(self, imr: IMR, plan: CompressionConfig) -> IMR:
        """Prune every eligible layer of ``imr`` toward
        ``plan.pruning_sparsity`` by dropping whole parameter groups.

        A parameter is eligible when it is in ``_PRUNABLE_DTYPES`` and its leading
        dimension has at least :data:`_MIN_GROUP_COUNT` groups; any
        other parameter is left unchanged.

        Args:
            imr: The model to prune.
            plan: The compression configuration. Only
                ``pruning_sparsity`` is consulted.

        Returns:
            A new ``IMR`` with every eligible parameter tensor's
            weakest groups dropped and sparse-encoded. The input ``imr``
            is not modified.

        Raises:
            CompressionError: If ``plan.pruning_sparsity`` is outside
                ``[0.0, 1.0)`` (see
                :meth:`~uaqe.compression.pruning.PruningApplier.
                compute_structured_mask`), or if a tensor cannot be
                scored or sparse-encoded.
        """
        new_layers = []
        total_pruned_elements = 0
        total_element_count = 0

        for layer in imr.layers:
            masks: Dict[str, PruningMask] = {}
            for param_name, tensor in layer.parameters.items():
                if tensor.dtype not in _PRUNABLE_DTYPES:
                    continue
                if not tensor.shape or tensor.shape[0] < _MIN_GROUP_COUNT:
                    continue
                masks[param_name] = self._pruning_applier.compute_structured_mask(
                    tensor, param_name, plan.pruning_sparsity
                )

            if not masks:
                new_layers.append(layer)
                continue

            new_layer, stats = self._pruning_applier.apply_mask(layer, masks)
            new_layers.append(new_layer)
            total_pruned_elements += stats.total_pruned_elements
            total_element_count += stats.total_element_count

        overall_sparsity = (
            total_pruned_elements / total_element_count
            if total_element_count
            else 0.0
        )
        self._logger.info(
            "Applied structured pruning.",
            strategy=self.name(),
            target_sparsity=plan.pruning_sparsity,
            achieved_sparsity=overall_sparsity,
        )
        return dataclasses.replace(imr, layers=new_layers)

    def name(self) -> str:
        """Return this strategy's unique registration name."""
        return STRATEGY_NAME
