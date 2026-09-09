"""Static, dataset-informed accuracy-retention scoring.

Like :class:`~uaqe.quantization.calibrator.Calibrator`, this module has
no forward-pass execution engine available to it (``uaqe`` performs no
framework-specific inference anywhere outside of
``uaqe.infrastructure.framework_adapters``, and even those only load
models — see that package's docstrings). It therefore cannot compute a
true "percent of labeled examples classified correctly" accuracy
figure. Instead it derives a documented, order-of-magnitude
**accuracy-retention proxy**: how much of the baseline model's fidelity
the optimized model is estimated to retain, from two static sources:

1. Each layer's :class:`~uaqe.common.types.Precision`, before and after
   optimization, scored via :data:`ACCURACY_RETENTION_BY_PRECISION` — a
   reference table of "how much signal is typically preserved at this
   numeric precision", the same style of reference-table estimation
   :class:`~uaqe.hardware.power_estimator.PowerEstimator` already uses
   for power draw.
2. Each layer's parameter volume (element count), used to weight that
   layer's contribution to the model-level score — a larger layer
   losing precision matters more than a small one.

When ``dataset_path`` is supplied, this module additionally counts the
representative samples found there (reusing the same ``.jsonl``-file-
or ``.json``-per-sample directory convention
:class:`~uaqe.quantization.calibration_dataset.CalibrationDataset`
already established) and records that count as
``AccuracyScore.sample_count`` — a transparency signal for how much
representative data backed this evaluation, without this module itself
depending on that package (``09_Architecture_Lock.md`` §8 permits
domain-to-domain imports, but a fresh, minimal loader here keeps this
package's only dependency on dataset conventions self-contained, not on
``uaqe.quantization`` internals).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from uaqe.common.imr import IMR, IMRLayer
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.types import Precision
from uaqe.evaluation.evaluation_result import AccuracyScore

#: Reference accuracy-retention fraction at each precision, relative to
#: an unquantized FP32 model. Order-of-magnitude figures representative
#: of typical post-training quantization error, not a measured value
#: for any specific model or dataset — a caller needing a precise
#: figure should score the exported artifact against a labeled
#: evaluation set using an external ML evaluation harness; this proxy
#: exists so a relative accuracy signal is available without one.
ACCURACY_RETENTION_BY_PRECISION: Dict[Precision, float] = {
    Precision.FP32: 1.0,
    Precision.FP16: 0.999,
    Precision.MIXED: 0.995,
    Precision.INT8: 0.97,
    Precision.INT4: 0.90,
}

#: File suffix recognized as one evaluation sample per line, mirroring
#: ``uaqe.quantization.calibration_dataset``'s on-disk convention.
_JSONL_SUFFIX = ".jsonl"

#: File suffix recognized as one evaluation sample per file, when
#: ``dataset_path`` is a directory.
_JSON_SUFFIX = ".json"


@dataclass(frozen=True)
class _LayerMatch:
    """One baseline/optimized layer pairing used during scoring.

    Attributes:
        name: The shared ``IMRLayer.name``.
        baseline_precision: The layer's precision in the baseline IMR.
        optimized_precision: The layer's precision in the optimized
            IMR.
        parameter_count: The layer's total parameter element count in
            the baseline IMR, used as this pairing's scoring weight.
    """

    name: str
    baseline_precision: Precision
    optimized_precision: Precision
    parameter_count: int


class AccuracyEvaluator:
    """Scores accuracy retention between a baseline and an optimized
    ``IMR``.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            evaluator operates silently.
    """

    def __init__(self, logger: Optional[ILogger] = None) -> None:
        """Initialize an ``AccuracyEvaluator``.

        Args:
            logger: Optional structured logging sink.
        """
        self.logger: Optional[ILogger] = logger

    def evaluate(
        self,
        baseline_imr: IMR,
        optimized_imr: IMR,
        dataset_path: Optional[str] = None,
    ) -> AccuracyScore:
        """Score accuracy retention between ``baseline_imr`` and
        ``optimized_imr``.

        Args:
            baseline_imr: The pre-optimization model.
            optimized_imr: The final, post-optimization model.
            dataset_path: Optional path to a representative evaluation
                dataset (see the module docstring for the accepted
                on-disk formats). Only its sample count is used;
                malformed or missing paths degrade to
                ``sample_count=0`` with a recorded assumption rather
                than raising, since dataset presence is not required
                for this proxy to produce a score.

        Returns:
            The resulting :class:`~uaqe.evaluation.evaluation_result.
            AccuracyScore`.
        """
        matches = self._match_layers(baseline_imr, optimized_imr)
        per_layer_error: Dict[str, float] = {}
        total_weight = 0
        weighted_retention = 0.0

        for match in matches:
            weight = max(match.parameter_count, 1)
            total_weight += weight
            baseline_retention = ACCURACY_RETENTION_BY_PRECISION.get(
                match.baseline_precision, 1.0
            )
            optimized_retention = ACCURACY_RETENTION_BY_PRECISION.get(
                match.optimized_precision, 1.0
            )
            weighted_retention += optimized_retention * weight
            if match.optimized_precision != match.baseline_precision:
                per_layer_error[match.name] = max(
                    0.0, baseline_retention - optimized_retention
                )

        optimized_accuracy = (
            weighted_retention / total_weight if total_weight > 0 else 1.0
        )
        baseline_accuracy = 1.0

        sample_count, dataset_assumption = self._count_samples(dataset_path)
        assumptions = [
            "accuracy figures are a static, order-of-magnitude "
            "precision-retention proxy weighted by parameter volume, "
            "not a measured labeled-dataset accuracy; uaqe has no "
            "forward-pass execution engine to compute the latter.",
            dataset_assumption,
        ]

        if self.logger is not None:
            self.logger.info(
                "Accuracy retention evaluated.",
                optimized_accuracy=optimized_accuracy,
                changed_layer_count=len(per_layer_error),
                sample_count=sample_count,
            )

        return AccuracyScore(
            baseline_accuracy=baseline_accuracy,
            optimized_accuracy=optimized_accuracy,
            accuracy_delta=optimized_accuracy - baseline_accuracy,
            per_layer_error=per_layer_error,
            sample_count=sample_count,
            assumptions=assumptions,
        )

    def _match_layers(
        self, baseline_imr: IMR, optimized_imr: IMR
    ) -> List[_LayerMatch]:
        """Pair up layers present in both IMRs by name.

        Layers removed by upstream structural passes (graph cleanup,
        fusion, pruning) have no counterpart in ``optimized_imr`` and
        are excluded — their contribution to the baseline is silently
        dropped rather than penalized, since a removed layer's
        precision is no longer a meaningful comparison point.

        Args:
            baseline_imr: The pre-optimization model.
            optimized_imr: The post-optimization model.

        Returns:
            One :class:`_LayerMatch` per layer name present in both
            IMRs.
        """
        optimized_by_name: Dict[str, IMRLayer] = {
            layer.name: layer for layer in optimized_imr.layers
        }
        matches: List[_LayerMatch] = []
        for baseline_layer in baseline_imr.layers:
            optimized_layer = optimized_by_name.get(baseline_layer.name)
            if optimized_layer is None:
                continue
            parameter_count = sum(
                self._element_count(tensor.shape)
                for tensor in baseline_layer.parameters.values()
            )
            matches.append(
                _LayerMatch(
                    name=baseline_layer.name,
                    baseline_precision=baseline_layer.precision,
                    optimized_precision=optimized_layer.precision,
                    parameter_count=parameter_count,
                )
            )
        return matches

    def _element_count(self, shape) -> int:
        """Compute the number of elements implied by a tensor shape.

        Args:
            shape: A tuple of dimension sizes.

        Returns:
            The product of every dimension, or ``0`` for an empty
            shape.
        """
        if not shape:
            return 0
        total = 1
        for dimension in shape:
            total *= dimension
        return total

    def _count_samples(self, dataset_path: Optional[str]) -> "tuple[int, str]":
        """Count representative samples under ``dataset_path``.

        Args:
            dataset_path: Path to a single ``.jsonl`` file or a
                directory of ``.json`` sample files, or ``None``.

        Returns:
            A ``(sample_count, assumption_note)`` tuple.
        """
        if not dataset_path:
            return 0, (
                "No dataset_path supplied; score derived from static "
                "per-layer precision/parameter analysis alone."
            )

        path = Path(dataset_path)
        try:
            if path.is_file() and path.suffix == _JSONL_SUFFIX:
                with path.open("r", encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())
                return count, f"Counted {count} sample(s) from {dataset_path!r}."
            if path.is_dir():
                count = sum(1 for _ in path.glob(f"*{_JSON_SUFFIX}"))
                return count, f"Counted {count} sample(s) under {dataset_path!r}."
            return 0, (
                f"dataset_path {dataset_path!r} is neither a {_JSONL_SUFFIX} "
                "file nor a directory of samples; score derived from "
                "static analysis alone."
            )
        except OSError as error:
            return 0, (
                f"dataset_path {dataset_path!r} could not be read "
                f"({error}); score derived from static analysis alone."
            )
