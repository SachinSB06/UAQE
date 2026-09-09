"""Loading and batching of representative calibration samples for
``uaqe.quantization.calibrator.Calibrator``.

``CalibrationDataset`` is the single place in this package that touches
the filesystem; every other class in ``uaqe.quantization`` receives an
already-loaded ``CalibrationDataset`` (or the ``CalibrationStatistics``
derived from one) via constructor/method injection rather than reading
``dataset_path`` itself, matching the "no module reads a config file
[or, here, a dataset file] directly" spirit of
``10_Module_Development_Guide.md`` §0 rule 7.

On-disk format: a calibration dataset is either

- a single ``.jsonl`` file, one JSON object per line, or
- a directory containing one ``.json`` file per sample,

where each JSON object maps an ``IMR`` input tensor name to a flat list
of ``float`` values for one representative sample, e.g.::

    {"input": [0.12, -0.44, 0.31, ...]}

This mirrors the simplest possible representative-data contract a
calibration dataset can have while staying entirely within the standard
library, consistent with every other first-party module in this
codebase (``uaqe.analyzer``, ``uaqe.hardware``, ``uaqe.common`` use no
third-party dependencies).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from uaqe.common.exceptions import QuantizationError

#: File suffix recognized as one calibration sample per line.
_JSONL_SUFFIX = ".jsonl"

#: File suffix recognized as one calibration sample per file, when
#: ``dataset_path`` is a directory.
_JSON_SUFFIX = ".json"


@dataclass(frozen=True)
class CalibrationBatch:
    """One batch of representative calibration samples.

    Attributes:
        inputs: Every sample's values for each input tensor name in
            this batch, concatenated in sample order — i.e.
            ``inputs["x"]`` holds ``batch_size`` samples' worth of
            values for input tensor ``"x"``, back to back.
        sample_count: The number of individual samples folded into this
            batch (``<= batch_size`` for the final, possibly partial,
            batch of a dataset).
    """

    inputs: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    sample_count: int = 0


class CalibrationDataset:
    """A loaded, batchable collection of representative calibration
    samples.

    Attributes:
        _dataset_path: The filesystem path this dataset was loaded
            from, retained for error messages and logging only.
        _batch_size: The number of samples grouped into each yielded
            :class:`CalibrationBatch`.
        _samples: Every loaded sample, in file order, each mapping
            input tensor name to that sample's flat value list.
    """

    def __init__(self, dataset_path: str, batch_size: int) -> None:
        """Load every calibration sample under ``dataset_path``.

        Args:
            dataset_path: Path to a single ``.jsonl`` file or to a
                directory of ``.json`` sample files.
            batch_size: The number of samples grouped into each batch
                yielded by iteration. Must be a positive integer.

        Raises:
            QuantizationError: If ``dataset_path`` does not exist, is
                empty, contains malformed JSON, or if ``batch_size`` is
                not positive.
        """
        if batch_size <= 0:
            raise QuantizationError(
                f"calibration_batch_size must be positive, got {batch_size}.",
                code="QUANT_INVALID_BATCH_SIZE",
                remediation_hint="Set QuantizationConfig.calibration_batch_size >= 1.",
            )
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._samples: List[Dict[str, List[float]]] = self._load(Path(dataset_path))

    @property
    def batch_size(self) -> int:
        """The configured number of samples per yielded batch."""
        return self._batch_size

    def sample_count(self) -> int:
        """Return the total number of individual samples loaded."""
        return len(self._samples)

    def input_names(self) -> List[str]:
        """Return every input tensor name observed across all samples,
        sorted for determinism.
        """
        names = set()
        for sample in self._samples:
            names.update(sample.keys())
        return sorted(names)

    def __len__(self) -> int:
        """Return the number of batches iteration will yield."""
        if not self._samples:
            return 0
        return -(-len(self._samples) // self._batch_size)  # ceiling division

    def __iter__(self) -> Iterator[CalibrationBatch]:
        """Yield every sample grouped into ``batch_size``-sized
        :class:`CalibrationBatch` instances, in load order.
        """
        for start in range(0, len(self._samples), self._batch_size):
            chunk = self._samples[start : start + self._batch_size]
            merged: Dict[str, List[float]] = {}
            for sample in chunk:
                for name, values in sample.items():
                    merged.setdefault(name, []).extend(values)
            yield CalibrationBatch(
                inputs={name: tuple(values) for name, values in merged.items()},
                sample_count=len(chunk),
            )

    def _load(self, path: Path) -> List[Dict[str, List[float]]]:
        """Dispatch loading to the ``.jsonl``-file or sample-directory
        reader based on ``path``, and validate the result is non-empty.
        """
        if not path.exists():
            raise QuantizationError(
                f"Calibration dataset path does not exist: {path}.",
                code="QUANT_DATASET_NOT_FOUND",
                remediation_hint=(
                    "Point QuantizationConfig at a valid .jsonl file or "
                    "sample directory."
                ),
            )

        if path.is_dir():
            samples = self._load_directory(path)
        elif path.suffix == _JSONL_SUFFIX:
            samples = self._load_jsonl(path)
        else:
            raise QuantizationError(
                f"Unrecognized calibration dataset format: {path}. "
                f"Expected a directory of {_JSON_SUFFIX} files or a "
                f"single {_JSONL_SUFFIX} file.",
                code="QUANT_DATASET_UNSUPPORTED_FORMAT",
            )

        if not samples:
            raise QuantizationError(
                f"Calibration dataset at {path} contains no samples.",
                code="QUANT_DATASET_EMPTY",
            )
        return samples

    def _load_directory(self, path: Path) -> List[Dict[str, List[float]]]:
        """Load one sample per ``.json`` file in ``path``, sorted by
        filename for determinism.
        """
        sample_files = sorted(path.glob(f"*{_JSON_SUFFIX}"))
        return [self._parse_sample(f.read_text(), source=str(f)) for f in sample_files]

    def _load_jsonl(self, path: Path) -> List[Dict[str, List[float]]]:
        """Load one sample per non-blank line of ``path``."""
        samples = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            samples.append(
                self._parse_sample(line, source=f"{path}:{line_number}")
            )
        return samples

    @staticmethod
    def _parse_sample(text: str, *, source: str) -> Dict[str, List[float]]:
        """Parse and validate a single sample's JSON text.

        Args:
            text: The raw JSON text for one sample.
            source: A human-readable origin (file path, optionally with
                line number) used in error messages.

        Returns:
            The parsed sample, mapping input tensor name to its flat
            list of float values.

        Raises:
            QuantizationError: If ``text`` is not valid JSON, is not a
                JSON object, or any value is not a list of numbers.
        """
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as err:
            raise QuantizationError(
                f"Malformed JSON in calibration sample at {source}: {err}.",
                code="QUANT_DATASET_MALFORMED_JSON",
            ) from err

        if not isinstance(parsed, dict):
            raise QuantizationError(
                f"Calibration sample at {source} must be a JSON object "
                f"mapping input tensor name to a list of values.",
                code="QUANT_DATASET_MALFORMED_SAMPLE",
            )

        sample: Dict[str, List[float]] = {}
        for name, values in parsed.items():
            if not isinstance(values, list) or not all(
                isinstance(value, (int, float)) for value in values
            ):
                raise QuantizationError(
                    f"Calibration sample at {source}, input {name!r} must "
                    f"be a list of numbers.",
                    code="QUANT_DATASET_MALFORMED_SAMPLE",
                )
            sample[name] = [float(value) for value in values]
        return sample
