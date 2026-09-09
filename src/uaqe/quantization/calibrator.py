"""Per-layer calibration statistics collection.

``Calibrator`` is the first stage of the quantization pipeline
(``Calibrator`` -> ``SensitivityAnalyzer`` -> ``PrecisionRecommender`` ->
``QuantizationPlanner``, per this package's ``__init__.py``). It has no
model-execution runtime available to it (``uaqe`` has no forward-pass
execution engine anywhere in this codebase — see
``uaqe.model_loader``/``uaqe.analyzer``, which are purely static), so it
derives per-layer numeric-range statistics from two static sources
instead of a true forward pass:

1. Each layer's own learned ``float32`` parameter tensors (weights,
   biases), which are always available directly on the ``IMR``.
2. The representative input values supplied by a
   ``uaqe.quantization.calibration_dataset.CalibrationDataset``, folded
   in wherever a dataset input name matches one of a layer's declared
   ``IMRLayer.inputs``.

This is a documented approximation, not a true activation-capture
calibration pass; it is sufficient for this package's purpose (choosing
a symmetric quantization range per layer) without requiring a
framework-specific execution runtime, which is out of scope for
``uaqe`` per ``01_Project_Architecture.md``.
"""

from __future__ import annotations

import array
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from uaqe.common.exceptions import QuantizationError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.domain.pipeline_stage import PipelineStage
from uaqe.quantization.calibration_dataset import CalibrationDataset

#: Default number of equal-width bins used for each layer's activation
#: histogram when a caller does not override it.
_DEFAULT_HISTOGRAM_BINS = 256


@dataclass(frozen=True)
class CalibrationStatistics:
    """Per-layer numeric-range statistics gathered by :class:`Calibrator`.

    Attributes:
        per_layer_activation_range: The observed ``(min, max)`` value
            range for each layer, keyed by ``IMRLayer.name``.
        per_layer_histogram: An equal-width value histogram (bin counts)
            for each layer, keyed by ``IMRLayer.name``.
        sample_count: The number of calibration-dataset samples folded
            into these statistics (``0`` if calibration ran from layer
            weights alone, with no dataset supplied).
        per_layer_abs_max: The absolute maximum observed activation
            value for each layer, keyed by ``IMRLayer.name``.
    """

    per_layer_activation_range: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )
    per_layer_histogram: Dict[str, List[int]] = field(default_factory=dict)
    sample_count: int = 0
    per_layer_abs_max: Dict[str, float] = field(default_factory=dict)


class Calibrator(PipelineStage):
    """Collects per-layer numeric-range statistics for an ``IMR``.

    When a model path and calibration dataset path are available,
    runs genuine forward passes with ONNX Runtime to accumulate
    empirical activation statistics. Otherwise falls back to static
    range collection from parameter buffers.

    Attributes:
        _logger: Structured logging sink.
        _num_histogram_bins: The number of equal-width bins used per
            layer's activation histogram.
    """

    def __init__(
        self,
        logger: ILogger,
        num_histogram_bins: int = _DEFAULT_HISTOGRAM_BINS,
        model_path: Optional[str] = None,
        calibration_dataset_path: Optional[str] = None,
    ) -> None:
        """Initialize the calibrator.

        Args:
            logger: Structured logging sink; every module logs through
                ``ILogger``, never ``print()``.
            num_histogram_bins: The number of equal-width bins to use
                per layer's activation histogram. Must be positive.
            model_path: Optional filesystem path to the model file to
                run forward passes against.
            calibration_dataset_path: Optional filesystem path to the
                representative calibration dataset.

        Raises:
            QuantizationError: If ``num_histogram_bins`` is not
                positive.
        """
        if num_histogram_bins <= 0:
            raise QuantizationError(
                f"num_histogram_bins must be positive, got {num_histogram_bins}.",
                code="QUANT_INVALID_HISTOGRAM_BINS",
            )
        self._logger = logger
        self._num_histogram_bins = num_histogram_bins
        self._model_path = model_path
        self._calibration_dataset_path = calibration_dataset_path

    def bind_model_path(self, path: Optional[str]) -> None:
        """Bind the model path for runtime execution."""
        self._model_path = path

    def bind_calibration_dataset_path(self, path: Optional[str]) -> None:
        """Bind the calibration dataset path for runtime execution."""
        self._calibration_dataset_path = path

    def execute(self, context: PipelineContext) -> StageResult:
        """Calibrate the current run's ``IMR`` and record the result."""
        start = time.monotonic()
        imr = context.get("model_loader").payload
        dataset = None
        if context.has("calibration_dataset"):
            dataset = context.get("calibration_dataset").payload

        model_path = self._model_path
        calibration_path = self._calibration_dataset_path
        if context.has("workflow_config"):
            w_config = context.get("workflow_config").payload
            if not model_path:
                model_path = getattr(w_config, "model_path", None)
            if not calibration_path:
                calibration_path = getattr(w_config, "calibration_dataset_path", None)

        self._logger.info(
            "Starting calibration.",
            stage_name=self.name(),
            layer_count=len(imr.layers),
            model_path=model_path,
            calibration_path=calibration_path,
        )

        stats: Optional[CalibrationStatistics] = None
        if model_path and os.path.exists(model_path) and calibration_path and os.path.exists(calibration_path):
            try:
                stats = self.calibrate_with_model(imr, model_path, calibration_path)
            except Exception as exc:
                self._logger.warning(
                    f"Real ONNX Runtime calibration failed; falling back to static parameter calibration: {exc}",
                    stage_name=self.name(),
                )

        if stats is None:
            stats = self.calibrate(imr, dataset)

        duration_ms = (time.monotonic() - start) * 1000.0
        self._logger.info(
            "Calibration complete.",
            stage_name=self.name(),
            sample_count=stats.sample_count,
            duration_ms=duration_ms,
        )
        return StageResult(
            stage_name=self.name(),
            success=True,
            payload=stats,
            duration_ms=duration_ms,
        )

    def calibrate_with_model(
        self, imr: IMR, model_path: str, calibration_path: str
    ) -> CalibrationStatistics:
        """Execute genuine ONNX Runtime forward passes to capture real
        activation statistics across all layers.
        """
        import onnx
        import onnxruntime as ort

        # 1. Load calibration inputs
        inputs_list = self._load_calibration_samples(calibration_path)
        if not inputs_list:
            raise QuantizationError(
                f"No valid calibration samples found at {calibration_path}",
                code="CALIBRATION_EMPTY_DATASET",
            )

        # 2. Build ONNX model with all intermediate node outputs exposed
        model = onnx.load(model_path)
        existing_outputs = {out.name for out in model.graph.output}
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        node_outputs = {out for node in model.graph.node for out in node.output}

        layer_output_map: Dict[str, str] = {}
        for layer in imr.layers:
            for out_name in layer.outputs:
                layer_output_map[out_name] = layer.name
                if out_name not in existing_outputs:
                    if out_name in value_infos:
                        model.graph.output.append(value_infos[out_name])
                        existing_outputs.add(out_name)
                    elif out_name in node_outputs:
                        tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, None)
                        vi = onnx.helper.make_value_info(out_name, tp)
                        model.graph.output.append(vi)
                        existing_outputs.add(out_name)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess = ort.InferenceSession(model.SerializeToString(), sess_options)
        input_meta = sess.get_inputs()[0]
        input_name = input_meta.name
        expected_shape = list(input_meta.shape)
        sess_output_names = [out.name for out in sess.get_outputs()]

        # 3. Accumulate per-layer min, max, abs_max and values
        per_layer_min: Dict[str, float] = {}
        per_layer_max: Dict[str, float] = {}
        per_layer_abs_max: Dict[str, float] = {}
        per_layer_values: Dict[str, List[float]] = {layer.name: [] for layer in imr.layers}

        sample_count = 0
        for sample in inputs_list:
            x = sample.astype(np.float32)
            if len(x.shape) == 3 and len(expected_shape) == 4:
                x = np.expand_dims(x, axis=0)

            # Match batch size 1 if needed
            if expected_shape[0] == 1 and x.shape[0] != 1:
                x = x[:1]

            results = sess.run(sess_output_names, {input_name: x})
            sample_count += 1

            for out_name, tensor_val in zip(sess_output_names, results):
                layer_name = layer_output_map.get(out_name)
                if not layer_name:
                    continue

                flat = tensor_val.flatten()
                t_min = float(np.min(flat))
                t_max = float(np.max(flat))
                t_abs = float(np.max(np.abs(flat)))

                if layer_name not in per_layer_min:
                    per_layer_min[layer_name] = t_min
                    per_layer_max[layer_name] = t_max
                    per_layer_abs_max[layer_name] = t_abs
                else:
                    per_layer_min[layer_name] = min(per_layer_min[layer_name], t_min)
                    per_layer_max[layer_name] = max(per_layer_max[layer_name], t_max)
                    per_layer_abs_max[layer_name] = max(per_layer_abs_max[layer_name], t_abs)

                # Subsample values for histogram to avoid excessive memory usage
                if len(flat) > 2048:
                    step = len(flat) // 2048
                    per_layer_values[layer_name].extend(flat[::step].tolist())
                else:
                    per_layer_values[layer_name].extend(flat.tolist())

        # 4. Construct histograms and final ranges
        per_layer_range: Dict[str, Tuple[float, float]] = {}
        per_layer_histogram: Dict[str, List[int]] = {}
        final_abs_max: Dict[str, float] = {}

        for layer in imr.layers:
            if layer.name in per_layer_min:
                l_min = per_layer_min[layer.name]
                l_max = per_layer_max[layer.name]
                l_abs = per_layer_abs_max[layer.name]
                vals = per_layer_values[layer.name]
                per_layer_range[layer.name] = (l_min, l_max)
                final_abs_max[layer.name] = l_abs
                per_layer_histogram[layer.name] = self._build_histogram(vals, l_min, l_max)
            else:
                # Layer not producing intermediate activation (e.g. initializers/identity)
                # Fall back to parameter tensors
                vals = []
                for p in layer.parameters.values():
                    if p.dtype == "float32":
                        arr = np.frombuffer(p.data, dtype=np.float32)
                        vals.extend(arr.tolist())
                if vals:
                    l_min, l_max = float(min(vals)), float(max(vals))
                    per_layer_range[layer.name] = (l_min, l_max)
                    final_abs_max[layer.name] = float(max(abs(l_min), abs(l_max)))
                    per_layer_histogram[layer.name] = self._build_histogram(vals, l_min, l_max)
                else:
                    per_layer_range[layer.name] = (0.0, 0.0)
                    final_abs_max[layer.name] = 0.0
                    per_layer_histogram[layer.name] = [0] * self._num_histogram_bins

        return CalibrationStatistics(
            per_layer_activation_range=per_layer_range,
            per_layer_histogram=per_layer_histogram,
            sample_count=sample_count,
            per_layer_abs_max=final_abs_max,
        )

    def _load_calibration_samples(self, calibration_path: str) -> List[np.ndarray]:
        """Load calibration samples from a manifest file or dataset directory."""
        if not os.path.exists(calibration_path):
            return []

        # Check if it is a JSON manifest file
        if calibration_path.endswith(".json") and os.path.isfile(calibration_path):
            with open(calibration_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if "images" in manifest:
                from uaqe.evaluation.real_dataset_adapter import RealDatasetAdapter

                cal_root = manifest.get("calibration_root", os.path.dirname(calibration_path))
                prep_ref = manifest.get("preprocessing_reference", {})
                class_mapping = prep_ref.get("class_mapping", {})
                prep_mode = prep_ref.get("preprocessing", "rgb_0_1")
                input_shape = tuple(prep_ref.get("input_shape", [1, 3, 128, 128]))

                adapter = RealDatasetAdapter(
                    dataset_path=cal_root if os.path.isabs(cal_root) else os.path.abspath(cal_root),
                    class_mapping=class_mapping,
                    preprocessing_mode=prep_mode,
                    input_shape=input_shape,
                )
                cur = os.path.dirname(os.path.abspath(calibration_path))
                project_root = cur
                while project_root and not os.path.exists(os.path.join(project_root, "datasets")):
                    parent = os.path.dirname(project_root)
                    if parent == project_root:
                        break
                    project_root = parent

                samples = []
                for img in manifest["images"]:
                    rel_p = img["relative_path"]
                    abs_p = os.path.abspath(rel_p)
                    if not os.path.exists(abs_p):
                        candidate = os.path.join(project_root, rel_p)
                        if os.path.exists(candidate):
                            abs_p = candidate
                        else:
                            abs_p = os.path.join(os.path.dirname(os.path.abspath(calibration_path)), os.path.basename(rel_p))
                    conf_class = img.get("mapped_configured_class", img.get("class", ""))
                    class_idx = class_mapping.get(conf_class, 0)
                    samples.append((abs_p, class_idx, img.get("class", "")))
                adapter.samples = samples

                return [adapter[i]["tensor"] for i in range(len(adapter))]

        # Check if directory of samples or JSONL
        try:
            ds = CalibrationDataset(calibration_path, batch_size=1)
            tensors = []
            for batch in ds:
                for name, vals in batch.inputs.items():
                    tensors.append(np.array(vals, dtype=np.float32))
            return tensors
        except Exception:
            return []

    def calibrate(
        self, imr: IMR, dataset: Optional[CalibrationDataset] = None
    ) -> CalibrationStatistics:
        """Compute per-layer activation-range and histogram statistics
        from layer weights and optional input batches (fallback mode).
        """
        dataset_values_by_input: Dict[str, List[float]] = {}
        sample_count = 0
        if dataset is not None:
            for batch in dataset:
                sample_count += batch.sample_count
                for name, values in batch.inputs.items():
                    dataset_values_by_input.setdefault(name, []).extend(values)

        per_layer_range: Dict[str, Tuple[float, float]] = {}
        per_layer_histogram: Dict[str, List[int]] = {}
        per_layer_abs_max: Dict[str, float] = {}

        for layer in imr.layers:
            values: List[float] = []
            for param_tensor in layer.parameters.values():
                if param_tensor.dtype != "float32":
                    continue
                try:
                    buffer = array.array("f")
                    buffer.frombytes(param_tensor.data)
                except ValueError as err:
                    raise QuantizationError(
                        f"Layer {layer.name!r} has a malformed float32 "
                        f"parameter buffer.",
                        code="QUANT_MALFORMED_BUFFER",
                        stage=self.name(),
                    ) from err
                values.extend(buffer)

            for input_name in layer.inputs:
                if input_name in dataset_values_by_input:
                    values.extend(dataset_values_by_input[input_name])

            if not values:
                per_layer_range[layer.name] = (0.0, 0.0)
                per_layer_histogram[layer.name] = [0] * self._num_histogram_bins
                per_layer_abs_max[layer.name] = 0.0
                continue

            layer_min, layer_max = min(values), max(values)
            per_layer_range[layer.name] = (layer_min, layer_max)
            per_layer_abs_max[layer.name] = max(abs(layer_min), abs(layer_max))
            per_layer_histogram[layer.name] = self._build_histogram(
                values, layer_min, layer_max
            )

        return CalibrationStatistics(
            per_layer_activation_range=per_layer_range,
            per_layer_histogram=per_layer_histogram,
            sample_count=sample_count,
            per_layer_abs_max=per_layer_abs_max,
        )

    def _build_histogram(
        self, values: List[float], value_min: float, value_max: float
    ) -> List[int]:
        """Bucket ``values`` into ``self._num_histogram_bins`` equal-width
        bins spanning ``[value_min, value_max]``.
        """
        bins = [0] * self._num_histogram_bins
        span = value_max - value_min
        if span == 0.0:
            bins[0] = len(values)
            return bins

        for value in values:
            index = int((value - value_min) / span * self._num_histogram_bins)
            if index >= self._num_histogram_bins:
                index = self._num_histogram_bins - 1
            bins[index] += 1
        return bins
