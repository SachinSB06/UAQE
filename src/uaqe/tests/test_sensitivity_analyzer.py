"""Smoke test / minimal reproduction for the SensitivityAnalyzer /
PrecisionRecommender "default INT8 silently promoted to FP32" bug.

Root cause (see ``sensitivity_analyzer.py`` module docstring and
``_estimate_sensitivity`` docstring for the full write-up): the
previous ``range_spread * peak_fraction`` formula degenerated to a
near-constant, always-positive score for almost any ordinary,
mixed-sign layer, so with ``QuantizationConfig.sensitivity_threshold``
at its dataclass default of ``0.0``, ``score > threshold`` was true for
nearly every layer, and ``PrecisionRecommender`` stepped every one of
them up away from the user's requested default precision.

Follows this package's existing plain-script test convention (see
``uaqe.analyzer.test_analyzer`` / ``test_analyzer_core``): run directly
with ``python3 -m uaqe.quantization.test_sensitivity_analyzer`` (or as
a script), assertions raise on failure, progress is printed.
"""

from __future__ import annotations

import random

from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.types import HardwareClass, Precision
from uaqe.common.value_objects import QuantizationConfig
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.quantization.calibrator import CalibrationStatistics
from uaqe.quantization.precision_recommender import PrecisionRecommender
from uaqe.quantization.sensitivity_analyzer import SensitivityAnalyzer


class FakeLogger:
    def debug(self, msg, **f):
        pass

    def info(self, msg, **f):
        pass

    def warning(self, msg, **f):
        print("WARN", msg, f)

    def error(self, msg, **f):
        print("ERROR", msg, f)

    def critical(self, msg, **f):
        print("CRIT", msg, f)


def _float32_tensor(values):
    import array

    buf = array.array("f", values)
    return IMRTensor(shape=(len(values),), dtype="float32", data=buf.tobytes())


def _make_histogram(values, bins=256):
    value_min, value_max = min(values), max(values)
    hist = [0] * bins
    span = value_max - value_min
    for value in values:
        index = int((value - value_min) / span * bins) if span else 0
        if index >= bins:
            index = bins - 1
        hist[index] += 1
    return (value_min, value_max), hist


def _profile():
    return HardwareProfile(
        profile_id="raspberrypi4",
        display_name="Raspberry Pi 4",
        hardware_class=HardwareClass.RASPBERRY_PI,
        ram_bytes=4 * 1024 * 1024 * 1024,
        flash_bytes=None,
        storage_bytes=32 * 1024 * 1024 * 1024,
        tensor_memory_bytes=512 * 1024 * 1024,
        runtime="tflite-runtime",
        default_runtime="tflite-runtime",
        supported_runtimes=["tflite-runtime"],
        max_model_size_bytes=200 * 1024 * 1024,
        schema_version="1.0",
        supported_precisions=[Precision.FP32, Precision.FP16, Precision.INT8],
    )


def main() -> None:
    random.seed(1234)

    # --- 1. Reproduce the raw scoring bug on ordinary mixed-sign data.
    ordinary_values = [random.gauss(0, 1) for _ in range(2000)]
    value_range, histogram = _make_histogram(ordinary_values)
    score = SensitivityAnalyzer._estimate_sensitivity(value_range, histogram)
    print(f"ordinary-layer score={score:.6f}")
    assert score == 0.0, (
        "An ordinary, well-conditioned mixed-sign layer must not score "
        f"above 0.0; got {score}."
    )

    # A layer with genuine outliers should still be flagged.
    outlier_values = [random.gauss(0, 0.05) for _ in range(2000)]
    outlier_values[0], outlier_values[1] = 50.0, -45.0
    outlier_range, outlier_hist = _make_histogram(outlier_values)
    outlier_score = SensitivityAnalyzer._estimate_sensitivity(
        outlier_range, outlier_hist
    )
    print(f"genuine-outlier-layer score={outlier_score:.6f}")
    assert outlier_score > 0.0, "A layer with genuine outliers must still be flagged."

    # --- 2. End-to-end: build an IMR with several ordinary layers and
    # confirm the default INT8 precision now survives, at the
    # QuantizationConfig dataclass default threshold of 0.0.
    layers = []
    for i in range(6):
        weight_values = [random.gauss(0, 1) for _ in range(512)]
        layers.append(
            IMRLayer(
                name=f"layer_{i}",
                op_type="Linear",
                inputs=[],
                outputs=[f"layer_{i}::out"],
                parameters={"weight": _float32_tensor(weight_values)},
                attributes={},
                precision=Precision.FP32,
            )
        )
    # One genuinely pathological layer, dominated by a couple of huge
    # outlier weights (e.g. a badly-initialized or diverging layer).
    pathological_values = [random.gauss(0, 0.02) for _ in range(512)]
    pathological_values[0], pathological_values[1] = 80.0, -75.0
    layers.append(
        IMRLayer(
            name="pathological_layer",
            op_type="Linear",
            inputs=[],
            outputs=["pathological_layer::out"],
            parameters={"weight": _float32_tensor(pathological_values)},
            attributes={},
            precision=Precision.FP32,
        )
    )

    imr = IMR(
        layers=layers,
        metadata=IMRMetadata(
            source_framework="pytorch",
            source_format=".pth",
            original_input_shapes={"input": (1, 16)},
            op_count=len(layers),
            total_parameters=sum(len(v) for v in [pathological_values] + [weight_values]),
        ),
    )

    per_layer_range = {}
    per_layer_hist = {}
    for layer in imr.layers:
        values = list(
            __import__("array").array("f", layer.parameters["weight"].data)
        )
        vr, hist = _make_histogram(values)
        per_layer_range[layer.name] = vr
        per_layer_hist[layer.name] = hist
    stats = CalibrationStatistics(
        per_layer_activation_range=per_layer_range,
        per_layer_histogram=per_layer_hist,
        sample_count=0,
    )

    config = QuantizationConfig(default_precision=Precision.INT8)
    assert config.sensitivity_threshold == 0.0, "test assumes the dataclass default"

    analyzer = SensitivityAnalyzer(logger=FakeLogger(), default_threshold=config.sensitivity_threshold)
    report = analyzer.analyze(imr, stats, config.sensitivity_threshold)
    print("flagged layers:", report.flagged_layers)
    assert report.flagged_layers == ["pathological_layer"], (
        "Only the genuinely pathological layer should be flagged at "
        f"threshold=0.0; got {report.flagged_layers}."
    )

    recommender = PrecisionRecommender(logger=FakeLogger())
    recommendation = recommender.recommend(imr, _profile(), config, sensitivity=report)
    for layer in imr.layers:
        precision = recommendation.per_layer_precision[layer.name]
        print(f"{layer.name}: {precision.value} -- {recommendation.rationale[layer.name]}")

    ordinary_precisions = {
        recommendation.per_layer_precision[f"layer_{i}"] for i in range(6)
    }
    assert ordinary_precisions == {Precision.INT8}, (
        "Ordinary layers must remain at the requested default INT8 "
        f"precision; got {ordinary_precisions}."
    )
    assert recommendation.per_layer_precision["pathological_layer"] != Precision.INT8, (
        "The genuinely sensitive layer should still be steppable to a "
        "higher precision."
    )

    print("OK — default INT8 preserved for ordinary layers; "
          "genuinely sensitive layer still promoted.")


if __name__ == "__main__":
    main()
