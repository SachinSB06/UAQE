"""Smoke test for uaqe.quantization, exercised end-to-end with a
synthetic IMR (no real model file / framework needed, since IMR is
framework-independent).
"""
import array
import json
import sys
import tempfile
import traceback
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, SRC)

from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.common.result_types import StageResult
from uaqe.common.types import Precision, HardwareClass
from uaqe.common.value_objects import QuantizationConfig
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext

from uaqe.quantization import (
    Calibrator,
    SensitivityAnalyzer,
    PrecisionRecommender,
    Int8Quantizer,
    Int4Quantizer,
    MixedPrecisionQuantizer,
    QuantizationPlanner,
    QuantizationReportRenderer,
    LayerQuantizer,
    CalibrationDataset,
)


class CollectingLogger(ILogger):
    def __init__(self):
        self.lines = []
    def debug(self, msg, **fields): self.lines.append(("DEBUG", msg, fields))
    def info(self, msg, **fields): self.lines.append(("INFO", msg, fields))
    def warning(self, msg, **fields): self.lines.append(("WARNING", msg, fields)); print("WARN:", msg, fields)
    def error(self, msg, **fields): self.lines.append(("ERROR", msg, fields)); print("ERROR:", msg, fields)
    def critical(self, msg, **fields): self.lines.append(("CRITICAL", msg, fields)); print("CRIT:", msg, fields)


def f32(values):
    return array.array("f", values).tobytes()


def make_imr():
    conv_w = IMRTensor(shape=(2, 2), dtype="float32", data=f32([0.5, -0.9, 1.2, 0.05]))
    conv_b = IMRTensor(shape=(2,), dtype="float32", data=f32([0.1, -0.1]))
    fc_w = IMRTensor(shape=(3,), dtype="float32", data=f32([10.0, 10.0, 10.0]))  # low sensitivity: flat histogram
    outlier_w = IMRTensor(shape=(5,), dtype="float32", data=f32([0.001, 0.001, 0.001, 0.001, 50.0]))  # peaky/sensitive
    zero_w = IMRTensor(shape=(4,), dtype="float32", data=f32([0.0, 0.0, 0.0, 0.0]))  # degenerate all-zero

    layers = [
        IMRLayer(name="conv1", op_type="Conv2D", inputs=["input"], outputs=["conv1_out"],
                  parameters={"weight": conv_w, "bias": conv_b}, attributes={}, precision=Precision.FP32),
        IMRLayer(name="relu1", op_type="ReLU", inputs=["conv1_out"], outputs=["relu1_out"],
                  parameters={}, attributes={}, precision=Precision.FP32),
        IMRLayer(name="fc1", op_type="MatMul", inputs=["relu1_out"], outputs=["fc1_out"],
                  parameters={"weight": fc_w}, attributes={}, precision=Precision.FP32),
        IMRLayer(name="fc2_sensitive", op_type="MatMul", inputs=["fc1_out"], outputs=["fc2_out"],
                  parameters={"weight": outlier_w}, attributes={}, precision=Precision.FP32),
        IMRLayer(name="bn_degenerate", op_type="BatchNorm", inputs=["fc2_out"], outputs=["out"],
                  parameters={"weight": zero_w}, attributes={}, precision=Precision.FP32),
    ]
    metadata = IMRMetadata(source_framework="synthetic", source_format=".synthetic",
                            original_input_shapes={"input": (1, 2, 2)}, op_count=len(layers),
                            total_parameters=sum(len(t.data)//4 for l in layers for t in l.parameters.values()))
    return IMR(layers=layers, metadata=metadata)


def make_profile(profile_id="esp32", supported=(Precision.FP32, Precision.INT8, Precision.INT4)):
    return HardwareProfile(
        profile_id=profile_id, display_name="Test Board", hardware_class=HardwareClass.EMBEDDED,
        ram_bytes=520_000, flash_bytes=4_000_000, storage_bytes=None, tensor_memory_bytes=300_000,
        runtime="tflite-micro", default_runtime="tflite-micro", supported_runtimes=["tflite-micro"],
        max_model_size_bytes=4_000_000, schema_version="1.0",
        supported_precisions=list(supported),
    )


def build_context(logger, imr, profile):
    ctx = PipelineContext(run_id="smoke-run-1")
    ctx.append("model_loader", StageResult(stage_name="model_loader", success=True, payload=imr))
    ctx.append("hardware_manager", StageResult(stage_name="hardware_manager", success=True, payload=profile))
    return ctx


def run_pipeline(logger, imr, profile, config):
    ctx = build_context(logger, imr, profile)

    calibrator = Calibrator(logger)
    calib_result = calibrator.execute(ctx)
    ctx.append("calibrator", calib_result)
    assert calibrator.name() == "calibrator", calibrator.name()

    analyzer = SensitivityAnalyzer(logger, default_threshold=config.sensitivity_threshold)
    sens_result = analyzer.execute(ctx)
    ctx.append("sensitivity_analyzer", sens_result)
    assert analyzer.name() == "sensitivity_analyzer", analyzer.name()

    recommender = PrecisionRecommender(logger)
    strategies = {
        "int8": Int8Quantizer(logger),
        "int4": Int4Quantizer(logger),
        "mixed_precision": MixedPrecisionQuantizer(logger),
    }
    planner = QuantizationPlanner(logger, recommender, config, strategies)
    assert planner.name() == "quantization_planner", planner.name()
    plan_result = planner.execute(ctx)
    ctx.append("quantization_planner", plan_result)

    plan, quantized_imr = plan_result.payload
    return ctx, calib_result.payload, sens_result.payload, plan, quantized_imr


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail if not condition else ''}")
    return condition


def main():
    bugs = []
    logger = CollectingLogger()
    imr = make_imr()
    profile = make_profile()

    print("=== Scenario 1: uniform INT8 default, no overrides ===")
    config = QuantizationConfig(default_precision=Precision.INT8, per_layer_overrides={},
                                 calibration_batch_size=4, sensitivity_threshold=0.05)
    try:
        ctx, calib, sens, plan, qimr = run_pipeline(logger, imr, profile, config)
        print("Selected strategy:", plan.selected_strategy_name)
        print("Per-layer precision:", {k: v.value for k, v in plan.per_layer_precision.items()})
        print("Flagged layers:", sens.flagged_layers)

        if not check("selected strategy is int8 for uniform INT8 plan",
                      plan.selected_strategy_name == "int8", plan.selected_strategy_name):
            bugs.append("Uniform INT8 recommendation did not select the int8 single-precision strategy")

        conv_w_q = qimr.get_layer("conv1").parameters["weight"]
        if not check("conv1 weight tensor dtype changed to int8", conv_w_q.dtype == "int8", conv_w_q.dtype):
            bugs.append(f"conv1 weight not quantized to int8 (dtype={conv_w_q.dtype})")

        relu_layer = qimr.get_layer("relu1")
        if not check("relu1 (no params) still present with precision updated",
                      relu_layer.precision == Precision.INT8, relu_layer.precision):
            bugs.append("Parameter-less layer precision not updated by strategy")

        # round-trip check via LayerQuantizer.dequantize_tensor
        lq = LayerQuantizer()
        orig_values = array.array("f")
        orig_values.frombytes(imr.get_layer("conv1").parameters["weight"].data)
        q_tensor, q_params = lq.quantize_tensor(imr.get_layer("conv1").parameters["weight"], 8)
        deq = lq.dequantize_tensor(q_tensor, q_params)
        deq_values = array.array("f")
        deq_values.frombytes(deq.data)
        max_err = max(abs(a - b) for a, b in zip(orig_values, deq_values))
        if not check("INT8 round-trip error within expected bound (<= 1 LSB)",
                      max_err <= q_params.scale * 1.0 + 1e-6, f"max_err={max_err}, scale={q_params.scale}"):
            bugs.append(f"INT8 dequantize round-trip error too large: {max_err} vs scale {q_params.scale}")

    except Exception:
        print("EXCEPTION in scenario 1:")
        traceback.print_exc()
        bugs.append("Scenario 1 (uniform INT8) raised an unexpected exception")

    print()
    print("=== Scenario 2: INT4 default, degenerate all-zero layer ===")
    logger2 = CollectingLogger()
    config2 = QuantizationConfig(default_precision=Precision.INT4, per_layer_overrides={},
                                  calibration_batch_size=4, sensitivity_threshold=0.05)
    try:
        ctx2, calib2, sens2, plan2, qimr2 = run_pipeline(logger2, imr, profile, config2)
        print("Selected strategy:", plan2.selected_strategy_name)
        zero_layer_q = qimr2.get_layer("bn_degenerate").parameters["weight"]
        print("bn_degenerate quantized dtype:", zero_layer_q.dtype)
        lq = LayerQuantizer()
        orig_zero_tensor = imr.get_layer("bn_degenerate").parameters["weight"]
        _, zero_params = lq.quantize_tensor(orig_zero_tensor, 4)
        print("degenerate all-zero tensor scale:", zero_params.scale)
    except Exception:
        print("EXCEPTION in scenario 2 (expected shape, checking manually):")
        traceback.print_exc()

    print()
    print("=== Scenario 3: MIXED default precision, per-layer overrides ===")
    logger3 = CollectingLogger()
    config3 = QuantizationConfig(
        default_precision=Precision.MIXED,
        per_layer_overrides={"conv1": Precision.INT8, "fc2_sensitive": Precision.FP32},
        calibration_batch_size=4, sensitivity_threshold=0.01,
    )
    try:
        ctx3, calib3, sens3, plan3, qimr3 = run_pipeline(logger3, imr, profile, config3)
        print("Selected strategy:", plan3.selected_strategy_name)
        print("Per-layer precision:", {k: v.value for k, v in plan3.per_layer_precision.items()})
        if not check("mixed_precision strategy selected for non-uniform plan",
                      plan3.selected_strategy_name == "mixed_precision", plan3.selected_strategy_name):
            bugs.append("Non-uniform precision plan did not route to mixed_precision strategy")
        conv1_q = qimr3.get_layer("conv1").parameters["weight"]
        fc2_q = qimr3.get_layer("fc2_sensitive").parameters["weight"]
        if not check("conv1 (explicit INT8 override) quantized to int8", conv1_q.dtype == "int8", conv1_q.dtype):
            bugs.append("Explicit per-layer INT8 override not honored under MIXED default")
        if not check("fc2_sensitive (explicit FP32 override) left unquantized", fc2_q.dtype == "float32", fc2_q.dtype):
            bugs.append("Explicit per-layer FP32 override not honored (layer was quantized anyway)")

        report = QuantizationReportRenderer().render(plan3, profile.profile_id, calib3, sens3)
        md = report.to_markdown()
        print(md[:600])
    except Exception:
        print("EXCEPTION in scenario 3:")
        traceback.print_exc()
        bugs.append("Scenario 3 (MIXED default with per-layer overrides) raised an unexpected exception")

    print()
    print("=== Scenario 4: unsupported precision requested -> hardware override ===")
    logger4 = CollectingLogger()
    restrictive_profile = make_profile(profile_id="artix7-strict", supported=(Precision.INT8,))
    config4 = QuantizationConfig(default_precision=Precision.FP32, per_layer_overrides={},
                                  calibration_batch_size=4, sensitivity_threshold=0.05)
    try:
        ctx4, calib4, sens4, plan4, qimr4 = run_pipeline(logger4, imr, restrictive_profile, config4)
        print("Selected strategy:", plan4.selected_strategy_name)
        print("Per-layer precision:", {k: v.value for k, v in plan4.per_layer_precision.items()})
        if not check("all layers overridden to INT8 (only supported precision)",
                      all(v == Precision.INT8 for v in plan4.per_layer_precision.values()),
                      plan4.per_layer_precision):
            bugs.append("FP32 default not overridden to the only hardware-supported precision (INT8)")
    except Exception:
        print("EXCEPTION in scenario 4:")
        traceback.print_exc()
        bugs.append("Scenario 4 (unsupported precision override) raised an unexpected exception")

    print()
    print("=== Scenario 5: CalibrationDataset from a .jsonl file ===")
    try:
        with tempfile.TemporaryDirectory(prefix="uaqe_smoketest_") as tmp_dir:
            tmp_path = Path(tmp_dir) / "calib.jsonl"
            tmp_path.write_text(
                json.dumps({"input": [0.1, 0.2, 0.3, 0.4]}) + "\n" +
                json.dumps({"input": [-0.5, 0.6, -0.7, 0.8]}) + "\n"
            )
            ds = CalibrationDataset(str(tmp_path), batch_size=1)
            print("sample_count:", ds.sample_count(), "len (batches):", len(ds), "input_names:", ds.input_names())
            batches = list(ds)
            if not check("2 samples -> 2 batches at batch_size=1", len(batches) == 2, len(batches)):
                bugs.append("CalibrationDataset batching produced wrong batch count")

            logger5 = CollectingLogger()
            calibrator5 = Calibrator(logger5)
            stats_with_ds = calibrator5.calibrate(imr, ds)
            print("conv1 range w/ dataset folded in:", stats_with_ds.per_layer_activation_range.get("conv1"))
            # conv1 has input_name "input" which matches dataset -> values should widen the range
    except Exception:
        print("EXCEPTION in scenario 5:")
        traceback.print_exc()
        bugs.append("Scenario 5 (CalibrationDataset) raised an unexpected exception")

    print()
    print("=== Scenario 6: LayerQuantizer INT4 pack/unpack round trip (direct) ===")
    try:
        lq = LayerQuantizer()
        vals = [-8, -1, 0, 1, 7, -5, 3, -8]  # covers full signed nibble range, odd count check too
        packed = lq._pack_int4(vals)
        unpacked = list(lq._unpack_int4(packed, len(vals)))
        if not check("INT4 pack/unpack round trip exact", unpacked == vals, f"{unpacked} != {vals}"):
            bugs.append(f"INT4 pack/unpack round trip mismatch: {unpacked} != {vals}")

        # odd number of elements
        vals_odd = [-8, 7, 3]
        packed_odd = lq._pack_int4(vals_odd)
        unpacked_odd = list(lq._unpack_int4(packed_odd, len(vals_odd)))
        if not check("INT4 pack/unpack odd-length round trip exact", unpacked_odd == vals_odd,
                      f"{unpacked_odd} != {vals_odd}"):
            bugs.append(f"INT4 odd-length pack/unpack mismatch: {unpacked_odd} != {vals_odd}")
    except Exception:
        print("EXCEPTION in scenario 6:")
        traceback.print_exc()
        bugs.append("Scenario 6 (INT4 pack/unpack) raised an unexpected exception")

    print()
    print("=== Scenario 7: Int8Quantizer rejects invalid per-layer override ===")
    try:
        strat = Int8Quantizer(logger)
        bad_config = QuantizationConfig(default_precision=Precision.INT8,
                                          per_layer_overrides={"conv1": Precision.INT4})
        try:
            strat.apply(imr, bad_config)
            bugs.append("Int8Quantizer did not raise QuantizationError for an INT4 override (contract violation)")
            print("[FAIL] expected QuantizationError not raised")
        except Exception as e:
            print(f"[PASS] Int8Quantizer correctly rejected INT4 override: {type(e).__name__}: {e}")
    except Exception:
        print("EXCEPTION in scenario 7 setup:")
        traceback.print_exc()

    print()
    print("=== Scenario 8: empty IMR (0 layers) ===")
    try:
        empty_imr = IMR(layers=[], metadata=IMRMetadata(
            source_framework="synthetic", source_format=".synthetic",
            original_input_shapes={}, op_count=0, total_parameters=0))
        config8 = QuantizationConfig(default_precision=Precision.INT8)
        ctx8, calib8, sens8, plan8, qimr8 = run_pipeline(CollectingLogger(), empty_imr, profile, config8)
        print("Selected strategy for empty IMR:", plan8.selected_strategy_name)
        if not check("empty-IMR plan falls back to mixed_precision (no crash)",
                      plan8.selected_strategy_name == "mixed_precision", plan8.selected_strategy_name):
            bugs.append("Empty IMR did not fall back to mixed_precision cleanly")
    except Exception:
        print("EXCEPTION in scenario 8 (empty IMR):")
        traceback.print_exc()
        bugs.append("Scenario 8 (empty IMR, 0 layers) raised an unexpected exception")

    print()
    print("=" * 70)
    print(f"TOTAL BUGS FOUND: {len(bugs)}")
    for i, b in enumerate(bugs, 1):
        print(f"  {i}. {b}")
    return bugs


if __name__ == "__main__":
    main()
