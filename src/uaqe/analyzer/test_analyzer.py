from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.types import Precision
from uaqe.analyzer import Analyzer, ModelAnalyzer, AnalysisResult
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.common.result_types import StageResult


class FakeLogger:
    def debug(self, msg, **f):
        pass

    def info(self, msg, **f):
        print("INFO", msg, f)

    def warning(self, msg, **f):
        print("WARN", msg, f)

    def error(self, msg, **f):
        print("ERROR", msg, f)

    def critical(self, msg, **f):
        print("CRIT", msg, f)


w1 = IMRTensor(shape=(64, 3, 3, 3), dtype="float32", data=b"x")
b1 = IMRTensor(shape=(64,), dtype="float32", data=b"y")
w2 = IMRTensor(shape=(128, 64), dtype="float32", data=b"z")

layers = [
    IMRLayer(name="conv1", op_type="Conv2d", inputs=[], outputs=["conv1::out"],
             parameters={"weight": w1, "bias": b1}, attributes={}, precision=Precision.FP32),
    IMRLayer(name="relu1", op_type="ReLU", inputs=["conv1::out"], outputs=["relu1::out"],
             parameters={}, attributes={}, precision=Precision.FP32),
    IMRLayer(name="fc1", op_type="Linear", inputs=["relu1::out"], outputs=["fc1::out"],
             parameters={"weight": w2}, attributes={}, precision=Precision.FP32),
    IMRLayer(name="weird", op_type="MysteryOp", inputs=["fc1::out"], outputs=["weird::out"],
             parameters={}, attributes={}, precision=Precision.FP32),
]
imr = IMR(
    layers=layers,
    metadata=IMRMetadata(
        source_framework="pytorch",
        source_format=".pth",
        original_input_shapes={"input": (1, 3, 8, 8)},
        op_count=len(layers),
        total_parameters=0,
    ),
)

report = Analyzer().analyze(imr)
print("params", report.parameter_count)
print("flops", report.estimated_flops)
print("mem", report.estimated_memory_bytes)
print("hist", report.op_type_histogram)
print("complexity", report.complexity)
print("unsupported", report.unsupported_layers)
print("warnings", report.warnings)

ctx = PipelineContext(run_id="r1")
ctx.append("model_loader", StageResult(stage_name="model_loader", success=True, payload=imr))
stage = ModelAnalyzer(logger=FakeLogger())
result = stage.execute(ctx)
print("stage payload type", type(result.payload))
assert isinstance(result.payload, AnalysisResult)
print("OK")
