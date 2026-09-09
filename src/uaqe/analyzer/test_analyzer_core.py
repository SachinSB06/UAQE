from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.types import Precision
from uaqe.analyzer.analyzer import Analyzer
from uaqe.analyzer.layer_analyzer import LayerAnalyzer
from uaqe.analyzer.parameter_counter import ParameterCounter
from uaqe.analyzer.flops_estimator import FlopsEstimator
from uaqe.analyzer.memory_analyzer import MemoryAnalyzer
from uaqe.analyzer.complexity_analyzer import ComplexityAnalyzer
from uaqe.analyzer.unsupported_layer_detector import UnsupportedLayerDetector

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

pc = ParameterCounter()
assert pc.count_total(imr) == (64 * 3 * 3 * 3 + 64 + 128 * 64)
print("param count OK:", pc.count_total(imr))

fe = FlopsEstimator()
flops = fe.estimate_total(imr)
print("flops:", flops, "warnings:", fe.lower_bound_warnings(imr))
assert flops == 2 * (64*3*3*3 + 64) + 2 * (128*64)  # includes conv bias params

ma = MemoryAnalyzer()
print("weight bytes:", ma.total_weight_bytes(imr))
print("estimated total mem:", ma.estimate_total(imr))

la = LayerAnalyzer()
summary = la.build_summary(imr)
print("summary:", summary)
assert summary["layer_count"] == 4
assert summary["graph_depth"] == 4

uld = UnsupportedLayerDetector()
print("unsupported:", uld.find_unsupported(imr))
assert uld.find_unsupported(imr) == ["weird"]

ca = ComplexityAnalyzer()
profile = ca.build_profile(pc.count_total(imr), flops, ma.estimate_total(imr), summary["graph_depth"])
print("complexity:", profile)

report = Analyzer().analyze(imr)
print("\nFull report:")
print(report)

print("\nALL CHECKS PASSED")
