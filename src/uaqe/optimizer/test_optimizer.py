import sys
sys.path.insert(0, "/home/claude/work/Quantization embedded/src")

from uaqe.common.imr import IMR, IMRLayer, IMRMetadata, IMRTensor
from uaqe.common.types import Precision, HardwareClass
from uaqe.common.value_objects import OptimizationConfig
from uaqe.domain.hardware_manager import HardwareProfile
from uaqe.domain.pipeline_context import PipelineContext
from uaqe.common.result_types import StageResult
from uaqe.optimizer import Optimizer, MemoryOptimizer, OptimizationReportRenderer


class PrintLogger:
    def debug(self, msg, **f): pass
    def info(self, msg, **f): print("INFO:", msg, f)
    def warning(self, msg, **f): print("WARN:", msg, f)
    def error(self, msg, **f): print("ERROR:", msg, f)
    def critical(self, msg, **f): print("CRIT:", msg, f)


def tensor(shape, n):
    return IMRTensor(shape=shape, dtype="float32", data=b"\x00" * (n * 4))


layers = [
    IMRLayer(name="conv1", op_type="Conv2D", inputs=["input"], outputs=["conv1_out"],
             parameters={"w": tensor((8, 3, 3, 3), 8*3*3*3)}, attributes={}, precision=Precision.FP32),
    IMRLayer(name="bn1", op_type="BatchNorm", inputs=["conv1_out"], outputs=["bn1_out"],
             parameters={}, attributes={}, precision=Precision.FP32),
    IMRLayer(name="relu1", op_type="ReLU", inputs=["bn1_out"], outputs=["relu1_out"],
             parameters={}, attributes={}, precision=Precision.FP32),
    IMRLayer(name="id1", op_type="Identity", inputs=["relu1_out"], outputs=["id1_out"],
             parameters={}, attributes={}, precision=Precision.FP32),
    IMRLayer(name="mm1", op_type="MatMul", inputs=["id1_out"], outputs=["mm1_out"],
             parameters={"w": tensor((64, 10), 640)}, attributes={}, precision=Precision.INT8),
    IMRLayer(name="add1", op_type="Add", inputs=["mm1_out"], outputs=["out"],
             parameters={"b": tensor((10,), 10)}, attributes={}, precision=Precision.INT8),
]
metadata = IMRMetadata(source_framework="pytorch", source_format=".pth",
                       original_input_shapes={"input": (1, 3, 32, 32)},
                       op_count=len(layers), total_parameters=1000)
imr = IMR(layers=layers, metadata=metadata)

profile = HardwareProfile(
    profile_id="test-mcu",
    display_name="Test MCU",
    hardware_class=HardwareClass.EMBEDDED,
    ram_bytes=256 * 1024,
    flash_bytes=1024 * 1024,
    storage_bytes=None,
    tensor_memory_bytes=128 * 1024,
    runtime="tflite-micro",
    max_model_size_bytes=1024 * 1024,
    schema_version="1.0",
    supported_precisions=[Precision.INT8, Precision.FP32],
)

logger = PrintLogger()
config = OptimizationConfig(
    objectives=["accuracy", "latency", "memory", "power"],
    objective_weights={"latency": 0.4, "memory": 0.3, "power": 0.2, "accuracy": 0.1},
    max_search_iterations=3,
)

context = PipelineContext(run_id="test-run")
context.append("model_loader", StageResult(stage_name="model_loader", success=True, payload=imr))
context.append("hardware_manager", StageResult(stage_name="hardware_manager", success=True, payload=profile))

optimizer = Optimizer(logger=logger, config=config)
stage_result = optimizer.execute(context)
context.append("optimizer", stage_result)

result, optimized_imr = stage_result.payload
print("\n--- Optimization result ---")
print("selected:", result.selected_candidate.name)
print("objective_scores:", result.objective_scores)
print("num layers before:", len(imr.layers), "after:", len(optimized_imr.layers))
print("rationale:", result.rationale)

memory_optimizer = MemoryOptimizer(logger=logger)
mem_stage_result = memory_optimizer.execute(context)
context.append("memory_optimizer", mem_stage_result)
memory_plan, final_imr = mem_stage_result.payload
print("\n--- Memory plan ---")
print("peak_memory_bytes:", memory_plan.peak_memory_bytes)

renderer = OptimizationReportRenderer()
doc = renderer.render(result, memory_plan)
print("\n--- Report (markdown) ---")
print(doc.to_markdown())
