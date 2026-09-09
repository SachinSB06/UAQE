"""FlatBuffer Mixed-Precision Transformation Engine.

Applies a `PrecisionPolicy` to a calibrated INT8 TFLite FlatBuffer,
selectively transforming targeted layers to FP16 or FP32 with valid
QUANTIZE / DEQUANTIZE graph boundaries, verified tensor datatypes,
and bit-exact FlatBuffer repacking.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

from uaqe.common.types import Precision
from uaqe.exporter.flatbuffer_inspector import FlatBufferDtypeSummary, FlatBufferInspector
from uaqe.quantization.precision_policy import PrecisionPolicy


class FlatBufferMixedPrecisionTransformer:
    """Transforms a full INT8 TFLite FlatBuffer into a verified mixed-precision model."""

    def __init__(
        self,
        base_tflite_path: str,
        sensitivity_report_path: Optional[str] = None,
    ) -> None:
        """Initialize transformer with baseline model and sensitivity report.

        Args:
            base_tflite_path: Path to baseline full-INT8 .tflite model.
            sensitivity_report_path: Optional path to Phase A.2 sensitivity report
                used to map layer names to tflite tensor indices.
        """
        if not os.path.exists(base_tflite_path):
            raise FileNotFoundError(f"Base TFLite model not found: '{base_tflite_path}'")

        self.base_tflite_path = base_tflite_path
        self.sensitivity_report_path = sensitivity_report_path
        self.layer_to_tensor_map: Dict[str, int] = {}
        self.tensor_to_layer_map: Dict[int, Dict[str, Any]] = {}

        if sensitivity_report_path and os.path.exists(sensitivity_report_path):
            self._load_sensitivity_mapping(sensitivity_report_path)

    def _load_sensitivity_mapping(self, path: str) -> None:
        """Load layer_name to tflite_tensor_idx mapping from sensitivity report."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for m in data.get("layer_metrics", []):
            layer_name = m.get("layer_name")
            tensor_idx = m.get("tflite_tensor_idx")
            if layer_name and tensor_idx is not None:
                self.layer_to_tensor_map[layer_name] = tensor_idx
                self.tensor_to_layer_map[tensor_idx] = m

    def transform(
        self,
        policy: PrecisionPolicy,
        output_path: str,
    ) -> FlatBufferDtypeSummary:
        """Apply precision policy to baseline TFLite model and save transformed model.

        Args:
            policy: PrecisionPolicy specifying per-layer precision overrides.
            output_path: Destination path for the transformed .tflite model.

        Returns:
            FlatBufferDtypeSummary with post-transformation verification metrics.
        """
        with open(self.base_tflite_path, "rb") as f:
            base_bytes = f.read()

        model_obj = schema_fb.Model.GetRootAsModel(base_bytes, 0)
        model_t = schema_fb.ModelT.InitFromObj(model_obj)
        subgraph = model_t.subgraphs[0]

        # 1. Resolve opcode indices for DEQUANTIZE and QUANTIZE
        dequant_opcode_idx, quant_opcode_idx = self._ensure_quant_opcodes(model_t)

        # 2. Determine targeted tensors from policy
        targeted_tensors: Dict[int, Tuple[str, Precision]] = {}
        layer_verification_targets: Dict[str, Tuple[int, Precision]] = {}

        for layer_name, prec in policy.layer_overrides.items():
            if prec in (Precision.FP16, Precision.FP32):
                # Resolve tensor index
                tensor_idx = self.layer_to_tensor_map.get(layer_name)
                if tensor_idx is None:
                    # Fuzzy match on canonical name
                    norm_target = layer_name.strip("/").replace("/", ".")
                    for k, idx in self.layer_to_tensor_map.items():
                        norm_k = k.strip("/").replace("/", ".")
                        if norm_target in norm_k or norm_k in norm_target:
                            tensor_idx = idx
                            break
                if tensor_idx is not None:
                    targeted_tensors[tensor_idx] = (layer_name, prec)
                    layer_verification_targets[layer_name] = (tensor_idx, prec)

        # 3. Transform operators
        new_operators = []
        transformed_op_count = 0

        for op_idx, op in enumerate(subgraph.operators):
            # Check if this operator produces any targeted tensor
            matching_target = None
            for out_idx in op.outputs:
                if out_idx in targeted_tensors:
                    matching_target = (out_idx, targeted_tensors[out_idx])
                    break

            if matching_target is not None:
                out_idx, (layer_name, target_prec) = matching_target
                dequant_ops, updated_op, quant_op = self._transform_single_operator(
                    model_t=model_t,
                    subgraph=subgraph,
                    op=op,
                    op_idx=op_idx,
                    target_precision=target_prec,
                    dequant_opcode_idx=dequant_opcode_idx,
                    quant_opcode_idx=quant_opcode_idx,
                )
                for d_op in dequant_ops:
                    new_operators.append(d_op)
                new_operators.append(updated_op)
                if quant_op is not None:
                    new_operators.append(quant_op)
                transformed_op_count += 1
            else:
                new_operators.append(op)

        subgraph.operators = new_operators

        # 4. Pack FlatBuffer with official TFLite identifier b"TFL3"
        builder = flatbuffers.Builder(1024 * 1024 * 8)
        model_offset = model_t.Pack(builder)
        builder.Finish(model_offset, file_identifier=b"TFL3")
        packed_bytes = builder.Output()

        # 5. Save model
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(packed_bytes)

        # 6. Verify with FlatBufferInspector
        summary = FlatBufferInspector.inspect(output_path, layer_verification_targets)
        return summary

    def _ensure_quant_opcodes(self, model_t: schema_fb.ModelT) -> Tuple[int, int]:
        """Resolve or register DEQUANTIZE and QUANTIZE opcode indices."""
        dequant_idx = None
        quant_idx = None

        for idx, op_code in enumerate(model_t.operatorCodes):
            code = op_code.deprecatedBuiltinCode
            if code == schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
                code = op_code.builtinCode
            if code == schema_fb.BuiltinOperator.DEQUANTIZE:
                dequant_idx = idx
            elif code == schema_fb.BuiltinOperator.QUANTIZE:
                quant_idx = idx

        if dequant_idx is None:
            code_t = schema_fb.OperatorCodeT()
            code_t.builtinCode = schema_fb.BuiltinOperator.DEQUANTIZE
            code_t.deprecatedBuiltinCode = schema_fb.BuiltinOperator.DEQUANTIZE
            model_t.operatorCodes.append(code_t)
            dequant_idx = len(model_t.operatorCodes) - 1

        if quant_idx is None:
            code_t = schema_fb.OperatorCodeT()
            code_t.builtinCode = schema_fb.BuiltinOperator.QUANTIZE
            code_t.deprecatedBuiltinCode = schema_fb.BuiltinOperator.QUANTIZE
            model_t.operatorCodes.append(code_t)
            quant_idx = len(model_t.operatorCodes) - 1

        return dequant_idx, quant_idx

    def _transform_single_operator(
        self,
        model_t: schema_fb.ModelT,
        subgraph: schema_fb.SubGraphT,
        op: schema_fb.OperatorT,
        op_idx: int,
        target_precision: Precision,
        dequant_opcode_idx: int,
        quant_opcode_idx: int,
    ) -> Tuple[List[schema_fb.OperatorT], schema_fb.OperatorT, Optional[schema_fb.OperatorT]]:
        """Transform an operator's weights and activations to higher precision."""
        orig_input_idx = op.inputs[0]
        orig_output_idx = op.outputs[0]

        input_tensor = subgraph.tensors[orig_input_idx]
        output_tensor = subgraph.tensors[orig_output_idx]

        input_scale = (
            input_tensor.quantization.scale[0]
            if (
                input_tensor.quantization is not None
                and input_tensor.quantization.scale is not None
                and len(input_tensor.quantization.scale) > 0
            )
            else 1.0
        )

        weight_scales = None
        new_inputs = list(op.inputs)
        dequant_ops: List[schema_fb.OperatorT] = []

        # Handle second input: either a constant weight or a second activation input
        if len(op.inputs) > 1:
            second_idx = op.inputs[1]
            second_tensor = subgraph.tensors[second_idx]
            is_const_weight = (
                second_tensor.buffer < len(model_t.buffers)
                and model_t.buffers[second_tensor.buffer].data is not None
                and len(model_t.buffers[second_tensor.buffer].data) > 0
                and second_tensor.quantization is not None
                and second_tensor.quantization.scale is not None
                and len(second_tensor.quantization.scale) > 0
            )

            if is_const_weight:
                weight_scales = np.array(second_tensor.quantization.scale, dtype=np.float32)
                raw_w = np.frombuffer(
                    model_t.buffers[second_tensor.buffer].data, dtype=np.int8
                ).reshape(second_tensor.shape)

                quantized_dim = (
                    second_tensor.quantization.quantizedDimension
                    if hasattr(second_tensor.quantization, "quantizedDimension") and second_tensor.quantization.quantizedDimension is not None
                    else 0
                )

                # Reconstruct full FP32 weights (respecting quantized_dimension)
                if len(weight_scales) > 1:
                    shape_broadcast = [1] * len(second_tensor.shape)
                    shape_broadcast[quantized_dim] = len(weight_scales)
                    w_f32 = raw_w.astype(np.float32) * weight_scales.reshape(shape_broadcast)
                else:
                    w_f32 = raw_w.astype(np.float32) * weight_scales[0]

                # Check operator type: TFLite's DEPTHWISE_CONV_2D requires filter->type == data_type (FLOAT32)
                op_code_obj = model_t.operatorCodes[op.opcodeIndex]
                b_code = op_code_obj.deprecatedBuiltinCode
                if b_code == schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
                    b_code = op_code_obj.builtinCode
                is_depthwise = (b_code == schema_fb.BuiltinOperator.DEPTHWISE_CONV_2D)

                w_buf = schema_fb.BufferT()
                if target_precision == Precision.FP16 and not is_depthwise:
                    w_f16 = w_f32.astype(np.float16)
                    w_buf.data = w_f16.tobytes()
                    second_tensor.type = schema_fb.TensorType.FLOAT16
                else:
                    w_buf.data = w_f32.tobytes()
                    second_tensor.type = schema_fb.TensorType.FLOAT32

                model_t.buffers.append(w_buf)
                second_tensor.buffer = len(model_t.buffers) - 1
                second_tensor.quantization = None

            elif second_tensor.type == schema_fb.TensorType.INT8:
                # Activation input (e.g. Mul, Add): insert DEQUANTIZE
                interm_in2 = schema_fb.TensorT()
                base_name2 = second_tensor.name.decode("utf-8") if second_tensor.name else f"tensor_{second_idx}"
                interm_in2.name = f"{base_name2}_f32_op{op_idx}".encode("utf-8")
                interm_in2.type = schema_fb.TensorType.FLOAT32
                interm_in2.shape = list(second_tensor.shape)
                interm_in2.buffer = 0
                subgraph.tensors.append(interm_in2)
                compute_in2_idx = len(subgraph.tensors) - 1

                dequant2 = schema_fb.OperatorT()
                dequant2.opcodeIndex = dequant_opcode_idx
                dequant2.inputs = [second_idx]
                dequant2.outputs = [compute_in2_idx]
                dequant_ops.append(dequant2)
                new_inputs[1] = compute_in2_idx

        # Handle bias if present
        if len(op.inputs) > 2:
            bias_idx = op.inputs[2]
            bias_tensor = subgraph.tensors[bias_idx]
            if (
                bias_tensor.type == schema_fb.TensorType.INT32
                and model_t.buffers[bias_tensor.buffer].data is not None
                and len(model_t.buffers[bias_tensor.buffer].data) > 0
                and weight_scales is not None
            ):
                raw_b = np.frombuffer(
                    model_t.buffers[bias_tensor.buffer].data, dtype=np.int32
                )
                b_scales = input_scale * weight_scales
                b_f32 = raw_b.astype(np.float32) * b_scales
                b_buf = schema_fb.BufferT()
                b_buf.data = b_f32.tobytes()
                model_t.buffers.append(b_buf)
                bias_tensor.buffer = len(model_t.buffers) - 1
                bias_tensor.type = schema_fb.TensorType.FLOAT32
                bias_tensor.quantization = None

        # Insert intermediate input tensor and DEQUANTIZE if input 0 is INT8
        compute_input_idx = orig_input_idx

        if input_tensor.type == schema_fb.TensorType.INT8:
            intermediate_in = schema_fb.TensorT()
            base_name = input_tensor.name.decode("utf-8") if input_tensor.name else f"tensor_{orig_input_idx}"
            intermediate_in.name = f"{base_name}_f32_op{op_idx}".encode("utf-8")
            intermediate_in.type = schema_fb.TensorType.FLOAT32
            intermediate_in.shape = list(input_tensor.shape)
            intermediate_in.buffer = 0
            subgraph.tensors.append(intermediate_in)
            compute_input_idx = len(subgraph.tensors) - 1

            dequant_op = schema_fb.OperatorT()
            dequant_op.opcodeIndex = dequant_opcode_idx
            dequant_op.inputs = [orig_input_idx]
            dequant_op.outputs = [compute_input_idx]
            dequant_ops.append(dequant_op)

        # Insert intermediate output tensor and QUANTIZE if original output is INT8
        quant_op = None
        compute_output_idx = orig_output_idx

        if output_tensor.type == schema_fb.TensorType.INT8:
            intermediate_out = schema_fb.TensorT()
            base_name = output_tensor.name.decode("utf-8") if output_tensor.name else f"tensor_{orig_output_idx}"
            intermediate_out.name = f"{base_name}_f32_op{op_idx}".encode("utf-8")
            intermediate_out.type = schema_fb.TensorType.FLOAT32
            intermediate_out.shape = list(output_tensor.shape)
            intermediate_out.buffer = 0
            subgraph.tensors.append(intermediate_out)
            compute_output_idx = len(subgraph.tensors) - 1

            quant_op = schema_fb.OperatorT()
            quant_op.opcodeIndex = quant_opcode_idx
            quant_op.inputs = [compute_output_idx]
            quant_op.outputs = [orig_output_idx]

        # Update compute op connections
        new_inputs[0] = compute_input_idx
        op.inputs = np.array(new_inputs, dtype=np.int32)

        new_outputs = list(op.outputs)
        new_outputs[0] = compute_output_idx
        op.outputs = np.array(new_outputs, dtype=np.int32)

        return dequant_ops, op, quant_op
