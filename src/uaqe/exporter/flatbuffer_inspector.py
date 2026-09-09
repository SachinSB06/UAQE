"""FlatBuffer dtype inspector and precision verifier.

Inspects raw and mixed-precision TensorFlow Lite FlatBuffers to extract:
- Exact tensor counts per datatype (INT8, FLOAT16, FLOAT32, INT32)
- Operator opcode breakdown
- Computational coverage percentages (INT8 coverage % vs Higher Precision %)
- Verification of requested precision overrides on specific layers/tensors
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

from uaqe.common.types import Precision


# Mapping from schema_fb.TensorType to human-readable names
_TENSOR_TYPE_MAP = {
    schema_fb.TensorType.FLOAT32: "FLOAT32",
    schema_fb.TensorType.FLOAT16: "FLOAT16",
    schema_fb.TensorType.INT32: "INT32",
    schema_fb.TensorType.UINT8: "UINT8",
    schema_fb.TensorType.INT64: "INT64",
    schema_fb.TensorType.STRING: "STRING",
    schema_fb.TensorType.BOOL: "BOOL",
    schema_fb.TensorType.INT16: "INT16",
    schema_fb.TensorType.COMPLEX64: "COMPLEX64",
    schema_fb.TensorType.INT8: "INT8",
}


@dataclass
class LayerVerificationResult:
    """Verification outcome for a single requested precision override."""

    layer_name: str
    target_tensor_idx: int
    requested_precision: str
    actual_precision: str
    status: str  # "PASS" or "FAIL"
    reason: str
    operator_type: str
    weight_tensor_idx: Optional[int] = None
    weight_dtype: Optional[str] = None


@dataclass
class FlatBufferDtypeSummary:
    """Summary of tensors, datatypes, and coverage in a TFLite FlatBuffer."""

    model_path: str
    file_size_bytes: int
    file_size_mb: float
    total_tensors: int
    total_operators: int
    int8_tensors: int
    int16_tensors: int
    fp16_tensors: int
    fp32_tensors: int
    int32_tensors: int
    other_tensors: int
    int8_coverage_percent: float
    higher_precision_coverage_percent: float
    operator_opcode_counts: Dict[str, int] = field(default_factory=dict)
    layer_verifications: List[LayerVerificationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        d = asdict(self)
        d["layer_verifications"] = [asdict(v) for v in self.layer_verifications]
        return d


class FlatBufferInspector:
    """Inspector for auditing TFLite FlatBuffers for genuine datatype allocation."""

    @staticmethod
    def inspect(
        model_path: str,
        layer_targets: Optional[Dict[str, Tuple[int, Precision]]] = None,
    ) -> FlatBufferDtypeSummary:
        """Inspect a TFLite FlatBuffer file.

        Args:
            model_path: Path to the .tflite FlatBuffer.
            layer_targets: Optional mapping of layer_name -> (tflite_tensor_idx, requested_precision).

        Returns:
            FlatBufferDtypeSummary with verified tensor breakdown and layer verifications.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at '{model_path}'")

        file_size_bytes = os.path.getsize(model_path)
        with open(model_path, "rb") as f:
            buf = f.read()

        model = schema_fb.Model.GetRootAsModel(buf, 0)
        subgraph = model.Subgraphs(0)

        total_tensors = subgraph.TensorsLength()
        total_operators = subgraph.OperatorsLength()

        int8_count = 0
        int16_count = 0
        fp16_count = 0
        fp32_count = 0
        int32_count = 0
        other_count = 0

        tensor_dtypes: Dict[int, str] = {}
        for i in range(total_tensors):
            t = subgraph.Tensors(i)
            ttype = t.Type()
            tname = _TENSOR_TYPE_MAP.get(ttype, f"UNKNOWN_{ttype}")
            tensor_dtypes[i] = tname
            if ttype == schema_fb.TensorType.INT8:
                int8_count += 1
            elif ttype == schema_fb.TensorType.INT16:
                int16_count += 1
            elif ttype == schema_fb.TensorType.FLOAT16:
                fp16_count += 1
            elif ttype == schema_fb.TensorType.FLOAT32:
                fp32_count += 1
            elif ttype == schema_fb.TensorType.INT32:
                int32_count += 1
            else:
                other_count += 1

        # Operator breakdown
        op_counts: Dict[str, int] = {}
        for i in range(total_operators):
            op = subgraph.Operators(i)
            op_code_obj = model.OperatorCodes(op.OpcodeIndex())
            code = op_code_obj.DeprecatedBuiltinCode()
            if code == schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
                code = op_code_obj.BuiltinCode()
            names = [k for k, v in schema_fb.BuiltinOperator.__dict__.items() if v == code]
            name = names[0] if names else f"OP_{code}"
            op_counts[name] = op_counts.get(name, 0) + 1

        # Calculate coverage (INT8 vs higher precision among compute tensors)
        int8_cov = (int8_count / total_tensors * 100.0) if total_tensors > 0 else 0.0
        hp_cov = ((fp16_count + fp32_count) / total_tensors * 100.0) if total_tensors > 0 else 0.0

        # Layer verifications
        verifications: List[LayerVerificationResult] = []
        if layer_targets:
            for l_name, (target_idx, req_prec) in layer_targets.items():
                # Find operator producing target_idx
                prod_op = None
                prod_op_idx = -1
                for o_idx in range(total_operators):
                    op = subgraph.Operators(o_idx)
                    for o in range(op.OutputsLength()):
                        if op.Outputs(o) == target_idx:
                            prod_op = op
                            prod_op_idx = o_idx
                            break
                    if prod_op:
                        break

                act_dtype = tensor_dtypes.get(target_idx, "UNKNOWN")
                weight_dtype = None
                weight_idx = None
                op_type_name = "UNKNOWN"

                if prod_op:
                    code_idx = prod_op.OpcodeIndex()
                    code_obj = model.OperatorCodes(code_idx)
                    b_code = code_obj.DeprecatedBuiltinCode()
                    if b_code == schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
                        b_code = code_obj.BuiltinCode()
                    names = [k for k, v in schema_fb.BuiltinOperator.__dict__.items() if v == b_code]
                    op_type_name = names[0] if names else str(b_code)

                    if prod_op.InputsLength() > 1:
                        weight_idx = prod_op.Inputs(1)
                        weight_dtype = tensor_dtypes.get(weight_idx, "UNKNOWN")

                req_str = req_prec.value
                # Evaluation: Did the weight or output achieve higher precision?
                is_pass = False
                reason = ""
                if req_str in ["FP16", "FLOAT16"]:
                    if weight_dtype in ["FLOAT16", "FLOAT32"] or act_dtype in ["FLOAT16", "FLOAT32"]:
                        is_pass = True
                        reason = f"Verified: target op uses higher-precision weights ({weight_dtype}) and/or activations ({act_dtype})"
                    else:
                        reason = f"Failed: target tensor remains {act_dtype}, weights remain {weight_dtype}"
                elif req_str in ["FP32", "FLOAT32"]:
                    if weight_dtype == "FLOAT32" or act_dtype == "FLOAT32":
                        is_pass = True
                        reason = f"Verified: target op uses FP32 weights ({weight_dtype}) and/or activations ({act_dtype})"
                    else:
                        reason = f"Failed: target tensor remains {act_dtype}, weights remain {weight_dtype}"
                else:
                    is_pass = (act_dtype == "INT8")
                    reason = f"INT8 retention verified ({act_dtype})"

                verifications.append(
                    LayerVerificationResult(
                        layer_name=l_name,
                        target_tensor_idx=target_idx,
                        requested_precision=req_str,
                        actual_precision=weight_dtype or act_dtype,
                        status="PASS" if is_pass else "FAIL",
                        reason=reason,
                        operator_type=op_type_name,
                        weight_tensor_idx=weight_idx,
                        weight_dtype=weight_dtype,
                    )
                )

        return FlatBufferDtypeSummary(
            model_path=model_path,
            file_size_bytes=file_size_bytes,
            file_size_mb=round(file_size_bytes / (1024 * 1024), 4),
            total_tensors=total_tensors,
            total_operators=total_operators,
            int8_tensors=int8_count,
            int16_tensors=int16_count,
            fp16_tensors=fp16_count,
            fp32_tensors=fp32_count,
            int32_tensors=int32_count,
            other_tensors=other_count,
            int8_coverage_percent=round(int8_cov, 2),
            higher_precision_coverage_percent=round(hp_cov, 2),
            operator_opcode_counts=op_counts,
            layer_verifications=verifications,
        )
