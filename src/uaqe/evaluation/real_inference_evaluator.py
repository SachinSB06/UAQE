import os
import time
import hashlib
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import psutil
from typing import Dict, List, Tuple, Optional, Any
from tensorflow.lite.python import schema_py_generated as tflite_schema
from uaqe.evaluation.real_dataset_adapter import RealDatasetAdapter

class RealInferenceEvaluator:
    """Model-agnostic evaluation runner that computes real labeled-dataset accuracy,
    similarities, latencies, memory, and stability metrics for FP32, ONNX, and TFLite.
    """

    def __init__(
        self,
        fp32_model_path: str,
        onnx_model_path: str,
        tflite_model_path: str,
        dataset: RealDatasetAdapter,
        config: Dict[str, Any],
        logger: Optional[Any] = None
    ) -> None:
        self.fp32_model_path = fp32_model_path
        self.onnx_model_path = onnx_model_path
        self.tflite_model_path = tflite_model_path
        self.dataset = dataset
        self.config = config
        self.logger = logger
        
        self.thresholds = config.get("evaluation_thresholds", {})
        
        # Verify model files exist
        verify_targets = [
            ("FP32 reference", fp32_model_path), 
            ("Optimized ONNX", onnx_model_path)
        ]
        if tflite_model_path and tflite_model_path != "NOT GENERATED":
            verify_targets.append(("TFLite FlatBuffer", tflite_model_path))
            
        for name, path in verify_targets:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required deployment model file {name} not found at: {path}")

    @staticmethod
    def get_file_sha256(file_path: str) -> str:
        """Compute the SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def run_reference_verification(self) -> Dict[str, Any]:
        """Run reference check on a single sample to verify channel order,
        shape compatibility, and logits alignment across the 3 paths.
        """
        if len(self.dataset) == 0:
            raise ValueError("Evaluation dataset is empty. Cannot run reference check.")
            
        sample = self.dataset[0]
        tensor = sample["tensor"]
        label = sample["label"]
        
        # 1. Run FP32 ONNX inference
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        ort_sess_fp32 = ort.InferenceSession(self.fp32_model_path, sess_options)
        
        input_name_fp32 = ort_sess_fp32.get_inputs()[0].name
        input_shape_fp32 = ort_sess_fp32.get_inputs()[0].shape
        output_name_fp32 = ort_sess_fp32.get_outputs()[0].name
        output_shape_fp32 = ort_sess_fp32.get_outputs()[0].shape
        
        # Adapt shape for inference batch size 1
        x = np.expand_dims(tensor, axis=0)
        
        # Verify shape compatibility
        expected_shape = [s if isinstance(s, int) else 1 for s in input_shape_fp32]
        expected_shape = [s if s > 0 else 1 for s in expected_shape]
        
        if list(x.shape) != expected_shape:
            # Check layout adaptation if dimensions are swapped
            if len(expected_shape) == 4 and expected_shape[3] == tensor.shape[0]: # NHWC case
                x = np.expand_dims(tensor.transpose(1, 2, 0), axis=0) # convert from CHW to HWC
            
        out_fp32 = ort_sess_fp32.run(None, {input_name_fp32: x})[0][0]
        
        # 2. Run Optimized ONNX inference
        ort_sess_opt = ort.InferenceSession(self.onnx_model_path, sess_options)
        input_name_opt = ort_sess_opt.get_inputs()[0].name
        out_opt = ort_sess_opt.run(None, {input_name_opt: x})[0][0]
        
        # 3. Run TFLite inference
        has_tflite = self.tflite_model_path and self.tflite_model_path != "NOT GENERATED"
        out_tflite = None
        tflite_pred = None
        tflite_output = []
        if has_tflite:
            interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path, num_threads=1)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            
            # Convert input array if TFLite expects NHWC or different dtype (e.g. uint8 vs float32)
            x_tflite = x.copy()
            if input_details["shape"][1] != x.shape[1]:
                # Expects NHWC layout
                if self.dataset.layout == "NCHW":
                    x_tflite = np.expand_dims(tensor.transpose(1, 2, 0), axis=0)
                    
            if input_details["dtype"] == np.uint8:
                # Apply quantization scale/zero-point
                scale, zero_point = input_details["quantization"]
                x_tflite = np.clip(np.round(x_tflite / scale + zero_point), 0, 255).astype(np.uint8)
            elif input_details["dtype"] == np.int8:
                scale, zero_point = input_details["quantization"]
                x_tflite = np.clip(np.round(x_tflite / scale + zero_point), -128, 127).astype(np.int8)
                
            interpreter.set_tensor(input_details["index"], x_tflite)
            interpreter.invoke()
            out_tflite = interpreter.get_tensor(output_details["index"])[0]
            
            # Dequantize output if it's integer
            if output_details["dtype"] == np.uint8 or output_details["dtype"] == np.int8:
                scale, zero_point = output_details["quantization"]
                out_tflite = (out_tflite.astype(np.float32) - zero_point) * scale
            
            tflite_pred = int(np.argmax(out_tflite))
            tflite_output = out_tflite.tolist()

        # Consistency verification
        has_nan = np.isnan(out_fp32).any() or np.isnan(out_opt).any()
        if out_tflite is not None:
            has_nan = has_nan or np.isnan(out_tflite).any()
            
        has_inf = np.isinf(out_fp32).any() or np.isinf(out_opt).any()
        if out_tflite is not None:
            has_inf = has_inf or np.isinf(out_tflite).any()
            
        # Output shapes validation
        expected_classes = self.config.get("expected_num_classes", 10)
        assert len(out_fp32) == expected_classes, f"FP32 output shape error: {len(out_fp32)} classes vs expected {expected_classes}"
        assert len(out_opt) == expected_classes, f"ONNX output shape error: {len(out_opt)} classes vs expected {expected_classes}"
        if out_tflite is not None:
            assert len(out_tflite) == expected_classes, f"TFLite output shape error: {len(out_tflite)} classes vs expected {expected_classes}"
        
        return {
            "fp32_output": out_fp32.tolist(),
            "onnx_output": out_opt.tolist(),
            "tflite_output": tflite_output if has_tflite else None,
            "fp32_pred": int(np.argmax(out_fp32)),
            "onnx_pred": int(np.argmax(out_opt)),
            "tflite_pred": tflite_pred,
            "label": label,
            "has_nan": has_nan,
            "has_inf": has_inf
        }

    def evaluate_accuracy_and_numerical_metrics(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Run all test samples on all three models, computing accuracy,
        classification reports, and numerical similarity metrics.
        """
        # Session options for single-threaded host runs
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        ort_sess_fp32 = ort.InferenceSession(self.fp32_model_path, sess_options)
        ort_sess_opt = ort.InferenceSession(self.onnx_model_path, sess_options)
        
        has_tflite = self.tflite_model_path and self.tflite_model_path != "NOT GENERATED"
        if has_tflite:
            interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path, num_threads=1)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
        
        input_name_fp32 = ort_sess_fp32.get_inputs()[0].name
        input_name_opt = ort_sess_opt.get_inputs()[0].name
        
        ground_truths = []
        preds_fp32 = []
        preds_onnx = []
        preds_tflite = []
        
        logits_fp32 = []
        logits_onnx = []
        logits_tflite = []
        
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            tensor = sample["tensor"]
            label = sample["label"]
            
            ground_truths.append(label)
            
            x = np.expand_dims(tensor, axis=0)
            
            # FP32 ONNX
            out_fp32 = ort_sess_fp32.run(None, {input_name_fp32: x})[0][0]
            preds_fp32.append(np.argmax(out_fp32))
            logits_fp32.append(out_fp32)
            
            # Optimized ONNX
            out_opt = ort_sess_opt.run(None, {input_name_opt: x})[0][0]
            preds_onnx.append(np.argmax(out_opt))
            logits_onnx.append(out_opt)
            
            # TFLite
            if has_tflite:
                x_tflite = x.copy()
                if input_details["shape"][1] != x.shape[1]:
                    # NCHW to NHWC layout swap
                    x_tflite = np.expand_dims(tensor.transpose(1, 2, 0), axis=0)
                    
                if input_details["dtype"] == np.uint8:
                    scale, zero_point = input_details["quantization"]
                    x_tflite = np.clip(np.round(x_tflite / scale + zero_point), 0, 255).astype(np.uint8)
                elif input_details["dtype"] == np.int8:
                    scale, zero_point = input_details["quantization"]
                    x_tflite = np.clip(np.round(x_tflite / scale + zero_point), -128, 127).astype(np.int8)
                    
                interpreter.set_tensor(input_details["index"], x_tflite)
                interpreter.invoke()
                out_tflite = interpreter.get_tensor(output_details["index"])[0]
                
                if output_details["dtype"] == np.uint8 or output_details["dtype"] == np.int8:
                    scale, zero_point = output_details["quantization"]
                    out_tflite = (out_tflite.astype(np.float32) - zero_point) * scale
                    
                preds_tflite.append(np.argmax(out_tflite))
                logits_tflite.append(out_tflite)

        logits_fp32 = np.array(logits_fp32)
        logits_onnx = np.array(logits_onnx)
        if has_tflite:
            logits_tflite = np.array(logits_tflite)
        
        # Compute accuracy
        acc_fp32 = np.mean(np.array(preds_fp32) == np.array(ground_truths))
        acc_onnx = np.mean(np.array(preds_onnx) == np.array(ground_truths))
        acc_tflite = np.mean(np.array(preds_tflite) == np.array(ground_truths)) if has_tflite else 0.0
        
        accuracy_dict = {
            "fp32_accuracy": float(acc_fp32),
            "onnx_accuracy": float(acc_onnx),
            "tflite_accuracy": float(acc_tflite) if has_tflite else "NOT GENERATED",
            "onnx_delta_vs_fp32_pp": float(acc_onnx - acc_fp32) * 100.0,
            "tflite_delta_vs_fp32_pp": float(acc_tflite - acc_fp32) * 100.0 if has_tflite else "NOT GENERATED",
            "tflite_delta_vs_onnx_pp": float(acc_tflite - acc_onnx) * 100.0 if has_tflite else "NOT GENERATED"
        }
        
        # Classification report for each model
        class_mapping = self.config["class_mapping"]
        num_classes = self.config.get("expected_num_classes", 10)
        
        # Invert class mapping for reports
        inv_map = {v: k for k, v in class_mapping.items()}
        class_names = [inv_map.get(i, f"Class_{i}") for i in range(num_classes)]
        
        classification_dict = {
            "fp32": self._compute_class_metrics(ground_truths, preds_fp32, class_names, num_classes),
            "onnx": self._compute_class_metrics(ground_truths, preds_onnx, class_names, num_classes),
            "tflite": self._compute_class_metrics(ground_truths, preds_tflite, class_names, num_classes) if has_tflite else None
        }
        
        # Numerical comparison metrics
        numerical_dict = {
            "fp32_vs_onnx": self._compute_similarity(logits_fp32, logits_onnx, preds_fp32, preds_onnx),
            "onnx_vs_tflite": self._compute_similarity(logits_onnx, logits_tflite, preds_onnx, preds_tflite) if has_tflite else None,
            "fp32_vs_tflite": self._compute_similarity(logits_fp32, logits_tflite, preds_fp32, preds_tflite) if has_tflite else None
        }
        
        return accuracy_dict, classification_dict, numerical_dict

    def _compute_class_metrics(self, y_true: List[int], y_pred: List[int], class_names: List[str], num_classes: int) -> Dict[str, Any]:
        """Compute precision, recall, F1, and confusion matrix."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
            
        per_class = {}
        total_support = 0
        sum_prec, sum_rec, sum_f1 = 0.0, 0.0, 0.0
        w_prec, w_rec, w_f1 = 0.0, 0.0, 0.0
        
        for i, name in enumerate(class_names):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            support = int(np.sum(cm[i, :]))
            
            total_support += support
            sum_prec += prec
            sum_rec += rec
            sum_f1 += f1
            
            w_prec += prec * support
            w_rec += rec * support
            w_f1 += f1 * support
            
            per_class[name] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
                "accuracy": float((tp + (len(y_true) - tp - fp - fn)) / len(y_true))
            }
            
        active_classes = len(class_names)
        
        return {
            "per_class": per_class,
            "macro_avg": {
                "precision": sum_prec / active_classes,
                "recall": sum_rec / active_classes,
                "f1": sum_f1 / active_classes,
                "support": int(total_support)
            },
            "weighted_avg": {
                "precision": w_prec / total_support if total_support > 0 else 0.0,
                "recall": w_rec / total_support if total_support > 0 else 0.0,
                "f1": w_f1 / total_support if total_support > 0 else 0.0,
                "support": int(total_support)
            },
            "confusion_matrix": cm.tolist()
        }

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray, pred_a: List[int], pred_b: List[int]) -> Dict[str, Any]:
        """Compute cosine similarity, MAE, RMSE, max error, and prediction agreement."""
        nan_count = int(np.isnan(a).sum() + np.isnan(b).sum())
        inf_count = int(np.isinf(a).sum() + np.isinf(b).sum())
        
        # Mask out any NaN/Infs to prevent similarity metric crashes
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.all():
            a = np.where(mask, a, 0.0)
            b = np.where(mask, b, 0.0)
            
        mae = float(np.mean(np.abs(a - b)))
        rmse = float(np.sqrt(np.mean((a - b)**2)))
        max_err = float(np.max(np.abs(a - b)))
        
        # Cosine similarity calculation
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        dot_prod = np.sum(a * b, axis=1)
        
        # Handle zero divisions
        sims = np.where((norm_a * norm_b) > 0.0, dot_prod / (norm_a * norm_b), 0.0)
        cos_sim = float(np.mean(sims))
        
        pred_agreement = float(np.mean(np.array(pred_a) == np.array(pred_b)))
        
        return {
            "cosine_similarity": cos_sim,
            "mean_absolute_error": mae,
            "rmse": rmse,
            "max_absolute_error": max_err,
            "prediction_agreement": pred_agreement,
            "nan_count": nan_count,
            "inf_count": inf_count
        }

    def audit_tflite(self) -> Dict[str, Any]:
        """Audit the structural properties of the TFLite FlatBuffer."""
        if not self.tflite_model_path or self.tflite_model_path == "NOT GENERATED":
            return {"status": "NOT GENERATED"}
        file_size = os.path.getsize(self.tflite_model_path)
        
        interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        tensor_details = interpreter.get_tensor_details()
        tensor_count = len(tensor_details)
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        float32_count = 0
        float16_count = 0
        int8_count = 0
        int32_count = 0
        quantized_count = 0
        quant_details = {}
        
        # Inspect tensor detail arrays
        for t in tensor_details:
            dtype = t["dtype"]
            if dtype == np.float32:
                float32_count += 1
            elif dtype == np.float16:
                float16_count += 1
            elif dtype == np.int8 or dtype == np.uint8:
                int8_count += 1
            elif dtype == np.int32:
                int32_count += 1
                
            scale, zero_point = t["quantization"]
            if scale != 0.0:
                quantized_count += 1
                # Save first few scales for verification
                if len(quant_details) < 10:
                    quant_details[t["name"]] = {"scale": float(scale), "zero_point": int(zero_point)}

        # Parse FlatBuffer table via tflite_schema
        with open(self.tflite_model_path, "rb") as f:
            model = tflite_schema.Model.GetRootAsModel(f.read(), 0)
        graph = model.Subgraphs(0)
        op_codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
        
        # Resolve BuiltinOperator names dynamically
        builtin_names = {}
        for attr in dir(tflite_schema.BuiltinOperator):
            val = getattr(tflite_schema.BuiltinOperator, attr)
            if isinstance(val, int):
                builtin_names[val] = attr

        operators_count = graph.OperatorsLength()
        op_types = {}
        for i in range(operators_count):
            op = graph.Operators(i)
            op_code = op_codes[op.OpcodeIndex()]
            builtin_code = op_code.BuiltinCode()
            if builtin_code == tflite_schema.BuiltinOperator.CUSTOM:
                op_name = op_code.CustomCode().decode("utf-8") if op_code.CustomCode() else "CUSTOM"
            else:
                op_name = builtin_names.get(builtin_code, f"UNKNOWN_{builtin_code}")
            op_types[op_name] = op_types.get(op_name, 0) + 1
            
        return {
            "file_size_bytes": file_size,
            "tensor_count": tensor_count,
            "operator_count": operators_count,
            "operator_types": op_types,
            "FLOAT32_tensors": float32_count,
            "FLOAT16_tensors": float16_count,
            "INT8_tensors": int8_count,
            "INT32_tensors": int32_count,
            "quantized_tensors": quantized_count,
            "sample_quantization_parameters": quant_details,
            "input_details": {
                "name": input_details["name"],
                "shape": input_details["shape"].tolist(),
                "dtype": str(input_details["dtype"])
            },
            "output_details": {
                "name": output_details["name"],
                "shape": output_details["shape"].tolist(),
                "dtype": str(output_details["dtype"])
            }
        }

    def audit_model_size(self) -> Dict[str, Any]:
        """Perform a file size audit comparing FP32 ONNX, Optimized ONNX, and TFLite."""
        fp32_size = os.path.getsize(self.fp32_model_path)
        onnx_size = os.path.getsize(self.onnx_model_path)
        
        has_tflite = self.tflite_model_path and self.tflite_model_path != "NOT GENERATED"
        tflite_size = os.path.getsize(self.tflite_model_path) if has_tflite else 0
        
        # Load ONNX models to extract parameters count and bytes
        import onnx
        from onnx import numpy_helper
        
        def get_onnx_params_bytes(path):
            try:
                model = onnx.load(path)
                return sum(int(numpy_helper.to_array(init).nbytes) for init in model.graph.initializer)
            except Exception:
                return 0
                
        fp32_param_bytes = get_onnx_params_bytes(self.fp32_model_path)
        onnx_param_bytes = get_onnx_params_bytes(self.onnx_model_path)
        
        tflite_param_bytes = 0
        if has_tflite:
            try:
                from tensorflow.lite.python import schema_py_generated as tflite_schema
                with open(self.tflite_model_path, "rb") as f:
                    model_fb = tflite_schema.Model.GetRootAsModel(f.read(), 0)
                graph = model_fb.Subgraphs(0)
                tensors_len = graph.TensorsLength()
                for i in range(tensors_len):
                    t = graph.Tensors(i)
                    buf_idx = t.Buffer()
                    if buf_idx > 0:
                        buf = model_fb.Buffers(buf_idx)
                        if buf.DataLength() > 0:
                            tflite_param_bytes += buf.DataLength()
            except Exception:
                tflite_param_bytes = 0

        res = {
            "fp32_onnx_size_bytes": fp32_size,
            "fp32_onnx_parameter_bytes": fp32_param_bytes,
            "optimized_onnx_size_bytes": onnx_size,
            "optimized_onnx_parameter_bytes": onnx_param_bytes,
            "tflite_size_bytes": tflite_size if has_tflite else "NOT GENERATED",
            "tflite_parameter_bytes": tflite_param_bytes if has_tflite else "NOT GENERATED",
            "fp32_to_onnx_reduction_percent": float(fp32_size - onnx_size) / fp32_size * 100.0,
            "fp32_to_tflite_reduction_percent": float(fp32_size - tflite_size) / fp32_size * 100.0 if has_tflite else "NOT GENERATED",
            "onnx_to_tflite_reduction_percent": float(onnx_size - tflite_size) / onnx_size * 100.0 if has_tflite else "NOT GENERATED",
            "fp32_to_tflite_compression_ratio": float(fp32_size) / tflite_size if has_tflite else "NOT GENERATED",
            "fp32_to_onnx_compression_ratio": float(fp32_size) / onnx_size
        }
        return res

    def benchmark_latency(self, warmup_runs: int = 10, benchmark_runs: int = 100) -> Dict[str, Any]:
        """Conduct controlled local latency and throughput benchmarking on the host."""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty. Cannot run benchmark.")
            
        # Select first sample for benchmark data
        sample = self.dataset[0]
        tensor = sample["tensor"]
        x = np.expand_dims(tensor, axis=0)
        
        # Set up sessions
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        # 1. Benchmark FP32 ONNX
        try:
            ort_sess_fp32 = ort.InferenceSession(self.fp32_model_path, sess_options)
            input_name_fp32 = ort_sess_fp32.get_inputs()[0].name
            
            for _ in range(warmup_runs):
                _ = ort_sess_fp32.run(None, {input_name_fp32: x})
                
            times_fp32 = []
            for _ in range(benchmark_runs):
                start = time.perf_counter()
                _ = ort_sess_fp32.run(None, {input_name_fp32: x})
                times_fp32.append((time.perf_counter() - start) * 1000.0) # ms
            fp32_stats = self._calculate_latencies(times_fp32)
            fp32_stats["benchmark_status"] = "PASS"
        except Exception as e:
            fp32_stats = {
                "benchmark_status": "FAILED",
                "benchmark_error": str(e),
                "mean_ms": "FAILED",
                "median_ms": "FAILED",
                "p95_ms": "FAILED",
                "min_ms": "FAILED",
                "max_ms": "FAILED",
                "throughput_ips": "FAILED"
            }

        # 2. Benchmark Optimized ONNX
        try:
            ort_sess_opt = ort.InferenceSession(self.onnx_model_path, sess_options)
            input_name_opt = ort_sess_opt.get_inputs()[0].name
            
            for _ in range(warmup_runs):
                _ = ort_sess_opt.run(None, {input_name_opt: x})
                
            times_opt = []
            for _ in range(benchmark_runs):
                start = time.perf_counter()
                _ = ort_sess_opt.run(None, {input_name_opt: x})
                times_opt.append((time.perf_counter() - start) * 1000.0)
            onnx_stats = self._calculate_latencies(times_opt)
            onnx_stats["benchmark_status"] = "PASS"
        except Exception as e:
            onnx_stats = {
                "benchmark_status": "FAILED",
                "benchmark_error": str(e),
                "mean_ms": "FAILED",
                "median_ms": "FAILED",
                "p95_ms": "FAILED",
                "min_ms": "FAILED",
                "max_ms": "FAILED",
                "throughput_ips": "FAILED"
            }

        # 3. Benchmark TFLite
        has_tflite = self.tflite_model_path and self.tflite_model_path != "NOT GENERATED"
        if has_tflite:
            try:
                interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path, num_threads=1)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
                
                x_tflite = x.copy()
                if input_details["shape"][1] != x.shape[1]:
                    x_tflite = np.expand_dims(tensor.transpose(1, 2, 0), axis=0)
                if input_details["dtype"] == np.uint8:
                    scale, zero_point = input_details["quantization"]
                    x_tflite = np.clip(np.round(x_tflite / scale + zero_point), 0, 255).astype(np.uint8)
                elif input_details["dtype"] == np.int8:
                    scale, zero_point = input_details["quantization"]
                    x_tflite = np.clip(np.round(x_tflite / scale + zero_point), -128, 127).astype(np.int8)
                
                for _ in range(warmup_runs):
                    interpreter.set_tensor(input_details["index"], x_tflite)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_details["index"])
                    
                times_tflite = []
                for _ in range(benchmark_runs):
                    start = time.perf_counter()
                    interpreter.set_tensor(input_details["index"], x_tflite)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(output_details["index"])
                    times_tflite.append((time.perf_counter() - start) * 1000.0)
                tflite_stats = self._calculate_latencies(times_tflite)
                tflite_stats["benchmark_status"] = "PASS"
            except Exception as e:
                tflite_stats = {
                    "benchmark_status": "FAILED",
                    "benchmark_error": str(e),
                    "mean_ms": "FAILED",
                    "median_ms": "FAILED",
                    "p95_ms": "FAILED",
                    "min_ms": "FAILED",
                    "max_ms": "FAILED",
                    "throughput_ips": "FAILED"
                }
        else:
            tflite_stats = {
                "benchmark_status": "FAILED",
                "benchmark_error": "TFLite model was not generated.",
                "mean_ms": "NOT GENERATED",
                "median_ms": "NOT GENERATED",
                "p95_ms": "NOT GENERATED",
                "min_ms": "NOT GENERATED",
                "max_ms": "NOT GENERATED",
                "throughput_ips": "NOT GENERATED"
            }

        # Environment details
        import platform
        env_details = {
            "cpu": platform.processor(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "onnxruntime_version": ort.__version__,
            "thread_count": 1,
            "warmup_runs": warmup_runs,
            "measured_runs": benchmark_runs
        }

        def get_speedup(num_stats, den_stats):
            if num_stats["benchmark_status"] == "PASS" and den_stats["benchmark_status"] == "PASS":
                return float(num_stats["mean_ms"] / den_stats["mean_ms"])
            return "FAILED"

        return {
            "environment": env_details,
            "fp32": fp32_stats,
            "onnx": onnx_stats,
            "tflite": tflite_stats,
            "speedup_onnx_vs_fp32": get_speedup(fp32_stats, onnx_stats),
            "speedup_tflite_vs_fp32": get_speedup(fp32_stats, tflite_stats)
        }

    def _calculate_latencies(self, times: List[float]) -> Dict[str, float]:
        """Compute summary statistics for a list of run latencies in ms."""
        arr = np.array(times)
        mean_lat = float(np.mean(arr))
        return {
            "mean_ms": mean_lat,
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "throughput_ips": float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
        }

    def benchmark_memory(self) -> Dict[str, Any]:
        """Measure RSS process memory metrics before/after loading models and during runs."""
        proc = psutil.Process()
        
        rss_baseline = proc.memory_info().rss
        
        # Load models sequentially, record memory delta, and unload
        # 1. FP32
        ort_sess_fp32 = ort.InferenceSession(self.fp32_model_path)
        rss_fp32 = proc.memory_info().rss
        # run a dummy inference to check memory during inference
        sample = self.dataset[0]
        x = np.expand_dims(sample["tensor"], axis=0)
        _ = ort_sess_fp32.run(None, {ort_sess_fp32.get_inputs()[0].name: x})
        rss_fp32_inf = proc.memory_info().rss
        del ort_sess_fp32
        
        # Garbage collect and record post-FP32 baseline
        import gc; gc.collect()
        rss_baseline_2 = proc.memory_info().rss
        
        # 2. ONNX
        ort_sess_opt = ort.InferenceSession(self.onnx_model_path)
        rss_onnx = proc.memory_info().rss
        _ = ort_sess_opt.run(None, {ort_sess_opt.get_inputs()[0].name: x})
        rss_onnx_inf = proc.memory_info().rss
        del ort_sess_opt
        
        # Garbage collect
        gc.collect()
        rss_baseline_3 = proc.memory_info().rss
        
        # 3. TFLite
        has_tflite = self.tflite_model_path and self.tflite_model_path != "NOT GENERATED"
        if has_tflite:
            interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
            interpreter.allocate_tensors()
            rss_tflite = proc.memory_info().rss
            
            input_details = interpreter.get_input_details()[0]
            x_tflite = x.copy()
            if input_details["shape"][1] != x.shape[1]:
                x_tflite = np.expand_dims(sample["tensor"].transpose(1, 2, 0), axis=0)
            if input_details["dtype"] == np.uint8:
                scale, zero_point = input_details["quantization"]
                x_tflite = np.clip(np.round(x_tflite / scale + zero_point), 0, 255).astype(np.uint8)
            elif input_details["dtype"] == np.int8:
                scale, zero_point = input_details["quantization"]
                x_tflite = np.clip(np.round(x_tflite / scale + zero_point), -128, 127).astype(np.int8)
                
            interpreter.set_tensor(input_details["index"], x_tflite)
            interpreter.invoke()
            rss_tflite_inf = proc.memory_info().rss
            del interpreter
            gc.collect()
        else:
            rss_tflite = rss_baseline_3
            rss_tflite_inf = rss_baseline_3

        return {
            "measurement_methodology": "RSS process memory is measured using psutil.Process().memory_info().rss before/after model loading and during inference runs.",
            "baseline_rss_bytes": rss_baseline,
            "fp32": {
                "baseline_rss_bytes": rss_baseline_2,
                "load_rss_bytes": rss_fp32,
                "inference_rss_bytes": rss_fp32_inf,
                "load_delta_bytes": rss_fp32 - rss_baseline_2,
                "inference_delta_bytes": rss_fp32_inf - rss_baseline_2
            },
            "onnx": {
                "baseline_rss_bytes": rss_baseline_3,
                "load_rss_bytes": rss_onnx,
                "inference_rss_bytes": rss_onnx_inf,
                "load_delta_bytes": rss_onnx - rss_baseline_3,
                "inference_delta_bytes": rss_onnx_inf - rss_baseline_3
            },
            "tflite": {
                "baseline_rss_bytes": rss_baseline_3 if has_tflite else "NOT GENERATED",
                "load_rss_bytes": rss_tflite if has_tflite else "NOT GENERATED",
                "inference_rss_bytes": rss_tflite_inf if has_tflite else "NOT GENERATED",
                "load_delta_bytes": rss_tflite - rss_baseline_3 if has_tflite else "NOT GENERATED",
                "inference_delta_bytes": rss_tflite_inf - rss_baseline_3 if has_tflite else "NOT GENERATED"
            }
        }

    def run_stability_test(self, iterations: int = 500) -> Dict[str, Any]:
        """Perform consecutive TFLite inferences to test output shape,
        prediction consistency, execution crashes, and memory leaks.
        """
        if not self.tflite_model_path or self.tflite_model_path == "NOT GENERATED":
            return {
                "iterations": iterations,
                "failures": "NOT GENERATED",
                "prediction_drift_occurrences": "NOT GENERATED",
                "rss_growth_bytes": "NOT GENERATED",
                "latency": {
                    "mean_ms": "NOT GENERATED",
                    "median_ms": "NOT GENERATED",
                    "p95_ms": "NOT GENERATED",
                    "min_ms": "NOT GENERATED",
                    "max_ms": "NOT GENERATED",
                    "throughput_ips": "NOT GENERATED"
                }
            }
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty. Cannot run stability test.")
            
        sample = self.dataset[0]
        tensor = sample["tensor"]
        x = np.expand_dims(tensor, axis=0)
        
        interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        x_tflite = x.copy()
        if input_details["shape"][1] != x.shape[1]:
            x_tflite = np.expand_dims(tensor.transpose(1, 2, 0), axis=0)
        if input_details["dtype"] == np.uint8:
            scale, zero_point = input_details["quantization"]
            x_tflite = np.clip(np.round(x_tflite / scale + zero_point), 0, 255).astype(np.uint8)
        elif input_details["dtype"] == np.int8:
            scale, zero_point = input_details["quantization"]
            x_tflite = np.clip(np.round(x_tflite / scale + zero_point), -128, 127).astype(np.int8)

        # Baseline memory
        proc = psutil.Process()
        rss_start = proc.memory_info().rss
        
        # Verify first output shape and value
        interpreter.set_tensor(input_details["index"], x_tflite)
        interpreter.invoke()
        out_first = interpreter.get_tensor(output_details["index"])[0].copy()
        first_pred = int(np.argmax(out_first))
        first_shape = out_first.shape
        
        failures = 0
        prediction_drift = 0
        times = []
        
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                interpreter.set_tensor(input_details["index"], x_tflite)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details["index"])[0]
                times.append((time.perf_counter() - start) * 1000.0)
                
                # Check output shape consistency
                if out.shape != first_shape:
                    failures += 1
                    
                # Check prediction consistency
                if int(np.argmax(out)) != first_pred:
                    prediction_drift += 1
            except Exception:
                failures += 1
                
        rss_end = proc.memory_info().rss
        
        return {
            "iterations": iterations,
            "failures": failures,
            "prediction_drift_occurrences": prediction_drift,
            "rss_growth_bytes": rss_end - rss_start,
            "latency": self._calculate_latencies(times)
        }
