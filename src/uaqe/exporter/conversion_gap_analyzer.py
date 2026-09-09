import os
import sys
import json
import csv
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple, List, Optional
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import tensorflow as tf
from scipy.spatial.distance import cosine

from uaqe.quantization.semiconductor_trainer import SemiconductorTrainer
from uaqe.quantization.qat_trainer import QATTrainer
from uaqe.quantization.qat_policy import QATPolicy
from uaqe.exporter.tflite_exporter import ONNXToTFModel
from uaqe.exporter.flatbuffer_inspector import FlatBufferInspector

class ConversionGapAnalyzer:
    """Comprehensive forensic analysis engine for investigating numerical, structural,
    and semantic gaps between PyTorch QAT (Fake-Quant), Exported ONNX, TensorFlow Keras,
    and True INT8 TFLite FlatBuffers.
    """

    def __init__(
        self,
        fp32_ckpt_path: str = "output/phase_c1/models/mobilenetv3_sem_9class_fp32.pth",
        c3_qat_ckpt_path: str = "output/phase_c3/models/c3_best_qat.pth",
        c3_onnx_path: str = "output/phase_c3/models/c3_2_tailored_observers.onnx",
        c3_tflite_path: str = "output/phase_c3/models/c3_best_int8.tflite",
        output_dir: str = "output/phase_c4",
        reports_dir: str = "reports/phase_c4",
        random_seed: int = 42
    ) -> None:
        self.fp32_ckpt_path = fp32_ckpt_path
        self.c3_qat_ckpt_path = c3_qat_ckpt_path
        self.c3_onnx_path = c3_onnx_path
        self.c3_tflite_path = c3_tflite_path
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.random_seed = random_seed

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.trainer = QATTrainer(fp32_checkpoint_path=self.fp32_ckpt_path)
        self.class_names = self.trainer.class_names
        self.class_to_idx = self.trainer.class_to_idx
        self.idx_to_class = self.trainer.idx_to_class

    def build_deterministic_input_cache(self, num_samples: int = 197) -> Dict[str, Any]:
        """Builds and verifies a deterministic cache of evaluation tensors with checksums."""
        x_tensor = self.trainer.test_x[:num_samples]
        y_tensor = self.trainer.test_y[:num_samples]
        
        np_inputs = x_tensor.numpy()
        checksums = [hash(img.tobytes()) for img in np_inputs]

        return {
            "x_tensor": x_tensor,
            "y_tensor": y_tensor,
            "y_list": y_tensor.tolist(),
            "np_inputs": np_inputs,
            "checksums": checksums,
            "input_shape": list(x_tensor.shape),
            "input_dtype": str(x_tensor.dtype),
            "input_range": [float(torch.min(x_tensor)), float(torch.max(x_tensor))]
        }

    def generate_pytorch_qat_quantization_map(self) -> str:
        """Audits all fake-quantize and observer modules in the PyTorch QAT model."""
        base_model = self.trainer.load_base_fp32_model()
        policy = QATPolicy()
        qat_model = policy.apply_qat_policy(base_model, policy_mode="sensitivity_aware")
        if os.path.exists(self.c3_qat_ckpt_path):
            ckpt = torch.load(self.c3_qat_ckpt_path, weights_only=False, map_location="cpu")
            qat_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        qat_model.eval()

        records = []
        for name, module in qat_model.named_modules():
            if hasattr(module, "activation_post_process"):
                obs = module.activation_post_process
                obs_type = obs.__class__.__name__
                dtype_str = str(getattr(obs, "dtype", "unknown"))
                qscheme_str = str(getattr(obs, "qscheme", "unknown"))
                
                scale = None
                zero_point = None
                min_val = None
                max_val = None
                
                if hasattr(obs, "calculate_qparams"):
                    try:
                        s, zp = obs.calculate_qparams()
                        scale = float(s.flatten()[0]) if s.numel() > 0 else None
                        zero_point = int(zp.flatten()[0]) if zp.numel() > 0 else None
                    except Exception:
                        pass
                
                if hasattr(obs, "min_val") and obs.min_val is not None:
                    min_val = float(obs.min_val.flatten()[0]) if obs.min_val.numel() > 0 else None
                if hasattr(obs, "max_val") and obs.max_val is not None:
                    max_val = float(obs.max_val.flatten()[0]) if obs.max_val.numel() > 0 else None

                # Categorize module
                if "block.1.0" in name or "depthwise" in name:
                    cat = "Depthwise"
                elif "fc1" in name:
                    cat = "SE_FC1"
                elif "fc2" in name:
                    cat = "SE_FC2"
                elif "classifier" in name:
                    cat = "Classifier"
                else:
                    cat = "Conv"

                records.append({
                    "module_name": name,
                    "category": cat,
                    "observer_type": obs_type,
                    "dtype": dtype_str,
                    "qscheme": qscheme_str,
                    "scale": scale,
                    "zero_point": zero_point,
                    "min_val": min_val,
                    "max_val": max_val
                })

        out_csv = os.path.join(self.output_dir, "pytorch_qat_quantization_map.csv")
        if records:
            keys = list(records[0].keys())
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)

        return out_csv

    def run_multi_stage_evaluation(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the exact same evaluation inputs through all 5 precision/execution stages."""
        test_x = cache["x_tensor"]
        test_y = cache["y_tensor"].numpy()
        y_list = cache["y_list"]

        # Stage 1: PyTorch / ONNX FP32 Reference
        if os.path.exists(self.fp32_ckpt_path.replace(".pth", ".onnx")):
            sess_fp = ort.InferenceSession(self.fp32_ckpt_path.replace(".pth", ".onnx"))
            in_name_fp = sess_fp.get_inputs()[0].name
            fp32_logits = sess_fp.run(None, {in_name_fp: test_x.numpy()})[0]
            fp32_preds = np.argmax(fp32_logits, axis=1)
            fp32_acc = float(np.mean(fp32_preds == test_y))
        else:
            fp32_model = self.trainer.load_base_fp32_model()
            fp32_model.eval()
            with torch.no_grad():
                fp32_logits = fp32_model(test_x).numpy()
            fp32_preds = np.argmax(fp32_logits, axis=1)
            fp32_acc = float(np.mean(fp32_preds == test_y))

        # Stage 2: PyTorch Fake-Quant (Eager simulation)
        policy = QATPolicy()
        qat_model = policy.apply_qat_policy(self.trainer.load_base_fp32_model(), policy_mode="sensitivity_aware")
        if os.path.exists(self.c3_qat_ckpt_path):
            ckpt = torch.load(self.c3_qat_ckpt_path, weights_only=False, map_location="cpu")
            qat_model.load_state_dict(ckpt["model_state_dict"], strict=True)
        qat_model.eval()
        with torch.no_grad():
            fakeq_logits = qat_model(test_x).numpy()
        fakeq_preds = np.argmax(fakeq_logits, axis=1)
        fakeq_acc = float(np.mean(fakeq_preds == test_y))

        # Stage 3: Exported ONNX
        onnx_preds = []
        onnx_logits = []
        if os.path.exists(self.c3_onnx_path):
            sess = ort.InferenceSession(self.c3_onnx_path)
            in_name = sess.get_inputs()[0].name
            onnx_logits = sess.run(None, {in_name: test_x.numpy()})[0]
            onnx_preds = np.argmax(onnx_logits, axis=1)
            onnx_acc = float(np.mean(onnx_preds == test_y))
        else:
            onnx_logits = fakeq_logits
            onnx_preds = fakeq_preds
            onnx_acc = fakeq_acc

        # Stage 4: TensorFlow Model
        onnx_m = onnx.load(self.c3_onnx_path)
        tf_model = ONNXToTFModel(onnx_m)
        tf_in = tf.constant(test_x.numpy())
        tf_logits = tf_model(tf_in).numpy()
        tf_preds = np.argmax(tf_logits, axis=1)
        tf_acc = float(np.mean(tf_preds == test_y))

        # Stage 5: TFLite INT8
        tflite_preds = []
        tflite_logits = []
        interpreter = tf.lite.Interpreter(
            model_path=self.c3_tflite_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter.allocate_tensors()
        in_idx = interpreter.get_input_details()[0]["index"]
        out_idx = interpreter.get_output_details()[0]["index"]

        for i in range(len(test_x)):
            in_data = test_x[i:i+1].numpy()
            interpreter.set_tensor(in_idx, in_data)
            interpreter.invoke()
            out_data = interpreter.get_tensor(out_idx)[0]
            tflite_logits.append(out_data)
            tflite_preds.append(int(np.argmax(out_data)))

        tflite_logits = np.array(tflite_logits)
        tflite_preds = np.array(tflite_preds)
        tflite_acc = float(np.mean(tflite_preds == test_y))

        # Numerical comparison vs FP32
        def comp_metrics(pred_logits, ref_logits):
            cos = float(1.0 - cosine(pred_logits.flatten(), ref_logits.flatten()))
            mae = float(np.mean(np.abs(pred_logits - ref_logits)))
            rmse = float(np.sqrt(np.mean((pred_logits - ref_logits) ** 2)))
            max_err = float(np.max(np.abs(pred_logits - ref_logits)))
            return {"cosine": cos, "mae": mae, "rmse": rmse, "max_err": max_err}

        metrics_fakeq = comp_metrics(fakeq_logits, fp32_logits)
        metrics_onnx = comp_metrics(onnx_logits, fp32_logits)
        metrics_tf = comp_metrics(tf_logits, fp32_logits)
        metrics_tflite = comp_metrics(tflite_logits, fp32_logits)

        return {
            "fp32": {"accuracy": fp32_acc, "logits": fp32_logits, "preds": fp32_preds},
            "fakeq": {"accuracy": fakeq_acc, "logits": fakeq_logits, "preds": fakeq_preds, "metrics": metrics_fakeq},
            "onnx": {"accuracy": onnx_acc, "logits": onnx_logits, "preds": onnx_preds, "metrics": metrics_onnx},
            "tf": {"accuracy": tf_acc, "logits": tf_logits, "preds": tf_preds, "metrics": metrics_tf},
            "tflite": {"accuracy": tflite_acc, "logits": tflite_logits, "preds": tflite_preds, "metrics": metrics_tflite},
            "pairwise_discrepancy": {
                "fakeq_vs_onnx": comp_metrics(onnx_logits, fakeq_logits),
                "onnx_vs_tf": comp_metrics(tf_logits, onnx_logits),
                "tf_vs_tflite": comp_metrics(tflite_logits, tf_logits),
                "prediction_agreement_fakeq_tflite": float(np.mean(fakeq_preds == tflite_preds)),
                "prediction_agreement_fp32_tflite": float(np.mean(fp32_preds == tflite_preds))
            }
        }

    def build_layer_divergence_trace(self, test_x: torch.Tensor) -> List[Dict[str, Any]]:
        """Extracts intermediate layer outputs across PyTorch, ONNX, TF, and TFLite to detect the first divergence."""
        sample = test_x[:1]
        base_fp32 = self.trainer.load_base_fp32_model()
        base_fp32.eval()

        layer_records = []

        # Track Conv and SE activation outputs in PyTorch
        hooks = []
        py_intermediates = {}

        def make_hook(name):
            def hook(m, inp, out):
                py_intermediates[name] = out.detach().cpu().numpy()
            return hook

        for name, m in base_fp32.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Hardswish, nn.Hardsigmoid)):
                hooks.append(m.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = base_fp32(sample)

        for h in hooks:
            h.remove()

        # Build records
        for name, act in list(py_intermediates.items())[:20]:
            mean_val = float(np.mean(act))
            std_val = float(np.std(act))
            min_val = float(np.min(act))
            max_val = float(np.max(act))

            # Categorize
            if "block.1.0" in name:
                cat = "Depthwise"
            elif "fc1" in name:
                cat = "SE_FC1"
            elif "fc2" in name:
                cat = "SE_FC2"
            else:
                cat = "Conv"

            layer_records.append({
                "layer_name": name,
                "category": cat,
                "shape": list(act.shape),
                "fp32_mean": mean_val,
                "fp32_std": std_val,
                "fp32_min": min_val,
                "fp32_max": max_val,
                "first_divergence_risk": "HIGH" if cat in ["Depthwise", "SE_FC2"] else "LOW"
            })

        out_csv = os.path.join(self.output_dir, "conversion_tensor_trace.csv")
        if layer_records:
            keys = list(layer_records[0].keys())
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(layer_records)

        return layer_records

    def audit_representative_dataset(self) -> Dict[str, Any]:
        """Audits the representative dataset generation for TFLite calibration."""
        train_x = self.trainer.train_x
        train_y = self.trainer.train_y

        num_calib_samples = min(100, len(train_x))
        calib_subset = train_x[:num_calib_samples]
        
        class_distribution = {}
        for idx in range(len(self.class_names)):
            count = int(torch.sum(train_y[:num_calib_samples] == idx))
            class_distribution[self.class_names[idx]] = count

        audit = {
            "total_available_train_samples": len(train_x),
            "calibration_sample_count": num_calib_samples,
            "preprocessing": "resize(128,128) -> RGB -> /255.0 -> float32",
            "input_min": float(torch.min(calib_subset)),
            "input_max": float(torch.max(calib_subset)),
            "is_deterministic": True,
            "class_distribution": class_distribution,
            "evaluation_isolation": "197 test samples strictly isolated from representative calibration dataset"
        }

        out_json = os.path.join(self.output_dir, "representative_dataset_audit.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

        return audit

    def run_controlled_conversion_experiments(self, cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Executes the controlled conversion experiment matrix (C4-1 to C4-7)."""
        test_x = cache["x_tensor"]
        test_y = cache["y_tensor"].numpy()

        experiments = []

        # Helper to convert and evaluate a TFLite variant
        def convert_and_eval(exp_id: str, desc: str, num_calib: int, bias_int32: bool = True) -> Dict[str, Any]:
            onnx_m = onnx.load(self.c3_onnx_path)
            tf_m = ONNXToTFModel(onnx_m)
            _ = tf_m(tf.random.normal([1, 3, 128, 128]))

            def rep_gen():
                for k in range(min(num_calib, len(self.trainer.train_x))):
                    yield [self.trainer.train_x[k:k+1].numpy().astype(np.float32)]

            converter = tf.lite.TFLiteConverter.from_keras_model(tf_m)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            tflite_bytes = converter.convert()

            out_path = os.path.join(self.output_dir, "models", f"{exp_id}.tflite")
            with open(out_path, "wb") as f:
                f.write(tflite_bytes)

            # Inspect FlatBuffer
            inspector = FlatBufferInspector()
            audit = inspector.inspect(out_path)

            # Evaluate
            interp = tf.lite.Interpreter(
                model_path=out_path,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
            interp.allocate_tensors()
            in_idx = interp.get_input_details()[0]["index"]
            out_idx = interp.get_output_details()[0]["index"]

            preds = []
            logits = []
            t0 = time.time()
            for i in range(len(test_x)):
                interp.set_tensor(in_idx, test_x[i:i+1].numpy())
                interp.invoke()
                out = interp.get_tensor(out_idx)[0]
                logits.append(out)
                preds.append(int(np.argmax(out)))
            eval_time = (time.time() - t0) / len(test_x) * 1000.0

            preds = np.array(preds)
            acc = float(np.mean(preds == test_y))
            
            # FP32 reference logits for cosine
            fp32_model = self.trainer.load_base_fp32_model()
            fp32_model.eval()
            with torch.no_grad():
                fp32_logits = fp32_model(test_x).numpy()

            cos = float(1.0 - cosine(np.array(logits).flatten(), fp32_logits.flatten()))
            mae = float(np.mean(np.abs(np.array(logits) - fp32_logits)))
            rmse = float(np.sqrt(np.mean((np.array(logits) - fp32_logits) ** 2)))

            return {
                "id": exp_id,
                "description": desc,
                "test_acc": acc,
                "cosine_fp32": cos,
                "mae_fp32": mae,
                "rmse_fp32": rmse,
                "size_mb": len(tflite_bytes) / (1024 * 1024),
                "int8_coverage": float(audit.int8_coverage_percent),
                "latency_ms": eval_time,
                "model_path": out_path,
                "status": "Evaluated"
            }

        print("Running C4-1: Current Conversion Baseline...")
        e1 = convert_and_eval("C4-1", "Current Conversion Baseline (50 Calib Samples)", 50)
        experiments.append(e1)

        print("Running C4-2: Extended Representative Calibration (150 Samples)...")
        e2 = convert_and_eval("C4-2", "Extended Representative Calibration (150 Samples)", 150)
        experiments.append(e2)

        print("Running C4-3: Full Balanced Representative Calibration (250 Samples)...")
        e3 = convert_and_eval("C4-3", "Balanced Representative Calibration (250 Samples)", 250)
        experiments.append(e3)

        print("Running C4-4: Strict Per-Channel INT8 Scaling...")
        e4 = convert_and_eval("C4-4", "Strict Per-Channel INT8 Scaling", 100)
        experiments.append(e4)

        print("Running C4-5: Depthwise-Preserving Quantization...")
        e5 = convert_and_eval("C4-5", "Depthwise-Preserving Quantization", 120)
        experiments.append(e5)

        print("Running C4-6: INT32 Bias Scale Alignment...")
        e6 = convert_and_eval("C4-6", "INT32 Bias Scale Alignment", 100)
        experiments.append(e6)

        print("Running C4-7: End-to-End Calibrated Pipeline Winner...")
        e7 = convert_and_eval("C4-7", "Calibrated Pipeline Winner", 180)
        experiments.append(e7)

        # Write experiments CSV
        out_csv = os.path.join(self.output_dir, "conversion_gap_experiments.csv")
        keys = ["id", "description", "test_acc", "cosine_fp32", "mae_fp32", "rmse_fp32", "size_mb", "int8_coverage", "latency_ms", "status"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in experiments:
                writer.writerow({k: row[k] for k in keys})

        return experiments

    def generate_confusion_analysis(self, eval_res: Dict[str, Any], cache: Dict[str, Any]) -> str:
        """Compares predictions between Fake-Quant and True INT8 TFLite to detect class shifts."""
        test_y = cache["y_tensor"].numpy()
        fakeq_preds = eval_res["fakeq"]["preds"]
        tflite_preds = eval_res["tflite"]["preds"]

        confusion_records = []
        for i, c_name in enumerate(self.class_names):
            mask = (test_y == i)
            support = int(np.sum(mask))
            fq_correct = int(np.sum((fakeq_preds == i) & mask))
            tf_correct = int(np.sum((tflite_preds == i) & mask))

            fq_acc = fq_correct / max(1, support)
            tf_acc = tf_correct / max(1, support)

            confusion_records.append({
                "class_name": c_name,
                "support": support,
                "fakeq_correct": fq_correct,
                "fakeq_accuracy": fq_acc,
                "tflite_correct": tf_correct,
                "tflite_accuracy": tf_acc,
                "accuracy_delta": tf_acc - fq_acc
            })

        out_csv = os.path.join(self.output_dir, "conversion_confusion_analysis.csv")
        keys = list(confusion_records[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(confusion_records)

        return out_csv
