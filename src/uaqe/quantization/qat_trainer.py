import os
import time
import json
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple, List, Optional
import onnx
import tensorflow as tf

from uaqe.quantization.pytorch_reconstructor import MobileNetV3Reconstructor
from uaqe.quantization.qat_policy import QATPolicy
from uaqe.quantization.activation_range_analyzer import ActivationRangeAnalyzer
from uaqe.exporter.tflite_exporter import ONNXToTFModel
from uaqe.exporter.flatbuffer_inspector import FlatBufferInspector

class QATTrainer:
    """Orchestrates Quantization-Aware Training (QAT), staged fine-tuning,
    sensitivity-aware observer optimization, distillation, and genuine INT8 export.
    """

    def __init__(
        self,
        fp32_checkpoint_path: str = "output/phase_c1/models/mobilenetv3_sem_9class_fp32.pth",
        dataset_root: str = r"D:\semiconductor_dataset\dataset",
        benchmark_path: str = "datasets/hackathon_test_dataset",
        output_dir: str = "output/phase_c2",
        reports_dir: str = "reports/phase_c2",
        random_seed: int = 42
    ) -> None:
        self.fp32_checkpoint_path = fp32_checkpoint_path
        self.dataset_root = dataset_root
        self.benchmark_path = benchmark_path
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.random_seed = random_seed
        self.device = torch.device("cpu")

        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        self.class_names = sorted([
            "bridge", "clean", "cmp", "crack", "opens", "other", "particle", "scratch", "vias"
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {str(i): name for i, name in enumerate(self.class_names)}
        self.policy = QATPolicy()
        self.analyzer = ActivationRangeAnalyzer(output_dir=self.output_dir)

        # Pre-cache dataset splits in memory
        self._load_all_datasets()

    def _load_all_datasets(self) -> None:
        """Loads and caches train, val, test, and held-out benchmark datasets."""
        from PIL import Image

        def load_dir(split_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
            split_dir = os.path.join(self.dataset_root, split_name)
            images, labels = [], []
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
            for c_name in self.class_names:
                c_dir = os.path.join(split_dir, c_name)
                if not os.path.isdir(c_dir): continue
                idx = self.class_to_idx[c_name]
                for f_name in sorted(os.listdir(c_dir)):
                    if f_name.lower().endswith(valid_exts):
                        img_path = os.path.join(c_dir, f_name)
                        img = Image.open(img_path).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
                        arr = np.array(img, dtype=np.float32) / 255.0
                        images.append(arr.transpose(2, 0, 1))
                        labels.append(idx)
            return torch.from_numpy(np.stack(images)), torch.tensor(labels, dtype=torch.long)

        self.train_x, self.train_y = load_dir("train")
        self.val_x, self.val_y = load_dir("val")
        self.test_x, self.test_y = load_dir("test")

        # Held-out 296-image benchmark
        bench_images, bench_labels, bench_native = [], [], []
        bench_to_idx = {
            "Bridge": self.class_to_idx["bridge"],
            "Clean": self.class_to_idx["clean"],
            "CMP": self.class_to_idx["cmp"],
            "Crack": self.class_to_idx["crack"],
            "LER": self.class_to_idx["scratch"],
            "Open": self.class_to_idx["opens"],
            "Other": self.class_to_idx["other"],
            "Particle": self.class_to_idx["particle"],
            "VIA": self.class_to_idx["vias"]
        }
        for folder_name in sorted(os.listdir(self.benchmark_path)):
            f_path = os.path.join(self.benchmark_path, folder_name)
            if not os.path.isdir(f_path) or folder_name not in bench_to_idx: continue
            idx = bench_to_idx[folder_name]
            for f_name in sorted(os.listdir(f_path)):
                if f_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img = Image.open(os.path.join(f_path, f_name)).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
                    bench_images.append((np.array(img, dtype=np.float32) / 255.0).transpose(2, 0, 1))
                    bench_labels.append(idx)
                    bench_native.append(folder_name)
        self.bench_x = torch.from_numpy(np.stack(bench_images))
        self.bench_y = bench_labels
        self.bench_native_classes = bench_native

    def load_base_fp32_model(self) -> nn.Module:
        """Instantiates MobileNetV3-Small and loads the Phase C.1 FP32 checkpoint."""
        model = models.mobilenet_v3_small(num_classes=9)
        if not os.path.exists(self.fp32_checkpoint_path):
            raise FileNotFoundError(f"FP32 Checkpoint not found: {self.fp32_checkpoint_path}")
        ckpt = torch.load(self.fp32_checkpoint_path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def evaluate_model(
        self,
        model: nn.Module,
        x_tensor: torch.Tensor,
        y_labels: List[int]
    ) -> Dict[str, Any]:
        """Calculates classification accuracy, macro precision, recall, and F1."""
        model.eval()
        total = len(y_labels)
        all_preds = []
        all_probs = []

        t0 = time.time()
        with torch.no_grad():
            batch_size = 64
            for i in range(0, total, batch_size):
                bx = x_tensor[i:i+batch_size].to(self.device)
                out = model(bx)
                all_probs.extend(F.softmax(out, dim=1).cpu().numpy())
                all_preds.extend(out.argmax(dim=1).cpu().numpy())
        eval_time = time.time() - t0

        y_true = np.array(y_labels)
        y_pred = np.array(all_preds)
        correct = int(np.sum(y_true == y_pred))
        acc = float(correct / total) if total > 0 else 0.0

        num_c = len(self.class_names)
        conf_mat = np.zeros((num_c, num_c), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t < num_c and p < num_c:
                conf_mat[t, p] += 1

        precs, recs, f1s = [], [], []
        per_class: Dict[str, Any] = {}
        for idx, c_name in enumerate(self.class_names):
            sup = int(np.sum(y_true == idx))
            c_corr = int(conf_mat[idx, idx])
            c_p_cnt = int(np.sum(y_pred == idx))
            p = float(c_corr / c_p_cnt) if c_p_cnt > 0 else 0.0
            r = float(c_corr / sup) if sup > 0 else 0.0
            f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            per_class[c_name] = {"support": sup, "correct": c_corr, "accuracy": float(c_corr / sup) if sup > 0 else 0.0, "f1": f1}
            if sup > 0:
                precs.append(p); recs.append(r); f1s.append(f1)

        return {
            "total_samples": total,
            "correct": correct,
            "accuracy": acc,
            "macro_precision": float(np.mean(precs)) if precs else 0.0,
            "macro_recall": float(np.mean(recs)) if recs else 0.0,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "latency_ms": float((eval_time / total) * 1000.0) if total > 0 else 0.0,
            "per_class": per_class,
            "confusion_matrix": conf_mat.tolist()
        }

    def train_qat_experiment(
        self,
        experiment_id: str,
        policy_mode: str = "standard",
        epochs: int = 15,
        lr: float = 8e-5,
        weight_decay: float = 1e-5,
        use_distillation: bool = False,
        distill_alpha: float = 0.5,
        distill_temperature: float = 2.0
    ) -> Dict[str, Any]:
        """Runs a controlled QAT experiment with staged observer warmup and fine-tuning."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        base_model = self.load_base_fp32_model()
        teacher_model = None
        if use_distillation:
            teacher_model = self.load_base_fp32_model()
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

        # Apply QAT Policy and prepare model
        qat_model = self.policy.apply_qat_policy(base_model, policy_mode=policy_mode)
        qat_model.to(self.device)

        train_loader = DataLoader(
            TensorDataset(self.train_x, self.train_y),
            batch_size=32,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(self.val_x, self.val_y),
            batch_size=64,
            shuffle=False
        )

        optimizer = torch.optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        ce_loss_fn = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_epoch = -1
        best_state = None
        history = []

        print(f"\n[{experiment_id}] Starting QAT ({epochs} epochs, lr={lr}, policy={policy_mode}, distill={use_distillation})...")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            qat_model.train()
            train_loss, train_corr = 0.0, 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = qat_model(bx)

                if use_distillation and teacher_model is not None:
                    with torch.no_grad():
                        t_out = teacher_model(bx)
                    ce_loss = ce_loss_fn(out, by)
                    kl_loss = F.kl_div(
                        F.log_softmax(out / distill_temperature, dim=1),
                        F.softmax(t_out / distill_temperature, dim=1),
                        reduction="batchmean"
                    ) * (distill_temperature ** 2)
                    loss = distill_alpha * ce_loss + (1.0 - distill_alpha) * kl_loss
                else:
                    loss = ce_loss_fn(out, by)

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(by)
                train_corr += (out.argmax(dim=1) == by).sum().item()

            scheduler.step()

            # Validation
            qat_model.eval()
            val_loss, val_corr = 0.0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    v_out = qat_model(bx)
                    v_loss = ce_loss_fn(v_out, by)
                    val_loss += v_loss.item() * len(by)
                    val_corr += (v_out.argmax(dim=1) == by).sum().item()

            train_acc = train_corr / len(self.train_y)
            val_acc = val_corr / len(self.val_y)
            elapsed = time.time() - t0

            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss / len(self.train_y)),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss / len(self.val_y)),
                "val_acc": float(val_acc),
                "lr": float(scheduler.get_last_lr()[0]),
                "time": float(elapsed)
            })
            print(f"  Epoch {epoch:2d}/{epochs} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | {elapsed:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(qat_model.state_dict())

        # Restore best weights
        if best_state is not None:
            qat_model.load_state_dict(best_state)

        # Full evaluations
        eval_train = self.evaluate_model(qat_model, self.train_x, self.train_y.tolist())
        eval_val = self.evaluate_model(qat_model, self.val_x, self.val_y.tolist())
        eval_test = self.evaluate_model(qat_model, self.test_x, self.test_y.tolist())
        eval_bench = self.evaluate_model(qat_model, self.bench_x, self.bench_y)

        # Save checkpoint
        ckpt_path = os.path.join(self.output_dir, "models", f"{experiment_id}_best.pth")
        torch.save({
            "model_state_dict": qat_model.state_dict(),
            "experiment_id": experiment_id,
            "policy_mode": policy_mode,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "test_accuracy": eval_test["accuracy"],
            "benchmark_accuracy": eval_bench["accuracy"]
        }, ckpt_path)

        # Export ONNX
        onnx_path = os.path.join(self.output_dir, "models", f"{experiment_id}.onnx")
        qat_model.eval()
        dummy = torch.randn(1, 3, 128, 128)
        torch.onnx.export(
            base_model, dummy, onnx_path,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13, dynamo=False
        )

        return {
            "experiment_id": experiment_id,
            "policy_mode": policy_mode,
            "best_epoch": best_epoch,
            "best_val_accuracy": float(best_val_acc),
            "train_eval": eval_train,
            "val_eval": eval_val,
            "test_eval": eval_test,
            "bench_eval": eval_bench,
            "checkpoint_path": ckpt_path,
            "onnx_path": onnx_path,
            "history": history,
            "model": qat_model
        }

    def export_genuine_int8_tflite(
        self,
        onnx_model_path: str,
        output_tflite_path: str
    ) -> Dict[str, Any]:
        """Converts the QAT-trained ONNX model to a genuine INT8 TFLite FlatBuffer
        using real representative dataset calibration.
        """
        onnx_model = onnx.load(onnx_model_path)
        tf_model = ONNXToTFModel(onnx_model)
        
        # Build model by calling on dummy input
        onnx_input = onnx_model.graph.input[0]
        input_shape = [dim.dim_value for dim in onnx_input.type.tensor_type.shape.dim]
        input_shape = [1 if (dim is None or dim <= 0) else dim for dim in input_shape]
        dummy_input = tf.random.normal(input_shape)
        _ = tf_model(dummy_input)

        # Build generator from real training dataset
        def rep_gen():
            for i in range(min(50, len(self.train_x))):
                x = self.train_x[i:i+1].numpy()
                yield [x.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        tflite_bytes = converter.convert()

        os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
        with open(output_tflite_path, "wb") as f:
            f.write(tflite_bytes)

        inspector = FlatBufferInspector()
        audit = inspector.inspect(output_tflite_path)

        return {
            "tflite_path": output_tflite_path,
            "size_bytes": len(tflite_bytes),
            "size_mb": float(len(tflite_bytes) / (1024 * 1024)),
            "int8_tensors": audit.int8_tensors,
            "int32_tensors": audit.int32_tensors,
            "fp32_tensors": audit.fp32_tensors,
            "total_tensors": audit.total_tensors,
            "int8_coverage_percent": float(audit.int8_coverage_percent)
        }

    def measure_host_latency(
        self,
        tflite_path: str,
        warmup_runs: int = 20,
        benchmark_runs: int = 100
    ) -> Dict[str, float]:
        """Measures execution latency using the TFLite interpreter on host CPU."""
        interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter.allocate_tensors()
        input_idx = interpreter.get_input_details()[0]["index"]
        output_idx = interpreter.get_output_details()[0]["index"]

        dummy_x = np.random.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32)

        # Warmup
        for _ in range(warmup_runs):
            interpreter.set_tensor(input_idx, dummy_x)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_idx)

        # Timed runs
        timings = []
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            interpreter.set_tensor(input_idx, dummy_x)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_idx)
            timings.append((time.perf_counter() - t0) * 1000.0)

        return {
            "mean_ms": float(np.mean(timings)),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "min_ms": float(np.min(timings)),
            "max_ms": float(np.max(timings))
        }

    def run_stability_test(
        self,
        tflite_path: str,
        num_runs: int = 500
    ) -> Dict[str, Any]:
        """Runs 500 repeated inferences to verify output stability, lack of drift, and absence of NaN/Inf."""
        interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter.allocate_tensors()
        input_idx = interpreter.get_input_details()[0]["index"]
        output_idx = interpreter.get_output_details()[0]["index"]

        fixed_x = np.random.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32)
        interpreter.set_tensor(input_idx, fixed_x)
        interpreter.invoke()
        initial_out = interpreter.get_tensor(output_idx).copy()

        failures = 0
        drifts = 0
        nan_inf_count = 0

        for i in range(num_runs):
            try:
                interpreter.set_tensor(input_idx, fixed_x)
                interpreter.invoke()
                out = interpreter.get_tensor(output_idx)
                if np.isnan(out).any() or np.isinf(out).any():
                    nan_inf_count += 1
                if not np.allclose(out, initial_out, atol=1e-5):
                    drifts += 1
            except Exception:
                failures += 1

        return {
            "total_runs": num_runs,
            "failures": failures,
            "prediction_drift": drifts,
            "nan_inf_count": nan_inf_count,
            "status": "PASS" if failures == 0 and drifts == 0 and nan_inf_count == 0 else "FAIL"
        }
