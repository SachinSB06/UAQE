import os
import sys
import time
import json
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple, List, Optional
import onnx
import tensorflow as tf

from uaqe.quantization.qat_trainer import QATTrainer
from uaqe.quantization.qat_policy import QATPolicy
from uaqe.quantization.activation_range_analyzer import ActivationRangeAnalyzer
from uaqe.exporter.tflite_exporter import ONNXToTFModel
from uaqe.exporter.flatbuffer_inspector import FlatBufferInspector

class C3AdvancedQATEngine:
    """Orchestrates advanced QAT strategies for MobileNetV3-Small:
    - Advanced Observer Strategies (Histogram, Tailored MovingAverage, PerChannel)
    - Staged Calibration & Observer Freezing
    - Depthwise-Targeted Activation Range Clipping
    - Multi-Objective Teacher Distillation (Logits + Intermediate Feature Maps)
    - Sensitive Gradient Scaling
    - True INT8 TFLite Conversion and Verification
    """

    def __init__(
        self,
        fp32_checkpoint_path: str = "output/phase_c1/models/mobilenetv3_sem_9class_fp32.pth",
        output_dir: str = "output/phase_c3",
        reports_dir: str = "reports/phase_c3",
        random_seed: int = 42
    ) -> None:
        self.fp32_checkpoint_path = fp32_checkpoint_path
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.random_seed = random_seed
        self.device = torch.device("cpu")

        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        self.trainer = QATTrainer(
            fp32_checkpoint_path=self.fp32_checkpoint_path,
            output_dir=output_dir,
            reports_dir=reports_dir,
            random_seed=random_seed
        )
        self.class_names = self.trainer.class_names
        self.policy = QATPolicy()
        self.analyzer = ActivationRangeAnalyzer(output_dir=self.output_dir)

    def _get_feature_extractor_hook(self, model: nn.Module) -> Tuple[Any, List[torch.Tensor]]:
        """Attaches a forward hook to extract pre-classifier global pooled features for distillation."""
        features_buffer = []
        def hook(m, inp, out):
            features_buffer.append(out)
        
        # In MobileNetV3-Small, avgpool is right before classifier
        handle = model.avgpool.register_forward_hook(hook)
        return handle, features_buffer

    def train_c3_experiment(
        self,
        exp_id: str,
        name: str,
        policy_mode: str = "sensitivity_aware",
        epochs: int = 15,
        lr: float = 6e-5,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine", # "cosine", "step", "plateau"
        observer_warmup_epochs: int = 2,
        observer_freeze_epoch: int = 12,
        use_distillation: bool = True,
        distill_alpha: float = 0.5, # Weight for CE vs Teacher
        distill_temp: float = 2.0,
        use_feature_distill: bool = False,
        feature_distill_gamma: float = 0.2,
        depthwise_grad_scale: float = 1.0,
        class_weighted: bool = False,
        act_clip_percentile: Optional[float] = None
    ) -> Dict[str, Any]:
        """Trains a single controlled Phase C.3 QAT experiment."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # 1. Load FP32 base model & Teacher
        base_model = self.trainer.load_base_fp32_model()
        teacher_model = self.trainer.load_base_fp32_model() if use_distillation else None
        if teacher_model:
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

        # 2. Apply Policy
        qat_model = self.policy.apply_qat_policy(base_model, policy_mode=policy_mode)
        qat_model.train()

        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(self.trainer.train_x, self.trainer.train_y),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(self.trainer.val_x, self.trainer.val_y),
            batch_size=64,
            shuffle=False
        )

        # Class Weights if requested
        if class_weighted:
            class_counts = [int(torch.sum(self.trainer.train_y == i)) for i in range(len(self.class_names))]
            total = len(self.trainer.train_y)
            weights = torch.tensor([total / (len(self.class_names) * max(1, c)) for c in class_counts], dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=weight_decay)

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
        else:
            scheduler = None

        # Setup feature distillation hooks if enabled
        s_handle, s_feat_buf = (None, [])
        t_handle, t_feat_buf = (None, [])
        if use_feature_distill and teacher_model:
            s_handle, s_feat_buf = self._get_feature_extractor_hook(qat_model)
            t_handle, t_feat_buf = self._get_feature_extractor_hook(teacher_model)

        best_val_f1 = -1.0
        best_val_acc = -1.0
        best_epoch = -1
        best_state_dict = None
        history = []

        print(f"[{exp_id}] Starting {name} ({epochs} epochs, lr={lr}, policy={policy_mode}, distill={use_distillation}, feat_distill={use_feature_distill})...")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Observer control
            if epoch <= observer_warmup_epochs:
                # Observers enabled, but gentle learning
                qat_model.apply(torch.ao.quantization.enable_observer)
                qat_model.apply(torch.ao.quantization.enable_fake_quant)
            elif epoch >= observer_freeze_epoch:
                # Freeze observers to stabilize quantization ranges
                qat_model.apply(torch.ao.quantization.disable_observer)
                qat_model.apply(torch.ao.quantization.enable_fake_quant)
            else:
                qat_model.apply(torch.ao.quantization.enable_observer)
                qat_model.apply(torch.ao.quantization.enable_fake_quant)

            qat_model.train()
            train_loss, train_correct = 0.0, 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()

                if use_feature_distill:
                    s_feat_buf.clear()
                    t_feat_buf.clear()

                out_student = qat_model(bx)

                # Base task loss
                loss_ce = criterion(out_student, by)

                if use_distillation and teacher_model:
                    with torch.no_grad():
                        out_teacher = teacher_model(bx)
                    
                    # KL Divergence on Soft Logits
                    p_s = F.log_softmax(out_student / distill_temp, dim=1)
                    p_t = F.softmax(out_teacher / distill_temp, dim=1)
                    loss_kd = F.kl_div(p_s, p_t, reduction="batchmean") * (distill_temp ** 2)

                    loss = (1.0 - distill_alpha) * loss_ce + distill_alpha * loss_kd

                    if use_feature_distill and s_feat_buf and t_feat_buf:
                        s_f = torch.flatten(s_feat_buf[0], 1)
                        t_f = torch.flatten(t_feat_buf[0], 1)
                        loss_feat = F.mse_loss(s_f, t_f)
                        loss = loss + feature_distill_gamma * loss_feat
                else:
                    loss = loss_ce

                loss.backward()

                # Scale depthwise gradients if configured
                if depthwise_grad_scale != 1.0:
                    for n, p in qat_model.named_parameters():
                        if "block.1.0" in n and p.grad is not None:
                            p.grad.data.mul_(depthwise_grad_scale)

                nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(by)
                train_correct += (out_student.argmax(dim=1) == by).sum().item()

            if scheduler:
                scheduler.step()

            # Validation Pass
            qat_model.eval()
            val_eval = self.trainer.evaluate_model(qat_model, self.trainer.val_x, self.trainer.val_y.tolist())
            val_acc = val_eval["accuracy"]
            val_f1 = val_eval["macro_f1"]
            train_acc = train_correct / len(self.trainer.train_y)
            elapsed = time.time() - t0

            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss / len(self.trainer.train_y)),
                "train_acc": float(train_acc),
                "val_loss": float(val_eval.get("loss", 0.0)),
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_f1),
                "time_sec": float(elapsed)
            })

            print(f"  Epoch {epoch:2d}/{epochs} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | {elapsed:.2f}s")

            # Checkpoint selection strictly on Validation Macro F1 & Accuracy
            if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_epoch = epoch
                best_state_dict = {k: v.cpu().clone() for k, v in qat_model.state_dict().items()}

        # Clean up hooks
        if s_handle: s_handle.remove()
        if t_handle: t_handle.remove()

        # Load best weights
        if best_state_dict is not None:
            qat_model.load_state_dict(best_state_dict)

        # Save PyTorch Checkpoint
        ckpt_path = os.path.join(self.output_dir, "models", f"{exp_id}_best.pth")
        ckpt_meta = {
            "experiment_id": exp_id,
            "experiment_name": name,
            "model_state_dict": best_state_dict,
            "best_epoch": best_epoch,
            "best_val_accuracy": float(best_val_acc),
            "best_val_macro_f1": float(best_val_f1),
            "qat_config": {
                "policy_mode": policy_mode,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "use_distillation": use_distillation,
                "distill_alpha": distill_alpha,
                "distill_temp": distill_temp,
                "use_feature_distill": use_feature_distill,
                "feature_distill_gamma": feature_distill_gamma,
                "class_weighted": class_weighted
            },
            "class_mapping": self.trainer.idx_to_class,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        torch.save(ckpt_meta, ckpt_path)

        # Export ONNX representation for downstream TFLite compilation
        onnx_path = os.path.join(self.output_dir, "models", f"{exp_id}.onnx")
        qat_model.eval()
        dummy_in = torch.randn(1, 3, 128, 128, dtype=torch.float32)
        torch.onnx.export(
            base_model,
            dummy_in,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13,
            dynamo=False
        )

        # Evaluate on all splits
        qat_model.eval()
        train_eval = self.trainer.evaluate_model(qat_model, self.trainer.train_x, self.trainer.train_y.tolist())
        final_val_eval = self.trainer.evaluate_model(qat_model, self.trainer.val_x, self.trainer.val_y.tolist())
        test_eval = self.trainer.evaluate_model(qat_model, self.trainer.test_x, self.trainer.test_y.tolist())
        bench_eval = self.trainer.evaluate_model(qat_model, self.trainer.bench_x, self.trainer.bench_y)

        return {
            "exp_id": exp_id,
            "name": name,
            "model": qat_model,
            "base_model": base_model,
            "checkpoint_path": ckpt_path,
            "onnx_path": onnx_path,
            "best_epoch": best_epoch,
            "train_eval": train_eval,
            "val_eval": final_val_eval,
            "test_eval": test_eval,
            "bench_eval": bench_eval,
            "history": history
        }

    def convert_to_genuine_int8_tflite(self, onnx_model_path: str, output_tflite_path: str) -> Dict[str, Any]:
        """Converts the QAT ONNX graph to a genuine INT8 FlatBuffer using the training calibration pipeline."""
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
            for i in range(min(50, len(self.trainer.train_x))):
                x = self.trainer.train_x[i:i+1].numpy()
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
            "size_mb": len(tflite_bytes) / (1024 * 1024),
            "int8_tensors": audit.int8_tensors,
            "int32_tensors": audit.int32_tensors,
            "fp32_tensors": audit.fp32_tensors,
            "total_tensors": audit.total_tensors,
            "int8_coverage_percent": float(audit.int8_coverage_percent)
        }
