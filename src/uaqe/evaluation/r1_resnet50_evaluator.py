"""UAQE Phase R1: ResNet-50 FP32 vs INT8 Evaluator & Benchmarker.

Provides independent evaluation, prediction recording, latency measurement,
prediction agreement, and numerical fidelity analysis for ResNet-50 on CIFAR-10.
"""

import os
import csv
import json
import time
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from uaqe.dataset.universal_dataset_loader import UniversalDatasetLoader, CIFAR10Dataset


class R1ResNet50Evaluator:
    """Evaluates and benchmarks FP32 and INT8 ResNet-50 models on the frozen CIFAR-10 test set."""

    def __init__(
        self,
        loader: UniversalDatasetLoader,
        max_test_samples: int = 1000,
        batch_size: int = 32
    ):
        self.loader = loader
        self.max_test_samples = max_test_samples
        self.batch_size = batch_size
        self.class_names = loader.class_names
        self.num_classes = len(self.class_names)

        # Build deterministic stratified test subset
        test_data = loader.splits["test"]
        raw_images = test_data["images"]
        raw_labels = test_data["labels"]

        if max_test_samples < len(raw_images):
            samples_per_class = max_test_samples // self.num_classes
            remainder = max_test_samples % self.num_classes
            selected_indices: List[int] = []
            for c in range(self.num_classes):
                c_idx = np.where(raw_labels == c)[0]
                count = samples_per_class + (1 if c < remainder else 0)
                selected_indices.extend(c_idx[:count].tolist())
            selected_indices = sorted(selected_indices)
            self.test_indices = selected_indices
            self.test_images = raw_images[selected_indices]
            self.test_labels = raw_labels[selected_indices]
        else:
            self.test_indices = list(range(len(raw_images)))
            self.test_images = raw_images
            self.test_labels = raw_labels

        self.dataset = CIFAR10Dataset(images=self.test_images, labels=self.test_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def evaluate_pytorch_fp32(
        self,
        model: torch.nn.Module,
        device: str = "cpu"
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
        """Evaluate PyTorch FP32 model independently on the test set."""
        model.eval()
        model.to(device)

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_confs: List[float] = []
        all_logits: List[np.ndarray] = []
        latencies_ms: List[float] = []

        # Warm-up pass
        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy)

        t0_total = time.time()
        with torch.no_grad():
            for bx, by in self.dataloader:
                bx = bx.to(device)
                t_start = time.perf_counter()
                logits = model(bx)
                t_end = time.perf_counter()

                batch_lat = (t_end - t_start) * 1000.0 / len(bx)
                latencies_ms.extend([batch_lat] * len(bx))

                probs = F.softmax(logits, dim=1)
                confs, preds = torch.max(probs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(by.numpy().tolist())
                all_confs.extend(confs.cpu().numpy().tolist())
                all_logits.append(logits.cpu().numpy())

        total_time = time.time() - t0_total
        all_preds_np = np.array(all_preds, dtype=np.int64)
        all_labels_np = np.array(all_labels, dtype=np.int64)
        concat_logits = np.concatenate(all_logits, axis=0)

        metrics = self._compute_metrics(
            all_labels_np,
            all_preds_np,
            latencies_ms,
            total_time,
            model_name="PyTorch ResNet-50 FP32"
        )

        prediction_rows: List[Dict[str, Any]] = []
        for idx in range(len(all_labels_np)):
            t_label = int(all_labels_np[idx])
            p_label = int(all_preds_np[idx])
            prediction_rows.append({
                "sample_index": idx,
                "global_test_index": self.test_indices[idx],
                "true_class_id": t_label,
                "true_class_name": self.class_names[t_label],
                "predicted_class_id": p_label,
                "predicted_class_name": self.class_names[p_label],
                "confidence": round(float(all_confs[idx]), 6),
                "is_correct": bool(t_label == p_label)
            })

        return metrics, prediction_rows, concat_logits

    def evaluate_onnx_int8(
        self,
        onnx_model_path: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
        """Evaluate exported ONNX INT8 model independently on the test set."""
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX INT8 model not found: {onnx_model_path}")

        sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        # Warm-up pass
        dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        for _ in range(3):
            _ = sess.run(None, {input_name: dummy_np})

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_confs: List[float] = []
        all_logits: List[np.ndarray] = []
        latencies_ms: List[float] = []

        t0_total = time.time()
        for bx, by in self.dataloader:
            bx_np = bx.numpy()
            t_start = time.perf_counter()
            logits_np = sess.run(None, {input_name: bx_np})[0]
            t_end = time.perf_counter()

            batch_lat = (t_end - t_start) * 1000.0 / len(bx_np)
            latencies_ms.extend([batch_lat] * len(bx_np))

            # Compute softmax probabilities
            exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(by.numpy().tolist())
            all_confs.extend(confs.tolist())
            all_logits.append(logits_np)

        total_time = time.time() - t0_total
        all_preds_np = np.array(all_preds, dtype=np.int64)
        all_labels_np = np.array(all_labels, dtype=np.int64)
        concat_logits = np.concatenate(all_logits, axis=0)

        metrics = self._compute_metrics(
            all_labels_np,
            all_preds_np,
            latencies_ms,
            total_time,
            model_name="ONNX Runtime ResNet-50 INT8 QDQ"
        )

        prediction_rows: List[Dict[str, Any]] = []
        for idx in range(len(all_labels_np)):
            t_label = int(all_labels_np[idx])
            p_label = int(all_preds_np[idx])
            prediction_rows.append({
                "sample_index": idx,
                "global_test_index": self.test_indices[idx],
                "true_class_id": t_label,
                "true_class_name": self.class_names[t_label],
                "predicted_class_id": p_label,
                "predicted_class_name": self.class_names[p_label],
                "confidence": round(float(all_confs[idx]), 6),
                "is_correct": bool(t_label == p_label)
            })

        return metrics, prediction_rows, concat_logits

    def _compute_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        latencies_ms: List[float],
        total_time: float,
        model_name: str
    ) -> Dict[str, Any]:
        """Compute standard classification and latency metrics."""
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(labels, preds).tolist()

        per_class_metrics = {}
        for c_idx, c_name in enumerate(self.class_names):
            mask = (labels == c_idx)
            class_total = int(mask.sum())
            class_correct = int((preds[mask] == c_idx).sum())
            class_acc = class_correct / class_total if class_total > 0 else 0.0
            per_class_metrics[c_name] = {
                "class_id": c_idx,
                "total_samples": class_total,
                "correct_samples": class_correct,
                "accuracy": round(float(class_acc), 4)
            }

        return {
            "model_name": model_name,
            "total_test_samples": len(labels),
            "correct_predictions": int((preds == labels).sum()),
            "top1_accuracy": round(float(acc), 6),
            "macro_precision": round(float(precision), 6),
            "macro_recall": round(float(recall), 6),
            "macro_f1": round(float(f1), 6),
            "weighted_precision": round(float(w_precision), 6),
            "weighted_recall": round(float(w_recall), 6),
            "weighted_f1": round(float(w_f1), 6),
            "latency": {
                "mean_ms": round(float(np.mean(latencies_ms)), 3),
                "median_ms": round(float(np.median(latencies_ms)), 3),
                "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
                "throughput_images_per_sec": round(len(labels) / max(total_time, 1e-4), 2)
            },
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm
        }

    @staticmethod
    def compute_prediction_agreement(
        fp32_rows: List[Dict[str, Any]],
        int8_rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute sample-by-sample prediction agreement between FP32 and INT8."""
        assert len(fp32_rows) == len(int8_rows), "Prediction row counts mismatch"

        agreed_count = 0
        disagreed_count = 0
        comparison_records: List[Dict[str, Any]] = []

        for f_row, i_row in zip(fp32_rows, int8_rows):
            assert f_row["sample_index"] == i_row["sample_index"], "Sample index mismatch"
            f_pred = f_row["predicted_class_id"]
            i_pred = i_row["predicted_class_id"]
            is_agree = bool(f_pred == i_pred)

            if is_agree:
                agreed_count += 1
            else:
                disagreed_count += 1

            comparison_records.append({
                "sample_index": f_row["sample_index"],
                "global_test_index": f_row["global_test_index"],
                "true_class_id": f_row["true_class_id"],
                "true_class_name": f_row["true_class_name"],
                "fp32_predicted_class_id": f_pred,
                "fp32_predicted_class_name": f_row["predicted_class_name"],
                "int8_predicted_class_id": i_pred,
                "int8_predicted_class_name": i_row["predicted_class_name"],
                "prediction_agreed": is_agree,
                "fp32_confidence": f_row["confidence"],
                "int8_confidence": i_row["confidence"]
            })

        total = len(fp32_rows)
        agreement_pct = round((agreed_count / max(total, 1)) * 100.0, 2)

        return {
            "total_test_samples": total,
            "agreed_predictions_count": agreed_count,
            "disagreed_predictions_count": disagreed_count,
            "agreement_percentage": agreement_pct,
            "sample_comparisons": comparison_records
        }

    @staticmethod
    def compute_numerical_fidelity(
        fp32_logits: np.ndarray,
        int8_logits: np.ndarray
    ) -> Dict[str, Any]:
        """Compute logit-level numerical fidelity metrics (cosine similarity, MAE, RMSE)."""
        assert fp32_logits.shape == int8_logits.shape, f"Shape mismatch: {fp32_logits.shape} vs {int8_logits.shape}"

        cos_sims: List[float] = []
        maes: List[float] = []
        rmses: List[float] = []
        max_errors: List[float] = []

        for i in range(len(fp32_logits)):
            f = fp32_logits[i].flatten()
            q = int8_logits[i].flatten()

            cos = float(np.dot(f, q) / (np.linalg.norm(f) * np.linalg.norm(q) + 1e-12))
            mae = float(np.mean(np.abs(f - q)))
            rmse = float(np.sqrt(np.mean((f - q) ** 2)))
            max_err = float(np.max(np.abs(f - q)))

            cos_sims.append(cos)
            maes.append(mae)
            rmses.append(rmse)
            max_errors.append(max_err)

        return {
            "total_samples_compared": len(fp32_logits),
            "mean_cosine_similarity": round(float(np.mean(cos_sims)), 6),
            "median_cosine_similarity": round(float(np.median(cos_sims)), 6),
            "min_cosine_similarity": round(float(np.min(cos_sims)), 6),
            "mean_mae": round(float(np.mean(maes)), 6),
            "median_mae": round(float(np.median(maes)), 6),
            "mean_rmse": round(float(np.mean(rmses)), 6),
            "max_absolute_error": round(float(np.max(max_errors)), 6),
            "interpretation": (
                "Numerical fidelity measures raw logit alignment between FP32 and INT8. "
                "High cosine similarity indicates consistent output distribution structure."
            )
        }

    @staticmethod
    def compute_size_report(
        fp32_onnx_path: str,
        int8_onnx_path: str,
        fp32_pt_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute physical model file size metrics and reduction percentages."""
        fp32_onnx_bytes = os.path.getsize(fp32_onnx_path)
        int8_onnx_bytes = os.path.getsize(int8_onnx_path)
        reduction_bytes = fp32_onnx_bytes - int8_onnx_bytes
        reduction_pct = round(((fp32_onnx_bytes - int8_onnx_bytes) / fp32_onnx_bytes) * 100.0, 2)
        compression_ratio = round(fp32_onnx_bytes / max(int8_onnx_bytes, 1), 2)

        report = {
            "fp32_onnx_path": fp32_onnx_path,
            "int8_onnx_path": int8_onnx_path,
            "fp32_onnx_size_bytes": fp32_onnx_bytes,
            "int8_onnx_size_bytes": int8_onnx_bytes,
            "size_reduction_bytes": reduction_bytes,
            "size_reduction_percentage": reduction_pct,
            "compression_ratio": f"{compression_ratio}x"
        }

        if fp32_pt_path and os.path.exists(fp32_pt_path):
            report["fp32_pt_size_bytes"] = os.path.getsize(fp32_pt_path)

        return report
