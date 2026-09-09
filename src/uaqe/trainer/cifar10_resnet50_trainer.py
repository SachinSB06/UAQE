"""CIFAR-10 ResNet-50 FP32 Baseline Trainer and Evaluator.

Provides clean training/fine-tuning routines for the adapted 10-class ResNet-50 model,
rigorous metric evaluation, prediction recording, and baseline artifact generation.
"""

import os
import csv
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from uaqe.dataset.universal_dataset_loader import UniversalDatasetLoader, CIFAR10Dataset
from uaqe.models.resnet50 import ResNetForImageClassification, build_resnet50_cifar10


class CIFAR10ResNet50Trainer:
    """Trains and establishes the FP32 reference baseline for ResNet-50 on CIFAR-10."""

    def __init__(
        self,
        dataset_loader: UniversalDatasetLoader,
        model: ResNetForImageClassification,
        model_metadata: Dict[str, Any],
        device: str = "cpu"
    ):
        """Initialize trainer.

        Args:
            dataset_loader: Ingested UniversalDatasetLoader instance.
            model: Adapted ResNetForImageClassification model.
            model_metadata: Model architecture and parameter metadata.
            device: 'cpu' or 'cuda'.
        """
        self.loader = dataset_loader
        self.model = model
        self.model_metadata = model_metadata
        self.device = torch.device(device)
        self.model.to(self.device)
        self.training_history: List[Dict[str, Any]] = []

    def extract_features(
        self,
        split: str,
        max_samples: Optional[int] = None,
        batch_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract bottleneck features from the frozen ResNet-50 backbone for fast convergence.

        Args:
            split: 'train', 'val', or 'test'.
            max_samples: Optional limit for fast stratified training.
            batch_size: Batch size for feature extraction.

        Returns:
            Tuple of (features_tensor [N, 2048], labels_tensor [N]).
        """
        self.model.eval()
        data = self.loader.splits[split]
        raw_images = data["images"]
        raw_labels = data["labels"]

        if max_samples is not None and max_samples < len(raw_images):
            # Deterministic stratified subsample
            num_classes = len(self.loader.class_names)
            samples_per_class = max_samples // num_classes
            selected_indices: List[int] = []
            for c in range(num_classes):
                c_idx = np.where(raw_labels == c)[0]
                selected_indices.extend(c_idx[:samples_per_class].tolist())
            selected_indices = sorted(selected_indices)
            images_subset = raw_images[selected_indices]
            labels_subset = raw_labels[selected_indices]
        else:
            images_subset = raw_images
            labels_subset = raw_labels

        dataset = CIFAR10Dataset(images=images_subset, labels=labels_subset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        feats_list = []
        labels_list = []

        with torch.no_grad():
            for bx, by in dataloader:
                bx = bx.to(self.device)
                f = self.model.resnet(bx)  # (B, 2048, 1, 1)
                f = torch.flatten(f, 1)    # (B, 2048)
                feats_list.append(f.cpu())
                labels_list.append(by)

        all_feats = torch.cat(feats_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        return all_feats, all_labels

    def train_classifier(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        train_samples: int = 5000,
        val_samples: int = 1000
    ) -> Dict[str, Any]:
        """Train the classifier head using extracted backbone features.

        Args:
            epochs: Number of training epochs.
            lr: Learning rate for AdamW.
            weight_decay: L2 regularization factor.
            batch_size: Mini-batch size.
            train_samples: Number of stratified training samples.
            val_samples: Number of stratified validation samples.

        Returns:
            Dictionary with training metadata and convergence history.
        """
        print(f"Extracting features for {train_samples} train and {val_samples} validation samples...", flush=True)
        t0_extract = time.time()
        train_feats, train_labels = self.extract_features("train", max_samples=train_samples, batch_size=batch_size)
        val_feats, val_labels = self.extract_features("val", max_samples=val_samples, batch_size=batch_size)
        t_extract = time.time() - t0_extract
        print(f"Feature extraction complete in {t_extract:.2f} s. Train shape: {train_feats.shape}", flush=True)

        train_dataset = TensorDataset(train_feats, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        self.training_history = []
        t0_train = time.time()

        for epoch in range(1, epochs + 1):
            self.model.classifier.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model.classifier(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(by)
                preds = out.argmax(dim=1)
                correct_train += (preds == by).sum().item()
                total_train += len(by)

            scheduler.step()
            epoch_loss = running_loss / total_train
            train_acc = correct_train / total_train

            # Validation evaluation
            self.model.classifier.eval()
            with torch.no_grad():
                val_out = self.model.classifier(val_feats.to(self.device))
                val_preds = val_out.argmax(dim=1).cpu()
                val_acc = (val_preds == val_labels).float().mean().item()

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(float(epoch_loss), 6),
                "train_accuracy": round(float(train_acc), 4),
                "val_accuracy": round(float(val_acc), 4),
                "lr": float(scheduler.get_last_lr()[0])
            }
            self.training_history.append(epoch_record)
            print(f"Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%", flush=True)

        t_train = time.time() - t0_train
        return {
            "epochs": epochs,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "feature_extraction_seconds": round(t_extract, 2),
            "training_seconds": round(t_train, 2),
            "history": self.training_history
        }

    def evaluate_test_set(
        self,
        batch_size: int = 64,
        max_test_samples: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Perform full end-to-end evaluation on the CIFAR-10 test set.

        Args:
            batch_size: Batch size for inference.
            max_test_samples: Optional limit (e.g. 10,000 for full test set).

        Returns:
            Tuple of (metrics_dict, prediction_rows_list).
        """
        self.model.eval()
        test_data = self.loader.splits["test"]
        raw_images = test_data["images"]
        raw_labels = test_data["labels"]

        if max_test_samples is not None and max_test_samples < len(raw_images):
            num_classes = len(self.loader.class_names)
            samples_per_class = max_test_samples // num_classes
            selected_indices: List[int] = []
            for c in range(num_classes):
                c_idx = np.where(raw_labels == c)[0]
                selected_indices.extend(c_idx[:samples_per_class].tolist())
            selected_indices = sorted(selected_indices)
            images_subset = raw_images[selected_indices]
            labels_subset = raw_labels[selected_indices]
        else:
            images_subset = raw_images
            labels_subset = raw_labels

        dataset = CIFAR10Dataset(images=images_subset, labels=labels_subset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_confs: List[float] = []
        latencies_ms: List[float] = []

        t0_eval = time.time()

        with torch.no_grad():
            for bx, by in dataloader:
                bx = bx.to(self.device)
                t_start = time.perf_counter()
                logits = self.model(bx)
                t_end = time.perf_counter()

                probs = F.softmax(logits, dim=1)
                confs, preds = torch.max(probs, dim=1)

                batch_lat = (t_end - t_start) * 1000.0 / len(bx)
                latencies_ms.extend([batch_lat] * len(bx))

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(by.numpy().tolist())
                all_confs.extend(confs.cpu().numpy().tolist())

        total_eval_time = time.time() - t0_eval

        all_preds_np = np.array(all_preds, dtype=np.int64)
        all_labels_np = np.array(all_labels, dtype=np.int64)

        # Metrics
        acc = accuracy_score(all_labels_np, all_preds_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average="macro", zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average="weighted", zero_division=0
        )

        cm = confusion_matrix(all_labels_np, all_preds_np).tolist()

        # Per-class metrics
        per_class_metrics = {}
        for c_idx, c_name in enumerate(self.loader.class_names):
            mask = (all_labels_np == c_idx)
            class_total = int(mask.sum())
            class_correct = int((all_preds_np[mask] == c_idx).sum())
            class_acc = class_correct / class_total if class_total > 0 else 0.0
            per_class_metrics[c_name] = {
                "class_id": c_idx,
                "total_samples": class_total,
                "correct_samples": class_correct,
                "accuracy": round(float(class_acc), 4)
            }

        metrics = {
            "total_test_samples": len(all_labels_np),
            "correct_predictions": int((all_preds_np == all_labels_np).sum()),
            "top1_accuracy": round(float(acc), 6),
            "macro_precision": round(float(precision), 6),
            "macro_recall": round(float(recall), 6),
            "macro_f1": round(float(f1), 6),
            "weighted_f1": round(float(weighted_f1), 6),
            "average_latency_ms_per_image": round(float(np.mean(latencies_ms)), 3),
            "p95_latency_ms_per_image": round(float(np.percentile(latencies_ms, 95)), 3),
            "throughput_images_per_second": round(len(all_labels_np) / total_eval_time, 2),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm
        }

        # Build predictions table
        prediction_rows: List[Dict[str, Any]] = []
        for idx in range(len(all_labels_np)):
            t_label = int(all_labels_np[idx])
            p_label = int(all_preds_np[idx])
            prediction_rows.append({
                "sample_index": idx,
                "true_class_id": t_label,
                "true_class_name": self.loader.class_names[t_label],
                "predicted_class_id": p_label,
                "predicted_class_name": self.loader.class_names[p_label],
                "confidence": round(float(all_confs[idx]), 6),
                "is_correct": bool(t_label == p_label)
            })

        return metrics, prediction_rows

    def save_baseline(
        self,
        output_dir: str,
        training_info: Dict[str, Any],
        test_metrics: Dict[str, Any],
        prediction_rows: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Export all baseline artifacts to output directory.

        Returns:
            Dictionary with paths to generated artifacts.
        """
        os.makedirs(output_dir, exist_ok=True)
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # 1. Save PyTorch baseline model checkpoint
        checkpoint_path = os.path.join(models_dir, "resnet50_cifar10_fp32_baseline.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "metadata": self.model_metadata,
            "metrics": test_metrics,
            "training_info": training_info
        }, checkpoint_path)

        # Compute checkpoint SHA-256
        hasher = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        checkpoint_hash = hasher.hexdigest()

        # 2. Save predictions CSV
        csv_path = os.path.join(output_dir, "resnet50_cifar10_predictions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sample_index",
                "true_class_id",
                "true_class_name",
                "predicted_class_id",
                "predicted_class_name",
                "confidence",
                "is_correct"
            ])
            writer.writeheader()
            writer.writerows(prediction_rows)

        # 3. Save baseline JSON report
        json_path = os.path.join(output_dir, "resnet50_cifar10_fp32_baseline.json")
        baseline_report = {
            "phase": "E.2",
            "model_name": "ResNetForImageClassification",
            "architecture": "ResNetForImageClassification (ResNet-50 v1.5)",
            "task": "CIFAR-10 10-Class Classification",
            "precision": "FP32",
            "baseline_type": "empirically_measured_reference",
            "pretrained_safetensors": self.model_metadata.get("safetensors_source", ""),
            "pretrained_safetensors_sha256": "9c6061af1f450bb0847e529fd742aa5066017be379c71bbf5546b198e5b13a1e",
            "checkpoint_config": self.model_metadata.get("checkpoint_config", {}),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_hash,
            "model_parameters": {
                "total_parameters": self.model_metadata.get("total_parameters", 0),
                "trainable_parameters": self.model_metadata.get("trainable_parameters", 0),
                "backbone_parameters": self.model_metadata.get("backbone_parameters", 0),
                "classifier_parameters": self.model_metadata.get("classifier_parameters", 0)
            },
            "dataset_split": {
                "train_samples": len(self.loader.splits["train"]["images"]),
                "val_samples": len(self.loader.splits["val"]["images"]),
                "test_samples": len(self.loader.splits["test"]["images"]),
                "evaluated_test_samples": test_metrics["total_test_samples"]
            },
            "preprocessing": {
                "input_resolution": [3, 224, 224],
                "color_format": "RGB",
                "interpolation": "bilinear",
                "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                "layout": "NCHW"
            },
            "training_hyperparameters": training_info,
            "baseline_metrics": test_metrics,
            "quantization_applied": False,
            "pruning_applied": False,
            "clustering_applied": False,
            "compression_applied": False
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(baseline_report, f, indent=2)

        return {
            "baseline_json": json_path,
            "predictions_csv": csv_path,
            "checkpoint_pt": checkpoint_path
        }
