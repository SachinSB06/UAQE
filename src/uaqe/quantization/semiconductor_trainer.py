import os
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional
import onnx

from uaqe.quantization.pytorch_reconstructor import MobileNetV3Reconstructor

class SemiconductorTrainer:
    """Trainer and evaluation pipeline for the 9-class semiconductor defect MobileNetV3 model."""

    def __init__(
        self,
        dataset_root: str = r"D:\semiconductor_dataset\dataset",
        benchmark_path: str = "datasets/hackathon_test_dataset",
        output_dir: str = "output/phase_c1",
        reports_dir: str = "reports/phase_c1",
        random_seed: int = 42
    ) -> None:
        self.dataset_root = dataset_root
        self.benchmark_path = benchmark_path
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.random_seed = random_seed
        self.device = torch.device("cpu") # Host CPU environment
        
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        self.class_names = sorted([
            "bridge", "clean", "cmp", "crack", "opens", "other", "particle", "scratch", "vias"
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {str(i): name for i, name in enumerate(self.class_names)}

        # Save class mapping immediately to guarantee single source of truth
        self.class_mapping_path = os.path.join(self.output_dir, "class_mapping.json")
        with open(self.class_mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.idx_to_class, f, indent=2)

    def load_dataset_split(self, split_name: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        split_dir = os.path.join(self.dataset_root, split_name)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")

        images, labels = [], []
        class_counts = {c: 0 for c in self.class_names}
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

        for c_name in sorted(os.listdir(split_dir)):
            c_dir = os.path.join(split_dir, c_name)
            if not os.path.isdir(c_dir) or c_name not in self.class_to_idx:
                continue
            idx = self.class_to_idx[c_name]
            for f_name in sorted(os.listdir(c_dir)):
                if f_name.lower().endswith(valid_exts):
                    img_path = os.path.join(c_dir, f_name)
                    img = Image.open(img_path).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    arr = arr.transpose(2, 0, 1) # NCHW
                    images.append(arr)
                    labels.append(idx)
                    class_counts[c_name] += 1

        x_tensor = torch.from_numpy(np.stack(images))
        y_tensor = torch.tensor(labels, dtype=torch.long)
        return x_tensor, y_tensor, class_counts

    def load_benchmark(self) -> Tuple[torch.Tensor, List[int], List[str], Dict[str, int]]:
        """Loads the fixed 296-image benchmark dataset for isolated evaluation."""
        if not os.path.exists(self.benchmark_path):
            raise FileNotFoundError(f"Benchmark directory not found: {self.benchmark_path}")

        images, labels, native_classes = [], [], []
        class_counts: Dict[str, int] = {}
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

        # Benchmark class folders map:
        # Bridge->bridge, Clean->clean, CMP->cmp, Crack->crack, LER->scratch, Open->opens, Particle->particle, VIA->vias, Other->other
        bench_to_model_idx = {
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
            folder_path = os.path.join(self.benchmark_path, folder_name)
            if not os.path.isdir(folder_path) or folder_name not in bench_to_model_idx:
                continue
            idx = bench_to_model_idx[folder_name]
            class_counts[folder_name] = 0
            for f_name in sorted(os.listdir(folder_path)):
                if f_name.lower().endswith(valid_exts):
                    img_path = os.path.join(folder_path, f_name)
                    img = Image.open(img_path).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    arr = arr.transpose(2, 0, 1)
                    images.append(arr)
                    labels.append(idx)
                    native_classes.append(folder_name)
                    class_counts[folder_name] += 1

        x_tensor = torch.from_numpy(np.stack(images))
        return x_tensor, labels, native_classes, class_counts

    def train_model(
        self,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 3e-4,
        weight_decay: float = 1e-4
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Trains / fine-tunes the 9-class MobileNetV3 model on the semiconductor training split."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Load splits
        train_x, train_y, train_counts = self.load_dataset_split("train")
        val_x, val_y, val_counts = self.load_dataset_split("val")
        test_x, test_y, test_counts = self.load_dataset_split("test")

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64, shuffle=False)

        # Build reconstructed model with transferred backbone
        reconstructor = MobileNetV3Reconstructor()
        model, recon_metadata = reconstructor.build_9class_trainable_model()
        model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        history = []
        best_val_acc = -1.0
        best_epoch = -1
        best_state_dict = None

        print(f"Starting 9-class model training ({epochs} epochs, lr={lr}, batch_size={batch_size})...")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            train_loss, train_correct = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
                train_correct += (out.argmax(dim=1) == y).sum().item()

            scheduler.step()

            # Validation
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * len(y)
                    val_correct += (out.argmax(dim=1) == y).sum().item()

            train_acc = train_correct / len(train_y)
            val_acc = val_correct / len(val_y)
            elapsed = time.time() - t0

            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss / len(train_y)),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss / len(val_y)),
                "val_accuracy": float(val_acc),
                "lr": float(scheduler.get_last_lr()[0]),
                "time_sec": float(elapsed)
            }
            history.append(epoch_record)
            print(f"Epoch {epoch:2d}/{epochs} | Train Acc: {train_acc*100:.2f}% (Loss: {epoch_record['train_loss']:.4f}) | Val Acc: {val_acc*100:.2f}% (Loss: {epoch_record['val_loss']:.4f}) | {elapsed:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best weights
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        training_summary = {
            "random_seed": self.random_seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "best_epoch": best_epoch,
            "best_val_accuracy": float(best_val_acc),
            "history": history,
            "reconstruction_metadata": recon_metadata,
            "dataset_counts": {
                "train": train_counts,
                "val": val_counts,
                "test": test_counts
            }
        }
        return model, training_summary

    def evaluate_model(
        self,
        model: nn.Module,
        x_tensor: torch.Tensor,
        y_labels: List[int],
        class_labels: List[str]
    ) -> Dict[str, Any]:
        """Calculates comprehensive classification metrics for any evaluation split."""
        model.eval()
        total_samples = len(y_labels)
        all_preds = []
        all_probs = []

        t0 = time.time()
        with torch.no_grad():
            batch_size = 64
            for i in range(0, total_samples, batch_size):
                batch_x = x_tensor[i:i+batch_size].to(self.device)
                out = model(batch_x)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
        eval_time = time.time() - t0

        y_true = np.array(y_labels)
        y_pred = np.array(all_preds)

        num_classes = len(class_labels)
        confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if t < num_classes and p < num_classes:
                confusion_mat[t, p] += 1

        correct = int(np.sum(y_true == y_pred))
        accuracy = float(correct / total_samples) if total_samples > 0 else 0.0

        # Per-class metrics
        per_class: Dict[str, Any] = {}
        precisions, recalls, f1s = [], [], []

        for idx, c_name in enumerate(class_labels):
            support = int(np.sum(y_true == idx))
            c_correct = int(confusion_mat[idx, idx])
            pred_count = int(np.sum(y_pred == idx))
            
            c_acc = float(c_correct / support) if support > 0 else 0.0
            prec = float(c_correct / pred_count) if pred_count > 0 else 0.0
            rec = float(c_correct / support) if support > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            per_class[c_name] = {
                "class_index": idx,
                "support": support,
                "correct": c_correct,
                "accuracy": c_acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            }
            if support > 0:
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)

        macro_prec = float(np.mean(precisions)) if precisions else 0.0
        macro_rec = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        return {
            "total_samples": total_samples,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "latency_ms_per_sample": float((eval_time / total_samples) * 1000.0) if total_samples > 0 else 0.0,
            "per_class_metrics": per_class,
            "confusion_matrix": confusion_mat.tolist()
        }

    def save_checkpoint(
        self,
        model: nn.Module,
        training_summary: Dict[str, Any],
        eval_results: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Saves PyTorch checkpoint (.pth) and metadata JSON."""
        pth_path = os.path.join(self.output_dir, "models", "mobilenetv3_sem_9class_fp32.pth")
        metadata_path = os.path.join(self.output_dir, "models", "mobilenetv3_sem_9class_fp32_metadata.json")

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "class_mapping": self.idx_to_class,
            "architecture": "MobileNetV3-Small",
            "input_size": [1, 3, 128, 128],
            "preprocessing": "rgb_0_1",
            "training_summary": training_summary,
            "eval_results": eval_results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pytorch_version": torch.__version__
        }
        torch.save(checkpoint_data, pth_path)

        metadata_json = {
            "checkpoint_path": pth_path,
            "checkpoint_size_bytes": os.path.getsize(pth_path),
            "architecture": "MobileNetV3-Small",
            "num_classes": 9,
            "class_mapping": self.idx_to_class,
            "input_shape": [1, 3, 128, 128],
            "preprocessing": "rgb_0_1",
            "best_epoch": training_summary["best_epoch"],
            "best_val_accuracy": training_summary["best_val_accuracy"],
            "test_accuracy": eval_results["test"]["accuracy"],
            "benchmark_accuracy": eval_results["benchmark"]["accuracy"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pytorch_version": torch.__version__
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_json, f, indent=2)

        return pth_path, metadata_path

    def export_onnx(self, model: nn.Module) -> str:
        """Exports the 9-class PyTorch model to ONNX."""
        onnx_out_path = os.path.join(self.output_dir, "models", "mobilenetv3_sem_9class_fp32.onnx")
        model.eval()
        dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float32)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_out_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13,
            dynamo=False
        )
        return onnx_out_path
