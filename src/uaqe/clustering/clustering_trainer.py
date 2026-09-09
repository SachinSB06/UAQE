"""
Clustering-Aware Trainer for MobileNetV3-Small.
Implements fine-tuning with Straight-Through Estimators (STE), Lloyd-Max centroid updates,
clustering regularization, and optional Knowledge Distillation from FP32 teacher.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.uaqe.clustering.clustering_module import (
    ClusteringConv2d,
    ClusteringLinear,
    ClusteringWeightModule
)


class ClusteringTrainer:
    """Orchestrates clustering-aware fine-tuning with early stopping on validation split."""

    def __init__(
        self,
        teacher_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        self.teacher_model = teacher_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

    def evaluate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 32
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """Evaluates model, returning (accuracy, precision_macro, recall_macro, f1_macro, preds, logits)."""
        model.eval()
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        all_logits = []
        all_preds = []

        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(self.device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1)
                all_logits.append(logits.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        preds_np = np.concatenate(all_preds, axis=0)
        y_np = y.numpy()

        acc = float(accuracy_score(y_np, preds_np))
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_np, preds_np, average="macro", zero_division=0)
        return acc, float(p_m), float(r_m), float(f1_m), preds_np, logits_np

    def train(
        self,
        model: nn.Module,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        epochs: int = 6,
        lr: float = 5e-5,
        batch_size: int = 16,
        weight_decay: float = 1e-4,
        cluster_loss_weight: float = 0.05,
        use_distillation: bool = False,
        distill_temperature: float = 4.0,
        distill_alpha: float = 0.5,
        seed: int = 42,
        patience: int = 3
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes clustering-aware fine-tuning with validation early stopping.
        
        Returns:
            Tuple of (best_model, training_summary_dict).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        model.to(self.device)
        train_ds = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        ce_loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # Baseline validation evaluation
        val_acc, val_p, val_r, val_f1, _, _ = self.evaluate(model, val_x, val_y, batch_size=batch_size)
        best_val_f1 = val_f1
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        history = [{
            "epoch": 0,
            "train_loss": 0.0,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": lr
        }]

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            # Freeze BatchNorm running statistics for fused Conv-BN stability
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            epoch_loss = 0.0
            num_batches = 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()

                student_logits = model(bx)
                ce_loss = ce_loss_fn(student_logits, by)

                if use_distillation and self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(bx)
                    t_s = F.log_softmax(student_logits / distill_temperature, dim=1)
                    t_t = F.softmax(teacher_logits / distill_temperature, dim=1)
                    kd_loss = F.kl_div(t_s, t_t, reduction="batchmean") * (distill_temperature ** 2)
                    task_loss = distill_alpha * ce_loss + (1.0 - distill_alpha) * kd_loss
                else:
                    task_loss = ce_loss

                # Clustering regularization loss
                c_loss = torch.tensor(0.0, device=self.device)
                for m in model.modules():
                    if isinstance(m, (ClusteringConv2d, ClusteringLinear)):
                        c_loss = c_loss + m.get_clustering_loss()

                total_loss = task_loss + (cluster_loss_weight * c_loss)
                total_loss.backward()

                # Zero gradient updates on pruned weight locations
                for m in model.modules():
                    if isinstance(m, (ClusteringConv2d, ClusteringLinear)):
                        if m.weight.grad is not None:
                            m.weight.grad.mul_(m.sparsity_mask)

                optimizer.step()

                # Update centroids (Lloyd-Max step)
                for m in model.modules():
                    if isinstance(m, (ClusteringConv2d, ClusteringLinear)):
                        m.update_centroids_from_weights()

                epoch_loss += float(total_loss.item())
                num_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(1, num_batches)

            # Validation evaluation
            val_acc, val_p, val_r, val_f1, _, _ = self.evaluate(model, val_x, val_y, batch_size=batch_size)
            current_lr = scheduler.get_last_lr()[0]

            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": current_lr
            })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        total_training_time = time.time() - start_time
        model.load_state_dict(best_state)

        summary = {
            "epochs_completed": len(history) - 1,
            "best_val_accuracy": best_val_acc,
            "best_val_f1": best_val_f1,
            "training_time_seconds": total_training_time,
            "history": history,
            "hyperparameters": {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "cluster_loss_weight": cluster_loss_weight,
                "use_distillation": use_distillation,
                "distill_temperature": distill_temperature,
                "distill_alpha": distill_alpha,
                "seed": seed,
                "patience": patience
            }
        }

        return model, summary
