"""CIFAR-10 Dataset Adapter for Python Pickle Format.

Ingests raw CIFAR-10 batch files non-destructively, creates deterministic
stratified splits (45k Train, 5k Validation, 10k Test), and provides PyTorch
Dataset instances with model-adaptive preprocessing.
"""

import os
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .base_adapter import BaseDatasetAdapter


class CIFAR10TorchDataset(Dataset):
    """PyTorch Dataset wrapper for pre-split CIFAR-10 in-memory arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        interpolation: str = "bilinear",
        layout: str = "NCHW"
    ):
        assert len(images) == len(labels), "Images and labels count mismatch"
        self.images = images
        self.labels = labels
        self.target_size = target_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.interpolation = interpolation
        self.layout = layout

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_np = self.images[idx]
        t = torch.from_numpy(img_np).float() / 255.0

        if (t.shape[1], t.shape[2]) != self.target_size:
            t = t.unsqueeze(0)
            t = F.interpolate(
                t,
                size=self.target_size,
                mode=self.interpolation,
                align_corners=False if self.interpolation == "bilinear" else None
            ).squeeze(0)

        t = (t - self.mean) / self.std

        if self.layout == "NHWC":
            t = t.permute(1, 2, 0)

        return t, int(self.labels[idx])


class CIFAR10PickleAdapter(BaseDatasetAdapter):
    """Adapter for raw CIFAR-10 python-pickled binary batches."""

    DEFAULT_CLASS_NAMES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.class_names = list(self.DEFAULT_CLASS_NAMES)

    @staticmethod
    def _unpickle(file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as f:
            return pickle.load(f, encoding="latin1")

    @staticmethod
    def _compute_sha256(file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load(
        self,
        train_val_split: Tuple[int, int] = (45000, 5000),
        seed: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Load CIFAR-10 batches from disk and partition deterministically."""
        meta_path = os.path.join(self.dataset_path, "batches.meta")
        if os.path.exists(meta_path):
            meta_dict = self._unpickle(meta_path)
            self.class_names = [str(k) for k in meta_dict.get("label_names", self.DEFAULT_CLASS_NAMES)]

        train_x_batches = []
        train_y_batches = []
        self.batch_manifest = []

        for b_idx in range(1, 6):
            batch_file = f"data_batch_{b_idx}"
            batch_path = os.path.join(self.dataset_path, batch_file)
            if not os.path.exists(batch_path):
                raise FileNotFoundError(f"Missing CIFAR-10 batch file: {batch_path}")

            sha256 = self._compute_sha256(batch_path)
            b_data = self._unpickle(batch_path)
            data_arr = b_data["data"]
            labels_arr = np.array(b_data["labels"], dtype=np.int64)

            train_x_batches.append(data_arr)
            train_y_batches.append(labels_arr)

            self.batch_manifest.append({
                "filename": batch_file,
                "path": batch_path,
                "samples": len(labels_arr),
                "sha256": sha256,
                "role": "train_raw"
            })

        test_file = "test_batch"
        test_path = os.path.join(self.dataset_path, test_file)
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing CIFAR-10 test batch file: {test_path}")

        test_sha256 = self._compute_sha256(test_path)
        test_data = self._unpickle(test_path)
        raw_test_x = test_data["data"].reshape(-1, 3, 32, 32)
        raw_test_y = np.array(test_data["labels"], dtype=np.int64)

        self.batch_manifest.append({
            "filename": test_file,
            "path": test_path,
            "samples": len(raw_test_y),
            "sha256": test_sha256,
            "role": "test"
        })

        raw_train_x = np.concatenate(train_x_batches, axis=0).reshape(-1, 3, 32, 32)
        raw_train_y = np.concatenate(train_y_batches, axis=0)

        assert len(raw_train_x) == 50000, f"Expected 50000 training samples, got {len(raw_train_x)}"
        assert len(raw_test_x) == 10000, f"Expected 10000 test samples, got {len(raw_test_x)}"

        target_train_count, target_val_count = train_val_split
        assert target_train_count + target_val_count == 50000, "Train + Val count must equal 50,000"

        val_per_class = target_val_count // len(self.class_names)
        rng = np.random.RandomState(seed)

        train_indices: List[int] = []
        val_indices: List[int] = []

        for c in range(len(self.class_names)):
            c_indices = np.where(raw_train_y == c)[0]
            rng.shuffle(c_indices)
            val_indices.extend(c_indices[:val_per_class].tolist())
            train_indices.extend(c_indices[val_per_class:].tolist())

        train_indices_arr = np.array(train_indices, dtype=np.int64)
        val_indices_arr = np.array(val_indices, dtype=np.int64)

        train_indices_arr.sort()
        val_indices_arr.sort()

        train_x = raw_train_x[train_indices_arr]
        train_y = raw_train_y[train_indices_arr]
        val_x = raw_train_x[val_indices_arr]
        val_y = raw_train_y[val_indices_arr]

        self.splits = {
            "train": {"images": train_x, "labels": train_y},
            "val": {"images": val_x, "labels": val_y},
            "test": {"images": raw_test_x, "labels": raw_test_y}
        }

        return self.splits

    def get_torch_dataset(
        self,
        split: str,
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        interpolation: str = "bilinear",
        layout: str = "NCHW"
    ) -> CIFAR10TorchDataset:
        if split not in self.splits:
            raise KeyError(f"Split '{split}' not found. Available splits: {list(self.splits.keys())}")

        data = self.splits[split]
        return CIFAR10TorchDataset(
            images=data["images"],
            labels=data["labels"],
            target_size=target_size,
            mean=mean,
            std=std,
            interpolation=interpolation,
            layout=layout
        )

    def get_metadata(self) -> Dict[str, Any]:
        class_distribution = {}
        for split_name, data in self.splits.items():
            labels = data["labels"]
            counts = {self.class_names[c]: int((labels == c).sum()) for c in range(len(self.class_names))}
            class_distribution[split_name] = counts

        return {
            "adapter_type": "CIFAR10PickleAdapter",
            "format": "cifar10_pickle",
            "dataset_root": self.dataset_path,
            "class_count": len(self.class_names),
            "class_names": self.class_names,
            "splits": {
                "train_count": len(self.splits["train"]["images"]) if "train" in self.splits else 0,
                "val_count": len(self.splits["val"]["images"]) if "val" in self.splits else 0,
                "test_count": len(self.splits["test"]["images"]) if "test" in self.splits else 0,
                "total_samples": sum(len(d["images"]) for d in self.splits.values()) if self.splits else 0
            },
            "class_distribution": class_distribution,
            "batch_manifest": self.batch_manifest
        }
