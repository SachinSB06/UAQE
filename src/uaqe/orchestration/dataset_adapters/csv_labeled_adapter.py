"""CSV Labeled Dataset Adapter for UAQE.

Supports flat image directories paired with a CSV annotations file (image_path/filename, label).
"""

import os
import csv
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from uaqe.dataset.base_adapter import BaseDatasetAdapter
from .image_folder_adapter import ImageFolderTorchDataset


class CSVLabeledImageAdapter(BaseDatasetAdapter):
    """Adapter for flat directory with CSV labels."""

    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.csv_path: Optional[str] = None
        self.split_samples: Dict[str, List[Tuple[str, int]]] = {}
        self._find_csv()

    def _find_csv(self) -> None:
        entries = os.listdir(self.dataset_path)
        for fname in entries:
            if fname.lower().endswith('.csv'):
                self.csv_path = os.path.join(self.dataset_path, fname)
                break

    def load(
        self,
        train_val_split: Tuple[int, int] = (45000, 5000),
        seed: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        if not self.csv_path or not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"No CSV labels file found in dataset directory: {self.dataset_path}")

        raw_rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    raw_rows.append((row[0].strip(), row[1].strip()))

        # Discover unique class names
        raw_classes = sorted(list(set(r[1] for r in raw_rows)))
        self.class_names = raw_classes
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        all_samples = []
        for img_name, label_name in raw_rows:
            img_path = os.path.join(self.dataset_path, img_name) if not os.path.isabs(img_name) else img_name
            if os.path.exists(img_path):
                all_samples.append((img_path, class_to_idx[label_name]))

        # Deterministic 80/10/10 split
        rng = np.random.RandomState(seed)
        train_samples, val_samples, test_samples = [], [], []
        for c_idx in range(len(self.class_names)):
            c_items = [s for s in all_samples if s[1] == c_idx]
            rng.shuffle(c_items)
            n_total = len(c_items)
            n_val = max(1, int(n_total * 0.1))
            n_test = max(1, int(n_total * 0.1))
            test_samples.extend(c_items[:n_test])
            val_samples.extend(c_items[n_test:n_test + n_val])
            train_samples.extend(c_items[n_test + n_val:])

        self.split_samples = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples
        }
        for sname, slist in self.split_samples.items():
            self.splits[sname] = {
                "images": np.array([s[0] for s in slist]),
                "labels": np.array([s[1] for s in slist], dtype=np.int64)
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
    ) -> ImageFolderTorchDataset:
        if split not in self.split_samples:
            raise KeyError(f"Split '{split}' not found. Available splits: {list(self.split_samples.keys())}")

        return ImageFolderTorchDataset(
            samples=self.split_samples[split],
            target_size=target_size,
            mean=mean,
            std=std,
            interpolation=interpolation,
            layout=layout
        )

    def get_metadata(self) -> Dict[str, Any]:
        class_dist = {}
        for sname, slist in self.split_samples.items():
            labels = [s[1] for s in slist]
            class_dist[sname] = {
                cname: int(labels.count(cidx)) for cidx, cname in enumerate(self.class_names)
            }

        return {
            "adapter_type": "CSVLabeledImageAdapter",
            "format": "csv_labeled_images",
            "dataset_root": self.dataset_path,
            "csv_file": self.csv_path,
            "class_count": len(self.class_names),
            "class_names": self.class_names,
            "splits": {
                "train_count": len(self.split_samples.get("train", [])),
                "val_count": len(self.split_samples.get("val", [])),
                "test_count": len(self.split_samples.get("test", [])),
                "total_samples": sum(len(v) for v in self.split_samples.values())
            },
            "class_distribution": class_dist
        }
