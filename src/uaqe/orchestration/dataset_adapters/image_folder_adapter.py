"""Image Folder Dataset Adapter for UAQE.

Supports standard image folder hierarchies (e.g. train/val/test subdirectories containing class folders,
or direct class folders), non-destructively scanning images and providing PyTorch Datasets.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from uaqe.dataset.base_adapter import BaseDatasetAdapter


class ImageFolderTorchDataset(Dataset):
    """PyTorch Dataset wrapper for ImageFolder samples."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        interpolation: str = "bilinear",
        layout: str = "NCHW"
    ):
        self.samples = samples
        self.target_size = target_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.interpolation = interpolation
        self.layout = layout

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        with Image.open(file_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Convert to numpy uint8 (H, W, C) -> (C, H, W)
            arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)

        t = torch.from_numpy(arr).float() / 255.0

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

        return t, label


class ImageFolderAdapter(BaseDatasetAdapter):
    """Adapter for directory-based image dataset hierarchies."""

    VALID_EXTENSIONS = (
        '.png', '.jpg', '.jpeg', '.bmp', '.webp',
        '.PNG', '.JPG', '.JPEG', '.BMP', '.WEBP'
    )

    def __init__(self, dataset_path: str):
        from uaqe.orchestration.universal_dataset_ingestor import UniversalDatasetIngestor
        resolved = UniversalDatasetIngestor.resolve_dataset_root(dataset_path)
        super().__init__(resolved)
        self.split_samples: Dict[str, List[Tuple[str, int]]] = {}
        self.has_predefined_splits = False

    def load(
        self,
        train_val_split: Tuple[int, int] = (45000, 5000),
        seed: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Scan directory and resolve train/val/test splits deterministically."""
        entries = sorted(os.listdir(self.dataset_path))
        has_train_subdir = "train" in entries and os.path.isdir(os.path.join(self.dataset_path, "train"))
        has_test_subdir = "test" in entries and os.path.isdir(os.path.join(self.dataset_path, "test"))

        if has_train_subdir:
            self.has_predefined_splits = True
            # Discover class names from train split (ignoring hidden entries)
            train_dir = os.path.join(self.dataset_path, "train")
            self.class_names = sorted([
                d for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
            ])
            class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

            # Load train, val, test subdirs
            for split_name in ["train", "val", "test"]:
                split_dir = os.path.join(self.dataset_path, split_name)
                samples = []
                labels = []
                if os.path.exists(split_dir):
                    for c_name in self.class_names:
                        c_dir = os.path.join(split_dir, c_name)
                        if os.path.exists(c_dir):
                            for fname in sorted(os.listdir(c_dir)):
                                if fname.lower().endswith(self.VALID_EXTENSIONS):
                                    fpath = os.path.join(c_dir, fname)
                                    c_idx = class_to_idx[c_name]
                                    samples.append((fpath, c_idx))
                                    labels.append(c_idx)
                self.split_samples[split_name] = samples
                self.splits[split_name] = {
                    "images": np.array([s[0] for s in samples]),
                    "labels": np.array(labels, dtype=np.int64)
                }

            # If val split empty, partition 10% from train
            if len(self.split_samples.get("val", [])) == 0 and len(self.split_samples.get("train", [])) > 0:
                train_all = self.split_samples["train"]
                rng = np.random.RandomState(seed)
                val_samples = []
                new_train_samples = []
                for c_idx in range(len(self.class_names)):
                    c_items = [s for s in train_all if s[1] == c_idx]
                    rng.shuffle(c_items)
                    n_val = max(1, int(len(c_items) * 0.1))
                    val_samples.extend(c_items[:n_val])
                    new_train_samples.extend(c_items[n_val:])
                self.split_samples["train"] = new_train_samples
                self.split_samples["val"] = val_samples
                self.splits["train"] = {
                    "images": np.array([s[0] for s in new_train_samples]),
                    "labels": np.array([s[1] for s in new_train_samples], dtype=np.int64)
                }
                self.splits["val"] = {
                    "images": np.array([s[0] for s in val_samples]),
                    "labels": np.array([s[1] for s in val_samples], dtype=np.int64)
                }
        else:
            # Direct class folders in root
            self.has_predefined_splits = False
            self.class_names = sorted([d for d in entries if os.path.isdir(os.path.join(self.dataset_path, d))])
            class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            all_samples = []
            for c_name in self.class_names:
                c_dir = os.path.join(self.dataset_path, c_name)
                for fname in sorted(os.listdir(c_dir)):
                    if fname.lower().endswith(self.VALID_EXTENSIONS):
                        all_samples.append((os.path.join(c_dir, fname), class_to_idx[c_name]))

            # Partition 80% train, 10% val, 10% test
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
        from collections import Counter
        for sname, slist in self.split_samples.items():
            counts = Counter(s[1] for s in slist)
            class_dist[sname] = {
                cname: counts.get(cidx, 0) for cidx, cname in enumerate(self.class_names)
            }

        return {
            "adapter_type": "ImageFolderAdapter",
            "format": "image_folder",
            "dataset_root": self.dataset_path,
            "has_predefined_splits": self.has_predefined_splits,
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
