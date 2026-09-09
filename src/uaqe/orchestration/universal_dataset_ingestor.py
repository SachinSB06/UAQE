"""Universal Dataset Ingestor for UAQE.

Provides framework-agnostic dataset format detection, adapter resolution,
deterministic partitioning, and descriptor generation across CIFAR-10 pickle,
ImageFolder hierarchies, and CSV-labeled datasets.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any, Type

from uaqe.dataset.base_adapter import BaseDatasetAdapter
from uaqe.dataset.cifar10_adapter import CIFAR10PickleAdapter
from .dataset_adapters.image_folder_adapter import ImageFolderAdapter
from .dataset_adapters.csv_labeled_adapter import CSVLabeledImageAdapter


class UniversalDatasetIngestor:
    """Universal Dataset Ingestor and Adapter Resolver."""

    # Adapter Registry
    ADAPTER_REGISTRY: Dict[str, Type[BaseDatasetAdapter]] = {
        "cifar10_pickle": CIFAR10PickleAdapter,
        "image_folder": ImageFolderAdapter,
        "csv_labeled_images": CSVLabeledImageAdapter
    }

    IMAGE_EXTENSIONS = (
        '.png', '.jpg', '.jpeg', '.bmp', '.webp',
        '.PNG', '.JPG', '.JPEG', '.BMP', '.WEBP'
    )

    def __init__(self, dataset_path: str, format_override: Optional[str] = None):
        self.dataset_path = self.resolve_dataset_root(dataset_path)
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        self.detected_format = format_override or self.detect_format(self.dataset_path)
        self.adapter: Optional[BaseDatasetAdapter] = None
        self._init_adapter()

    @classmethod
    def _has_image_files(cls, dir_path: str) -> bool:
        """Return True if dir_path directly contains at least one recognized image file."""
        try:
            if not os.path.isdir(dir_path):
                return False
            with os.scandir(dir_path) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        return True
        except Exception:
            pass
        return False

    @classmethod
    def _is_split_directory(cls, dir_path: str) -> bool:
        """Return True if dir_path is a split directory (e.g. train) containing class subdirectories with images."""
        if not os.path.isdir(dir_path):
            return False
        try:
            subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and not d.startswith(".")]
            if subdirs:
                # Check if at least one class directory directly has image files
                return any(cls._has_image_files(os.path.join(dir_path, d)) for d in subdirs)
        except Exception:
            pass
        return False

    @classmethod
    def _is_direct_class_root(cls, dir_path: str) -> bool:
        """Return True if dir_path directly contains class directories with images (Case 3: no train/val/test)."""
        if not os.path.isdir(dir_path):
            return False
        try:
            entries = os.listdir(dir_path)
            subdirs = [d for d in entries if os.path.isdir(os.path.join(dir_path, d)) and not d.startswith(".")]
            if any(d.lower() in ("train", "val", "test") for d in subdirs):
                return False
            if subdirs:
                classes_with_images = sum(1 for d in subdirs if cls._has_image_files(os.path.join(dir_path, d)))
                return classes_with_images >= 1
        except Exception:
            pass
        return False

    @classmethod
    def resolve_dataset_root(cls, path: str) -> str:
        """Resolve the true dataset root directory across Case 1, Case 2, and Case 3.
        
        Case 1: root directly contains train/val/test with class subfolders.
        Case 2: root contains an inner dataset directory that contains train/val/test.
        Case 3: ImageFolder has class_a/class_b/class_c with no train/val/test split.
        """
        root = os.path.abspath(path)
        if not os.path.isdir(root):
            return root

        # Case 1: root directly contains 'train' split directory
        train_dir = os.path.join(root, "train")
        if cls._is_split_directory(train_dir):
            return root

        # Check if root is directly a CIFAR-10 batch directory
        cifar_files = [f for f in os.listdir(root) if not os.path.isdir(os.path.join(root, f))]
        if any(f.startswith("data_batch_") for f in cifar_files) or "batches.meta" in cifar_files:
            return root

        # Search candidates breadth-first up to max depth (depth 5)
        candidates_with_train: List[Tuple[int, str]] = []
        candidates_with_classes: List[Tuple[int, str]] = []
        candidates_cifar: List[Tuple[int, str]] = []

        for current_root, dirs, files in os.walk(root):
            rel = os.path.relpath(current_root, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 5:
                dirs.clear()
                continue

            # Check for CIFAR-10
            if any(f.startswith("data_batch_") for f in files) or "batches.meta" in files:
                candidates_cifar.append((depth, current_root))

            # Check for Case 2: candidate contains train split
            if "train" in dirs and cls._is_split_directory(os.path.join(current_root, "train")):
                candidates_with_train.append((depth, current_root))

            # Check for Case 3: candidate contains direct class folders
            if cls._is_direct_class_root(current_root):
                base = os.path.basename(current_root).lower()
                if base not in ("train", "val", "test"):
                    candidates_with_classes.append((depth, current_root))

        if candidates_with_train:
            candidates_with_train.sort(key=lambda x: x[0])
            return candidates_with_train[0][1]

        if candidates_cifar:
            candidates_cifar.sort(key=lambda x: x[0])
            return candidates_cifar[0][1]

        if candidates_with_classes:
            candidates_with_classes.sort(key=lambda x: x[0])
            return candidates_with_classes[0][1]

        return root

    @classmethod
    def _unwrap_dataset_dir(cls, path: str) -> str:
        """Backward-compatible alias for resolve_dataset_root."""
        return cls.resolve_dataset_root(path)

    @classmethod
    def detect_format(cls, path: str) -> str:
        """Detect dataset format dynamically."""
        if not os.path.exists(path):
            return "unknown"

        # Resolve true root first
        path = cls.resolve_dataset_root(path)

        if os.path.isdir(path):
            files = os.listdir(path)

            # 1. CIFAR-10 pickle batches (direct)
            has_batches = any(f.startswith("data_batch_") for f in files)
            has_test = "test_batch" in files
            has_meta = "batches.meta" in files
            if (has_batches and has_test) or has_meta:
                return "cifar10_pickle"

            # 1b. CIFAR-10 pickle batches (nested in subfolder)
            for sub in files:
                sub_path = os.path.join(path, sub)
                if os.path.isdir(sub_path):
                    sub_files = os.listdir(sub_path)
                    if any(f.startswith("data_batch_") for f in sub_files) or "batches.meta" in sub_files:
                        return "cifar10_pickle"

            # 2. CSV labeled images
            has_csv = any(f.lower().endswith(".csv") for f in files)
            has_images = any(f.lower().endswith(cls.IMAGE_EXTENSIONS) for f in files)
            if has_csv and has_images:
                return "csv_labeled_images"

            # 3. ImageFolder hierarchy (subdirs)
            subdirs = [f for f in files if os.path.isdir(os.path.join(path, f))]
            if len(subdirs) > 0:
                return "image_folder"

        return "unknown"

    def _init_adapter(self) -> None:
        self.dataset_path = self.resolve_dataset_root(self.dataset_path)

        if self.detected_format not in self.ADAPTER_REGISTRY:
            raise ValueError(
                f"Dataset format '{self.detected_format}' unrecognized or unsupported for path: {self.dataset_path}. "
                f"Supported formats: {list(self.ADAPTER_REGISTRY.keys())}"
            )

        # If cifar10_pickle is inside a subdirectory, resolve path to subdirectory
        if self.detected_format == "cifar10_pickle" and os.path.isdir(self.dataset_path):
            files = os.listdir(self.dataset_path)
            if not any(f.startswith("data_batch_") for f in files) and not ("batches.meta" in files):
                for sub in files:
                    sub_path = os.path.join(self.dataset_path, sub)
                    if os.path.isdir(sub_path):
                        sub_files = os.listdir(sub_path)
                        if any(f.startswith("data_batch_") for f in sub_files) or "batches.meta" in sub_files:
                            self.dataset_path = sub_path
                            break

        adapter_cls = self.ADAPTER_REGISTRY[self.detected_format]
        self.adapter = adapter_cls(self.dataset_path)

    def load(
        self,
        train_val_split: Tuple[int, int] = (45000, 5000),
        seed: int = 42
    ) -> Dict[str, Dict[str, Any]]:
        """Load dataset non-destructively and create deterministic splits."""
        if self.adapter is None:
            raise RuntimeError("Dataset adapter not initialized")
        return self.adapter.load(train_val_split=train_val_split, seed=seed)

    def get_descriptor(self) -> Dict[str, Any]:
        """Return normalized dataset descriptor."""
        if self.adapter is None:
            raise RuntimeError("Dataset adapter not initialized")
        meta = self.adapter.get_metadata()
        if not meta.get("splits") or not meta["splits"].get("train_count"):
            try:
                self.load()
                meta = self.adapter.get_metadata()
            except Exception:
                pass
        splits = meta.get("splits", {})

        return {
            "dataset_name": os.path.basename(self.dataset_path),
            "detected_format": self.detected_format,
            "adapter_type": self.adapter.__class__.__name__,
            "dataset_root": self.dataset_path,
            "class_count": meta.get("class_count", len(self.adapter.class_names)),
            "class_names": self.adapter.class_names,
            "splits": splits,
            "class_distribution": meta.get("class_distribution", {}),
            "original_files_modified": False
        }

    def inspect(self) -> Dict[str, Any]:
        """Return normalized dataset descriptor (alias for get_descriptor)."""
        return self.get_descriptor()

    def generate_descriptor(self, output_path: str) -> Dict[str, Any]:
        """Export dataset_descriptor.json."""
        descriptor = self.get_descriptor()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptor, f, indent=2)
        return descriptor
