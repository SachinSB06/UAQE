"""Universal Dataset Loader for UAQE.

Acts as a dataset adapter registry and dispatcher that automatically detects
dataset formats (e.g. CIFAR-10 pickle batches, image folder hierarchies) and delegates
ingestion, deterministic partitioning, and PyTorch dataset creation to dedicated adapters.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any, Type
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .base_adapter import BaseDatasetAdapter
from .cifar10_adapter import CIFAR10PickleAdapter, CIFAR10TorchDataset
# Alias for backwards compatibility
CIFAR10Dataset = CIFAR10TorchDataset


class UniversalDatasetLoader:
    """Universal Dataset Loader registry and dispatcher."""

    FORMAT_CIFAR10_PICKLE = "cifar10_pickle"
    FORMAT_IMAGE_FOLDER = "image_folder"
    FORMAT_CSV_LABELED = "csv_labeled_images"
    FORMAT_UNKNOWN = "unknown"

    # Adapter Registry
    _ADAPTER_REGISTRY: Dict[str, Type[BaseDatasetAdapter]] = {
        FORMAT_CIFAR10_PICKLE: CIFAR10PickleAdapter,
    }

    @classmethod
    def _resolve_adapter_class(cls, format_name: str) -> Optional[Type[BaseDatasetAdapter]]:
        """Resolve adapter class dynamically without top-level circular imports."""
        if format_name in cls._ADAPTER_REGISTRY:
            return cls._ADAPTER_REGISTRY[format_name]

        if format_name == cls.FORMAT_IMAGE_FOLDER:
            from uaqe.orchestration.dataset_adapters.image_folder_adapter import ImageFolderAdapter
            cls._ADAPTER_REGISTRY[cls.FORMAT_IMAGE_FOLDER] = ImageFolderAdapter
            return ImageFolderAdapter

        if format_name == cls.FORMAT_CSV_LABELED:
            from uaqe.orchestration.dataset_adapters.csv_labeled_adapter import CSVLabeledImageAdapter
            cls._ADAPTER_REGISTRY[cls.FORMAT_CSV_LABELED] = CSVLabeledImageAdapter
            return CSVLabeledImageAdapter

        return None

    def __init__(self, dataset_path: str, format_override: Optional[str] = None):
        """Initialize loader with dataset directory and resolve appropriate adapter.

        Args:
            dataset_path: Path to dataset root directory.
            format_override: Optional explicit format specifier.
        """
        self.dataset_path = os.path.abspath(dataset_path)
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        self.detected_format = format_override or self.detect_format(self.dataset_path)
        adapter_cls = self._resolve_adapter_class(self.detected_format)

        if adapter_cls is not None:
            self.adapter: Optional[BaseDatasetAdapter] = adapter_cls(self.dataset_path)
        else:
            self.adapter = None

    @classmethod
    def register_adapter(cls, format_name: str, adapter_cls: Type[BaseDatasetAdapter]) -> None:
        """Register a new dataset adapter dynamically."""
        cls._ADAPTER_REGISTRY[format_name] = adapter_cls

    @classmethod
    def detect_format(cls, path: str) -> str:
        """Auto-detect format of dataset at path."""
        if not os.path.exists(path):
            return cls.FORMAT_UNKNOWN

        if os.path.isdir(path):
            files = os.listdir(path)
            # Check for CIFAR-10 pickle format
            has_batches = any(f.startswith("data_batch_") for f in files)
            has_test = "test_batch" in files
            has_meta = "batches.meta" in files
            if (has_batches and has_test) or has_meta:
                return cls.FORMAT_CIFAR10_PICKLE

            # Check for subdirectories (image_folder format)
            subdirs = [f for f in files if os.path.isdir(os.path.join(path, f))]
            if len(subdirs) > 0:
                return cls.FORMAT_IMAGE_FOLDER

        return cls.FORMAT_UNKNOWN

    @property
    def class_names(self) -> List[str]:
        return self.adapter.class_names if self.adapter else []

    @property
    def splits(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self.adapter.splits if self.adapter else {}

    def load_cifar10(
        self,
        train_val_split: Tuple[int, int] = (45000, 5000),
        seed: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Convenience method for CIFAR-10 loading."""
        return self.load(train_val_split=train_val_split, seed=seed)

    def load(
        self,
        train_val_split: Tuple[int, int] = (45000, 5000),
        seed: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Load dataset using the resolved adapter."""
        if self.adapter is None:
            raise ValueError(f"No adapter registered for format '{self.detected_format}' at {self.dataset_path}")
        return self.adapter.load(train_val_split=train_val_split, seed=seed)

    def get_torch_dataset(
        self,
        split: str,
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        interpolation: str = "bilinear",
        layout: str = "NCHW"
    ) -> Dataset:
        """Delegate PyTorch Dataset creation to the resolved adapter."""
        if self.adapter is None:
            raise ValueError("No adapter available")
        return self.adapter.get_torch_dataset(
            split=split,
            target_size=target_size,
            mean=mean,
            std=std,
            interpolation=interpolation,
            layout=layout
        )

    def get_dataloader(
        self,
        split: str,
        batch_size: int = 64,
        shuffle: bool = False,
        num_workers: int = 0,
        **preprocess_kwargs
    ) -> DataLoader:
        """Delegate PyTorch DataLoader creation to the resolved adapter."""
        if self.adapter is None:
            raise ValueError("No adapter available")
        return self.adapter.get_dataloader(
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **preprocess_kwargs
        )

    def generate_ingestion_report(
        self,
        output_path: str,
        model_name: str = "ResNet-50 v1.5",
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate and save standardized dataset ingestion report JSON."""
        if self.adapter is None or not self.adapter.splits:
            raise RuntimeError("Cannot generate ingestion report before loading dataset splits.")

        meta = self.adapter.get_metadata()

        if preprocessing_config is None:
            preprocessing_config = {
                "target_resolution": [224, 224],
                "interpolation": "bilinear",
                "color_format": "RGB",
                "normalization": "ImageNet",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "layout": "NCHW"
            }

        report = {
            "dataset_name": "CIFAR-10",
            "detected_format": self.detected_format,
            "adapter_type": meta.get("adapter_type", "Unknown"),
            "dataset_root": self.dataset_path,
            "class_count": meta["class_count"],
            "class_names": meta["class_names"],
            "splits": meta["splits"],
            "class_distribution": meta["class_distribution"],
            "raw_batch_manifest": meta.get("batch_manifest", []),
            "target_model": model_name,
            "model_adaptive_preprocessing": preprocessing_config,
            "original_files_modified": False
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
