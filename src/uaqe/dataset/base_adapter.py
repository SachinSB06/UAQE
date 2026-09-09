"""Base Dataset Adapter Interface for UAQE.

Defines the abstract interface that all dataset adapters must implement
to support universal dataset ingestion, partitioning, and model-adaptive preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset-specific ingestion adapters."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.class_names: List[str] = []
        self.splits: Dict[str, Dict[str, np.ndarray]] = {}
        self.batch_manifest: List[Dict[str, Any]] = []

    @abstractmethod
    def load(
        self,
        train_val_split: Tuple[int, int] = (45000, 5000),
        seed: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Load raw dataset files non-destructively and create deterministic splits."""
        pass

    @abstractmethod
    def get_torch_dataset(
        self,
        split: str,
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        interpolation: str = "bilinear",
        layout: str = "NCHW"
    ) -> Dataset:
        """Return a PyTorch Dataset for the specified split with model-adaptive preprocessing."""
        pass

    def get_dataloader(
        self,
        split: str,
        batch_size: int = 64,
        shuffle: bool = False,
        num_workers: int = 0,
        **preprocess_kwargs
    ) -> DataLoader:
        """Return a standard PyTorch DataLoader for the specified split."""
        dataset = self.get_torch_dataset(split, **preprocess_kwargs)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False
        )

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata describing dataset format, classes, splits, and manifests."""
        pass
