"""UAQE Dataset Management Package."""
from .base_adapter import BaseDatasetAdapter
from .cifar10_adapter import CIFAR10PickleAdapter, CIFAR10TorchDataset
from .universal_dataset_loader import UniversalDatasetLoader

# Backward compatibility alias
CIFAR10Dataset = CIFAR10TorchDataset

__all__ = [
    "BaseDatasetAdapter",
    "CIFAR10PickleAdapter",
    "CIFAR10TorchDataset",
    "CIFAR10Dataset",
    "UniversalDatasetLoader"
]
