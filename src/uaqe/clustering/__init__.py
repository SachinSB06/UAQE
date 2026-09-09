"""
UAQE Clustering Package (Phase D.3)
Implements Clustering-Aware Fine-Tuning, Straight-Through Estimators (STE),
Codebook Centroid Optimization, and Storage Serialization.
"""

from src.uaqe.clustering.clustering_module import (
    ClusteringWeightModule,
    apply_clustering_to_model,
    freeze_clustered_weights
)
from src.uaqe.clustering.clustering_trainer import ClusteringTrainer
from src.uaqe.clustering.clustering_experimenter import ClusteringExperimenter

__all__ = [
    "ClusteringWeightModule",
    "apply_clustering_to_model",
    "freeze_clustered_weights",
    "ClusteringTrainer",
    "ClusteringExperimenter"
]
