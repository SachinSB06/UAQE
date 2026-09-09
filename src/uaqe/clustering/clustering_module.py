"""
Clustering-Aware Weight Modules and Straight-Through Estimators (STE).
Enables clustering-aware fine-tuning of sparse MobileNetV3 architectures,
preserving structural zero positions while optimizing non-zero weights and codebook centroids.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class ClusteringConv2d(nn.Conv2d):
    """Conv2d layer with straight-through clustering-aware weight quantization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        num_clusters: int = 16
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype
        )
        self.num_clusters = num_clusters
        self.register_buffer("sparsity_mask", torch.ones_like(self.weight, dtype=torch.bool))
        self.register_buffer("centroids", torch.zeros(num_clusters, dtype=torch.float32))
        self.centroids_initialized = False

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, num_clusters: int = 16, mask: Optional[torch.Tensor] = None) -> ClusteringConv2d:
        """Constructs a ClusteringConv2d from an existing standard nn.Conv2d."""
        c_layer = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
            num_clusters=num_clusters
        )
        c_layer.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            c_layer.bias.data.copy_(conv.bias.data)

        if mask is not None:
            c_layer.sparsity_mask.copy_(mask.bool())
        else:
            c_layer.sparsity_mask.copy_((conv.weight.data != 0).bool())

        c_layer.init_centroids()
        return c_layer

    def init_centroids(self) -> None:
        """Initializes cluster centroids on non-zero weights using K-Means or quantiles."""
        with torch.no_grad():
            w_nz = self.weight[self.sparsity_mask].detach().cpu().numpy()
            if len(w_nz) == 0:
                self.centroids.zero_()
                self.centroids_initialized = True
                return

            k = min(self.num_clusters, len(w_nz))
            if k >= len(w_nz):
                cents = np.sort(w_nz)
                if len(cents) < self.num_clusters:
                    pad = np.zeros(self.num_clusters - len(cents), dtype=np.float32)
                    cents = np.concatenate([cents, pad])
            else:
                # K-Means clustering initialization
                kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
                kmeans.fit(w_nz.reshape(-1, 1))
                cents = np.sort(kmeans.cluster_centers_.flatten())
                if len(cents) < self.num_clusters:
                    pad = np.zeros(self.num_clusters - len(cents), dtype=np.float32)
                    cents = np.concatenate([cents, pad])

            self.centroids.copy_(torch.from_numpy(cents.astype(np.float32)).to(self.weight.device))
            self.centroids_initialized = True

    def get_clustered_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes STE clustered weights and returns (w_clustered, cluster_indices)."""
        w = self.weight * self.sparsity_mask
        if not self.centroids_initialized:
            self.init_centroids()

        shape = w.shape
        w_flat = w.flatten()
        mask_flat = self.sparsity_mask.flatten()

        nz_indices = torch.where(mask_flat)[0]
        if len(nz_indices) == 0:
            return w, torch.zeros_like(mask_flat, dtype=torch.long)

        w_nz = w_flat[nz_indices]  # (N_nz,)
        # Pairwise distance: |w_nz[:, None] - centroids[None, :]| -> (N_nz, K)
        dists = torch.abs(w_nz.unsqueeze(1) - self.centroids.unsqueeze(0))
        assign = torch.argmin(dists, dim=1)  # (N_nz,)
        c_vals = self.centroids[assign]      # (N_nz,)

        # STE approximation: forward uses centroid values, backward passes straight through to w
        w_nz_ste = (c_vals - w_nz).detach() + w_nz

        w_out_flat = torch.zeros_like(w_flat)
        w_out_flat[nz_indices] = w_nz_ste
        w_out = w_out_flat.reshape(shape)

        assign_full = torch.zeros_like(mask_flat, dtype=torch.long)
        assign_full[nz_indices] = assign
        return w_out, assign_full.reshape(shape)

    def update_centroids_from_weights(self) -> None:
        """Updates centroids to the mean of assigned weights (Lloyd-Max step)."""
        with torch.no_grad():
            w = self.weight * self.sparsity_mask
            mask_flat = self.sparsity_mask.flatten()
            nz_indices = torch.where(mask_flat)[0]
            if len(nz_indices) == 0:
                return

            w_nz = w.flatten()[nz_indices]
            dists = torch.abs(w_nz.unsqueeze(1) - self.centroids.unsqueeze(0))
            assign = torch.argmin(dists, dim=1)

            for k_idx in range(self.num_clusters):
                k_mask = (assign == k_idx)
                if k_mask.any():
                    self.centroids[k_idx] = w_nz[k_mask].mean()

    def get_clustering_loss(self) -> torch.Tensor:
        """Computes L2 penalty between trainable weights and their assigned centroids."""
        w = self.weight * self.sparsity_mask
        mask_flat = self.sparsity_mask.flatten()
        nz_indices = torch.where(mask_flat)[0]
        if len(nz_indices) == 0:
            return torch.tensor(0.0, device=self.weight.device)

        w_nz = w.flatten()[nz_indices]
        dists = torch.abs(w_nz.unsqueeze(1) - self.centroids.unsqueeze(0))
        assign = torch.argmin(dists, dim=1)
        c_vals = self.centroids[assign]
        return F.mse_loss(w_nz, c_vals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ste, _ = self.get_clustered_weights()
        return F.conv2d(
            x, w_ste, self.bias, self.stride, self.padding,
            self.dilation, self.groups
        )

    def to_standard_conv2d(self) -> nn.Conv2d:
        """Freezes clustered weights into a standard nn.Conv2d."""
        conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=(self.bias is not None),
            padding_mode=self.padding_mode
        )
        with torch.no_grad():
            w_frozen, _ = self.get_clustered_weights()
            # Enforce exact centroid values without STE gradient detached delta
            w = self.weight * self.sparsity_mask
            mask_flat = self.sparsity_mask.flatten()
            nz_indices = torch.where(mask_flat)[0]
            if len(nz_indices) > 0:
                w_nz = w.flatten()[nz_indices]
                dists = torch.abs(w_nz.unsqueeze(1) - self.centroids.unsqueeze(0))
                assign = torch.argmin(dists, dim=1)
                c_vals = self.centroids[assign]
                w_exact = torch.zeros_like(w.flatten())
                w_exact[nz_indices] = c_vals
                conv.weight.data.copy_(w_exact.reshape(self.weight.shape))
            else:
                conv.weight.data.copy_(w)

            if self.bias is not None:
                conv.bias.data.copy_(self.bias.data)
        return conv


class ClusteringLinear(nn.Linear):
    """Linear layer with straight-through clustering-aware weight quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        num_clusters: int = 16
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.num_clusters = num_clusters
        self.register_buffer("sparsity_mask", torch.ones_like(self.weight, dtype=torch.bool))
        self.register_buffer("centroids", torch.zeros(num_clusters, dtype=torch.float32))
        self.centroids_initialized = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, num_clusters: int = 16, mask: Optional[torch.Tensor] = None) -> ClusteringLinear:
        """Constructs a ClusteringLinear from an existing standard nn.Linear."""
        c_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            num_clusters=num_clusters
        )
        c_layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            c_layer.bias.data.copy_(linear.bias.data)

        if mask is not None:
            c_layer.sparsity_mask.copy_(mask.bool())
        else:
            c_layer.sparsity_mask.copy_((linear.weight.data != 0).bool())

        c_layer.init_centroids()
        return c_layer

    def init_centroids(self) -> None:
        """Initializes cluster centroids on non-zero weights."""
        with torch.no_grad():
            w_nz = self.weight[self.sparsity_mask].detach().cpu().numpy()
            if len(w_nz) == 0:
                self.centroids.zero_()
                self.centroids_initialized = True
                return

            k = min(self.num_clusters, len(w_nz))
            if k >= len(w_nz):
                cents = np.sort(w_nz)
                if len(cents) < self.num_clusters:
                    pad = np.zeros(self.num_clusters - len(cents), dtype=np.float32)
                    cents = np.concatenate([cents, pad])
            else:
                kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
                kmeans.fit(w_nz.reshape(-1, 1))
                cents = np.sort(kmeans.cluster_centers_.flatten())
                if len(cents) < self.num_clusters:
                    pad = np.zeros(self.num_clusters - len(cents), dtype=np.float32)
                    cents = np.concatenate([cents, pad])

            self.centroids.copy_(torch.from_numpy(cents.astype(np.float32)).to(self.weight.device))
            self.centroids_initialized = True

    def get_clustered_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.weight * self.sparsity_mask
        if not self.centroids_initialized:
            self.init_centroids()

        shape = w.shape
        w_flat = w.flatten()
        mask_flat = self.sparsity_mask.flatten()

        nz_indices = torch.where(mask_flat)[0]
        if len(nz_indices) == 0:
            return w, torch.zeros_like(mask_flat, dtype=torch.long)

        w_nz = w_flat[nz_indices]
        dists = torch.abs(w_nz.unsqueeze(1) - self.centroids.unsqueeze(0))
        assign = torch.argmin(dists, dim=1)
        c_vals = self.centroids[assign]

        w_nz_ste = (c_vals - w_nz).detach() + w_nz

        w_out_flat = torch.zeros_like(w_flat)
        w_out_flat[nz_indices] = w_nz_ste
        w_out = w_out_flat.reshape(shape)

        assign_full = torch.zeros_like(mask_flat, dtype=torch.long)
        assign_full[nz_indices] = assign
        return w_out, assign_full.reshape(shape)

    def update_centroids_from_weights(self) -> None:
        with torch.no_grad():
            w = self.weight * self.sparsity_mask
            mask_flat = self.sparsity_mask.flatten()
            nz_indices = torch.where(mask_flat)[0]
            if len(nz_indices) == 0:
                return

            w_nz = w.flatten()[nz_indices]
            dists = torch.abs(w_nz.unsqueeze(1) - self.centroids.unsqueeze(0))
            assign = torch.argmin(dists, dim=1)

            for k_idx in range(self.num_clusters):
                k_mask = (assign == k_idx)
                if k_mask.any():
                    self.centroids[k_idx] = w_nz[k_mask].mean()

    def get_clustering_loss(self) -> torch.Tensor:
        w = self.weight * self.sparsity_mask
        mask_flat = self.sparsity_mask.flatten()
        nz_indices = torch.where(mask_flat)[0]
        if len(nz_indices) == 0:
            return torch.tensor(0.0, device=self.weight.device)

        w_nz = w.flatten()[nz_indices]
        dists = torch.abs(w_nz.unsqueeze(1) - self.centroids.unsqueeze(0))
        assign = torch.argmin(dists, dim=1)
        c_vals = self.centroids[assign]
        return F.mse_loss(w_nz, c_vals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ste, _ = self.get_clustered_weights()
        return F.linear(x, w_ste, self.bias)

    def to_standard_linear(self) -> nn.Linear:
        linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=(self.bias is not None)
        )
        with torch.no_grad():
            w = self.weight * self.sparsity_mask
            mask_flat = self.sparsity_mask.flatten()
            nz_indices = torch.where(mask_flat)[0]
            if len(nz_indices) > 0:
                w_nz = w.flatten()[nz_indices]
                dists = torch.abs(w_nz.unsqueeze(1) - self.centroids.unsqueeze(0))
                assign = torch.argmin(dists, dim=1)
                c_vals = self.centroids[assign]
                w_exact = torch.zeros_like(w.flatten())
                w_exact[nz_indices] = c_vals
                linear.weight.data.copy_(w_exact.reshape(self.weight.shape))
            else:
                linear.weight.data.copy_(w)

            if self.bias is not None:
                linear.bias.data.copy_(self.bias.data)
        return linear


ClusteringWeightModule = Union[ClusteringConv2d, ClusteringLinear]


def apply_clustering_to_model(
    model: nn.Module,
    num_clusters: int = 16,
    mask_dict: Optional[Dict[str, torch.Tensor]] = None
) -> nn.Module:
    """Recursively replaces Conv2d and Linear layers with clustering-aware versions."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            mask = mask_dict.get(name) if mask_dict else None
            c_layer = ClusteringConv2d.from_conv2d(child, num_clusters=num_clusters, mask=mask)
            setattr(model, name, c_layer)
        elif isinstance(child, nn.Linear):
            mask = mask_dict.get(name) if mask_dict else None
            c_layer = ClusteringLinear.from_linear(child, num_clusters=num_clusters, mask=mask)
            setattr(model, name, c_layer)
        else:
            apply_clustering_to_model(child, num_clusters=num_clusters, mask_dict=mask_dict)
    return model


def freeze_clustered_weights(model: nn.Module) -> nn.Module:
    """Converts all ClusteringConv2d and ClusteringLinear layers back to standard PyTorch layers."""
    for name, child in list(model.named_children()):
        if isinstance(child, ClusteringConv2d):
            std_layer = child.to_standard_conv2d()
            setattr(model, name, std_layer)
        elif isinstance(child, ClusteringLinear):
            std_layer = child.to_standard_linear()
            setattr(model, name, std_layer)
        else:
            freeze_clustered_weights(child)
    return model
