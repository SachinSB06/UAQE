"""ResNet-50 Checkpoint Model Architecture and Loader for UAQE.

Implements the exact ResNetForImageClassification architecture matching the downloaded
safetensors checkpoint (src/models/resnet50/model.safetensors) and its configuration
(src/models/resnet50/config.json), with strict safetensors tensor mapping and
customizable classification head.
"""

import os
import json
import struct
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetEmbeddings(nn.Module):
    """Initial convolutional stem: 7x7 conv stride 2 + BatchNorm + ReLU + MaxPool 3x3 stride 2."""

    def __init__(self, embedding_size: int = 64, num_channels: int = 3):
        super().__init__()
        self.embedder = nn.ModuleDict({
            "convolution": nn.Conv2d(num_channels, embedding_size, kernel_size=7, stride=2, padding=3, bias=False),
            "normalization": nn.BatchNorm2d(embedding_size, eps=1e-5, momentum=0.1)
        })
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder["convolution"](x)
        x = self.embedder["normalization"](x)
        x = F.relu(x)
        x = self.pooler(x)
        return x


class ResNetBottleneckLayer(nn.Module):
    """ResNet Bottleneck Layer (v1.5 standard: 1x1 stride 1, 3x3 stride S, 1x1 stride 1)."""

    def __init__(self, in_channels: int, intermediate_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.layer = nn.ModuleList([
            nn.ModuleDict({
                "convolution": nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False),
                "normalization": nn.BatchNorm2d(intermediate_channels, eps=1e-5, momentum=0.1)
            }),
            nn.ModuleDict({
                "convolution": nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                "normalization": nn.BatchNorm2d(intermediate_channels, eps=1e-5, momentum=0.1)
            }),
            nn.ModuleDict({
                "convolution": nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, bias=False),
                "normalization": nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
            })
        ])

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.ModuleDict({
                "convolution": nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                "normalization": nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
            })
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 1x1 reduction
        out = self.layer[0]["convolution"](x)
        out = self.layer[0]["normalization"](out)
        out = F.relu(out)

        # 3x3 convolution
        out = self.layer[1]["convolution"](out)
        out = self.layer[1]["normalization"](out)
        out = F.relu(out)

        # 1x1 expansion
        out = self.layer[2]["convolution"](out)
        out = self.layer[2]["normalization"](out)

        if self.shortcut is not None:
            residual = self.shortcut["convolution"](residual)
            residual = self.shortcut["normalization"](residual)

        out = out + residual
        out = F.relu(out)
        return out


class ResNetStage(nn.Module):
    """ResNet Stage containing multiple Bottleneck Layers."""

    def __init__(self, in_channels: int, intermediate_channels: int, out_channels: int, depth: int, stride: int = 2):
        super().__init__()
        layers = []
        # First block handles downsampling
        layers.append(ResNetBottleneckLayer(in_channels, intermediate_channels, out_channels, stride=stride))
        for _ in range(1, depth):
            layers.append(ResNetBottleneckLayer(out_channels, intermediate_channels, out_channels, stride=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResNetEncoder(nn.Module):
    """ResNet Encoder containing 4 stages."""

    def __init__(self, depths: List[int] = [3, 4, 6, 3], hidden_sizes: List[int] = [256, 512, 1024, 2048]):
        super().__init__()
        stages = []
        in_c = 64
        for idx, (depth, hidden_size) in enumerate(zip(depths, hidden_sizes)):
            inter_c = hidden_size // 4
            stride = 1 if idx == 0 else 2
            stages.append(ResNetStage(in_c, inter_c, hidden_size, depth, stride=stride))
            in_c = hidden_size
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class ResNetBackbone(nn.Module):
    """ResNet-50 Feature Extractor Backbone."""

    def __init__(
        self,
        depths: List[int] = [3, 4, 6, 3],
        hidden_sizes: List[int] = [256, 512, 1024, 2048],
        embedding_size: int = 64,
        num_channels: int = 3
    ):
        super().__init__()
        self.embedder = ResNetEmbeddings(embedding_size=embedding_size, num_channels=num_channels)
        self.encoder = ResNetEncoder(depths=depths, hidden_sizes=hidden_sizes)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.pooler(x)
        return x


class ResNetForImageClassification(nn.Module):
    """Complete ResNet-50 for Image Classification."""

    def __init__(
        self,
        num_classes: int = 1000,
        depths: List[int] = [3, 4, 6, 3],
        hidden_sizes: List[int] = [256, 512, 1024, 2048],
        embedding_size: int = 64,
        num_channels: int = 3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.resnet = ResNetBackbone(
            depths=depths,
            hidden_sizes=hidden_sizes,
            embedding_size=embedding_size,
            num_channels=num_channels
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_sizes[-1], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.resnet(x)
        logits = self.classifier(feats)
        return logits


def load_safetensors_state_dict(safetensors_path: str) -> Dict[str, torch.Tensor]:
    """Load safetensors file directly into a PyTorch state_dict dictionary."""
    with open(safetensors_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_meta = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len

        state_dict: Dict[str, torch.Tensor] = {}
        for k, v in header_meta.items():
            if k == "__metadata__":
                continue
            offsets = v["data_offsets"]
            byte_len = offsets[1] - offsets[0]
            f.seek(data_start + offsets[0])
            raw = f.read(byte_len)
            dtype_np = np.float32 if v["dtype"] == "F32" else np.int64
            arr = np.frombuffer(raw, dtype=dtype_np)
            if len(v["shape"]) > 0:
                arr = arr.reshape(v["shape"])
            state_dict[k] = torch.from_numpy(arr.copy())
    return state_dict


def build_resnet50_cifar10(
    safetensors_path: str,
    num_classes: int = 10,
    checkpoint_mode: str = "USER_UPLOAD",
    benchmark_checkpoint_path: Optional[str] = None,
    device: str = "cpu"
) -> Tuple[ResNetForImageClassification, Dict[str, Any]]:
    """Instantiate ResNet-50, load pretrained ImageNet weights, and adapt classifier to num_classes.

    Args:
        safetensors_path: Path to model.safetensors.
        num_classes: Target number of classes (10 for CIFAR-10).
        checkpoint_mode: 'USER_UPLOAD' or 'VERIFIED_BENCHMARK'.
            In USER_UPLOAD mode, never loads the local benchmark checkpoint.
            In VERIFIED_BENCHMARK mode, loads verified trained classifier weights if available.
        benchmark_checkpoint_path: Optional explicit path to verified baseline checkpoint.
        device: 'cpu' or 'cuda'.

    Returns:
        Tuple of (model, metadata_dict).
    """
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors file not found at: {safetensors_path}")

    # Load pretrained weights (ImageNet-1k, 1000 classes)
    raw_state_dict = load_safetensors_state_dict(safetensors_path)

    # Initialize 1000-class model to load weights strictly
    model = ResNetForImageClassification(num_classes=1000)
    incompatible = model.load_state_dict(raw_state_dict, strict=True)
    assert len(incompatible.missing_keys) == 0, f"Missing keys during weight loading: {incompatible.missing_keys}"
    assert len(incompatible.unexpected_keys) == 0, f"Unexpected keys during weight loading: {incompatible.unexpected_keys}"

    weight_source = "ORIGINAL_MODEL"
    adaptation_status = "NOT_REQUIRED"
    adaptation_note = "Model classes match target classes; no adaptation performed."

    if num_classes != 1000:
        hidden_dim = model.classifier[1].in_features
        new_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, num_classes)
        )

        loaded_verified = False
        if checkpoint_mode == "VERIFIED_BENCHMARK":
            # Search for verified trained baseline checkpoint
            candidate_paths = []
            if benchmark_checkpoint_path:
                candidate_paths.append(benchmark_checkpoint_path)
            candidate_paths.extend([
                "output/phase_e2/models/resnet50_cifar10_fp32_baseline.pt",
                os.path.join(os.getcwd(), "output", "phase_e2", "models", "resnet50_cifar10_fp32_baseline.pt"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "phase_e2", "models", "resnet50_cifar10_fp32_baseline.pt"))
            ])

            for p in candidate_paths:
                if os.path.exists(p):
                    try:
                        ckpt = torch.load(p, map_location="cpu", weights_only=False)
                        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                        if "classifier.1.weight" in sd and "classifier.1.bias" in sd:
                            w = sd["classifier.1.weight"]
                            b = sd["classifier.1.bias"]
                            if w.shape == (num_classes, hidden_dim) and b.shape == (num_classes,):
                                new_head[1].weight.data.copy_(w)
                                new_head[1].bias.data.copy_(b)
                                loaded_verified = True
                                weight_source = "VERIFIED_BASELINE_CHECKPOINT"
                                adaptation_status = "VERIFIED_CHECKPOINT"
                                adaptation_note = f"Loaded verified trained CIFAR-10 baseline classifier weights from {p}."
                                break
                    except Exception as ex:
                        pass

        if not loaded_verified:
            # USER_UPLOAD mode or no verified checkpoint found
            nn.init.kaiming_normal_(new_head[1].weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(new_head[1].bias, 0.0)
            weight_source = "RANDOM_INITIALIZATION"
            adaptation_status = "RANDOM_HEAD"
            adaptation_note = (
                "Classifier head randomly initialized (Kaiming normal); model requires training/adaptation "
                "before optimization."
            )

        model.classifier = new_head
        model.num_classes = num_classes

    model.to(device)

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.resnet.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    # Verification forward pass
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (2, num_classes), f"Expected forward pass output shape (2, {num_classes}), got {dummy_output.shape}"

    metadata = {
        "architecture": "ResNetForImageClassification",
        "model_type": "resnet",
        "safetensors_source": safetensors_path,
        "checkpoint_config": {
            "depths": [3, 4, 6, 3],
            "hidden_sizes": [256, 512, 1024, 2048],
            "embedding_size": 64,
            "layer_type": "bottleneck",
            "downsample_in_first_stage": False,
            "downsample_in_bottleneck": False
        },
        "num_classes": num_classes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "backbone_parameters": backbone_params,
        "classifier_parameters": classifier_params,
        "input_resolution": [3, 224, 224],
        "output_shape": list(dummy_output.shape),
        "device": device,
        "checkpoint_mode": checkpoint_mode,
        "weight_source": weight_source,
        "adaptation_status": adaptation_status,
        "adaptation_note": adaptation_note
    }

    return model, metadata
