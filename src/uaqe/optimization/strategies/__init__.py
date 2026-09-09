"""UAQE Optimization Strategies Package."""

from .xnnpack_int8 import XNNPACKCompatibleINT8Strategy, build_xnnpack_compatible_artifact

__all__ = [
    "XNNPACKCompatibleINT8Strategy",
    "build_xnnpack_compatible_artifact",
]
