"""Hardware Target Profile Registry for UAQE.

Provides canonical hardware constraints, supported precisions, memory limits,
and export formats for target hardware classes (ESP32, STM32, Artix-7, Zynq-7000,
Raspberry Pi 4, Raspberry Pi 5).
"""

from typing import Dict, List, Tuple, Optional, Any


class HardwareTargetRegistry:
    """Registry of hardware profiles used for constraint planning."""

    _PROFILES: Dict[str, Dict[str, Any]] = {
        "esp32": {
            "target_id": "esp32",
            "name": "Espressif ESP32-S3",
            "hardware_class": "EMBEDDED_MCU",
            "sram_bytes": 524288,        # 512 KB
            "flash_bytes": 8388608,      # 8 MB
            "supported_precisions": ["INT8"],
            "preferred_export_format": "tflite",
            "max_model_size_bytes": 4194304, # 4 MB
            "is_physical_measurement": False,
            "description": "Embedded micro-controller target. Strictly requires INT8 quantization and small memory footprint."
        },
        "stm32": {
            "target_id": "stm32",
            "name": "STMicroelectronics STM32H7",
            "hardware_class": "EMBEDDED_MCU",
            "sram_bytes": 1048576,       # 1 MB
            "flash_bytes": 2097152,      # 2 MB
            "supported_precisions": ["INT8", "FP16"],
            "preferred_export_format": "tflite",
            "max_model_size_bytes": 1572864, # 1.5 MB
            "is_physical_measurement": False,
            "description": "High-performance ARM Cortex-M7 MCU. Supports INT8 and limited FP16."
        },
        "artix7": {
            "target_id": "artix7",
            "name": "Xilinx Artix-7 XC7A100T",
            "hardware_class": "FPGA",
            "lut_count": 63400,
            "bram_blocks": 135,
            "supported_precisions": ["INT8", "INT4", "FP16"],
            "preferred_export_format": "uaqe",
            "max_model_size_bytes": 8388608,
            "is_physical_measurement": False,
            "description": "Cost-optimized FPGA target. Excellent for INT8 and custom sparse bit-serial arithmetic."
        },
        "zynq7000": {
            "target_id": "zynq7000",
            "name": "Xilinx Zynq-7020 SoC",
            "hardware_class": "FPGA_SOC",
            "lut_count": 53200,
            "bram_blocks": 140,
            "dram_bytes": 1073741824,   # 1 GB
            "supported_precisions": ["INT8", "FP16", "FP32"],
            "preferred_export_format": "uaqe",
            "max_model_size_bytes": 33554432,
            "is_physical_measurement": False,
            "description": "Dual ARM Cortex-A9 + FPGA fabric SoC. Supports heterogeneous acceleration."
        },
        "raspberrypi4": {
            "target_id": "raspberrypi4",
            "name": "Raspberry Pi 4 Model B (Broadcom BCM2711, 4x Cortex-A72)",
            "hardware_class": "SINGLE_BOARD_COMPUTER",
            "ram_bytes": 4294967296,    # 4 GB
            "supported_precisions": ["INT8", "FP16", "FP32"],
            "preferred_export_format": "tflite",
            "max_model_size_bytes": 268435456, # 256 MB
            "is_physical_measurement": False,
            "description": "Quad-core ARM Cortex-A72 SBC with NEON SIMD vector extensions."
        },
        "raspberrypi5": {
            "target_id": "raspberrypi5",
            "name": "Raspberry Pi 5 (Broadcom BCM2712, 4x Cortex-A76)",
            "hardware_class": "SINGLE_BOARD_COMPUTER",
            "ram_bytes": 8589934592,    # 8 GB
            "supported_precisions": ["INT8", "FP16", "FP32"],
            "preferred_export_format": "tflite",
            "max_model_size_bytes": 536870912, # 512 MB
            "is_physical_measurement": False,
            "description": "Quad-core ARM Cortex-A76 SBC (2.4 GHz) with Dot-Product SIMD extensions."
        },
        "hostcpu": {
            "target_id": "host_cpu",
            "name": "Host x86_64 CPU",
            "hardware_class": "HOST_CPU",
            "supported_precisions": ["INT8", "FP16", "FP32"],
            "preferred_export_format": "onnx",
            "max_model_size_bytes": 1073741824,
            "is_physical_measurement": True,
            "description": "Host x86_64 CPU runtime environment with SIMD vector execution (AVX2/AVX-512)."
        },
        "jetsonorinnano": {
            "target_id": "jetson_orin_nano",
            "name": "NVIDIA Jetson Orin Nano (Ampere GPU + ARM Cortex-A78AE)",
            "hardware_class": "EMBEDDED_GPU",
            "ram_bytes": 8589934592,
            "supported_precisions": ["INT8", "FP16", "FP32"],
            "preferred_export_format": "onnx",
            "max_model_size_bytes": 1073741824,
            "is_physical_measurement": False,
            "description": "Embedded AI system with Ampere GPU and TensorRT execution capabilities."
        }
    }

    @classmethod
    def get_profile(cls, target_id: str) -> Dict[str, Any]:
        normalized = target_id.lower().replace("-", "").replace("_", "").replace(" ", "")
        for k, v in cls._PROFILES.items():
            if k == normalized or k == target_id.lower():
                return v
        raise ValueError(
            f"Hardware target '{target_id}' not found in registry. "
            f"Available targets: {list(cls._PROFILES.keys())}"
        )

    @classmethod
    def list_targets(cls) -> List[str]:
        targets = []
        for k, v in cls._PROFILES.items():
            if k not in targets:
                targets.append(k)
            tid = v.get("target_id")
            if tid and tid not in targets:
                targets.append(tid)
        return targets
