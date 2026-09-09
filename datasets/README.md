# UAQE Dataset Guidelines & Calibration Registry

This directory contains calibration datasets, validation manifests, and documentation for dataset formats supported by the **Universal AI Quantization Engine (UAQE)**.

---

## 1. Supported Dataset Formats

UAQE includes an autonomous dataset ingestion subsystem (`UniversalDatasetIngestor`) that detects structure, splits, image dimensions, and class mappings across the following formats:

| Format | Structure | Ingestion Adapter | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **ImageFolder** | Subfolders per class (`class_name/image.png`) | `ImageFolderDatasetAdapter` | Vision classification, representative PTQ calibration |
| **Split ImageFolder** | `train/`, `val/`, `test/` directories with class subfolders | `ImageFolderDatasetAdapter` | Quantization-aware training (QAT), evaluation |
| **CIFAR-10 Binary** | Python pickle batches (`data_batch_*`, `test_batch`) | `CIFAR10Adapter` | ResNet-50 transfer learning & benchmark verification |
| **Manifest-Driven** | JSON manifest specifying image relative paths & labels | `UniversalDatasetLoader` | Deterministic reproducible calibration sampling |

---

## 2. Expected Directory Structure

For custom image classification datasets, organize folders in the standard directory format:

```text
my_dataset/
├── train/
│   ├── class_a/
│   │   ├── img001.png
│   │   └── img002.png
│   └── class_b/
│       ├── img003.png
│       └── img004.png
├── val/
│   ├── class_a/
│   │   └── img005.png
│   └── class_b/
│       └── img006.png
└── test/ (optional)
    ├── class_a/
    └── class_b/
```

Single-level class folders without `train`/`val` splits are also automatically recognized; UAQE will partition them into calibration and validation sets dynamically using stratified sampling.

---

## 3. Included Reference Datasets

### A. Semiconductor (SEM) 9-Class Calibration Set (`datasets/calibration/`)
- **Purpose**: Representative calibration dataset for MobileNetV3 INT8 post-training quantization and QAT.
- **Classes**: 9 defect categories (`bridge`, `clean`, `cmp`, `crack`, `opens`, `other`, `particle`, `scratch`, `vias`).
- **Resolution**: 128×128 RGB images.
- **Manifest**: `datasets/calibration/calibration_manifest.json` provides deterministic sample provenance.

### B. Hackathon Test Dataset (`datasets/hackathon_test_dataset/`)
- **Purpose**: Lightweight smoke-test dataset for end-to-end API and UI pipeline verification.

---

## 4. Datasets Intentionally Excluded from Version Control

To maintain a lean, GitHub-compliant repository (<100 MB), large multi-gigabyte datasets are **strictly excluded**:
- **ImageNet-1K / ImageNet-10K**: Full-scale ImageNet archives (>150 GB) used during ViT capability boundary testing are not stored in Git.
- **Raw High-Resolution Wafer Scans**: Uncompressed industrial wafer inspection raw imagery.

To run experiments requiring ImageNet or custom external datasets, mount the directory locally and specify its path via CLI or the Web UI:
```bash
python uaqe.py optimize \
  --model "models/mobilenetv3_sem_9class_qat_int8.tflite" \
  --dataset "path/to/my_external_dataset" \
  --target raspberrypi5 \
  --profile balanced
```

---

## 5. Dataset Adapters & Ingestion Workflow

When a dataset is provided, UAQE:
1. Validates directory permissions and scans image integrity via `Pillow`.
2. Computes a deterministic SHA-256 hash across samples (`manifest_hash`) to ensure strict provenance.
3. Automatically maps tensor channel ordering (`NCHW` vs `NHWC`) and normalizes pixel intensities based on the ingested model's input specification.
4. Generates a structured descriptor recording class counts, split ratios, and sample resolutions.
