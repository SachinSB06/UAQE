# Calibration Dataset Inspection & Validation Report

This report summarizes the structure, integrity, and preprocessing compatibility of the calibration dataset.
It was generated for the training-free `CalibrationGuidedReconstructionPruner`.

## Dataset Integrity Summary

**DATASET INTEGRITY = READY**

- **Total Calibration Images**: 1258
  - **Train split**: 877
  - **Val split**: 184
  - **Test split**: 197
- **Calibration Subset Selected**: 200 (from `train` split)
- **Corrupt/Unreadable Images**: 0
- **Internal Duplicate Groups**: 3
- **Cross-Dataset Leakage (vs Hackathon Test)**: 0 overlaps

## Preprocessing Verification

- **Preprocessing Mode**: `rgb_0_1` (rgb_0_1, values divided by 255.0 to be in range `[0.0, 1.0]`)
- **Input Layout**: `NCHW` (NCHW)
- **Dimensions**: 122x106, 136x142, 138x140, 163x167, 165x170, 169x166, 169x169, 176x179, 182x179, 184x176, 189x168, 191x171, 191x182, 194x170, 196x177, 197x176, 199x164, 199x199, 200x173, 202x201, 214x195, 215x187, 215x206, 216x200, 222x200, 226x191, 230x233, 233x243, 256x256, 258x203, 318x317, 323x326, 323x331, 326x327, 330x323, 331x305, 332x299, 335x300, 336x330, 343x287, 351x289, 353x284, 358x389, 359x295, 363x363, 372x378, 373x367, 376x365, 394x348, 398x380, 402x394, 409x352, 411x416, 426x426, 449x364, 450x364, 454x370, 455x446, 459x363, 464x455, 466x362, 467x470, 504x356, 556x362, 561x426, 612x443, 641x313, 641x627, 675x614, 75x72
- **Color Channels**: L, RGB, RGBA (RGB: 33, Grayscale/L: 1188)
- **Compatibility Status**: ✅ COMPATIBLE

*All verified images successfully resize to 128x128, convert to RGB, layout as NCHW, and scale to [0, 1].*

## Split and Class Distribution

### Split Counts
| Split | Image Count |
| --- | --- |
| train | 877 |
| val | 184 |
| test | 197 |

### Class Distribution
| Calibration Class | Train Count | Val Count | Test Count | Mapped Configured Class | Index |
| --- | --- | --- | --- | --- | --- |
| `bridge` | 111 | 23 | 25 | `Bridge` | 1 |
| `clean` | 51 | 11 | 12 | `Clean` | 0 |
| `cmp` | 115 | 24 | 26 | `CMP` | 2 |
| `crack` | 111 | 23 | 25 | `Crack` | 3 |
| `opens` | 111 | 23 | 25 | `Open` | 5 |
| `other` | 56 | 12 | 12 | `Other` | 9 |
| `particle` | 105 | 22 | 23 | `Particle` | 6 |
| `scratch` | 107 | 23 | 24 | `LER` | 4 |
| `vias` | 110 | 23 | 25 | `VIA` | 8 |

## Class Taxonomy Mapping Analysis

The folders in the calibration dataset are lowercase, whereas the evaluation configuration and hackathon test dataset use Title/UPPER case names. The reconstruction algorithm does not use class labels directly, but the taxonomy matches are analyzed here to detect any domain or task mismatch.

| Calibration folder name | Configured Class Name | Index | Match Status |
| --- | --- | --- | --- |
| `bridge` | `Bridge` | 1 | ✅ MATCH |
| `clean` | `Clean` | 0 | ✅ MATCH |
| `cmp` | `CMP` | 2 | ✅ MATCH |
| `crack` | `Crack` | 3 | ✅ MATCH |
| `opens` | `Open` | 5 | ⚠️ MISMATCH |
| `other` | `Other` | 9 | ✅ MATCH |
| `particle` | `Particle` | 6 | ✅ MATCH |
| `scratch` | `LER` | 4 | ⚠️ MISMATCH |
| `vias` | `VIA` | 8 | ⚠️ MISMATCH |

- **Missing Configured Classes in Calibration**: None

## Duplicate & Leakage Analysis

### Internal Duplicates
Detected 3 groups of duplicate images within the calibration dataset.

| Hash (Prefix) | Duplicate Files |
| --- | --- |
| `f3a9252e` | `datasets/calibration/dataset/train/cmp/cmp101.png`, `datasets/calibration/dataset/train/cmp/cmp36.png` |
| `b178de3e` | `datasets/calibration/dataset/train/cmp/cmp146.png`, `datasets/calibration/dataset/train/cmp/cmp92.png` |
| `b2ceda8a` | `datasets/calibration/dataset/train/opens/open150.png`, `datasets/calibration/dataset/test/opens/open133.png` |


### Cross-Dataset Leakage (vs `hackathon_test_dataset`)
No cross-dataset leakage detected. Perfect separation between calibration split and hackathon test dataset.

## Selected Calibration Subset Details

- **Source Split**: `train`
- **Subset Size**: 200 images
- **Sampling Policy**: `stratified_class_proportional` (Stratified class-proportional)
- **Random Seed**: `42`
- **Manifest Location**: [`calibration_manifest.json`](file:///D:/Quantization embedded/datasets/calibration/calibration_manifest.json)

---
*Report generated on local time: 2026-08-25*
