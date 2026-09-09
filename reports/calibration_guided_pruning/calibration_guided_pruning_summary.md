# Calibration-Guided Reconstruction Pruning Report

## 1. Status Summary
- **Status**: `BLOCKED`
- **Reason**: `CALIBRATION DATA = UNAVAILABLE` — A separate unlabeled calibration dataset is required. The final test dataset is protected and cannot be used for iterative candidate selection, threshold tuning, or reconstruction fitting.

## 2. Reusable Framework
- Fully implemented `ActivationCollector` for forward-pass activation tracking.
- Fully implemented `ClosedFormReconstructor` for regularized least-squares survivors reconstruction.
- Recalibration and proxy evaluation pipelines are registered and verified.
