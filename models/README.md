# UAQE Model Repository & Artifact Registry

This directory documents the model formats, capabilities, ingestion procedures, and verified demonstration artifacts supported by the **Universal AI Quantization Engine (UAQE)**.

---

## 1. Supported Model Formats

UAQE incorporates an autonomous, capability-driven model ingestor (`UniversalModelIngestor`) that inspects layer topologies, tensor shapes, and parameters across the following formats:

| Format | File Extension | Ingestion Adapter | Primary Capabilities |
| :--- | :--- | :--- | :--- |
| **ONNX** | `.onnx` | `OnnxModelAdapter` | Static graph analysis, operator topology inspection, INT8 PTQ conversion |
| **TensorFlow Lite** | `.tflite` | `TFLiteModelAdapter` | FlatBuffer inspection, operator delegation analysis, benchmark execution |
| **SafeTensors** | `.safetensors` | `SafeTensorsModelAdapter` | Zero-copy weight inspection, header parsing, parameter verification |
| **PyTorch Checkpoint** | `.pt`, `.pth`, `.bin` | `PyTorchModelAdapter` | State-dict parameter inspection, transfer learning fine-tuning |

---

## 2. Architecture Capabilities & Guardrails

UAQE enforces strict **capability-driven orchestration**:
- **MobileNetV3 (Small / Large)**: Fully supported for Post-Training Quantization (PTQ INT8), Quantization-Aware Training (QAT), Structured Sensitivity Pruning, and XNNPACK CPU acceleration.
- **ResNet-50**: Supported for PTQ INT8, layer sensitivity profiling, Fisher information approximation, and transfer learning workflows.
- **Vision Transformers (ViT-Large / ViT-Base)**: Ingested and inspected for tensor topology. However, UAQE enforces an **explicit fail-safe guardrail** that flags dynamic attention operators as unsupported for edge micro-kernel quantization, rejecting them cleanly with HTTP 400 and structured diagnostics rather than causing pipeline crashes.

---

## 3. Included Demonstration Artifacts

To facilitate offline testing and regression verification without requiring external downloads or multi-hour training, this directory includes two verified, small (~1.85 MB) demonstration artifacts:

### A. Reference QAT INT8 Model
- **Filename**: `models/mobilenetv3_sem_9class_qat_int8.tflite`
- **Architecture**: MobileNetV3-Small (9-class semiconductor wafer defect classifier)
- **Size**: 1,855,816 bytes (~1.77 MiB)
- **Runtime**: TensorFlow Lite Standard CPU
- **SHA-256**:
  ```text
  10d51d4f243c554734e7347f9010297fb182a64c4ad89d5551e24830846dab1d
  ```

### B. XNNPACK-Compatible INT8 Model
- **Filename**: `models/mobilenetv3_sem_9class_xnnpack_int8.tflite`
- **Architecture**: MobileNetV3-Small (XNNPACK-aligned quantization parameters)
- **Size**: 1,855,816 bytes (~1.77 MiB)
- **Runtime Engine**: TensorFlow Lite with XNNPACK Delegate (204/204 operators delegated, 0 fallbacks)
- **SHA-256**:
  ```text
  93a54e8f1408b62c41fe595e4a88b7d6ac2dbf9b368569bc506fb5f1f5da78fd
  ```

---

## 4. Checkpoint Policy & What NOT to Commit

- **Large Checkpoints (>50 MB)**: Pretrained models such as unquantized ResNet-50 (`model.safetensors`, ~97.7 MB) or ViT checkpoints (~1.1 GB) should **not** be committed directly to Git. Instead, download or mount them locally.
- **Intermediate Training Checkpoints**: Ephemeral epoch weights (`.ckpt`, `optimizer.pt`) generated during QAT or reconstruction are excluded via `.gitignore`.
- **Generated Job Artifacts**: Candidate models produced during optimization runs are written to `output/jobs/<job_id>/candidates/` and are strictly excluded from version control.

---

## 5. How to Provide Your Own Models

### Via Web Dashboard
1. Open the UAQE dashboard at `http://localhost:3000`.
2. Navigate to **New Optimization**.
3. Drag and drop your `.onnx`, `.tflite`, or `.safetensors` model file.
4. UAQE inspects the graph, reports tensor dimensions, and generates an optimization plan.

### Via Command-Line Interface (CLI)
```bash
python uaqe.py inspect-model --model "path/to/your_model.onnx"

python uaqe.py plan \
  --model "path/to/your_model.onnx" \
  --dataset "path/to/calibration_dataset" \
  --target raspberrypi5 \
  --profile balanced
```

---

## 6. Licensing Considerations

Demonstration models provided in this repository are derived from open-source MobileNet architectures and trained on non-proprietary synthetic semiconductor inspection datasets. When using custom weights, ensure compliance with the respective model architecture and training dataset licenses.
