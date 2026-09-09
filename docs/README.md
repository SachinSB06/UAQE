# UAQE Documentation

Welcome to the technical documentation for the **Universal AI Quantization Engine (UAQE)**.

---

## Documentation Index

1. **[Project Structure](project_structure.md)**
   - Complete directory layout and explanations of repository components.

2. **[Architecture Overview](architecture/overview.md)**
   - High-level system architecture, engine pipeline, FastAPI backend, Vite dashboard, and SSE event streaming.

3. **[Optimization Strategies](optimization/strategies.md)**
   - Supported candidate generation strategies: Post-Training Quantization (PTQ), Sensitivity-Guided Pruning, Layer Reconstruction, Quantization-Aware Training (QAT), and Experimental XNNPACK delegation.

4. **[Benchmarking & Timing Protocol](benchmarking/protocol.md)**
   - Canonical benchmark methodology, multi-iteration warmup, timing isolation, candidate metric normalization, and latency provenance verification.

5. **[Hardware Validation Status](deployment/hardware_validation.md)**
   - Separation of Host CPU benchmark execution from target edge hardware deployment (Raspberry Pi 5 physical validation status).

---

## Authoritative Reports

- **[Project Cleanup Report](../output/reports/project_cleanup_report.md)**: Zero-change cleanup verification and before/after repository inventory.
- **[Candidate Latency Provenance Audit](../output/reports/candidate_latency_provenance_audit.md)**: Forensic verification of Candidate 4 XNNPACK delegate provenance and UI display correctness.
- **[Candidate Metric Identity Audit](../output/reports/candidate_metric_identity_audit.md)**: Verification of candidate metric isolation and anti-collision keying.
