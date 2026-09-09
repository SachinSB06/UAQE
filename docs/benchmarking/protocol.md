# Canonical Benchmarking & Timing Protocol

To ensure truthful, scientifically reproducible performance metrics across diverse architectures, UAQE implements strict timing isolation and benchmark governance rules.

---

## 1. Timing Isolation & Pre-Warming

```text
[Load Model] ──▶ [Warmup Iterations (5-10)] ──▶ [GC Collection] ──▶ [Timed Runs (30-100)] ──▶ [Median / Percentiles]
```

1. **Interpreter Initialization**: A dedicated interpreter instance is created for each benchmark evaluation to prevent state pollution.
2. **Warmup Iterations**: A minimum of 5 to 10 warmup inferences are performed with representative input tensors to populate instruction and data caches.
3. **Memory Garbage Collection**: Full Python garbage collection is invoked immediately prior to the timed execution window.
4. **Isolated Timing Window**: Inference runs use high-resolution system timers (`time.perf_counter_ns()`). Benchmark statistics record mean, median, 90th percentile (P90), and 99th percentile (P99) latency.

---

## 2. Telemetry Sampling Protocol

- **Resident Set Size (RSS)**: Measured before, during, and after inference execution.
- **Peak RAM Delta**: Tracked using process memory handles (`psutil.Process().memory_info()`).
- **CPU Utilization**: Sampled at discrete intervals to detect background scheduling contention.

---

## 3. Candidate Metric Isolation & Provenance Rules

- **Anti-Collision Keying**: Every candidate is indexed and retrieved by its explicit `candidate_id` (e.g., `candidate_1`, `candidate_4`). Array position is never used as candidate identity.
- **Per-Candidate Provenance**: Runtime delegate attributes (`runtime_backend`, `delegate_type`, `delegate_acceleration`, `delegated_operators`) are stored directly within each candidate's record.
- **Zero Cross-Contamination**: Winner metadata (e.g., Candidate 1's standard CPU status) is never displayed on non-winner cards (e.g., Candidate 4's XNNPACK status).
