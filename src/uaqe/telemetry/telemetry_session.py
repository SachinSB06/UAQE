"""Telemetry Session Manager for UAQE Jobs.

Maintains FP32 baseline telemetry, candidate telemetry records, final model telemetry,
and session-wide continuous execution samples.
Serializes structured telemetry to telemetry.json within the job artifact directory.
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

from .process_metrics import get_host_environment_info
from .runtime_monitor import RuntimeMonitor


@dataclass
class PhaseTelemetry:
    """Telemetry record for a specific evaluation phase or candidate."""
    phase_id: str
    phase_name: str
    model_state: str  # 'FP32_BASELINE', 'CANDIDATE', 'FINAL_OPTIMIZED'
    candidate_id: Optional[str]
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    throughput_ips: float
    batch_size: int
    warmup_runs: int
    measured_runs: int
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_ram_mb: float
    peak_ram_mb: float
    duration_sec: float
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convenience aliases for standardized protocol comparisons
        d["cpu_avg_pct"] = self.avg_cpu_percent
        d["cpu_peak_pct"] = self.peak_cpu_percent
        d["ram_avg_mb"] = self.avg_ram_mb
        d["ram_peak_mb"] = self.peak_ram_mb
        d["duration_ms"] = round(self.duration_sec * 1000.0, 1)
        return d


class TelemetrySession:
    """Manages telemetry collection across all phases of an autonomous optimization job."""

    def __init__(self, job_id: str, job_dir: Optional[str] = None):
        self.job_id = job_id
        self.job_dir = job_dir
        self.environment_info = get_host_environment_info()
        self.phases: Dict[str, PhaseTelemetry] = {}
        self.execution_samples: List[Dict[str, Any]] = []
        self.benchmark_provenance: Optional[Dict[str, Any]] = None
        self.start_time = time.time()
        self.protocol = {
            "sample_interval_ms": 50,
            "warmup_runs": 10,
            "measurement_runs": 1000,
            "phases_demarcated": [
                "BASELINE_START",
                "BASELINE_WARMUP",
                "BASELINE_MEASUREMENT",
                "BASELINE_END",
                "OPTIMIZED_START",
                "OPTIMIZED_WARMUP",
                "OPTIMIZED_MEASUREMENT",
                "OPTIMIZED_END"
            ]
        }

    def add_execution_sample(self, sample: Dict[str, Any]) -> None:
        """Add a continuous execution sample to the session."""
        self.execution_samples.append(sample)

    def record_phase(
        self,
        phase_id: str,
        phase_name: str,
        model_state: str,
        candidate_id: Optional[str],
        latency_mean_ms: float,
        latency_median_ms: float,
        latency_p95_ms: float,
        throughput_ips: float,
        batch_size: int,
        warmup_runs: int,
        measured_runs: int,
        monitor_summary: Dict[str, Any],
        samples: Optional[List[Dict[str, Any]]] = None
    ) -> PhaseTelemetry:
        """Record telemetry for an evaluated phase."""
        record = PhaseTelemetry(
            phase_id=phase_id,
            phase_name=phase_name,
            model_state=model_state,
            candidate_id=candidate_id,
            latency_mean_ms=round(latency_mean_ms, 3),
            latency_median_ms=round(latency_median_ms, 3),
            latency_p95_ms=round(latency_p95_ms, 3),
            throughput_ips=round(throughput_ips, 2),
            batch_size=batch_size,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            avg_cpu_percent=monitor_summary.get("avg_cpu_percent", 0.0),
            peak_cpu_percent=monitor_summary.get("peak_cpu_percent", 0.0),
            avg_ram_mb=monitor_summary.get("avg_ram_mb", 0.0),
            peak_ram_mb=monitor_summary.get("peak_ram_mb", 0.0),
            duration_sec=monitor_summary.get("duration_sec", 0.0),
            samples=samples or []
        )
        self.phases[phase_id] = record
        if self.job_dir:
            self.save_to_disk()
        return record

    def save_to_disk(self) -> str:
        """Persist session telemetry to telemetry.json in the job directory."""
        if not self.job_dir:
            return ""
        os.makedirs(self.job_dir, exist_ok=True)
        path = os.path.join(self.job_dir, "telemetry.json")
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to JSON-serializable dictionary with resource comparison."""
        baseline_phase = next((p.to_dict() for p in self.phases.values() if p.model_state == "FP32_BASELINE"), None)
        final_phase = next((p.to_dict() for p in self.phases.values() if p.model_state == "FINAL_OPTIMIZED"), None)

        comparison = None
        if baseline_phase and final_phase:
            b_cpu = baseline_phase.get("avg_cpu_percent") or 0.0
            f_cpu = final_phase.get("avg_cpu_percent") or 0.0
            cpu_chg = round(((f_cpu - b_cpu) / b_cpu) * 100.0, 2) if b_cpu > 0 else 0.0

            b_ram = baseline_phase.get("avg_ram_mb") or 0.0
            f_ram = final_phase.get("avg_ram_mb") or 0.0
            ram_chg = round(((f_ram - b_ram) / b_ram) * 100.0, 2) if b_ram > 0 else 0.0

            b_lat = baseline_phase.get("latency_mean_ms") or 0.0
            f_lat = final_phase.get("latency_mean_ms") or 0.0
            # Canonical: ((baseline - optimized) / baseline) * 100
            lat_chg = round(((b_lat - f_lat) / b_lat) * 100.0, 2) if b_lat > 0 else 0.0

            b_tput = baseline_phase.get("throughput_ips") or 0.0
            f_tput = final_phase.get("throughput_ips") or 0.0
            tput_chg = round(((f_tput - b_tput) / b_tput) * 100.0, 2) if b_tput > 0 else 0.0

            comparison = {
                "cpu_change_percent": cpu_chg,
                "ram_change_percent": ram_chg,
                "latency_change_percent": lat_chg,
                "throughput_change_percent": tput_chg,
            }

        return {
            "job_id": self.job_id,
            "environment_info": self.environment_info,
            "target_hardware_status": "PENDING",
            "host_validation_status": "MEASURED",
            "start_time": self.start_time,
            "protocol": self.protocol,
            "benchmark_provenance": self.benchmark_provenance,
            "baseline": baseline_phase,
            "final": final_phase,
            "comparison": comparison,
            "phases": {k: v.to_dict() for k, v in self.phases.items()},
            "execution_samples": self.execution_samples
        }
