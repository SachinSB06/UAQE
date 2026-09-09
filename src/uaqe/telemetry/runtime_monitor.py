"""Continuous Runtime Telemetry Monitor for UAQE Benchmark Executions.

Samples CPU and RAM utilization at periodic intervals using a lightweight background daemon thread.
Supports live sample callbacks for SSE streaming, benchmark phase tagging, and statistical aggregation.
"""

from __future__ import annotations

import os
import time
import threading
from typing import List, Dict, Any, Optional, Callable
import psutil

from .process_metrics import sample_current_metrics

_ACTIVE_MONITORS: List["RuntimeMonitor"] = []
_ACTIVE_MONITORS_LOCK = threading.Lock()


def pause_active_monitors() -> None:
    """Pause all running monitors to ensure zero telemetry interference during benchmarking."""
    with _ACTIVE_MONITORS_LOCK:
        for m in _ACTIVE_MONITORS:
            m.pause()


def resume_active_monitors() -> None:
    """Resume all running monitors after benchmarking."""
    with _ACTIVE_MONITORS_LOCK:
        for m in _ACTIVE_MONITORS:
            m.resume()


class RuntimeMonitor:
    """Background sampler collecting time-series CPU and RAM data during evaluation."""

    def __init__(
        self,
        sample_interval_sec: float = 0.1,
        sample_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        job_id: Optional[str] = None
    ):
        self.sample_interval_sec = sample_interval_sec
        self.sample_callback = sample_callback
        self.job_id = job_id
        self.samples: List[Dict[str, Any]] = []
        self._is_running = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._current_phase: str = "INITIALIZING"
        self._process: Optional[psutil.Process] = None
        try:
            self._process = psutil.Process(os.getpid())
            # Prime CPU percent measurement
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def pause(self) -> None:
        """Temporarily pause telemetry sampling."""
        with self._lock:
            self._is_paused = True

    def resume(self) -> None:
        """Resume telemetry sampling."""
        with self._lock:
            self._is_paused = False

    def is_paused(self) -> bool:
        with self._lock:
            return self._is_paused

    def set_phase(self, phase: str) -> None:
        """Tag the current benchmark or execution phase."""
        with self._lock:
            self._current_phase = phase

    def get_phase(self) -> str:
        """Return the current benchmark phase."""
        with self._lock:
            return self._current_phase

    def start(self) -> None:
        """Start background telemetry collection."""
        with self._lock:
            if self._is_running:
                return
            self.samples.clear()
            self._is_running = True
            self._is_paused = False
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        with _ACTIVE_MONITORS_LOCK:
            if self not in _ACTIVE_MONITORS:
                _ACTIVE_MONITORS.append(self)

    def stop(self) -> List[Dict[str, Any]]:
        """Stop background telemetry collection and return all recorded samples."""
        with self._lock:
            self._is_running = False
            self._is_paused = False
        with _ACTIVE_MONITORS_LOCK:
            if self in _ACTIVE_MONITORS:
                _ACTIVE_MONITORS.remove(self)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            return list(self.samples)

    def _sample_loop(self) -> None:
        while True:
            with self._lock:
                if not self._is_running:
                    break
                if self._is_paused:
                    paused = True
                else:
                    paused = False
                current_phase = self._current_phase

            if paused:
                time.sleep(0.05)
                continue

            sample = sample_current_metrics(self._process)
            sample["phase"] = current_phase
            if self.job_id:
                sample["job_id"] = self.job_id

            with self._lock:
                self.samples.append(sample)

            if self.sample_callback:
                try:
                    self.sample_callback(sample)
                except Exception:
                    pass

            time.sleep(self.sample_interval_sec)

    def get_summary(self) -> Dict[str, Any]:
        """Compute genuine statistical summary from sampled telemetry."""
        with self._lock:
            if not self.samples:
                return {
                    "sample_count": 0,
                    "avg_cpu_percent": 0.0,
                    "peak_cpu_percent": 0.0,
                    "avg_process_cpu_percent": 0.0,
                    "peak_process_cpu_percent": 0.0,
                    "avg_ram_mb": 0.0,
                    "peak_ram_mb": 0.0,
                    "avg_system_ram_percent": 0.0,
                    "duration_sec": 0.0,
                }

            sys_cpu_vals = [s.get("system_cpu_percent", 0.0) for s in self.samples]
            proc_cpu_vals = [s.get("process_cpu_percent", 0.0) for s in self.samples]
            ram_vals = [s.get("process_ram_mb", 0.0) for s in self.samples]
            sys_ram_vals = [s.get("system_ram_percent", 0.0) for s in self.samples]
            start_time = self.samples[0]["timestamp"]
            end_time = self.samples[-1]["timestamp"]

            return {
                "sample_count": len(self.samples),
                "avg_cpu_percent": round(sum(sys_cpu_vals) / len(sys_cpu_vals), 1) if sys_cpu_vals else 0.0,
                "peak_cpu_percent": round(max(sys_cpu_vals), 1) if sys_cpu_vals else 0.0,
                "avg_process_cpu_percent": round(sum(proc_cpu_vals) / len(proc_cpu_vals), 1) if proc_cpu_vals else 0.0,
                "peak_process_cpu_percent": round(max(proc_cpu_vals), 1) if proc_cpu_vals else 0.0,
                "avg_ram_mb": round(sum(ram_vals) / len(ram_vals), 2) if ram_vals else 0.0,
                "peak_ram_mb": round(max(ram_vals), 2) if ram_vals else 0.0,
                "avg_system_ram_percent": round(sum(sys_ram_vals) / len(sys_ram_vals), 1) if sys_ram_vals else 0.0,
                "duration_sec": round(end_time - start_time, 3),
            }
