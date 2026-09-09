"""Tests for UAQE Telemetry, Runtime Monitor, and Telemetry Session."""

import os
import sys
import time
import json
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from uaqe.telemetry.process_metrics import get_host_environment_info, sample_current_metrics
from uaqe.telemetry.runtime_monitor import RuntimeMonitor
from uaqe.telemetry.telemetry_session import TelemetrySession


class TestUAQETelemetry(unittest.TestCase):
    """Test suite for UAQE telemetry collection subsystem."""

    def test_get_host_environment_info(self):
        """Verify host environment hardware info is genuine and non-empty."""
        info = get_host_environment_info()
        self.assertIn("os", info)
        self.assertIn("architecture", info)
        self.assertIn("cpu_count_logical", info)
        self.assertGreater(info["cpu_count_logical"], 0)
        self.assertEqual(info["environment"], "HOST")

    def test_sample_current_metrics(self):
        """Verify instantaneous CPU and RAM metrics can be sampled."""
        sample = sample_current_metrics()
        self.assertIn("timestamp", sample)
        self.assertIn("process_ram_mb", sample)
        self.assertIn("system_cpu_percent", sample)
        self.assertGreater(sample["timestamp"], 0)

    def test_runtime_monitor_lifecycle(self):
        """Verify RuntimeMonitor background thread starts, collects samples, and stops."""
        monitor = RuntimeMonitor(sample_interval_sec=0.02)
        monitor.start()
        time.sleep(0.1)
        samples = monitor.stop()
        self.assertGreater(len(samples), 0)
        summary = monitor.get_summary()
        self.assertIn("sample_count", summary)
        self.assertGreater(summary["sample_count"], 0)
        self.assertIn("avg_cpu_percent", summary)
        self.assertIn("peak_ram_mb", summary)

    def test_telemetry_session_persistence(self):
        """Verify TelemetrySession records phases and serializes telemetry.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TelemetrySession(job_id="TEST-JOB-001", job_dir=tmpdir)
            session.record_phase(
                phase_id="fp32_baseline",
                phase_name="FP32 Reference Baseline",
                model_state="FP32_BASELINE",
                candidate_id=None,
                latency_mean_ms=100.0,
                latency_median_ms=98.0,
                latency_p95_ms=110.0,
                throughput_ips=10.0,
                batch_size=1,
                warmup_runs=5,
                measured_runs=100,
                monitor_summary={"avg_cpu_percent": 15.0, "peak_cpu_percent": 25.0, "avg_ram_mb": 300.0, "peak_ram_mb": 350.0, "duration_sec": 1.0}
            )
            saved_path = session.save_to_disk()
            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertEqual(data["job_id"], "TEST-JOB-001")
                self.assertEqual(data["target_hardware_status"], "PENDING")
                self.assertIsNotNone(data["baseline"])


if __name__ == "__main__":
    unittest.main()
