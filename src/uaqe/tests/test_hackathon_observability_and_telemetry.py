"""Comprehensive Test Suite for UAQE Real-Time Telemetry, Observability, and Comparison.

Validates genuine host CPU/RAM collection, live sample dispatch, baseline vs optimized
measurement protocols, telemetry persistence, zero dummy values, and two-job isolation.
"""

import os
import json
import time
import shutil
import tempfile
import unittest

from uaqe.telemetry.process_metrics import (
    get_host_environment_info,
    sample_current_metrics,
    reset_peak_process_ram,
    get_peak_process_ram_mb,
)
from uaqe.telemetry.runtime_monitor import RuntimeMonitor
from uaqe.telemetry.telemetry_session import TelemetrySession
from uaqe.optimization.accuracy_safety_policy import (
    AccuracySafetyPolicy,
    BaselineStatus,
    AccuracyClassification,
)


class TestHackathonObservabilityAndTelemetry(unittest.TestCase):
    """Test suite for hackathon observability and real-time telemetry."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="uaqe_test_tel_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_real_cpu_and_ram_sampling(self):
        """Verify that process and system CPU/RAM metrics are genuinely sampled without fabrication."""
        info = get_host_environment_info()
        self.assertIn("os", info)
        self.assertIn("cpu_count_logical", info)
        self.assertGreaterEqual(info["cpu_count_logical"], 1)
        self.assertGreater(info["total_ram_mb"], 0)

        # Instantaneous sample
        sample = sample_current_metrics()
        self.assertIn("timestamp", sample)
        self.assertIn("system_cpu_percent", sample)
        self.assertIn("process_cpu_percent", sample)
        self.assertIn("process_ram_mb", sample)
        self.assertIn("system_ram_percent", sample)
        self.assertIn("system_ram_used_mb", sample)
        self.assertIn("system_ram_available_mb", sample)

        # Genuine values are within legitimate hardware ranges
        self.assertGreaterEqual(sample["system_cpu_percent"], 0.0)
        self.assertLessEqual(sample["system_cpu_percent"], 100.0)
        self.assertGreater(sample["process_ram_mb"], 0.0)
        self.assertGreater(sample["system_ram_percent"], 0.0)
        self.assertLessEqual(sample["system_ram_percent"], 100.0)

    def test_02_peak_process_ram_tracking(self):
        """Verify peak process RAM tracks highest observed RSS."""
        reset_peak_process_ram()
        initial_peak = get_peak_process_ram_mb()
        self.assertEqual(initial_peak, 0.0)

        s1 = sample_current_metrics()
        peak_after_s1 = get_peak_process_ram_mb()
        self.assertGreater(peak_after_s1, 0.0)
        self.assertGreaterEqual(peak_after_s1, s1["process_ram_mb"])

    def test_03_runtime_monitor_live_callback_and_phases(self):
        """Verify RuntimeMonitor invokes live sample callback and tags benchmark phases."""
        dispatched_samples = []

        def on_sample(s):
            dispatched_samples.append(s)

        monitor = RuntimeMonitor(
            sample_interval_sec=0.05,
            sample_callback=on_sample,
            job_id="TEST-JOB-101"
        )
        monitor.set_phase("BASELINE_MEASUREMENT")
        self.assertEqual(monitor.get_phase(), "BASELINE_MEASUREMENT")

        monitor.start()
        time.sleep(0.25)
        monitor.set_phase("OPTIMIZED_MEASUREMENT")
        time.sleep(0.20)
        samples = monitor.stop()
        summary = monitor.get_summary()

        self.assertGreater(len(samples), 3)
        self.assertGreater(len(dispatched_samples), 3)
        self.assertEqual(samples[0]["job_id"], "TEST-JOB-101")
        self.assertEqual(dispatched_samples[0]["job_id"], "TEST-JOB-101")

        # Phases tagged correctly
        phases_seen = {s.get("phase") for s in dispatched_samples}
        self.assertIn("BASELINE_MEASUREMENT", phases_seen)

        # Summary statistics computed correctly
        self.assertIn("avg_cpu_percent", summary)
        self.assertIn("peak_cpu_percent", summary)
        self.assertIn("avg_ram_mb", summary)
        self.assertIn("peak_ram_mb", summary)
        self.assertGreater(summary["avg_ram_mb"], 0.0)
        self.assertGreaterEqual(summary["peak_ram_mb"], summary["avg_ram_mb"])

    def test_04_telemetry_session_protocol_and_comparison(self):
        """Verify TelemetrySession records structured protocol and computes exact comparison deltas."""
        session = TelemetrySession(job_id="TEST-JOB-202", job_dir=self.test_dir)

        # 1. Record baseline phase
        session.record_phase(
            phase_id="fp32_baseline",
            phase_name="FP32 Reference Baseline",
            model_state="FP32_BASELINE",
            candidate_id=None,
            latency_mean_ms=80.0,
            latency_median_ms=79.5,
            latency_p95_ms=85.0,
            throughput_ips=12.5,
            batch_size=1,
            warmup_runs=10,
            measured_runs=50,
            monitor_summary={
                "avg_cpu_percent": 60.0,
                "peak_cpu_percent": 85.0,
                "avg_ram_mb": 1500.0,
                "peak_ram_mb": 1600.0,
                "duration_sec": 4.0,
            }
        )

        # 2. Record optimized phase
        session.record_phase(
            phase_id="final_optimized",
            phase_name="Sensitivity-Aware Mixed Precision",
            model_state="FINAL_OPTIMIZED",
            candidate_id="cand_003",
            latency_mean_ms=40.0,
            latency_median_ms=39.5,
            latency_p95_ms=43.0,
            throughput_ips=25.0,
            batch_size=1,
            warmup_runs=10,
            measured_runs=50,
            monitor_summary={
                "avg_cpu_percent": 45.0,
                "peak_cpu_percent": 70.0,
                "avg_ram_mb": 1100.0,
                "peak_ram_mb": 1200.0,
                "duration_sec": 2.0,
            }
        )

        tel_dict = session.to_dict()
        comp = tel_dict.get("comparison")
        self.assertIsNotNone(comp)

        # Latency speedup: ((80 - 40) / 80) * 100 = +50.0%
        self.assertEqual(comp["latency_change_percent"], 50.0)

        # CPU change: ((45 - 60) / 60) * 100 = -25.0%
        self.assertEqual(comp["cpu_change_percent"], -25.0)

        # RAM change: ((1100 - 1500) / 1500) * 100 = -26.67%
        self.assertAlmostEqual(comp["ram_change_percent"], -26.67, places=1)

        # Throughput change: ((25 - 12.5) / 12.5) * 100 = +100.0%
        self.assertEqual(comp["throughput_change_percent"], 100.0)

    def test_05_telemetry_persistence_and_no_dummy_values(self):
        """Verify saved telemetry.json on disk contains real data and zero dummy fallback values."""
        session = TelemetrySession(job_id="TEST-JOB-303", job_dir=self.test_dir)
        session.record_phase(
            phase_id="fp32_baseline",
            phase_name="FP32 Reference Baseline",
            model_state="FP32_BASELINE",
            candidate_id=None,
            latency_mean_ms=75.0,
            latency_median_ms=74.0,
            latency_p95_ms=80.0,
            throughput_ips=13.33,
            batch_size=1,
            warmup_runs=10,
            measured_runs=50,
            monitor_summary={
                "avg_cpu_percent": 72.4,
                "peak_cpu_percent": 91.2,
                "avg_ram_mb": 1240.5,
                "peak_ram_mb": 1310.0,
                "duration_sec": 3.75,
            }
        )
        tel_path = session.save_to_disk()
        self.assertTrue(os.path.exists(tel_path))

        with open(tel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["job_id"], "TEST-JOB-303")
        base = data["baseline"]
        self.assertEqual(base["avg_cpu_percent"], 72.4)
        self.assertEqual(base["peak_cpu_percent"], 91.2)
        # Ensure no placeholder 14.2% or 24.5%
        self.assertNotEqual(base["avg_cpu_percent"], 14.2)
        self.assertNotEqual(base["peak_cpu_percent"], 24.5)

    def test_06_two_job_telemetry_isolation(self):
        """Verify Job A telemetry is completely isolated from Job B telemetry."""
        dir_a = os.path.join(self.test_dir, "JOB_A")
        dir_b = os.path.join(self.test_dir, "JOB_B")
        os.makedirs(dir_a, exist_ok=True)
        os.makedirs(dir_b, exist_ok=True)

        session_a = TelemetrySession(job_id="JOB_A", job_dir=dir_a)
        session_a.record_phase(
            phase_id="fp32_baseline",
            phase_name="Baseline A",
            model_state="FP32_BASELINE",
            candidate_id=None,
            latency_mean_ms=100.0,
            latency_median_ms=100.0,
            latency_p95_ms=105.0,
            throughput_ips=10.0,
            batch_size=1,
            warmup_runs=10,
            measured_runs=50,
            monitor_summary={"avg_cpu_percent": 50.0, "peak_cpu_percent": 60.0, "avg_ram_mb": 800.0, "peak_ram_mb": 900.0, "duration_sec": 2.0}
        )
        session_a.save_to_disk()

        session_b = TelemetrySession(job_id="JOB_B", job_dir=dir_b)
        session_b.record_phase(
            phase_id="fp32_baseline",
            phase_name="Baseline B",
            model_state="FP32_BASELINE",
            candidate_id=None,
            latency_mean_ms=30.0,
            latency_median_ms=29.0,
            latency_p95_ms=32.0,
            throughput_ips=33.3,
            batch_size=1,
            warmup_runs=10,
            measured_runs=50,
            monitor_summary={"avg_cpu_percent": 85.0, "peak_cpu_percent": 98.0, "avg_ram_mb": 1400.0, "peak_ram_mb": 1500.0, "duration_sec": 1.5}
        )
        session_b.save_to_disk()

        with open(os.path.join(dir_a, "telemetry.json")) as fa:
            data_a = json.load(fa)
        with open(os.path.join(dir_b, "telemetry.json")) as fb:
            data_b = json.load(fb)

        self.assertEqual(data_a["job_id"], "JOB_A")
        self.assertEqual(data_b["job_id"], "JOB_B")
        self.assertNotEqual(data_a["baseline"]["latency_mean_ms"], data_b["baseline"]["latency_mean_ms"])
        self.assertNotEqual(data_a["baseline"]["avg_cpu_percent"], data_b["baseline"]["avg_cpu_percent"])

    def test_07_canonical_speedup_math(self):
        """Verify positive latency change means speedup and negative means slower."""
        # 100ms -> 50ms: speedup = +50.0%
        speedup = AccuracySafetyPolicy.calculate_latency_change_pct(100.0, 50.0)
        self.assertEqual(speedup, 50.0)

        # 50ms -> 100ms: slowdown = -100.0%
        slowdown = AccuracySafetyPolicy.calculate_latency_change_pct(50.0, 100.0)
        self.assertEqual(slowdown, -100.0)


if __name__ == "__main__":
    unittest.main()
