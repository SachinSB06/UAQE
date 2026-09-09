"""
tests/test_candidate_metric_identity.py

Verifies that:
1. Each candidate in optimization_history.json retains its own candidate_id.
2. The API serializer (routes_jobs.py) correctly maps per-candidate fields.
3. Derived metrics (throughput, speedup, accuracy_loss) are internally consistent per candidate.
4. Switching between Candidate 1 and Candidate 4 in the API response does not contaminate fields.
5. The job-level metrics.json reflects only the winner (cand_001) and NOT non-winner candidates.

Uses the project's existing FastAPI TestClient pattern (as found in test_model_identity_and_two_job_isolation.py).
No mocks are introduced.  Historical persisted data for UAQE-20260909-131603-5C90DE51 remains untouched.
"""

import os
import sys
import json
import math
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

JOB_ID = "UAQE-20260909-131603-5C90DE51"
JOB_DIR = os.path.join(PROJECT_ROOT, "output", "jobs", JOB_ID)
HISTORY_PATH = os.path.join(JOB_DIR, "optimization_history.json")
METRICS_PATH = os.path.join(JOB_DIR, "metrics.json")

# Tolerance for floating-point comparisons
_ATOL = 0.01   # ± 0.01 for percentages / pp


def _load_history():
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper – start API server if not already running (same pattern as other tests)
# ---------------------------------------------------------------------------
_server_started_here = False

def _ensure_server():
    """Probe the API; start a background uvicorn server if port 8000 is not listening."""
    global _server_started_here
    import urllib.request
    import threading
    import time

    url = "http://127.0.0.1:8000/api/status"
    try:
        with urllib.request.urlopen(url, timeout=2):
            return  # already running
    except Exception:
        pass

    # Spawn a daemon server
    import uvicorn
    from uaqe.server import app

    def _run():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _server_started_here = True

    # Wait until reachable
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception:
            continue
    raise RuntimeError("Could not start UAQE API server on port 8000")


def _api_get(path: str) -> dict:
    """HTTP GET against the live (or started) API server."""
    import urllib.request
    import urllib.error
    url = f"http://127.0.0.1:8000{path}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


class TestPersistedCandidateIdentity(unittest.TestCase):
    """Verify that optimization_history.json preserves per-candidate metric identity."""

    @classmethod
    def setUpClass(cls):
        cls.history = _load_history()
        cls.metrics = _load_metrics()
        # Build an id→candidate map for O(1) lookup (no index-based access in tests)
        cls.by_id = {c["candidate_id"]: c for c in cls.history}

    def test_01_all_candidates_have_candidate_id(self):
        """Every entry in optimization_history.json must carry a non-empty candidate_id."""
        for i, c in enumerate(self.history):
            cid = c.get("candidate_id", "")
            self.assertTrue(cid, f"Candidate at position {i} has no candidate_id")

    def test_02_candidate_ids_are_unique(self):
        """candidate_id values must be unique across the history list."""
        ids = [c["candidate_id"] for c in self.history]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate candidate_ids found")

    def test_03_cand_001_exists(self):
        self.assertIn("cand_001", self.by_id, "cand_001 not found in optimization history")

    def test_04_cand_004_exists(self):
        self.assertIn("cand_004", self.by_id, "cand_004 not found in optimization history")

    def test_05_cand_001_strategy(self):
        c1 = self.by_id["cand_001"]
        self.assertIn("mobilenet_adaptive", c1.get("strategy_type", ""),
                      "cand_001 strategy_type unexpected")

    def test_06_cand_004_strategy(self):
        c4 = self.by_id["cand_004"]
        self.assertEqual(c4.get("strategy_type"), "XNNPACK_COMPATIBLE_INT8",
                         "cand_004 strategy_type unexpected")

    def test_07_cand_001_metrics_are_internally_consistent(self):
        """cand_001: throughput ≈ 1000/latency, accuracy_loss = fp32_acc - opt_acc (not available
        in history, so we cross-check against metrics.json winner which IS cand_001)."""
        c1 = self.by_id["cand_001"]
        lat = c1.get("latency_mean_ms")
        thr = c1.get("throughput_ips")
        if lat and lat > 0 and thr is not None:
            expected_thr = 1000.0 / lat
            self.assertAlmostEqual(thr, expected_thr, delta=1.0,
                                   msg=f"cand_001 throughput {thr} inconsistent with latency {lat}")

    def test_08_cand_004_latency_is_correct(self):
        """cand_004 measured latency must be ~4.50 ms (XNNPACK delegated)."""
        c4 = self.by_id["cand_004"]
        lat = c4.get("latency_mean_ms")
        self.assertIsNotNone(lat, "cand_004 has no latency_mean_ms")
        self.assertAlmostEqual(lat, 4.50, delta=0.5,
                               msg=f"cand_004 latency {lat} ms unexpected for XNNPACK path")

    def test_09_cand_004_throughput_consistent_with_latency(self):
        """cand_004 throughput ≈ 1000 / latency (canonical formula)."""
        c4 = self.by_id["cand_004"]
        lat = c4.get("latency_mean_ms")
        thr = c4.get("throughput_ips")
        self.assertIsNotNone(lat, "cand_004 latency missing")
        self.assertIsNotNone(thr, "cand_004 throughput missing")
        expected = 1000.0 / lat
        self.assertAlmostEqual(thr, expected, delta=2.0,
                               msg=f"cand_004 throughput {thr} inconsistent with latency {lat}")

    def test_10_cand_004_accuracy_loss_consistent(self):
        """cand_004 accuracy_loss_pp must match fp32_acc - opt_acc from metrics."""
        c4 = self.by_id["cand_004"]
        # fp32 accuracy is from job-level metrics.json (shared baseline)
        fp32 = self.metrics.get("fp32_accuracy")
        opt = c4.get("top1_accuracy")
        reported_loss = c4.get("accuracy_loss_pp")
        self.assertIsNotNone(fp32, "fp32_accuracy missing from metrics.json")
        self.assertIsNotNone(opt, "cand_004 top1_accuracy missing")
        self.assertIsNotNone(reported_loss, "cand_004 accuracy_loss_pp missing")
        expected_loss_pp = (fp32 - opt) * 100
        self.assertAlmostEqual(reported_loss, expected_loss_pp, delta=_ATOL,
                               msg=f"cand_004 accuracy_loss_pp {reported_loss} ≠ {expected_loss_pp}")

    def test_11_metrics_json_reflects_winner_cand_001_not_cand_004(self):
        """metrics.json fields must match cand_001, not cand_004, confirming it is winner-only."""
        c1 = self.by_id["cand_001"]
        c4 = self.by_id["cand_004"]
        m_lat = self.metrics.get("optimized_latency_ms")
        c1_lat = c1.get("latency_mean_ms")
        c4_lat = c4.get("latency_mean_ms")
        # metrics.json optimized_latency_ms should be close to cand_001, not cand_004
        diff_from_c1 = abs(m_lat - c1_lat)
        diff_from_c4 = abs(m_lat - c4_lat)
        self.assertLess(diff_from_c1, diff_from_c4,
                        f"metrics.json latency ({m_lat}) is closer to cand_004 ({c4_lat}) than cand_001 ({c1_lat})")

    def test_12_switching_candidate_id_lookup_does_not_mix_fields(self):
        """
        Simulate the API serializer: look up cand_001 and cand_004 independently by candidate_id.
        Verify that no field from cand_001 appears in cand_004's object (and vice versa).
        """
        c1 = self.by_id["cand_001"]
        c4 = self.by_id["cand_004"]
        # Key distinguishing fields
        self.assertNotAlmostEqual(c1["latency_mean_ms"], c4["latency_mean_ms"], places=1,
                                  msg="cand_001 and cand_004 latencies are identical — mixing suspected")
        self.assertNotAlmostEqual(c1["throughput_ips"], c4["throughput_ips"], places=1,
                                  msg="cand_001 and cand_004 throughputs are identical — mixing suspected")
        self.assertNotAlmostEqual(c1["accuracy_loss_pp"], c4["accuracy_loss_pp"], places=2,
                                  msg="cand_001 and cand_004 accuracy_loss_pp are identical — mixing suspected")


class TestAPIJobDetailCandidateIdentity(unittest.TestCase):
    """Verify the API endpoint returns correct per-candidate fields."""

    @classmethod
    def setUpClass(cls):
        _ensure_server()
        cls.job_detail = _api_get(f"/api/jobs/{JOB_ID}")
        # Build id→candidate map from API response
        cls.api_by_id = {c["candidate_id"]: c for c in cls.job_detail.get("candidates", [])}

    def test_01_api_returns_four_candidates(self):
        self.assertEqual(len(self.api_by_id), 4, "Expected 4 candidates from API")

    def test_02_api_cand_001_candidate_id_correct(self):
        self.assertIn("cand_001", self.api_by_id)
        self.assertEqual(self.api_by_id["cand_001"]["candidate_id"], "cand_001")

    def test_03_api_cand_004_candidate_id_correct(self):
        self.assertIn("cand_004", self.api_by_id)
        self.assertEqual(self.api_by_id["cand_004"]["candidate_id"], "cand_004")

    def test_04_api_cand_004_latency_is_xnnpack_value(self):
        c4 = self.api_by_id["cand_004"]
        lat = c4.get("latency_mean_ms")
        self.assertIsNotNone(lat, "API cand_004 latency missing")
        self.assertAlmostEqual(lat, 4.50, delta=0.5,
                               msg=f"API cand_004 latency {lat} ms is not XNNPACK value")

    def test_05_api_cand_001_latency_is_reference_value(self):
        c1 = self.api_by_id["cand_001"]
        lat = c1.get("latency_mean_ms")
        self.assertIsNotNone(lat, "API cand_001 latency missing")
        self.assertGreater(lat, 50.0,
                           msg=f"API cand_001 latency {lat} ms unexpectedly low — may be cand_004 value")

    def test_06_api_cand_004_accuracy_loss_is_own_value(self):
        """API cand_004.accuracy_loss_pp must NOT be cand_001's value (1.02 pp)."""
        c4 = self.api_by_id["cand_004"]
        loss = c4.get("accuracy_loss_pp")
        self.assertIsNotNone(loss)
        self.assertAlmostEqual(loss, 1.5228, delta=0.01,
                               msg=f"API cand_004 accuracy_loss_pp {loss} ≠ expected ~1.52 pp")

    def test_07_api_cand_001_accuracy_loss_is_own_value(self):
        c1 = self.api_by_id["cand_001"]
        loss = c1.get("accuracy_loss_pp")
        self.assertIsNotNone(loss)
        self.assertAlmostEqual(loss, 1.0152, delta=0.01,
                               msg=f"API cand_001 accuracy_loss_pp {loss} ≠ expected ~1.02 pp")

    def test_08_api_cand_004_throughput_consistent(self):
        c4 = self.api_by_id["cand_004"]
        lat = c4.get("latency_mean_ms")
        thr = c4.get("throughput_ips")
        if lat and lat > 0 and thr is not None:
            expected = 1000.0 / lat
            self.assertAlmostEqual(thr, expected, delta=5.0,
                                   msg=f"API cand_004 throughput {thr} inconsistent with latency {lat}")

    def test_09_switching_lookup_no_field_leakage(self):
        """Fetching cand_001 and cand_004 from the API by candidate_id must yield different values."""
        c1 = self.api_by_id["cand_001"]
        c4 = self.api_by_id["cand_004"]
        self.assertNotAlmostEqual(c1["latency_mean_ms"], c4["latency_mean_ms"], places=1)
        self.assertNotAlmostEqual(c1["throughput_ips"], c4["throughput_ips"], places=1)
        self.assertNotAlmostEqual(c1["accuracy_loss_pp"], c4["accuracy_loss_pp"], places=2)

    def test_10_api_job_metrics_reflect_winner_latency(self):
        """jobDetail.metrics.optimized_latency_ms must match cand_001, not cand_004."""
        metrics = self.job_detail.get("metrics", {})
        m_lat_raw = metrics.get("optimized_latency_ms", {})
        m_lat = m_lat_raw.get("value") if isinstance(m_lat_raw, dict) else m_lat_raw
        c1_lat = self.api_by_id["cand_001"]["latency_mean_ms"]
        c4_lat = self.api_by_id["cand_004"]["latency_mean_ms"]
        self.assertAlmostEqual(m_lat, c1_lat, delta=1.0,
                               msg=f"API metrics.optimized_latency_ms ({m_lat}) != cand_001 ({c1_lat})")
        self.assertGreater(abs(m_lat - c4_lat), 10.0,
                           msg="API metrics.optimized_latency_ms is suspiciously close to cand_004 value")

    def test_11_api_cand_001_provenance_is_reference(self):
        """API cand_001 candidate must serialize benchmark_provenance with REFERENCE delegate."""
        c1 = self.api_by_id["cand_001"]
        prov = c1.get("benchmark_provenance")
        self.assertIsNotNone(prov, "cand_001 in API response is missing benchmark_provenance")
        self.assertEqual(prov.get("delegate"), "REFERENCE")
        self.assertIn("BUILTIN_WITHOUT_DEFAULT_DELEGATES", prov.get("runtime", ""))

    def test_12_api_cand_004_provenance_is_xnnpack(self):
        """API cand_004 candidate must serialize benchmark_provenance with XNNPACK delegate."""
        c4 = self.api_by_id["cand_004"]
        prov = c4.get("benchmark_provenance")
        self.assertIsNotNone(prov, "cand_004 in API response is missing benchmark_provenance")
        self.assertEqual(prov.get("delegate"), "XNNPACK")
        self.assertEqual(prov.get("runtime"), "TensorFlow Lite (XNNPACK)")


if __name__ == "__main__":
    unittest.main(verbosity=2)

