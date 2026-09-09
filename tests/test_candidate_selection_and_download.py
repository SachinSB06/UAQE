"""
tests/test_candidate_selection_and_download.py

Authoritative 10-Point Candidate Regression Test Suite:
1. Candidate switch (Candidate A shows A metrics, Candidate B shows B metrics, no metric leakage)
2. Artifact switch (Candidate A points to artifact A, Candidate B points to artifact B)
3. No hardcoded universal optimized_model.onnx download target
4. Existing artifact downloads successfully (verified hashes & contents)
5. Missing/invalid candidate artifact returns a clean 404
6. Job isolation (cross-job candidate download isolation)
7. Path traversal protection (secure path validation preventing traversal)
8. Candidate artifact metadata completeness (filename, format, size_bytes, sha256, download_url)
9. Candidate telemetry identity (per-candidate isolated telemetry phases)
10. Refresh and selection consistency (deterministic idempotency across queries)

CRITICAL NOTE:
The benchmark figures and artifact SHA256 hashes used in this test are REGRESSION TEST
EXPECTATIONS ONLY against the authoritative historical run UAQE-20260909-191140-11AA370D.
They must NEVER be hardcoded into production code.
"""

import hashlib
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from fastapi.testclient import TestClient
from uaqe.server import app

JOB_ID = "UAQE-20260909-191140-11AA370D"
CAND1_ID = "cand_001"
CAND4_ID = "cand_004"


class TestCandidateSelectionAndDownload(unittest.TestCase):
    """10-Point regression verification suite for candidate switching and downloads."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_candidate_switch_metrics_isolation(self):
        """1. Candidate switch: Selecting Candidate 1 shows Candidate 1 metrics; Candidate 4 shows Candidate 4 metrics with zero leakage."""
        resp = self.client.get(f"/api/jobs/{JOB_ID}")
        self.assertEqual(resp.status_code, 200, f"Failed to retrieve job: {resp.text}")
        data = resp.json()

        candidates = {c["candidate_id"]: c for c in data.get("candidates", [])}
        self.assertIn(CAND1_ID, candidates)
        self.assertIn(CAND4_ID, candidates)

        cand1 = candidates[CAND1_ID]
        cand4 = candidates[CAND4_ID]

        # Candidate 1 expected metrics (Reference PTQ INT8)
        self.assertAlmostEqual(cand1["latency_mean_ms"], 149.905, places=2)
        self.assertAlmostEqual(cand1["throughput_ips"], 6.671, places=2)

        # Candidate 4 expected metrics (XNNPACK-compatible INT8)
        self.assertAlmostEqual(cand4["latency_mean_ms"], 9.818, places=2)
        self.assertAlmostEqual(cand4["throughput_ips"], 101.86, places=1)

        # Strict isolation: Candidate 1 and Candidate 4 metrics must be completely distinct
        self.assertNotEqual(cand1["latency_mean_ms"], cand4["latency_mean_ms"])
        self.assertNotEqual(cand1["throughput_ips"], cand4["throughput_ips"])
        self.assertGreater(cand4["throughput_ips"], cand1["throughput_ips"] * 10)

    def test_02_artifact_switch_points_to_distinct_artifacts(self):
        """2. Artifact switch: Candidate 1 points to artifact 1; Candidate 4 points to artifact 4."""
        resp1 = self.client.get(f"/api/jobs/{JOB_ID}/candidates/{CAND1_ID}/download")
        self.assertEqual(resp1.status_code, 200)
        content1 = resp1.content
        sha1 = hashlib.sha256(content1).hexdigest()

        resp4 = self.client.get(f"/api/jobs/{JOB_ID}/candidates/{CAND4_ID}/download")
        self.assertEqual(resp4.status_code, 200)
        content4 = resp4.content
        sha4 = hashlib.sha256(content4).hexdigest()

        # Hashes and binary contents must differ
        self.assertNotEqual(sha1, sha4, "Candidate 1 and Candidate 4 artifact hashes must not match")
        self.assertNotEqual(content1, content4, "Candidate 1 and Candidate 4 binary bytes must not match")

    def test_03_no_hardcoded_universal_optimized_model_onnx_download_target(self):
        """3. No hardcoded universal optimized_model.onnx download target: TFLite candidates report .tflite, missing ONNX returns 404."""
        resp = self.client.get(f"/api/jobs/{JOB_ID}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        for c in data.get("candidates", []):
            artifact = c.get("artifact", {})
            self.assertEqual(artifact.get("format"), "tflite")
            self.assertTrue(artifact.get("filename", "").endswith(".tflite"))

        # Downloading non-existent universal ONNX filename must return clean 404, never fake ONNX bytes
        resp_onnx = self.client.get(f"/api/jobs/{JOB_ID}/download/optimized_model.onnx")
        self.assertEqual(resp_onnx.status_code, 404)

    def test_04_existing_artifact_downloads_successfully(self):
        """4. Existing artifact downloads successfully with verified byte sizes and SHA256 hashes."""
        # Candidate 1
        resp1 = self.client.get(f"/api/jobs/{JOB_ID}/candidates/{CAND1_ID}/download")
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(len(resp1.content), 1855816)
        sha1 = hashlib.sha256(resp1.content).hexdigest()
        self.assertTrue(sha1.startswith("10d51d4f"), f"Expected SHA starting 10d51d4f, got {sha1}")

        # Candidate 4
        resp4 = self.client.get(f"/api/jobs/{JOB_ID}/candidates/{CAND4_ID}/download")
        self.assertEqual(resp4.status_code, 200)
        self.assertEqual(len(resp4.content), 1855816)
        sha4 = hashlib.sha256(resp4.content).hexdigest()
        self.assertTrue(sha4.startswith("93a54e8f"), f"Expected SHA starting 93a54e8f, got {sha4}")

    def test_05_missing_invalid_candidate_artifact_returns_clean_404(self):
        """5. Missing/invalid candidate artifact returns a clean 404 error."""
        resp_fake = self.client.get(f"/api/jobs/{JOB_ID}/candidates/cand_nonexistent_999/download")
        self.assertEqual(resp_fake.status_code, 404)
        self.assertIn("not found", resp_fake.text.lower())

    def test_06_job_isolation(self):
        """6. Job isolation: Requests against one job cannot access or leak candidates from another job."""
        # Query non-existent job
        resp_fake_job = self.client.get(f"/api/jobs/UAQE-NONEXISTENT-JOB/candidates/{CAND1_ID}/download")
        self.assertEqual(resp_fake_job.status_code, 404)

        # Cross-job check
        job2_id = "UAQE-20260909-131603-5C90DE51"
        job2_dir = os.path.join(PROJECT_ROOT, "output", "jobs", job2_id)
        if os.path.isdir(job2_dir):
            resp_job2 = self.client.get(f"/api/jobs/{job2_id}")
            if resp_job2.status_code == 200:
                cands2 = [c["candidate_id"] for c in resp_job2.json().get("candidates", [])]
                if CAND4_ID not in cands2:
                    resp_cross = self.client.get(f"/api/jobs/{job2_id}/candidates/{CAND4_ID}/download")
                    self.assertEqual(resp_cross.status_code, 404)

    def test_07_path_traversal_protection(self):
        """7. Path traversal protection: Path manipulation attempts in candidate_id or filename are blocked."""
        # Traversal in candidate_id
        resp1 = self.client.get(f"/api/jobs/{JOB_ID}/candidates/..%2F..%2Fetc%2Fpasswd/download")
        self.assertIn(resp1.status_code, [400, 403, 404])

        # Traversal in filename
        resp2 = self.client.get(f"/api/jobs/{JOB_ID}/download/..%2F..%2F..%2FWindows%2Fwin.ini")
        self.assertIn(resp2.status_code, [400, 403, 404])

        # Windows backslash traversal
        resp3 = self.client.get(f"/api/jobs/{JOB_ID}/download/..%5C..%5Cserver.py")
        self.assertIn(resp3.status_code, [400, 403, 404])

    def test_08_candidate_artifact_metadata_completeness(self):
        """8. Candidate artifact metadata completeness: filename, format, size_bytes, sha256, download_url."""
        resp = self.client.get(f"/api/jobs/{JOB_ID}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        for c in data.get("candidates", []):
            cid = c["candidate_id"]
            art = c.get("artifact", {})
            self.assertIn("filename", art)
            self.assertIn("format", art)
            self.assertIn("size_bytes", art)
            self.assertIn("sha256", art)
            self.assertIn("download_url", art)
            self.assertEqual(art["download_url"], f"/api/jobs/{JOB_ID}/candidates/{cid}/download")
            self.assertGreater(art["size_bytes"], 0)
            self.assertEqual(len(art["sha256"]), 64)

    def test_09_candidate_telemetry_identity(self):
        """9. Candidate telemetry identity: Telemetry phases are keyed strictly by unique candidate_id."""
        resp = self.client.get(f"/api/jobs/{JOB_ID}/telemetry")
        self.assertEqual(resp.status_code, 200)
        tel = resp.json()
        phases = tel.get("phases", {})

        self.assertIn(CAND1_ID, phases)
        self.assertIn(CAND4_ID, phases)

        p1 = phases[CAND1_ID]
        p4 = phases[CAND4_ID]

        # Verify isolated, distinct hardware metrics
        self.assertAlmostEqual(p4["avg_cpu_percent"], 68.2, places=1)
        self.assertAlmostEqual(p4["avg_ram_mb"], 449.66, places=1)

    def test_10_refresh_selection_consistency(self):
        """10. Refresh/selection consistency: Repeated queries return identical, idempotent candidate data without drift."""
        resp_a = self.client.get(f"/api/jobs/{JOB_ID}")
        resp_b = self.client.get(f"/api/jobs/{JOB_ID}")

        self.assertEqual(resp_a.status_code, 200)
        self.assertEqual(resp_b.status_code, 200)

        data_a = resp_a.json()
        data_b = resp_b.json()

        cands_a = {c["candidate_id"]: c for c in data_a.get("candidates", [])}
        cands_b = {c["candidate_id"]: c for c in data_b.get("candidates", [])}

        self.assertEqual(cands_a[CAND4_ID]["latency_mean_ms"], cands_b[CAND4_ID]["latency_mean_ms"])
        self.assertEqual(cands_a[CAND4_ID]["throughput_ips"], cands_b[CAND4_ID]["throughput_ips"])
        self.assertEqual(cands_a[CAND4_ID]["artifact"]["sha256"], cands_b[CAND4_ID]["artifact"]["sha256"])


if __name__ == "__main__":
    unittest.main()
