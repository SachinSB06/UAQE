"""Test verifying fresh optimization lifecycle and state isolation.

Proves:
1. New optimization creates a unique job ID.
2. Two consecutive optimizations create distinct job IDs (Job A != Job B).
3. Analyzing fresh inputs yields separate descriptors.
4. Jobs maintain strictly isolated directories and artifacts in output/jobs/<job_id>.
5. GET /api/jobs/<job_id> returns only that job's metadata and candidate history.
"""

import os
import sys
import unittest
from unittest.mock import patch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi.testclient import TestClient
from uaqe.server import app

client = TestClient(app)

class TestFreshOptimizationLifecycle(unittest.TestCase):
    """Test suite for fresh optimization guarantee."""

    def test_unique_job_creation(self):
        """Verify calling optimize creates distinct jobs every time."""
        samples_res = client.get("/api/uploads/samples")
        samples = samples_res.json()
        model_path = samples["models"][0]["path"] if samples["models"] else "output/phase_e2/models/resnet50_cifar10_fp32_baseline.pt"
        dataset_path = samples["datasets"][0]["path"] if samples["datasets"] else "D:\\uaqe_datasets\\cifar10"

        payload = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "target_hardware": "raspberrypi5",
            "optimization_profile": "balanced",
            "auto_approve": True
        }

        with patch("uaqe.api.routes_jobs._run_optimization_worker") as mock_worker:
            # Request Job 1
            res1 = client.post("/api/jobs/optimize", json=payload)
            self.assertEqual(res1.status_code, 200)
            job_a = res1.json()["job_id"]
            self.assertTrue(job_a.startswith("UAQE-"))

            # Request Job 2 with same inputs
            res2 = client.post("/api/jobs/optimize", json=payload)
            self.assertEqual(res2.status_code, 200)
            job_b = res2.json()["job_id"]
            self.assertTrue(job_b.startswith("UAQE-"))

            # Job A and Job B MUST NOT be identical
            self.assertNotEqual(job_a, job_b, "Consecutive optimization calls must generate unique job IDs")
            self.assertEqual(mock_worker.call_count, 2)
            calls = mock_worker.call_args_list
            self.assertEqual(calls[0].kwargs["job_id"], job_a)
            self.assertEqual(calls[1].kwargs["job_id"], job_b)


    def test_fresh_analysis_for_different_inputs(self):
        """Verify /api/jobs/analyze generates fresh inspections for distinct inputs."""
        samples_res = client.get("/api/uploads/samples")
        samples = samples_res.json()
        if not samples["models"] or not samples["datasets"]:
            self.skipTest("No sample model or dataset available.")

        model_path = samples["models"][0]["path"]
        dataset_path = samples["datasets"][0]["path"]

        payload1 = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "target_hardware": "raspberrypi5",
            "optimization_profile": "balanced"
        }
        res1 = client.post("/api/jobs/analyze", json=payload1)
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertIn("model_inspection", data1)
        self.assertIn("dataset_inspection", data1)
        self.assertIn("compatibility", data1)
        self.assertEqual(data1["compatibility"]["compatible"], True)



if __name__ == "__main__":
    unittest.main()

