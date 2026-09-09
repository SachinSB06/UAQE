"""Tests for UAQE FastAPI Backend API Server and Telemetry Endpoints."""

import os
import sys
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure src is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from uaqe.server import app


class TestUAQEApiServer(unittest.TestCase):
    """Test suite for FastAPI backend API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_get_system_status(self):
        """Verify /api/status returns READY and hardware targets."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "READY")
        self.assertIn("pytorch_checkpoint", data["supported_model_formats"])
        self.assertIn("balanced", data["supported_profiles"])
        self.assertTrue(len(data["supported_targets"]) > 0)
        self.assertIn("os", data["host_environment"])

    def test_list_jobs(self):
        """Verify /api/jobs returns a list of historical jobs."""
        response = self.client.get("/api/jobs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        if len(data) > 0:
            first = data[0]
            self.assertIn("job_id", first)
            self.assertIn("status", first)
            self.assertIn("verdict", first)

    def test_get_job_detail_existing(self):
        """Verify /api/jobs/{id} returns full job details with provenance metadata."""
        # Find an existing job directory
        jobs_response = self.client.get("/api/jobs")
        self.assertEqual(jobs_response.status_code, 200)
        jobs = jobs_response.json()
        if not jobs:
            self.skipTest("No jobs found in output/jobs/ directory.")

        job_id = jobs[0]["job_id"]
        response = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(response.status_code, 200)
        detail = response.json()
        self.assertEqual(detail["job_id"], job_id)
        self.assertIn("model_inspection", detail)
        self.assertIn("dataset_inspection", detail)
        self.assertIn("optimization_plan", detail)
        self.assertIn("metrics", detail)
        self.assertIn("candidates", detail)
        self.assertEqual(detail["target_hardware_status"], "PENDING")
        self.assertEqual(detail["host_telemetry_status"], "MEASURED")

        # Verify provenance label on metrics
        if "fp32_accuracy" in detail["metrics"]:
            fp32_metric = detail["metrics"]["fp32_accuracy"]
            self.assertIn("source", fp32_metric)
            self.assertIn("value", fp32_metric)

    def test_get_job_telemetry(self):
        """Verify /api/jobs/{id}/telemetry returns telemetry data with environment metadata."""
        jobs_response = self.client.get("/api/jobs")
        jobs = jobs_response.json()
        if not jobs:
            self.skipTest("No jobs found in output/jobs/ directory.")

        job_id = jobs[0]["job_id"]
        response = self.client.get(f"/api/jobs/{job_id}/telemetry")
        self.assertEqual(response.status_code, 200)
        telemetry = response.json()
        self.assertEqual(telemetry["job_id"], job_id)
        self.assertIn("environment_info", telemetry)
        self.assertEqual(telemetry["target_hardware_status"], "PENDING")
        self.assertEqual(telemetry["host_validation_status"], "MEASURED")

    def test_get_verified_samples(self):
        """Verify /api/uploads/samples returns pre-verified model & dataset sample paths."""
        response = self.client.get("/api/uploads/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("datasets", data)

    def test_analyze_model_and_dataset(self):
        """Verify /api/jobs/analyze performs fresh model & dataset inspection."""
        model_p = "output/phase_e2/models/resnet50_cifar10_fp32_baseline.pt"
        dataset_p = "D:\\uaqe_datasets\\cifar10"
        if not os.path.exists(model_p) or not os.path.exists(dataset_p):
            self.skipTest("Required benchmark files missing on host.")

        response = self.client.post("/api/jobs/analyze", json={
            "model_path": model_p,
            "dataset_path": dataset_p,
            "target": "raspberrypi5",
            "profile": "balanced"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_inspection", data)
        self.assertIn("dataset_inspection", data)
        self.assertIn("compatibility", data)
        self.assertEqual(data["model_inspection"]["framework"], "PyTorch")
        self.assertIn("cifar", data["dataset_inspection"]["dataset_name"].lower())
        self.assertTrue(data["compatibility"]["compatible"])

    @patch("uaqe.api.routes_jobs._run_optimization_worker")
    def test_unique_job_creation_and_isolation(self, mock_worker):
        """Verify POST /api/jobs/optimize always generates distinct, unique job IDs."""
        model_p = "output/phase_e2/models/resnet50_cifar10_fp32_baseline.pt"
        dataset_p = "D:\\uaqe_datasets\\cifar10"
        if not os.path.exists(model_p) or not os.path.exists(dataset_p):
            self.skipTest("Required benchmark files missing on host.")

        # Request 1
        resp1 = self.client.post("/api/jobs/optimize", json={
            "model_path": model_p,
            "dataset_path": dataset_p,
            "target": "raspberrypi5",
            "profile": "balanced"
        })
        self.assertEqual(resp1.status_code, 200)
        job1_id = resp1.json()["job_id"]

        # Request 2
        resp2 = self.client.post("/api/jobs/optimize", json={
            "model_path": model_p,
            "dataset_path": dataset_p,
            "target": "raspberrypi5",
            "profile": "balanced"
        })
        self.assertEqual(resp2.status_code, 200)
        job2_id = resp2.json()["job_id"]

        # Assert job IDs are strictly distinct and follow naming convention
        self.assertNotEqual(job1_id, job2_id)
        self.assertTrue(job1_id.startswith("UAQE-"))
        self.assertTrue(job2_id.startswith("UAQE-"))

    def test_get_nonexistent_job_returns_404(self):
        """Verify nonexistent job ID returns 404 error."""
        response = self.client.get("/api/jobs/UAQE-NONEXISTENT-999999")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
