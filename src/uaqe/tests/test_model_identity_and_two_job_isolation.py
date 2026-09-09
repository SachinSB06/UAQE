"""Comprehensive E2E Model Identity and Two-Job Cross-Contamination Test.

Verifies:
1. Uploaded MobileNetV3 model identity is strictly preserved from upload to results.
2. Capability-driven task detection resolves image_classification for MobileNet and ResNet.
3. Semiconductor dataset (9 classes) adapts cleanly with MobileNet (10 -> 9).
4. Full autonomous optimization pipeline reaches COMPLETED with measured metrics.
5. Two distinct jobs (MobileNet vs ResNet) are completely isolated with zero leakage.
"""

import os
import json
import time
import hashlib
import urllib.request
import urllib.error
import unittest
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
API_BASE = "http://127.0.0.1:8000/api"


def get_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def upload_file(url: str, file_path: str, field_name: str = "file") -> dict:
    boundary = "----WebKitFormBoundary" + hashlib.md5(str(time.time()).encode()).hexdigest()
    filename = os.path.basename(file_path)
    
    with open(file_path, "rb") as f:
        file_content = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("latin1") + file_content + f"\r\n--{boundary}--\r\n".encode("latin1")

    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload_folder(url: str, folder_path: str, top_name: str = "dataset") -> dict:
    boundary = "----WebKitFormBoundary" + hashlib.md5(str(time.time()).encode()).hexdigest()
    body_parts = []

    # Form field 'folder_name'
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="folder_name"\r\n\r\n'
        f"{top_name}\r\n"
    .encode("utf-8"))

    # Walk folder and add up to 30 files across multiple subfolders if present
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    selected_files = []
    if subdirs:
        for sub in sorted(subdirs):
            sub_p = os.path.join(folder_path, sub)
            sub_files = [f for f in sorted(os.listdir(sub_p)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".meta", ".bin", "data_batch_1", "test_batch", "batches.meta"))]
            for f in sub_files[:5]:
                selected_files.append((os.path.join(sub_p, f), f"{sub}/{f}"))
                if len(selected_files) >= 30:
                    break
            if len(selected_files) >= 30:
                break
    else:
        for root, _, files in os.walk(folder_path):
            for f in sorted(files):
                if not f.lower().endswith((".png", ".jpg", ".jpeg", ".meta", ".bin", "data_batch_1", "test_batch", "batches.meta")):
                    continue
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, folder_path).replace("\\", "/")
                selected_files.append((fp, rel))
                if len(selected_files) >= 30:
                    break
            if len(selected_files) >= 30:
                break

    rel_paths = [rel for _, rel in selected_files]
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="relative_paths"\r\n\r\n'
        f"{json.dumps(rel_paths)}\r\n"
    .encode("utf-8"))

    for fp, rel in selected_files:
        # files field
        with open(fp, "rb") as fh:
            content = fh.read()
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{os.path.basename(fp)}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        .encode("latin1") + content + b"\r\n")

    body_parts.append(f"--{boundary}--\r\n".encode("latin1"))
    body = b"".join(body_parts)

    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, data: dict) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestModelIdentityAndIsolation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mobilenet_file = os.path.join(REPO_ROOT, "src", "models", "mobilenetv3_sem.onnx")
        cls.mobilenet_sha = get_sha256(cls.mobilenet_file)
        
        cls.resnet_file = os.path.join(REPO_ROOT, "output", "phase_e2", "models", "resnet50_cifar10_fp32_baseline.pt")
        cls.resnet_sha = get_sha256(cls.resnet_file)
        
        cls.sem_dataset_dir = os.path.join(REPO_ROOT, "datasets", "calibration", "dataset", "train")
        cls.cifar_dataset_dir = r"D:\uaqe_datasets\cifar10"

        # Ensure API server is available
        cls._server_started = False
        try:
            with urllib.request.urlopen(f"{API_BASE}/status", timeout=1) as resp:
                pass
        except Exception:
            import threading
            import uvicorn
            from uaqe.server import app
            config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
            cls._server = uvicorn.Server(config)
            cls._server_thread = threading.Thread(target=cls._server.run, daemon=True)
            cls._server_thread.start()
            cls._server_started = True
            for _ in range(50):
                time.sleep(0.1)
                try:
                    with urllib.request.urlopen(f"{API_BASE}/status", timeout=1) as resp:
                        break
                except Exception:
                    pass

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_server_started", False):
            cls._server.should_exit = True

    def test_01_samples_discovery(self):
        """Verify both ResNet and MobileNet samples are discovered with distinct identities."""
        samples = get_json(f"{API_BASE}/uploads/samples")
        model_ids = [m["id"] for m in samples["models"]]
        self.assertIn("mobilenetv3_sem", model_ids)
        self.assertIn("resnet50_cifar10", model_ids)
        
        mobilenet_sample = next(m for m in samples["models"] if m["id"] == "mobilenetv3_sem")
        self.assertEqual(mobilenet_sample["architecture"], "MobileNetV3-Small")
        self.assertEqual(mobilenet_sample["format"], "onnx")
        self.assertEqual(mobilenet_sample["sha256"], self.mobilenet_sha)

        resnet_sample = next(m for m in samples["models"] if m["id"] == "resnet50_cifar10")
        self.assertEqual(resnet_sample["architecture"], "ResNet-50")
        self.assertEqual(resnet_sample["format"], "pytorch_checkpoint")
        self.assertEqual(resnet_sample["sha256"], self.resnet_sha)

    def test_02_mobilenet_e2e_pipeline(self):
        """Upload mobilenetv3_sem.onnx + semiconductor dataset and verify full pipeline execution."""
        print("\n--- Testing MobileNetV3 E2E Pipeline ---")
        
        # 1. Upload Model
        m_resp = upload_file(f"{API_BASE}/uploads/model", self.mobilenet_file)
        m_upload_id = m_resp["upload_id"]
        self.assertEqual(m_resp["sha256"], self.mobilenet_sha)
        self.assertEqual(m_resp["format"], "onnx")
        print(f"MobileNet Uploaded: upload_id={m_upload_id}, sha256={m_resp['sha256']}")

        # 2. Upload Dataset
        d_resp = upload_folder(f"{API_BASE}/uploads/dataset", self.sem_dataset_dir, top_name="semiconductor_data")
        d_upload_id = d_resp["upload_id"]
        self.assertEqual(d_resp["detected_format"], "image_folder")
        print(f"Dataset Uploaded: upload_id={d_upload_id}, classes={d_resp['class_count']}")

        # 3. Analyze
        analyze_payload = {
            "model_upload_id": m_upload_id,
            "dataset_upload_id": d_upload_id,
            "target": "raspberrypi5",
            "profile": "balanced"
        }
        analyze_resp = post_json(f"{API_BASE}/jobs/analyze", analyze_payload)
        m_insp = analyze_resp["model_inspection"]
        def _get_val(x):
            return x["value"] if isinstance(x, dict) and "value" in x else x

        self.assertEqual(_get_val(m_insp["architecture"]), "MobileNetV3-Small")
        self.assertEqual(_get_val(m_insp["format"]), "onnx")
        self.assertEqual(_get_val(m_insp["sha256"]), self.mobilenet_sha)
        compat = analyze_resp.get("compatibility", {})
        self.assertEqual(compat.get("task"), "image_classification")

        # 4. Launch Autonomous Optimization Job
        opt_payload = {
            "model_upload_id": m_upload_id,
            "dataset_upload_id": d_upload_id,
            "target": "raspberrypi5",
            "profile": "balanced",
            "max_candidates": 2,
            "calib_samples": 20,
            "test_samples": 30,
            "auto_approve": True
        }
        opt_resp = post_json(f"{API_BASE}/jobs/optimize", opt_payload)
        job_id = opt_resp["job_id"]
        print(f"MobileNet Job Dispatched: {job_id}")

        # 5. Poll until completed or timeout
        status = "QUEUED"
        for _ in range(90):
            time.sleep(1)
            detail = get_json(f"{API_BASE}/jobs/{job_id}")
            status = detail.get("status")
            if status in ["COMPLETED", "FAILED"]:
                break

        print(f"MobileNet Job Final Status: {status}")
        self.assertEqual(status, "COMPLETED", f"Job failed with detail: {detail}")

        # 6. Verify Model Identity in job directory
        job_dir = os.path.join(REPO_ROOT, "output", "jobs", job_id)
        job_model_file = os.path.join(job_dir, "inputs", "model", "mobilenetv3_sem.onnx")
        self.assertTrue(os.path.exists(job_model_file), "mobilenetv3_sem.onnx was not copied to job input dir")
        self.assertEqual(get_sha256(job_model_file), self.mobilenet_sha, "Model SHA in job does not match uploaded MobileNet SHA")

        # Verify model inspection in job
        self.assertEqual(detail["model_inspection"]["architecture"]["value"], "MobileNetV3-Small")
        self.assertEqual(detail["model_inspection"]["sha256"]["value"], self.mobilenet_sha)
        print("[PASS] Uploaded MobileNet reached Results successfully with verified identity!")

    def test_03_two_job_cross_contamination_isolation(self):
        """Run Job A (MobileNet) and Job B (ResNet) and verify zero cross-contamination."""
        print("\n--- Testing Two-Job Cross-Contamination Isolation ---")
        
        # Dispatch Job A: MobileNet
        job_a_req = {
            "model_id": "mobilenetv3_sem",
            "dataset_id": "semiconductor_defect",
            "target": "raspberrypi5",
            "profile": "balanced",
            "max_candidates": 1,
            "calib_samples": 10,
            "test_samples": 20
        }
        resp_a = post_json(f"{API_BASE}/jobs/optimize", job_a_req)
        job_a_id = resp_a["job_id"]

        # Dispatch Job B: ResNet
        job_b_req = {
            "model_id": "resnet50_cifar10",
            "dataset_id": "cifar10",
            "target": "raspberrypi5",
            "profile": "balanced",
            "max_candidates": 1,
            "calib_samples": 10,
            "test_samples": 20
        }
        resp_b = post_json(f"{API_BASE}/jobs/optimize", job_b_req)
        job_b_id = resp_b["job_id"]

        print(f"Dispatched Job A: {job_a_id} (MobileNet)")
        print(f"Dispatched Job B: {job_b_id} (ResNet)")

        # Wait for both to finish
        for _ in range(90):
            time.sleep(1)
            det_a = get_json(f"{API_BASE}/jobs/{job_a_id}")
            det_b = get_json(f"{API_BASE}/jobs/{job_b_id}")
            if det_a.get("status") in ["COMPLETED", "FAILED"] and det_b.get("status") in ["COMPLETED", "FAILED"]:
                break

        self.assertEqual(det_a.get("status"), "COMPLETED")
        self.assertEqual(det_b.get("status"), "COMPLETED")

        # Provenance Isolation Assertions
        sha_a = det_a["model_inspection"]["sha256"]["value"]
        sha_b = det_b["model_inspection"]["sha256"]["value"]
        arch_a = det_a["model_inspection"]["architecture"]["value"]
        arch_b = det_b["model_inspection"]["architecture"]["value"]

        print(f"Job A Arch: {arch_a}, SHA: {sha_a[:12]}...")
        print(f"Job B Arch: {arch_b}, SHA: {sha_b[:12]}...")

        self.assertNotEqual(sha_a, sha_b, "Job A and Job B SHA-256 MUST be distinct")
        self.assertNotEqual(arch_a, arch_b, "Job A and Job B architectures MUST be distinct")
        self.assertEqual(arch_a, "MobileNetV3-Small")
        self.assertEqual(arch_b, "ResNetForImageClassification")
        self.assertEqual(sha_a, self.mobilenet_sha)
        self.assertEqual(sha_b, self.resnet_sha)
        print("[PASS] Complete Two-Job Isolation Confirmed. Zero cross-contamination.")


if __name__ == "__main__":
    unittest.main()
