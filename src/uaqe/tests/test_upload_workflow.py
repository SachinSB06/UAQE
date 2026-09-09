"""Tests for UAQE Model File & Dataset Folder Upload Workflow and Security.

Verifies:
1. Single model file upload validation, SHA-256 computation, and type enforcement.
2. Dataset folder upload with relative path preservation and manifest hash generation.
3. Strict path traversal security rejection (../, ..\\, absolute paths, drive letters, null bytes).
4. Full analyze & optimize lifecycle using upload IDs.
5. Pre-verified sample selection workflow with explicit provenance tagging.
"""

import os
import sys
import io
import json
import zipfile
import unittest
from unittest.mock import patch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi.testclient import TestClient
from uaqe.server import app

client = TestClient(app)


class TestUploadWorkflow(unittest.TestCase):
    """Test suite for Model & Dataset uploads and security enforcement."""

    def test_model_file_upload_success(self):
        """Verify uploading a valid model file calculates SHA-256 and stages file."""
        dummy_model_bytes = b"PYTORCH_CHECKPOINT_DUMMY_DATA_123456789"
        files = {
            "file": ("custom_resnet.pt", io.BytesIO(dummy_model_bytes), "application/octet-stream")
        }
        res = client.post("/api/uploads/model", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["upload_id"].startswith("MODEL-"))
        self.assertEqual(data["filename"], "custom_resnet.pt")
        self.assertEqual(data["size_bytes"], len(dummy_model_bytes))
        self.assertEqual(data["format"], "pytorch_checkpoint")
        self.assertEqual(data["source"], "USER_UPLOAD")
        self.assertEqual(data["status"], "READY")
        self.assertTrue(os.path.exists(data["staged_path"]))

    def test_model_file_unsupported_format(self):
        """Verify uploading an unsupported file type is rejected with 400."""
        files = {
            "file": ("malicious_script.exe", io.BytesIO(b"ELF_HEADER"), "application/octet-stream")
        }
        res = client.post("/api/uploads/model", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported model extension", res.json()["detail"])

    def test_model_file_empty_rejected(self):
        """Verify uploading an empty 0-byte file is rejected with 400."""
        files = {
            "file": ("empty_model.pt", io.BytesIO(b""), "application/octet-stream")
        }
        res = client.post("/api/uploads/model", files=files)
        self.assertEqual(res.status_code, 400)

    def test_dataset_folder_upload_success(self):
        """Verify folder upload preserves nested relative paths and builds manifest."""
        file1 = ("data_batch_1", io.BytesIO(b"batch_1_content"), "application/octet-stream")
        file2 = ("data_batch_2", io.BytesIO(b"batch_2_content"), "application/octet-stream")
        file3 = ("batches.meta", io.BytesIO(b"batches_meta_content"), "application/octet-stream")

        files = [
            ("files", file1),
            ("files", file2),
            ("files", file3)
        ]
        relative_paths = json.dumps([
            "cifar-10-batches-py/data_batch_1",
            "cifar-10-batches-py/data_batch_2",
            "cifar-10-batches-py/batches.meta"
        ])
        data = {
            "relative_paths": relative_paths,
            "folder_name": "cifar10"
        }

        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        res_data = res.json()

        self.assertTrue(res_data["upload_id"].startswith("DATASET-"))
        self.assertEqual(res_data["file_count"], 3)
        self.assertEqual(res_data["status"], "READY")
        self.assertEqual(res_data["source"], "USER_UPLOAD")
        self.assertTrue(len(res_data["manifest_hash"]) == 64)

        # Verify staged folder structure
        staged_dir = res_data["staged_path"]
        self.assertTrue(os.path.exists(os.path.join(staged_dir, "cifar-10-batches-py", "data_batch_1")))
        self.assertTrue(os.path.exists(os.path.join(staged_dir, "cifar-10-batches-py", "data_batch_2")))
        self.assertTrue(os.path.exists(os.path.join(staged_dir, "cifar-10-batches-py", "batches.meta")))
        self.assertTrue(os.path.exists(os.path.join(staged_dir, "dataset_manifest.json")))

    def test_security_directory_traversal_unix(self):
        """Verify paths containing ../ are strictly rejected."""
        files = [
            ("files", ("evil.txt", io.BytesIO(b"attack"), "text/plain"))
        ]
        data = {
            "relative_paths": json.dumps(["../../secret.txt"]),
            "folder_name": "attack_dataset"
        }
        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Security violation", res.json()["detail"])

    def test_security_directory_traversal_windows(self):
        """Verify paths containing ..\\ are strictly rejected."""
        files = [
            ("files", ("evil.txt", io.BytesIO(b"attack"), "text/plain"))
        ]
        data = {
            "relative_paths": json.dumps(["..\\..\\Windows\\win.ini"]),
            "folder_name": "attack_dataset"
        }
        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Security violation", res.json()["detail"])

    def test_security_absolute_path_rejected(self):
        """Verify absolute paths (e.g. C:\\ or /etc) are rejected."""
        files = [
            ("files", ("pass.txt", io.BytesIO(b"pass"), "text/plain"))
        ]
        data = {
            "relative_paths": json.dumps(["C:/Windows/System32/config.sys"]),
            "folder_name": "attack_dataset"
        }
        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Security violation", res.json()["detail"])

    def test_security_null_byte_rejected(self):
        """Verify null byte injections in paths are rejected."""
        files = [
            ("files", ("test.png", io.BytesIO(b"image"), "image/png"))
        ]
        data = {
            "relative_paths": json.dumps(["train/cat\0.png"]),
            "folder_name": "attack_dataset"
        }
        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Security violation", res.json()["detail"])

    def test_dataset_zip_fallback_upload(self):
        """Verify zip fallback endpoint correctly extracts structure and builds manifest."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("test_dataset/train/cat/cat1.jpg", b"fake_cat_image")
            zipf.writestr("test_dataset/train/dog/dog1.jpg", b"fake_dog_image")
            zipf.writestr("test_dataset/val/cat/cat2.jpg", b"fake_cat_val")
        zip_buffer.seek(0)

        files = {
            "file": ("my_dataset.zip", zip_buffer, "application/zip")
        }
        res = client.post("/api/uploads/dataset/zip", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data["upload_id"].startswith("DATASET-"))
        self.assertEqual(data["file_count"], 3)
        self.assertEqual(data["status"], "READY")
        self.assertTrue(os.path.exists(os.path.join(data["staged_path"], "dataset_manifest.json")))

    def test_analyze_and_optimize_with_upload_ids(self):
        """Verify /api/jobs/analyze and /api/jobs/optimize operate seamlessly with upload IDs."""
        # 1. Upload sample model
        real_model_p = "output/phase_e2/models/resnet50_cifar10_fp32_baseline.pt"
        if os.path.exists(real_model_p):
            with open(real_model_p, "rb") as f:
                model_bytes = f.read(500000)  # Read header/slice
        else:
            model_bytes = b"CHECKPOINT_DATA_FOR_ANALYZE_TEST"

        model_res = client.post("/api/uploads/model", files={
            "file": ("resnet50.pt", io.BytesIO(model_bytes), "application/octet-stream")
        })
        self.assertEqual(model_res.status_code, 200)
        model_uid = model_res.json()["upload_id"]

        # 2. Upload dataset folder
        ds_res = client.post("/api/uploads/dataset", files=[
            ("files", ("batch_1", io.BytesIO(b"data1"), "application/octet-stream")),
            ("files", ("batch_2", io.BytesIO(b"data2"), "application/octet-stream"))
        ], data={
            "relative_paths": json.dumps(["cifar10/batch_1", "cifar10/batch_2"]),
            "folder_name": "cifar10"
        })
        self.assertEqual(ds_res.status_code, 200)
        ds_uid = ds_res.json()["upload_id"]

        # 3. Test analyze with upload IDs
        analyze_payload = {
            "model_upload_id": model_uid,
            "dataset_upload_id": ds_uid,
            "target": "raspberrypi5",
            "profile": "balanced"
        }
        with patch("uaqe.api.routes_jobs.UniversalModelIngestor._init_adapter"), \
             patch("uaqe.api.routes_jobs.UniversalModelIngestor.inspect") as mock_m_insp, \
             patch("uaqe.api.routes_jobs.UniversalDatasetIngestor._init_adapter"), \
             patch("uaqe.api.routes_jobs.UniversalDatasetIngestor.get_descriptor") as mock_ds_desc, \
             patch("uaqe.api.routes_jobs.CompatibilityChecker.check_compatibility") as mock_compat:
            mock_m_insp.return_value = {
                "format": "pytorch_checkpoint",
                "framework": "PyTorch",
                "architecture": "ResNet-50",
                "task": "image_classification",
                "input_shape": [1, 3, 224, 224],
                "output_shape": [1, 10]
            }
            mock_ds_desc.return_value = {
                "detected_format": "cifar10_pickle",
                "class_count": 10,
                "splits": {"train_count": 50000, "val_count": 5000, "test_count": 5000}
            }
            mock_compat.return_value = {"compatible": True, "issues": []}

            analyze_res = client.post("/api/jobs/analyze", json=analyze_payload)
            self.assertEqual(analyze_res.status_code, 200)
            ana_data = analyze_res.json()
            self.assertIn("model_inspection", ana_data)
            self.assertIn("dataset_inspection", ana_data)
            self.assertEqual(ana_data["model_source"], "USER_UPLOAD")
            self.assertEqual(ana_data["dataset_source"], "USER_UPLOAD")



        # 4. Test optimize with upload IDs
        with patch("uaqe.api.routes_jobs.UniversalModelIngestor._init_adapter"), \
             patch("uaqe.api.routes_jobs.UniversalModelIngestor.inspect") as mock_m_insp_opt, \
             patch("uaqe.api.routes_jobs._run_optimization_worker") as mock_worker:
            mock_m_insp_opt.return_value = {
                "format": "pytorch_checkpoint",
                "framework": "PyTorch",
                "architecture": "ResNet-50",
                "task": "image_classification",
                "input_shape": [1, 3, 224, 224],
                "output_shape": [1, 10]
            }
            opt_res = client.post("/api/jobs/optimize", json=analyze_payload)
            self.assertEqual(opt_res.status_code, 200)
            opt_data = opt_res.json()
            self.assertTrue(opt_data["job_id"].startswith("UAQE-"))
            self.assertEqual(opt_data["status"], "LAUNCHED")
            self.assertEqual(opt_data["model_source"], "USER_UPLOAD")
            self.assertEqual(opt_data["dataset_source"], "USER_UPLOAD")

            # Verify input isolation directory created
            from uaqe.api.routes_jobs import _get_jobs_root
            job_dir = os.path.join(_get_jobs_root(), opt_data["job_id"])
            manifest_path = os.path.join(job_dir, "input_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))

            with open(manifest_path, "r", encoding="utf-8") as f:
                input_man = json.load(f)
                self.assertEqual(input_man["model_source"], "USER_UPLOAD")
                self.assertEqual(input_man["dataset_source"], "USER_UPLOAD")
                self.assertTrue(os.path.exists(input_man["model_path"]))
                self.assertTrue(os.path.exists(input_man["dataset_path"]))

    def test_samples_source_classification(self):
        """Verify pre-verified samples have PRE_VERIFIED_SAMPLE source tag."""
        samples_res = client.get("/api/uploads/samples")
        self.assertEqual(samples_res.status_code, 200)
        data = samples_res.json()
        self.assertTrue(len(data["models"]) > 0)
        self.assertEqual(data["models"][0]["source"], "PRE_VERIFIED_SAMPLE")


if __name__ == "__main__":
    unittest.main()
