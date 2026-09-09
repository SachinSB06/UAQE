"""Comprehensive test suite for UAQE Dataset Folder Upload Workflow.

Implements all required tests:
- TEST A: 3-file multipart upload with nested directories.
- TEST B: ~40 MB multipart upload.
- TEST C: Nested directory structure preservation.
- TEST D: 2,000+ files upload exceeding standard 1,000 file limit.
- TEST E: Windows-style relative path normalization (\\ -> /).
- TEST F: Strict path traversal security rejection (../, C:\\, absolute, UNC, null bytes).
- TEST G: FormData contract and file/relative_path consistency verification.
- TEST H: Staging isolation and deterministic manifest hashing.
"""

from __future__ import annotations

import os
import sys
import io
import json
import shutil
import hashlib
import unittest

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi.testclient import TestClient
from uaqe.server import app

client = TestClient(app)


class TestDatasetFolderUpload(unittest.TestCase):
    """Test suite covering all aspects of UAQE dataset folder upload."""

    def test_a_three_file_multipart_upload_nested(self):
        """TEST A: 3-file multipart upload with nested directories.

        Expected: HTTP 200, file_count=3, READY, correct relative paths.
        """
        files = [
            ("files", ("img_a.png", io.BytesIO(b"PNG_DATA_A"), "image/png")),
            ("files", ("img_b.png", io.BytesIO(b"PNG_DATA_B"), "image/png")),
            ("files", ("img_c.png", io.BytesIO(b"PNG_DATA_C"), "image/png")),
        ]
        relative_paths = [
            "train/class_0/img_a.png",
            "train/class_1/img_b.png",
            "val/class_0/img_c.png",
        ]
        data = {
            "relative_paths": json.dumps(relative_paths),
            "folder_name": "test_3_files",
        }

        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 200, f"Upload failed: {res.text}")
        payload = res.json()

        self.assertEqual(payload["file_count"], 3)
        self.assertEqual(payload["status"], "READY")
        self.assertTrue(payload["upload_id"].startswith("DATASET-"))
        self.assertTrue(payload["manifest_hash"])
        staged_path = payload["staged_path"]
        self.assertTrue(os.path.exists(staged_path))

        # Verify staged files exist at their respective relative paths
        for rel in relative_paths:
            full_path = os.path.join(staged_path, rel)
            self.assertTrue(os.path.exists(full_path), f"Missing staged file: {rel}")

    def test_b_forty_mb_multipart_upload(self):
        """TEST B: ~40 MB multipart upload.

        Expected: HTTP 200, correct total size, manifest generated.
        """
        # Create 10 files of 4 MB each = 40 MB
        chunk_size = 4 * 1024 * 1024
        file_count = 10
        total_expected_bytes = chunk_size * file_count

        files = []
        rel_paths = []
        for i in range(file_count):
            content = b"X" * chunk_size
            files.append(("files", (f"file_{i}.bin", io.BytesIO(content), "application/octet-stream")))
            rel_paths.append(f"batch_data/shard_{i}.bin")

        data = {
            "relative_paths": json.dumps(rel_paths),
            "folder_name": "test_40mb",
        }

        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 200, f"40MB upload failed: {res.text}")
        payload = res.json()

        self.assertEqual(payload["file_count"], file_count)
        self.assertEqual(payload["total_size_bytes"], total_expected_bytes)
        self.assertEqual(payload["status"], "READY")
        self.assertTrue(payload["manifest_hash"])

    def test_c_nested_structure_preservation(self):
        """TEST C: Nested directory structure preservation.

        Example:
        dataset/
          class_a/
            img1.png
          class_b/
            nested/
              img2.png
        """
        files = [
            ("files", ("img1.png", io.BytesIO(b"CLASS_A_IMG"), "image/png")),
            ("files", ("img2.png", io.BytesIO(b"CLASS_B_NESTED_IMG"), "image/png")),
        ]
        relative_paths = [
            "class_a/img1.png",
            "class_b/nested/img2.png",
        ]
        data = {
            "relative_paths": json.dumps(relative_paths),
            "folder_name": "nested_dataset",
        }

        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 200, f"Upload failed: {res.text}")
        payload = res.json()

        staged_path = payload["staged_path"]
        self.assertTrue(os.path.isfile(os.path.join(staged_path, "class_a", "img1.png")))
        self.assertTrue(os.path.isfile(os.path.join(staged_path, "class_b", "nested", "img2.png")))

    def test_d_two_thousand_plus_files(self):
        """TEST D: 2,000+ files upload exceeding old 1,000 limit.

        Expected: HTTP 200, no max_files=1000 failure.
        """
        file_count = 2100
        files = []
        rel_paths = []
        for i in range(file_count):
            files.append(("files", (f"img_{i}.txt", io.BytesIO(f"content_{i}".encode("utf-8")), "text/plain")))
            rel_paths.append(f"class_{i % 50}/img_{i}.txt")

        data = {
            "relative_paths": json.dumps(rel_paths),
            "folder_name": "test_many_files",
        }

        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 200, f"Large file count upload failed: {res.text}")
        payload = res.json()

        self.assertEqual(payload["file_count"], file_count)
        self.assertEqual(payload["status"], "READY")

    def test_e_windows_path_normalization(self):
        """TEST E: Windows path normalization (\\ -> /).

        Example:
        sub\\class\\img.jpg
        Expected normalized:
        sub/class/img.jpg
        """
        files = [
            ("files", ("img.jpg", io.BytesIO(b"WINDOWS_PATH_JPEG"), "image/jpeg")),
        ]
        relative_paths = [
            "sub\\class\\img.jpg",
        ]
        data = {
            "relative_paths": json.dumps(relative_paths),
            "folder_name": "win_dataset",
        }

        res = client.post("/api/uploads/dataset", files=files, data=data)
        self.assertEqual(res.status_code, 200, f"Upload failed: {res.text}")
        payload = res.json()

        staged_path = payload["staged_path"]
        expected_file = os.path.join(staged_path, "sub", "class", "img.jpg")
        self.assertTrue(os.path.isfile(expected_file))

        # Check manifest.json has POSIX relative path
        with open(os.path.join(staged_path, "dataset_manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["files"][0]["relative_path"], "sub/class/img.jpg")

    def test_f_security_path_traversal_rejection(self):
        """TEST F: Security path traversal rejection.

        Verify these are rejected:
        ../file.txt
        ../../file.txt
        C:\\file.txt
        D:\\file.txt
        \\file.txt
        /file.txt
        \\\\server\\share\\file.txt
        null-byte path
        """
        malicious_paths = [
            "../file.txt",
            "../../file.txt",
            "C:\\file.txt",
            "D:\\file.txt",
            "\\file.txt",
            "/file.txt",
            "\\\\server\\share\\file.txt",
            "safe_dir/\0evil.txt",
        ]

        for bad_path in malicious_paths:
            files = [
                ("files", ("payload.txt", io.BytesIO(b"EVIL"), "text/plain")),
            ]
            data = {
                "relative_paths": json.dumps([bad_path]),
                "folder_name": "attack",
            }
            res = client.post("/api/uploads/dataset", files=files, data=data)
            self.assertEqual(
                res.status_code, 400,
                f"Malicious path '{bad_path}' was NOT rejected with HTTP 400! Got: {res.status_code} {res.text}"
            )

    def test_g_formdata_contract_and_count_mismatch(self):
        """TEST G: FormData contract & consistency check.

        Verify required fields:
        - files
        - relative_paths
        - folder_name
        And verify file count mismatch between files and relative_paths is rejected.
        """
        # Missing relative_paths
        files = [
            ("files", ("img.png", io.BytesIO(b"DATA"), "image/png")),
        ]
        res = client.post("/api/uploads/dataset", files=files, data={})
        self.assertEqual(res.status_code, 400)
        self.assertIn("relative_paths", res.json()["detail"])

        # Count mismatch: 2 files but 1 relative path
        files2 = [
            ("files", ("img1.png", io.BytesIO(b"DATA1"), "image/png")),
            ("files", ("img2.png", io.BytesIO(b"DATA2"), "image/png")),
        ]
        data2 = {
            "relative_paths": json.dumps(["class_a/img1.png"]),
            "folder_name": "mismatch_test",
        }
        res2 = client.post("/api/uploads/dataset", files=files2, data=data2)
        self.assertEqual(res2.status_code, 400)
        self.assertIn("File count mismatch", res2.json()["detail"])

    def test_h_staging_isolation_and_deterministic_manifest(self):
        """TEST H: Staging isolation.

        Verify:
        - unique upload ID
        - files remain inside staging root
        - manifest hash is deterministic
        - no cross-upload contamination
        """
        files1 = [
            ("files", ("a.txt", io.BytesIO(b"SAME_CONTENT_A"), "text/plain")),
            ("files", ("b.txt", io.BytesIO(b"SAME_CONTENT_B"), "text/plain")),
        ]
        data1 = {
            "relative_paths": json.dumps(["f1.txt", "f2.txt"]),
            "folder_name": "run_1",
        }

        files2 = [
            ("files", ("a.txt", io.BytesIO(b"SAME_CONTENT_A"), "text/plain")),
            ("files", ("b.txt", io.BytesIO(b"SAME_CONTENT_B"), "text/plain")),
        ]
        data2 = {
            "relative_paths": json.dumps(["f1.txt", "f2.txt"]),
            "folder_name": "run_2",
        }

        res1 = client.post("/api/uploads/dataset", files=files1, data=data1)
        res2 = client.post("/api/uploads/dataset", files=files2, data=data2)

        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res2.status_code, 200)

        p1 = res1.json()
        p2 = res2.json()

        # Different upload IDs
        self.assertNotEqual(p1["upload_id"], p2["upload_id"])
        # Different isolated staged directories
        self.assertNotEqual(p1["staged_path"], p2["staged_path"])
        # Identical deterministic manifest hash
        self.assertEqual(p1["manifest_hash"], p2["manifest_hash"])

    def test_i_dataset_root_discovery_nested(self):
        """TEST I: Nested dataset directory layout (Requirement 11).

        Structure:
        temporary_root/
            imagenet_10k_split/
                train/
                    class_a/
                        a1.JPEG
                        a2.JPEG
                val/
                    class_a/
                        v1.JPEG
                test/
                    class_a/
                        t1.JPEG

        Verify:
        - resolved dataset root = temporary_root/imagenet_10k_split
        - train split != empty
        - val split != empty
        - test split != empty
        """
        import tempfile
        from PIL import Image
        from uaqe.orchestration.universal_dataset_ingestor import UniversalDatasetIngestor

        with tempfile.TemporaryDirectory() as temp_dir:
            ds_root = os.path.join(temp_dir, "imagenet_10k_split")
            for split, count, prefix in [("train", 2, "a"), ("val", 1, "v"), ("test", 1, "t")]:
                class_dir = os.path.join(ds_root, split, "class_a")
                os.makedirs(class_dir, exist_ok=True)
                for idx in range(1, count + 1):
                    img = Image.new("RGB", (32, 32), color=(idx * 40, idx * 50, idx * 60))
                    img.save(os.path.join(class_dir, f"{prefix}{idx}.JPEG"))

            # Also add an arbitrary metadata/manifest file at temporary_root to test robustness
            with open(os.path.join(temp_dir, "manifest.json"), "w") as f:
                f.write('{"info": "test"}')

            ingestor = UniversalDatasetIngestor(temp_dir)
            resolved = ingestor.dataset_path
            self.assertEqual(os.path.normpath(resolved), os.path.normpath(ds_root))

            desc = ingestor.get_descriptor()
            self.assertEqual(desc["dataset_name"], "imagenet_10k_split")
            self.assertEqual(desc["class_count"], 1)
            self.assertEqual(desc["splits"]["train_count"], 2)
            self.assertEqual(desc["splits"]["val_count"], 1)
            self.assertEqual(desc["splits"]["test_count"], 1)

    def test_j_dataset_root_discovery_direct(self):
        """TEST J: Direct train/val/test under root (Requirement 11).

        Verify:
        - resolved dataset root = temporary_root
        - train split != empty
        - val split != empty
        - test split != empty
        """
        import tempfile
        from PIL import Image
        from uaqe.orchestration.universal_dataset_ingestor import UniversalDatasetIngestor

        with tempfile.TemporaryDirectory() as temp_dir:
            for split, prefix in [("train", "tr"), ("val", "va"), ("test", "te")]:
                class_dir = os.path.join(temp_dir, split, "class_b")
                os.makedirs(class_dir, exist_ok=True)
                img = Image.new("RGB", (32, 32), color=(10, 20, 30))
                img.save(os.path.join(class_dir, f"{prefix}1.JPEG"))

            ingestor = UniversalDatasetIngestor(temp_dir)
            resolved = ingestor.dataset_path
            self.assertEqual(os.path.normpath(resolved), os.path.normpath(temp_dir))

            desc = ingestor.get_descriptor()
            self.assertEqual(desc["class_count"], 1)
            self.assertEqual(desc["splits"]["train_count"], 1)
            self.assertEqual(desc["splits"]["val_count"], 1)
            self.assertEqual(desc["splits"]["test_count"], 1)


if __name__ == "__main__":
    unittest.main()

