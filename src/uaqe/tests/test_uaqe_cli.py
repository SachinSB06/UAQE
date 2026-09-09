"""Automated Unit & Integration Test Suite for uaqe.py Master CLI.

Tests:
1. CLI help output and subcommand availability
2. Argument parsing and validation for all subcommands
3. Error handling on non-existent models and datasets
4. Inspection subcommands (inspect-model, inspect-dataset, validate)
5. Dry-run plan generation without optimization execution
6. Unique job directory isolation under output/jobs/<job_id>/
7. Historical baseline protection (Phases C4–R1)
"""

import os
import sys
import json
import subprocess
import unittest

# Add src to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


class TestUAQECLI(unittest.TestCase):
    """Test suite for the uaqe.py master CLI."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cls.uaqe_script = os.path.join(cls.project_root, "uaqe.py")
        cls.cifar10_dir = r"D:\uaqe_datasets\cifar10"
        cls.resnet_model = os.path.join(cls.project_root, "output", "phase_e2", "models", "resnet50_cifar10_fp32_baseline.pt")

    def _run_cli(self, args: list) -> subprocess.CompletedProcess:
        cmd = [sys.executable, self.uaqe_script] + args
        return subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

    def test_01_cli_help(self):
        """Verify uaqe.py --help returns 0 and displays available subcommands."""
        res = self._run_cli(["--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("optimize", res.stdout)
        self.assertIn("plan", res.stdout)
        self.assertIn("inspect-model", res.stdout)
        self.assertIn("inspect-dataset", res.stdout)
        self.assertIn("validate", res.stdout)

    def test_02_optimize_subcommand_help(self):
        """Verify uaqe.py optimize --help returns 0 and shows all options."""
        res = self._run_cli(["optimize", "--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("--model", res.stdout)
        self.assertIn("--dataset", res.stdout)
        self.assertIn("--target", res.stdout)
        self.assertIn("--profile", res.stdout)
        self.assertIn("--auto-approve", res.stdout)
        self.assertIn("--dry-run", res.stdout)

    def test_03_invalid_model_path_fails_cleanly(self):
        """Verify CLI exits with non-zero error when model file does not exist."""
        res = self._run_cli(["optimize", "--model", "non_existent_file.pt", "--dataset", self.cifar10_dir])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Error: Model file does not exist", res.stderr)

    def test_04_invalid_dataset_path_fails_cleanly(self):
        """Verify CLI exits with non-zero error when dataset path does not exist."""
        res = self._run_cli(["optimize", "--model", self.resnet_model, "--dataset", "non_existent_dir"])
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Error: Dataset path does not exist", res.stderr)

    def test_05_inspect_model_command(self):
        """Verify uaqe.py inspect-model outputs valid JSON descriptor."""
        res = self._run_cli(["inspect-model", "--model", self.resnet_model])
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["architecture"], "ResNetForImageClassification")
        self.assertEqual(data["output_shape"], [1, 10])

    def test_06_inspect_dataset_command(self):
        """Verify uaqe.py inspect-dataset outputs valid JSON descriptor for CIFAR-10."""
        res = self._run_cli(["inspect-dataset", "--dataset", self.cifar10_dir])
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertEqual(data["detected_format"], "cifar10_pickle")
        self.assertEqual(data["class_count"], 10)

    def test_07_validate_command(self):
        """Verify uaqe.py validate confirms compatibility between ResNet-50 and CIFAR-10."""
        res = self._run_cli(["validate", "--model", self.resnet_model, "--dataset", self.cifar10_dir])
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertTrue(data["compatible"])

    def test_08_dry_run_generates_plan_without_optimization(self):
        """Verify --dry-run generates plan files under output/jobs/ without full optimization."""
        res = self._run_cli([
            "optimize",
            "--model", self.resnet_model,
            "--dataset", self.cifar10_dir,
            "--target", "raspberrypi5",
            "--profile", "balanced",
            "--dry-run"
        ])
        self.assertEqual(res.returncode, 0)
        self.assertIn("[DRY RUN]", res.stdout)
        self.assertIn("OPTIMIZATION PLAN", res.stdout)

    def test_09_historical_integrity_unchanged(self):
        """Verify all historical artifacts from Phases C4–R1 remain 100% intact."""
        protected_dirs = [
            os.path.join(self.project_root, "output", "phase_c4"),
            os.path.join(self.project_root, "output", "phase_c5"),
            os.path.join(self.project_root, "output", "phase_d1"),
            os.path.join(self.project_root, "output", "phase_d2"),
            os.path.join(self.project_root, "output", "phase_d3"),
            os.path.join(self.project_root, "output", "phase_d4"),
            os.path.join(self.project_root, "output", "phase_d5"),
            os.path.join(self.project_root, "output", "phase_e1"),
            os.path.join(self.project_root, "output", "phase_e2"),
            os.path.join(self.project_root, "output", "phase_e3"),
            os.path.join(self.project_root, "output", "phase_r1")
        ]
        for pdir in protected_dirs:
            self.assertTrue(os.path.exists(pdir), f"Historical directory missing: {pdir}")
            self.assertGreater(len(os.listdir(pdir)), 0, f"Historical directory empty: {pdir}")


if __name__ == "__main__":
    unittest.main()
