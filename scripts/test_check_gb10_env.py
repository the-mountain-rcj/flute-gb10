"""CPU-only unit tests; these never claim to test a real CUDA driver/GPU."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "check_gb10_env", Path(__file__).with_name("check_gb10_env.py"))
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


class PreflightTests(unittest.TestCase):
    def test_linux_arm64(self):
        with mock.patch.object(check.platform, "system", return_value="Linux"), \
             mock.patch.object(check.platform, "machine", return_value="aarch64"):
            self.assertIn("Linux/aarch64", check.check_platform())

    def test_wrong_platform(self):
        with mock.patch.object(check.platform, "system", return_value="Windows"):
            with self.assertRaises(check.CheckError):
                check.check_platform()

    def test_driver_capability_is_required(self):
        with mock.patch.object(check, "inspect_driver", return_value={
            "name": "B200", "capability": (10, 0), "driver_cuda": 13000, "count": 1,
        }):
            with self.assertRaisesRegex(check.CheckError, "expected GB10"):
                check.check_driver()

    def test_driver_version_is_required(self):
        with mock.patch.object(check, "inspect_driver", return_value={
            "name": "GB10", "capability": (12, 1), "driver_cuda": 12090, "count": 1,
        }):
            with self.assertRaisesRegex(check.CheckError, "below 13000"):
                check.check_driver()

    def test_gb10_driver_passes(self):
        with mock.patch.object(check, "inspect_driver", return_value={
            "name": "GB10", "capability": (12, 1), "driver_cuda": 13010, "count": 1,
        }):
            self.assertIn("SM121", check.check_driver())

    def test_missing_driver_is_failure(self):
        with mock.patch.object(check.ctypes, "CDLL", side_effect=OSError("missing")):
            with self.assertRaisesRegex(check.CheckError, "libcuda"):
                check.inspect_driver()

    def test_absent_venv_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("setup will create", check.check_existing_venv(Path(tmp)))

    def test_non_venv_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".venv-gb10-flute").mkdir()
            with self.assertRaisesRegex(check.CheckError, "without pyvenv.cfg"):
                check.check_existing_venv(Path(tmp))

    def test_shared_venv_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / ".venv-gb10-flute"
            directory.mkdir()
            (directory / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
            with self.assertRaisesRegex(check.CheckError, "not explicitly isolated"):
                check.check_existing_venv(Path(tmp))

    def test_isolated_venv_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / ".venv-gb10-flute"
            directory.mkdir()
            (directory / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
            with mock.patch.object(check, "inspect_python", return_value={
                "prefix": str(directory), "base_prefix": "/usr",
            }):
                self.assertIn("reusable", check.check_existing_venv(Path(tmp)))

    def test_wrong_python_version_rejected(self):
        with mock.patch.object(check, "run", return_value='{"version": [3, 13]}'):
            with self.assertRaisesRegex(check.CheckError, "Python 3.12"):
                check.inspect_python("python")

    def test_missing_python_dev_package_rejected(self):
        with mock.patch.object(check.shutil, "which", return_value="python3.12"), \
             mock.patch.object(check, "inspect_python", return_value={
                 "venv": True, "ensurepip": True, "python_h": False,
             }):
            with self.assertRaisesRegex(check.CheckError, "python_h"):
                check.check_base_python()

    def test_wrong_cutlass_tag_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(check, "run", side_effect=["wrong", "v3.4.1commit"]):
            with self.assertRaisesRegex(check.CheckError, "not checked out"):
                check.verify_cutlass_checkout(Path(tmp))

    def test_dirty_cutlass_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(check, "run", side_effect=["abc", "abc", " M header.h"]):
            with self.assertRaisesRegex(check.CheckError, "local changes"):
                check.verify_cutlass_checkout(Path(tmp))

    def test_missing_cutlass_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "flute"
            self.assertIn("setup will download", check.check_cutlass(root, Path(tmp) / "workspace"))

    def test_nonzero_process_rejected(self):
        result = check.subprocess.CompletedProcess(["nvcc"], 1, "", "bad config")
        with mock.patch.object(check.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(check.CheckError, "bad config"):
                check.run(["nvcc"])

    def test_wrong_cuda_version_rejected(self):
        with mock.patch.object(check.shutil, "which", return_value="nvcc"), \
             mock.patch.object(check, "run", return_value="Cuda compilation tools, release 12.9, V12.9"):
            with self.assertRaisesRegex(check.CheckError, "not CUDA 13.0"):
                check.check_toolkit()

    def test_missing_sm121_rejected(self):
        with mock.patch.object(check.shutil, "which", return_value="nvcc"), \
             mock.patch.object(check, "run", side_effect=["release 13.0,", "sm_120"]):
            with self.assertRaisesRegex(check.CheckError, "sm_121"):
                check.check_toolkit()

    def test_valid_toolkit_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            toolkit = Path(tmp)
            (toolkit / "bin").mkdir()
            (toolkit / "bin" / "nvcc").touch()
            (toolkit / "include").mkdir()
            (toolkit / "include" / "cuda.h").touch()
            (toolkit / "include" / "cuda_runtime.h").touch()
            (toolkit / "lib64").mkdir()
            (toolkit / "lib64" / "libcudart.so").touch()
            with mock.patch.object(check.shutil, "which", return_value=str(toolkit / "bin" / "nvcc")), \
                 mock.patch.object(check, "run", side_effect=["release 13.0,", "sm_120\nsm_121"]):
                self.assertIn("CUDA 13.0 with sm_121", check.check_toolkit())

    def test_report_does_not_hide_failures(self):
        report = check.Report()
        with contextlib.redirect_stdout(io.StringIO()):
            report.check("GPU", mock.Mock(side_effect=check.CheckError("no GPU")))
            report.emit("WARN", "disk", "low")
        self.assertEqual(report.failures, 1)
        self.assertEqual(report.warnings, 1)


if __name__ == "__main__":
    unittest.main()
