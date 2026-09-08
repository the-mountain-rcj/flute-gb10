"""CPU/mocked tests for the explicit CUDA 13.4 trial; no GPU validation."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "check_gb10_env_cuda134", Path(__file__).with_name("check_gb10_env.py"))
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


@contextlib.contextmanager
def fake_toolkit(version="13.4", targets="sm_120\nsm_121", missing=(), arm_layout=False):
    """Create only empty test fixtures; no compiler is ever invoked."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        executable = root / "bin" / "nvcc"
        executable.parent.mkdir()
        executable.touch()
        development = root / "targets" / "aarch64-linux" if arm_layout else root
        include = development / "include"
        library = development / ("lib" if arm_layout else "lib64")
        include.mkdir(parents=True)
        library.mkdir(parents=True)
        for name in ("cuda.h", "cuda_runtime.h"):
            if name not in missing:
                (include / name).touch()
        if "libcudart.so" not in missing:
            (library / "libcudart.so").touch()
        with mock.patch.object(check.shutil, "which", return_value=str(executable)), \
             mock.patch.object(check, "run", side_effect=[
                 f"Cuda compilation tools, release {version}, V{version}.59", targets,
             ]) as run:
            yield run


def driver_info(version=13040, capability=(12, 1)):
    return {"name": "Test GPU", "capability": capability,
            "driver_cuda": version, "count": 1}


@contextlib.contextmanager
def successful_checks(keep_driver=False):
    """Mock all external checks while retaining the real reporting/control flow."""
    names = ["check_platform", "check_base_python", "check_existing_venv",
             "check_command", "check_toolkit", "check_cutlass"]
    if not keep_driver:
        names.append("check_driver")
    with contextlib.ExitStack() as stack:
        mocks = {name: stack.enter_context(mock.patch.object(check, name, return_value="ok"))
                 for name in names}
        mocks["report_resources"] = stack.enter_context(
            mock.patch.object(check, "report_resources"))
        yield mocks


class ToolkitTrialTests(unittest.TestCase):
    def test_default_still_rejects_cuda134(self):
        with fake_toolkit(), self.assertRaises(check.CheckError):
            check.check_toolkit()

    def test_explicit_cuda134_passes_complete_toolkit(self):
        with fake_toolkit() as run:
            self.assertIn("CUDA 13.4", check.check_toolkit(cuda_version="13.4"))
            self.assertEqual(run.call_count, 2)

    def test_explicit_cuda134_passes_arm_target_layout(self):
        with fake_toolkit(arm_layout=True):
            self.assertIn("sm_121", check.check_toolkit(cuda_version="13.4"))

    def test_default_cuda130_still_passes(self):
        with fake_toolkit(version="13.0"):
            self.assertIn("CUDA 13.0", check.check_toolkit())

    def test_trial_requires_exact_version(self):
        for version in ("13.0", "13.3", "14.0"):
            with self.subTest(version=version), fake_toolkit(version=version), \
                 self.assertRaises(check.CheckError):
                check.check_toolkit(cuda_version="13.4")

    def test_unsupported_recipe_cannot_be_requested_programmatically(self):
        for version in ("13.3", "14.0"):
            with self.subTest(version=version), self.assertRaises(check.CheckError):
                check.check_toolkit(cuda_version=version)

    def test_experimental_mode_still_requires_nvcc(self):
        with mock.patch.object(check.shutil, "which", return_value=None), \
             self.assertRaises(check.CheckError):
            check.check_toolkit(cuda_version="13.4")

    def test_experimental_mode_still_requires_sm121(self):
        for targets in ("sm_120", "sm_1210", "compute_121", ""):
            with self.subTest(targets=targets), fake_toolkit(targets=targets), \
                 self.assertRaisesRegex(check.CheckError, "sm_121"):
                check.check_toolkit(cuda_version="13.4")

    def test_experimental_mode_still_requires_both_headers(self):
        for missing in ("cuda.h", "cuda_runtime.h"):
            with self.subTest(missing=missing), fake_toolkit(missing=(missing,)), \
                 self.assertRaises(check.CheckError):
                check.check_toolkit(cuda_version="13.4")

    def test_experimental_mode_still_requires_development_library(self):
        with fake_toolkit(missing=("libcudart.so",)), self.assertRaises(check.CheckError):
            check.check_toolkit(cuda_version="13.4")


class DriverTrialTests(unittest.TestCase):
    def test_default_accepts_cuda130_driver(self):
        with mock.patch.object(check, "inspect_driver", return_value=driver_info(13000)):
            self.assertIn("SM121", check.check_driver())

    def test_experimental_rejects_cuda130_driver(self):
        with mock.patch.object(check, "inspect_driver", return_value=driver_info(13000)), \
             self.assertRaises(check.CheckError):
            check.check_driver(minimum_cuda=13040)

    def test_experimental_rejects_below_cuda134_boundary(self):
        with mock.patch.object(check, "inspect_driver", return_value=driver_info(13039)), \
             self.assertRaises(check.CheckError):
            check.check_driver(minimum_cuda=13040)

    def test_experimental_accepts_cuda134_driver(self):
        with mock.patch.object(check, "inspect_driver", return_value=driver_info(13040)):
            self.assertIn("CUDA 13.4", check.check_driver(minimum_cuda=13040))

    def test_experimental_still_rejects_non_gb10_gpu(self):
        with mock.patch.object(check, "inspect_driver", return_value=driver_info(13040, (10, 0))), \
             self.assertRaises(check.CheckError):
            check.check_driver(minimum_cuda=13040)


class CommandLineTrialTests(unittest.TestCase):
    def test_default_passes_pinned_parameters_without_experimental_warning(self):
        output = io.StringIO()
        with successful_checks() as mocks, contextlib.redirect_stdout(output):
            self.assertEqual(check.main([]), 0)
        mocks["check_toolkit"].assert_called_once_with(cuda_version="13.0")
        mocks["check_driver"].assert_called_once_with(minimum_cuda=13000)
        self.assertIn("Summary: 0 failure(s), 0 warning(s).", output.getvalue())
        self.assertNotIn("Experimental CUDA mode", output.getvalue())

    def test_flag_propagates_trial_parameters_without_skipping_any_check(self):
        output = io.StringIO()
        with successful_checks() as mocks, contextlib.redirect_stdout(output):
            self.assertEqual(check.main(["--experimental-cuda-13-4"]), 0)
        mocks["check_toolkit"].assert_called_once_with(cuda_version="13.4")
        mocks["check_driver"].assert_called_once_with(minimum_cuda=13040)
        for name in ("check_platform", "check_base_python", "check_existing_venv",
                     "check_cutlass", "report_resources"):
            mocks[name].assert_called_once()
        self.assertEqual(mocks["check_command"].call_args_list,
                         [mock.call("git"), mock.call("c++")])
        self.assertIn("[WARN] Experimental CUDA mode", output.getvalue())
        self.assertIn("PyTorch 2.9.1/cu130", output.getvalue())
        self.assertIn("Summary: 0 failure(s), 1 warning(s).", output.getvalue())

    def test_trial_does_not_turn_failures_into_warnings(self):
        output = io.StringIO()
        with successful_checks() as mocks, contextlib.redirect_stdout(output):
            mocks["check_toolkit"].side_effect = check.CheckError("headers missing")
            self.assertEqual(check.main(["--experimental-cuda-13-4"]), 1)
        mocks["check_driver"].assert_called_once()
        mocks["check_cutlass"].assert_called_once()
        self.assertIn("Summary: 1 failure(s), 1 warning(s).", output.getvalue())
        self.assertIn("STOP:", output.getvalue())

    def test_trial_flag_does_not_bypass_wrong_gpu(self):
        output = io.StringIO()
        with successful_checks(keep_driver=True), \
             mock.patch.object(check, "inspect_driver", return_value=driver_info(13040, (12, 0))), \
             contextlib.redirect_stdout(output):
            self.assertEqual(check.main(["--experimental-cuda-13-4"]), 1)
        self.assertIn("[FAIL] GPU / NVIDIA driver", output.getvalue())

    def test_help_lists_trial_flag_without_running_checks(self):
        output = io.StringIO()
        with successful_checks() as mocks, contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as caught:
                check.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--experimental-cuda-13-4", output.getvalue())
        for action in mocks.values():
            action.assert_not_called()

    def test_unknown_flag_is_error_before_running_checks(self):
        output = io.StringIO()
        with successful_checks() as mocks, contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit) as caught:
                check.main(["--experimental-cuda-13-3"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("unrecognized arguments", output.getvalue())
        for action in mocks.values():
            action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
