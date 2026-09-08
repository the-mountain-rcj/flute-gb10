#!/usr/bin/env python3
"""Read-only, stdlib-only prerequisites check for the pinned GB10 recipe.

This does not install packages, compile FLUTE, change repositories, or prove
kernel compatibility. Run the GPU smoke test after a successful installation.
"""

import ctypes
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


class CheckError(RuntimeError):
    pass


def run(command, timeout=20):
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckError(f"{command[0]} could not run: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise CheckError(f"{command[0]} exited {result.returncode}: "
                         f"{detail[-1] if detail else 'no diagnostic'}")
    return result.stdout.strip()


class Report:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def emit(self, status, label, message):
        self.failures += status == "FAIL"
        self.warnings += status == "WARN"
        print(f"[{status}] {label}: {message}")

    def check(self, label, action):
        try:
            self.emit("PASS", label, action())
        except (CheckError, OSError, ValueError, KeyError) as exc:
            self.emit("FAIL", label, str(exc))


def check_platform():
    system, machine = platform.system(), platform.machine().lower()
    if system != "Linux" or machine not in {"aarch64", "arm64"}:
        raise CheckError(f"found {system}/{machine}; this recipe requires Linux ARM64 GB10")
    return f"{system}/{machine}"


PYTHON_PROBE = """
import importlib.util, json, pathlib, sys, sysconfig
print(json.dumps({
    'version': list(sys.version_info[:2]),
    'venv': importlib.util.find_spec('venv') is not None,
    'ensurepip': importlib.util.find_spec('ensurepip') is not None,
    'python_h': (pathlib.Path(sysconfig.get_path('include')) / 'Python.h').is_file(),
    'prefix': sys.prefix,
    'base_prefix': sys.base_prefix,
}))
"""


def inspect_python(executable):
    info = json.loads(run([str(executable), "-I", "-c", PYTHON_PROBE]))
    if info["version"] != [3, 12]:
        raise CheckError(f"{executable} must be Python 3.12, found {info['version']}")
    return info


def check_base_python():
    executable = shutil.which("python3.12")
    if not executable:
        raise CheckError("python3.12 not in PATH; install Python 3.12, python3.12-venv and python3.12-dev")
    info = inspect_python(executable)
    missing = [name for name in ("venv", "ensurepip", "python_h") if not info[name]]
    if missing:
        raise CheckError(f"Python 3.12 missing {', '.join(missing)}; check python3.12-venv / python3.12-dev")
    return f"{executable}; venv, ensurepip and Python.h found"


def check_existing_venv(root=ROOT):
    directory = root / ".venv-gb10-flute"
    if not directory.exists() and not directory.is_symlink():
        return "absent; setup will create a separate Python 3.12 environment"
    config_path = directory / "pyvenv.cfg"
    if not config_path.is_file():
        raise CheckError(f"{directory} exists without pyvenv.cfg; will not overwrite it")
    config = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            config[key.strip().lower()] = value.strip().lower()
    if config.get("include-system-site-packages") != "false":
        raise CheckError("existing venv is not explicitly isolated (include-system-site-packages must be false)")
    info = inspect_python(directory / "bin" / "python")
    if info["prefix"] == info["base_prefix"]:
        raise CheckError("existing bin/python does not identify itself as a virtual environment")
    if Path(info["prefix"]).resolve() != directory.resolve():
        raise CheckError("existing bin/python belongs to a different virtual environment")
    return "existing isolated Python 3.12 environment is reusable"


def check_command(name):
    executable = shutil.which(name)
    if not executable:
        raise CheckError(f"{name} not in PATH; install git / build-essential as appropriate")
    version = run([executable, "--version"]).splitlines()
    return f"{executable}; {version[0] if version else 'command available'}"


def check_toolkit():
    executable = shutil.which("nvcc")
    if not executable:
        raise CheckError("nvcc not in PATH; select CUDA Toolkit 13.0 (nvidia-smi alone is insufficient)")
    version = run([executable, "--version"])
    if not re.search(r"release\s+13\.0,", version):
        raise CheckError("selected nvcc is not CUDA 13.0; this installation recipe is pinned to 13.0")
    targets = run([executable, "--list-gpu-code"])
    if "sm_121" not in targets.split():
        raise CheckError("selected nvcc does not list the sm_121 compilation target")
    toolkit = Path(executable).resolve().parent.parent
    headers = [toolkit / "include", toolkit / "targets" / "aarch64-linux" / "include"]
    if not any((path / "cuda.h").is_file() and (path / "cuda_runtime.h").is_file()
               for path in headers):
        raise CheckError(f"CUDA development headers missing under {toolkit}")
    libraries = [toolkit / "lib64", toolkit / "lib",
                 toolkit / "targets" / "aarch64-linux" / "lib"]
    if not any((path / "libcudart.so").is_file() for path in libraries):
        raise CheckError(f"CUDA development library libcudart.so missing under {toolkit}")
    return f"CUDA 13.0 with sm_121, headers and libcudart.so; root={toolkit}"


def inspect_driver():
    """Query libcuda directly; no PyTorch or CUDA Toolkit Python package needed."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise CheckError(f"cannot load NVIDIA driver libcuda.so.1: {exc}") from exc
    integer_pointer = ctypes.POINTER(ctypes.c_int)
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [integer_pointer],
        "cuDeviceGet": [integer_pointer, ctypes.c_int],
        "cuDeviceGetName": [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
        "cuDeviceGetAttribute": [integer_pointer, ctypes.c_int, ctypes.c_int],
        "cuDriverGetVersion": [integer_pointer],
    }
    for name, arguments in signatures.items():
        try:
            function = getattr(driver, name)
        except AttributeError as exc:
            raise CheckError(f"driver lacks {name}") from exc
        function.argtypes, function.restype = arguments, ctypes.c_int

    def invoke(name, *arguments):
        result = getattr(driver, name)(*arguments)
        if result:
            raise CheckError(f"{name} returned CUDA error {result}; check the driver, device access and CUDA visibility")

    invoke("cuInit", 0)
    count, device, major, minor, version = (ctypes.c_int() for _ in range(5))
    invoke("cuDeviceGetCount", ctypes.byref(count))
    if count.value < 1:
        raise CheckError("CUDA driver reports no visible GPU")
    invoke("cuDeviceGet", ctypes.byref(device), 0)
    name = ctypes.create_string_buffer(256)
    invoke("cuDeviceGetName", name, len(name), device.value)
    # CUDA Driver API CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR}.
    invoke("cuDeviceGetAttribute", ctypes.byref(major), 75, device.value)
    invoke("cuDeviceGetAttribute", ctypes.byref(minor), 76, device.value)
    invoke("cuDriverGetVersion", ctypes.byref(version))
    return {"name": name.value.decode("utf-8", errors="replace"),
            "capability": (major.value, minor.value), "driver_cuda": version.value,
            "count": count.value}


def check_driver():
    info = inspect_driver()
    if info["capability"] != (12, 1):
        raise CheckError(f"GPU 0 is {info['name']}, capability {info['capability']}; expected GB10 SM121")
    if info["driver_cuda"] < 13000:
        raise CheckError(f"driver CUDA API version {info['driver_cuda']} is below 13000; update the GB10 driver for CUDA 13.0")
    version = info["driver_cuda"]
    return (f"GPU 0: {info['name']}, SM121; {info['count']} visible GPU(s); "
            f"driver supports CUDA {version // 1000}.{version % 1000 // 10}")


def verify_cutlass_checkout(directory):
    if not directory.is_dir():
        raise CheckError(f"{directory} is not a readable directory")
    prefix = ["git", "--no-optional-locks", "-C", str(directory)]
    head = run(prefix + ["rev-parse", "HEAD"])
    tag = run(prefix + ["rev-parse", "refs/tags/v3.4.1^{commit}"])
    if head != tag:
        raise CheckError(f"{directory} is not checked out at v3.4.1; no files were changed")
    if run(prefix + ["status", "--porcelain", "--untracked-files=all"]):
        raise CheckError(f"{directory} has local changes/untracked files; inspect it before using its headers")
    return head


def check_cutlass(root=ROOT, workspace=Path("/workspace/cutlass")):
    sibling = root.parent / "cutlass-v3.4.1"
    existing = []
    for directory in (sibling, workspace):
        if directory.exists() or directory.is_symlink():
            if directory.resolve() not in [path.resolve() for path in existing]:
                verify_cutlass_checkout(directory)
                existing.append(directory)
    if not existing:
        return "not present; setup will download stock CUTLASS v3.4.1"
    return "clean v3.4.1: " + ", ".join(str(path) for path in existing)


def report_resources(report, root=ROOT):
    free_gib = shutil.disk_usage(root).free / GIB
    report.emit("PASS" if free_gib >= 20 else "WARN", "Free disk",
                f"{free_gib:.1f} GiB at repository; >=20 GiB is a planning recommendation, not a measured minimum")
    try:
        memory = Path("/proc/meminfo").read_text(encoding="utf-8")
        match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", memory, re.MULTILINE)
        if not match:
            raise ValueError("MemAvailable not found")
        available_gib = int(match.group(1)) * 1024 / GIB
        report.emit("PASS" if available_gib >= 16 else "WARN", "Available RAM",
                    f"{available_gib:.1f} GiB; >=16 GiB headroom recommended for build/autotuning, not a measured minimum")
    except (OSError, ValueError) as exc:
        report.emit("WARN", "Available RAM", f"could not inspect /proc/meminfo: {exc}")


def main():
    report = Report()
    print("GB10 preflight: read-only; no install/download/build and no PyTorch requirement.")
    for label, action in [
        ("Platform", check_platform),
        ("Python build prerequisites", check_base_python),
        ("Existing environment", check_existing_venv),
        ("git", lambda: check_command("git")),
        ("C++ compiler", lambda: check_command("c++")),
        ("CUDA Toolkit", check_toolkit),
        ("GPU / NVIDIA driver", check_driver),
        ("CUTLASS conflict check", check_cutlass),
    ]:
        report.check(label, action)
    report_resources(report)
    print("[INFO] Setup installs PyTorch 2.9.1/cu130 in its own venv; no current Torch install is required.")
    print(f"Summary: {report.failures} failure(s), {report.warnings} warning(s).")
    if report.failures:
        print("STOP: fix FAIL items before running setup. No environment changes were made.")
        return 1
    print("Prerequisites passed. Continue to setup; FLUTE compilation and GPU correctness are NOT yet validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
