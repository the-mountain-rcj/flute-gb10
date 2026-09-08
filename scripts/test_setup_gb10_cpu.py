"""Bash argument/preflight plumbing only; never install, compile, or use CUDA."""

from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/setup_gb10_a16_fp4.sh"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = (str(GIT_BASH) if sys.platform == "win32" and GIT_BASH.is_file()
        else shutil.which("bash"))


@unittest.skipUnless(BASH, "Bash unavailable")
class SetupShellTests(unittest.TestCase):
    def run_bash(self, *args):
        return subprocess.run([BASH, *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=20, check=False)

    def test_syntax(self):
        result = self.run_bash("-n", SCRIPT)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_does_not_need_linux_or_cuda(self):
        result = self.run_bash(SCRIPT, "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--experimental-cuda-13-4", result.stdout)
        self.assertIn("PyTorch 2.9.1/cu130", result.stdout)

    def test_unknown_argument_rejected_before_install(self):
        result = self.run_bash(SCRIPT, "--skip-checks")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Unknown argument", result.stderr)

    def test_wrong_platform_rejected_with_experimental_flag(self):
        result = self.run_bash("-c", r'''
uname() { printf '%s\n' Windows; }
source "$1" --experimental-cuda-13-4
''', "test-setup", SCRIPT)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Linux ARM64", result.stderr)

    def test_failed_preflight_stops_both_modes(self):
        # Source in a fresh Bash process so the function mocks are inherited
        # without altering any real executable. The first Python invocation
        # always fails, before the installer can create its venv or download.
        preamble = r'''
uname() {
    if [[ "$1" == -m ]]; then printf '%s\n' aarch64;
    else printf '%s\n' Linux; fi
}
python3.12() {
    printf 'MOCK_PREFLIGHT:%s\n' "$@"
    return 42
}
nvcc() { printf '%s\n' 'Cuda compilation tools, release 13.4, V13.4.59'; }
c++() { return 99; }
git() { return 99; }
task_mock_script="$1"
shift
source "$task_mock_script" "$@"
'''
        for flags in ([], ["--experimental-cuda-13-4"]):
            with self.subTest(flags=flags):
                result = self.run_bash("-c", preamble, "test-setup", SCRIPT, *flags)
                self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
                lines = result.stdout.splitlines()
                self.assertEqual(len(lines), 1 + len(flags), result.stdout)
                self.assertIn("scripts/check_gb10_env.py", lines[0])
                if flags:
                    self.assertEqual(lines[1], "MOCK_PREFLIGHT:--experimental-cuda-13-4")


if __name__ == "__main__":
    unittest.main()
