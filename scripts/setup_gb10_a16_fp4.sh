#!/usr/bin/env bash
# Build the stock FLUTE library for this GB10 experiment, without kernel edits.
# This script is syntax-checked only; a GB10 build/run is still required.
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_machine="$(uname -m)"
if [[ "$(uname -s)" != Linux || ( "$task_machine" != aarch64 && "$task_machine" != arm64 ) ]]; then
    echo "This setup targets Linux ARM64 GB10 / SM121 only." >&2
    exit 2
fi
for task_command in python3.12 nvcc c++ git; do
    if ! command -v "$task_command" >/dev/null 2>&1; then
        echo "Missing $task_command. Install Python 3.12 + venv, build-essential, git and CUDA Toolkit 13.0 first." >&2
        exit 2
    fi
done
task_nvcc_version="$(nvcc --version)"
if [[ "$task_nvcc_version" != *"release 13.0,"* ]]; then
    echo "This recipe is pinned to CUDA Toolkit 13.0; select its nvcc in PATH." >&2
    exit 2
fi

# Check driver/device/toolkit/headers and directory conflicts BEFORE creating
# the isolated environment, downloading dependencies or compiling the library.
python3.12 "$task_root/scripts/check_gb10_env.py"

task_venv="$task_root/.venv-gb10-flute"
if [[ -e "$task_venv" && ! -f "$task_venv/pyvenv.cfg" ]]; then
    echo "$task_venv exists but is not a virtual environment; refusing to overwrite it." >&2
    exit 2
fi
if [[ ! -e "$task_venv" ]]; then
    python3.12 -m venv "$task_venv"
fi
task_python="$task_venv/bin/python"
if [[ ! -x "$task_python" ]]; then
    echo "Missing $task_python; inspect the existing environment before retrying." >&2
    exit 2
fi
# Never install/downgrade PyTorch into the user's active DeepGEMM environment.
# Also reject site-package-sharing environments to keep this isolation explicit.
"$task_python" - <<'PY'
import pathlib, sys
config = (pathlib.Path(sys.prefix) / "pyvenv.cfg").read_text().lower()
if "include-system-site-packages = true" in config:
    raise SystemExit("This setup requires an isolated venv, not --system-site-packages. Use the manual guide for an existing shared environment.")
PY
"$task_python" -m pip install setuptools packaging ninja wheel click jaxtyping
"$task_python" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
"$task_python" - <<'PY'
import torch
import triton.testing
assert torch.__version__.split('+')[0] == '2.9.1', torch.__version__
assert torch.version.cuda == '13.0', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA PyTorch is not working'
assert torch.cuda.get_device_capability(0) == (12, 1), 'Expected GB10 SM121'
torch.ones(16, device='cuda').add_(1)
torch.cuda.synchronize()
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY

task_cutlass="$task_root/../cutlass-v3.4.1"
if [[ ! -e "$task_cutlass" ]]; then
    git clone --branch v3.4.1 --depth 1 https://github.com/NVIDIA/cutlass.git "$task_cutlass"
fi
task_cutlass="$(cd -- "$task_cutlass" && pwd)"
task_tag="$(git -C "$task_cutlass" rev-parse 'refs/tags/v3.4.1^{commit}')"
task_head="$(git -C "$task_cutlass" rev-parse HEAD)"
if [[ "$task_head" != "$task_tag" || -n "$(git -C "$task_cutlass" status --porcelain --untracked-files=no)" ]]; then
    echo "CUTLASS directory must be an unmodified v3.4.1 checkout: $task_cutlass" >&2
    exit 2
fi
# setup.py has a hardcoded include path; refuse a conflicting copy rather than
# silently mixing headers or changing/removing the user's /workspace directory.
if [[ -e /workspace/cutlass && "$(readlink -f /workspace/cutlass)" != "$task_cutlass" ]]; then
    if [[ "$(git -C /workspace/cutlass rev-parse HEAD)" != "$task_tag" || -n "$(git -C /workspace/cutlass status --porcelain --untracked-files=no)" ]]; then
        echo "Conflicting /workspace/cutlass detected. See docs/gb10_a16_fp4_g128.md; no files were changed there." >&2
        exit 2
    fi
fi
export CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f "$(command -v nvcc)")")")"
export TORCH_CUDA_ARCH_LIST=12.1
export MAX_JOBS="${MAX_JOBS:-1}"
export CPATH="$task_cutlass/include:$task_cutlass/tools/util/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$task_cutlass/include:$task_cutlass/tools/util/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export NVCC_PREPEND_FLAGS="-I$task_cutlass/include -I$task_cutlass/tools/util/include${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"
cd -- "$task_root"
"$task_python" -m pip install -v -e . --no-build-isolation --no-deps 2>&1 | tee build-gb10.log
"$task_python" - <<'PY'
import flute
import flute._C
import flute.tune
assert flute.TEMPLATE_CONFIGS
print('FLUTE:', flute.__file__)
print('Extension:', flute._C.__file__)
print('Build/import passed. GPU GEMM correctness and timing still need the smoke test.')
PY
echo "Next: source \"$task_venv/bin/activate\""
echo "Then run the smoke command in README.md. Keep build-gb10.log if anything fails."
