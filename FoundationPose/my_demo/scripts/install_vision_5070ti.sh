#!/usr/bin/env bash
# Install the ROS 2 vision process on an Ubuntu 22.04 RTX 5070 Ti workstation.
set -euo pipefail

_sb_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
_sb_env="${SB_5070TI_ENV:-$_sb_root/.envs/sb_vision_5070ti}"
_sb_log_dir="$_sb_root/runs/install_5070ti"
_sb_log="$_sb_log_dir/install.log"

_sb_die() { echo "ERROR: $*" >&2; exit 1; }
_sb_step() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
_sb_need_file() { [ -f "$1" ] || _sb_die "Missing $1"; }

if [ "${1:-}" = "--help" ]; then
  echo "Usage: bash my_demo/scripts/install_vision_5070ti.sh"
  echo "Requires Ubuntu 22.04 x86_64, RTX 5070 Ti driver, Conda, ROS 2 Humble,"
  echo "internet access, and the local model files listed in README_5070ti_install.md."
  echo "Installs to: $_sb_env (override with SB_5070TI_ENV)."
  exit 0
fi
[ "$#" -eq 0 ] || _sb_die "Unknown option: $*"
case "$_sb_env" in /*) ;; *) _sb_die "SB_5070TI_ENV must be an absolute path" ;; esac
[ "$(uname -m)" = x86_64 ] || _sb_die "This installer needs Linux x86_64"
_sb_need_file /etc/os-release
# shellcheck disable=SC1091
source /etc/os-release
[ "${ID:-}" = ubuntu ] && [ "${VERSION_ID:-}" = 22.04 ] || _sb_die "ROS 2 Humble installer expects Ubuntu 22.04"
_sb_need_file /opt/ros/humble/setup.bash
command -v nvidia-smi >/dev/null || _sb_die "NVIDIA driver/nvidia-smi is missing"
_sb_gpu_list="$(nvidia-smi --query-gpu=name --format=csv,noheader)" || _sb_die "nvidia-smi cannot query the GPU; check the NVIDIA driver"
_sb_gpu="${_sb_gpu_list%%$'\n'*}"
case "$_sb_gpu" in *"RTX 5070 Ti"*) ;; *) _sb_die "Expected RTX 5070 Ti, found: $_sb_gpu" ;; esac
command -v conda >/dev/null || _sb_die "Conda is missing (the control computer already uses Conda)"
command -v colcon >/dev/null || _sb_die "Install ROS 2 colcon before running this installer"

# Fail before modifying the environment if the private model files were not transferred.
_sb_need_file "$_sb_root/runs/yolov8/best.pt"
_sb_need_file "$_sb_root/runs/mesh/bottle.obj"
_sb_need_file "$_sb_root/weights/2023-10-28-18-33-37/model_best.pth"
_sb_need_file "$_sb_root/weights/2024-01-11-20-02-45/model_best.pth"
_sb_need_file "$_sb_root/sb_vision_handoff_20260910/sb_communication/msg/ObjectPose.msg"
_sb_need_file "$_sb_root/my_demo/config/bottle_current.json"
_sb_need_file "$_sb_root/my_demo/requirements-5070ti.txt"

(
  cd "$_sb_root"
  printf '%s\n' \
    '1d66903956f9fe8913f0d75bc71392af351893f4e5ff555ba9d34cb9680f0910  runs/yolov8/best.pt' \
    '566773b675e80bda7a76b0715336c2b144663492c0cd06cc0b4e6e5292862bc2  runs/mesh/bottle.obj' \
    '774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60  weights/2023-10-28-18-33-37/model_best.pth' \
    '81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26  weights/2024-01-11-20-02-45/model_best.pth' \
    '03463e448beaa0829223ed5afd90993ece5f307b9217562b8c163e8d853f0bda  sb_vision_handoff_20260910/sb_communication/msg/ObjectPose.msg' \
    | sha256sum --check --status
) || _sb_die "One or more model/message checksums differ from the validated project"

mkdir -p "$_sb_log_dir"
exec > >(tee -a "$_sb_log") 2>&1
_sb_step "GPU: $_sb_gpu; environment: $_sb_env"
_sb_step "Project assets and ObjectPose definition: SHA-256 verified"

# Conda's CUDA toolkit supplies nvcc locally; no system CUDA or sudo is needed.
_sb_conda_exe="$(command -v conda)"
eval "$("$_sb_conda_exe" shell.bash hook)"
if [ ! -x "$_sb_env/bin/python" ]; then
  _sb_step "Creating Python 3.10 environment"
  conda create -y -p "$_sb_env" -c conda-forge \
    python=3.10 pip cmake ninja eigen boost-cpp pybind11 \
    gcc_linux-64=12 gxx_linux-64=12
fi
if [ ! -x "$_sb_env/bin/nvcc" ]; then
  _sb_step "Installing CUDA 12.8 toolkit in Conda environment"
  conda install -y -p "$_sb_env" -c nvidia -c conda-forge cuda-toolkit=12.8
fi

conda activate "$_sb_env"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export FORCE_CUDA=1
export MAX_JOBS="${SB_BUILD_JOBS:-2}"
[ -x "$CUDA_HOME/bin/nvcc" ] || _sb_die "CUDA toolkit has no nvcc at $CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version

_sb_step "Installing PyTorch 2.7.1 with CUDA 12.8"
python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
assert torch.__version__.startswith('2.7.1+cu128'), torch.__version__
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.get_device_capability(0) == (12, 0), torch.cuda.get_device_capability(0)
x = torch.randn(32, device='cuda')
torch.cuda.synchronize()
assert bool(torch.isfinite((x * x).sum()))
print('PyTorch CUDA kernel passed:', torch.__version__, torch.cuda.get_device_name(0))
PY

_sb_step "Installing Python runtime dependencies"
python -m pip install -r "$_sb_root/my_demo/requirements-5070ti.txt"
python -m pip check

_sb_step "Compiling PyTorch3D v0.7.9 and nvdiffrast v0.4.0 for sm_120"
python -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'
python -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0'
python -m pip check

_sb_step "Building local FoundationPose mycpp extension"
cmake -S "$_sb_root/mycpp" -B "$_sb_root/mycpp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPython3_ROOT_DIR="$CONDA_PREFIX" \
  -DPYBIND11_PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build "$_sb_root/mycpp/build" --clean-first -j "$MAX_JOBS"

# Build ROS message bindings with the OS Python used by ROS 2 Humble.
_sb_step "Building sb_communication with ROS system Python"
env -u CONDA_PREFIX -u CMAKE_PREFIX_PATH -u PYTHONPATH -u PYTHONHOME \
  -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  PATH="/usr/bin:/bin:/opt/ros/humble/bin:$PATH" \
  bash "$_sb_root/my_demo/scripts/build_sb_interfaces.sh"
# shellcheck disable=SC1091
source "$_sb_root/my_demo/ros2_handoff_env.sh"

_sb_step "Checking ROS bindings and real CUDA extension kernels"
export SB_VISION_PYTHON="$CONDA_PREFIX/bin/python"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_EXTENSIONS_DIR="$_sb_root/.cache/torch_extensions_5070ti"
mkdir -p "$TORCH_EXTENSIONS_DIR"
python - <<'PY'
import torch
import nvdiffrast.torch as dr
from pytorch3d.ops import knn_points
from sb_communication.msg import ObjectPose
from ultralytics import YOLO
from pathlib import Path
import sys
root = Path.cwd()
sys.path.insert(0, str(root / 'my_demo'))
sys.path.insert(0, str(root / 'mycpp/build'))
import mycpp
from sb_pose_contract import load_object_profile
assert load_object_profile(root / 'my_demo/config/bottle_current.json')['object_id'] == 'bottle'
assert len(ObjectPose().covariance) == 36
p = torch.randn(1, 8, 3, device='cuda')
assert knn_points(p, p, K=1).idx.is_cuda
ctx = dr.RasterizeCudaContext()
vertices = torch.tensor([[[-0.5, -0.5, 0.5, 1.], [0.5, -0.5, 0.5, 1.], [0., 0.5, 0.5, 1.]]], device='cuda')
triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
image, _ = dr.rasterize(ctx, vertices, triangles, resolution=[16, 16])
assert image.is_cuda and image.shape == (1, 16, 16, 4)
model = YOLO(str(root / 'runs/yolov8/best.pt'))
assert 0 in model.names, model.names
print('CUDA extensions, ObjectPose, mesh, and YOLO checkpoint: OK')
PY

_sb_step "Running local vision unit tests and GPU pose smoke test"
cd "$_sb_root"
python -m unittest discover -s my_demo/tests -p 'test_*.py'
python my_demo/scripts/gpu_smoke.py
_sb_step "Checking compressed RGB-D and ObjectPose over local CycloneDDS"
export SB_ROS_DOMAIN_ID=93
export SB_DDS_INTERFACE=lo
unset SB_DDS_PEERS
# shellcheck disable=SC1091
source "$_sb_root/my_demo/ros2_handoff_env.sh"
python my_demo/scripts/test_ros_handoff.py --image-transport compressed
python -m pip freeze > "$_sb_log_dir/pip-freeze.txt"
_sb_step "Installation complete. Log: $_sb_log"
_sb_step "Use my_demo/scripts/run_sb_vision_5070ti.sh with your chosen ROS Domain and clock source"
