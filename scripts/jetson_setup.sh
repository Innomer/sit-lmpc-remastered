#!/usr/bin/env bash
# Set up SIT-LMPC on the car's Jetson: venv with ROS Python visible, the package, GPU JAX, and a check.
# Run from the repo root after sourcing ROS:  source /opt/ros/$ROS_DISTRO/setup.bash && bash scripts/jetson_setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
echo "== system"
uname -m
cat /etc/nv_tegra_release 2>/dev/null || echo "no /etc/nv_tegra_release (not L4T?)"
echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
python3 --version

if [ -z "${ROS_DISTRO:-}" ]; then
  echo "Source ROS first (source /opt/ros/<distro>/setup.bash)." >&2
  exit 1
fi
ros_python=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! python3 -c 'import rclpy, ackermann_msgs.msg' 2>/dev/null; then
  echo "rclpy or ackermann_msgs is not importable from python3; source the car workspace too." >&2
  exit 1
fi
case "$ros_python" in
  3.12|3.13) ;;
  *)
    echo "ROS $ROS_DISTRO uses Python $ros_python, but SIT-LMPC needs Python 3.12+ (JAX 0.11)." >&2
    echo "rclpy cannot be imported across Python versions. Options (docs/CAR_DEPLOYMENT.md, 'Python and ROS'):" >&2
    echo "  1. a ROS 2 Jazzy container (Ubuntu 24.04, Python 3.12) with --runtime nvidia --network host" >&2
    echo "  2. a ROS 2 Jazzy install on the Jetson" >&2
    exit 1
    ;;
esac

cuda_major=""
if [ -f /usr/local/cuda/version.json ]; then
  cuda_major=$(python3 -c 'import json; print(json.load(open("/usr/local/cuda/version.json"))["cuda"]["version"].split(".")[0])')
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  cuda_major=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
elif command -v nvidia-smi >/dev/null; then
  cuda_major=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -1)
fi
echo "CUDA major version: ${cuda_major:-unknown}"

echo "== venv (.venv, with system site packages for rclpy)"
if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e vendor/dyna-dist -e .
case "$cuda_major" in
  13) python -m pip install "jax[cuda13]==0.11.2" ;;
  12) python -m pip install "jax[cuda12]==0.11.2" ;;
  *) echo "Unknown CUDA version; JAX stays CPU only (far too slow for LMPPI: ~500 ms per step)." >&2 ;;
esac

echo "== check"
python - <<'EOF'
import time
import jax
import jax.numpy as jnp
import rclpy
import sit_lmpc.adapters.ros_car
from sit_lmpc.config.schema import load_config

devices = jax.devices()
print("JAX devices:", devices)
x = jnp.ones((2048, 2048))
jax.block_until_ready(x @ x)
started = time.perf_counter()
for _ in range(10):
    jax.block_until_ready(x @ x)
print(f"2048x2048 matmul: {(time.perf_counter() - started) / 10 * 1000:.1f} ms")
for name in ("car-levine-known", "car-levine-bnn-nochance", "car-levine-bnn"):
    load_config(f"configs/{name}.yaml").validate()
print("configs OK")
if devices[0].platform == "cpu":
    print("WARNING: JAX runs on the CPU. LMPPI needs the GPU to fit the 100 ms control period.")
EOF
echo "Done. Next: python scripts/car_check.py --config configs/car-levine-known.yaml (docs/CAR_DEPLOYMENT.md)"
