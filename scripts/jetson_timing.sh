#!/usr/bin/env bash
# World-model planning-time test on the Jetson, in one command, from a world-model checkout:
#   bash scripts/jetson_timing.sh [python]
# Uses this checkout's code (PYTHONPATH) with any existing venv that has JAX, so the car's irl checkout and venv stay
# untouched. Writes results/wm-timing/jetson-{parts,planner-known}.json and prints the variant table. ~20-40 min.
set -euo pipefail
cd "$(dirname "$0")/.."
python=${1:-}
if [ -z "$python" ]; then
  for candidate in .venv/bin/python ~/sit-lmpc-remastered/.venv/bin/python ~/sim_ws/.venv/bin/python; do
    if [ -x "$candidate" ] && "$candidate" -c "import jax" 2>/dev/null; then python=$candidate; break; fi
  done
fi
[ -n "$python" ] || { echo "no python with jax found; pass one: bash scripts/jetson_timing.sh /path/to/python"; exit 1; }
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}
mkdir -p results/wm-timing
echo "== python: $python"
"$python" -c "import jax; print('devices:', jax.devices())"
echo "== world-model parts"
"$python" scripts/time_world_model.py --out results/wm-timing/jetson-parts.json
echo "== planner with the known model (capped safe set)"
"$python" scripts/benchmark_planning.py --models known --samples 1024 512 256 --horizons 15 10 \
  --out results/wm-timing/jetson-planner-known.json
echo "== variant table"
"$python" scripts/combine_timing.py results/wm-timing/jetson-planner-known.json results/wm-timing/jetson-parts.json \
  | tee results/wm-timing/jetson-table.txt
