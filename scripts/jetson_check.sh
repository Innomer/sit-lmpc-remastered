#!/usr/bin/env bash
# Jetson check in one command, from this checkout: JAX GPU check, then planning p50/p95 for every config in
# configs/car/ with suggested cuts when p95 > 90 ms. Results in results/jetson-check.json.
#   bash scripts/jetson_check.sh [python]
set -euo pipefail
cd "$(dirname "$0")/.."
python=${1:-}
if [ -z "$python" ]; then
  for candidate in .venv/bin/python ~/sit-lmpc-remastered/.venv/bin/python ~/sim_ws/.venv/bin/python python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import jax" 2>/dev/null; then python=$candidate; break; fi
  done
fi
[ -n "$python" ] || { echo "no python with jax found; pass one: bash scripts/jetson_check.sh /path/to/python"; exit 1; }
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}
echo "== python: $python"
"$python" scripts/jetson_check.py "${@:2}"
