#!/usr/bin/env bash
# Root-cause runs: one sim run with the code of a given checkout (e.g. an old commit in a git worktree).
#   bash scripts/run_with_code.sh <code dir> <config> <bridge config> <scenario> <run name> <seed> [max laps]
# Uses <code dir>/src (PYTHONPATH ahead of the installed package) and <code dir>/scripts/condition_scheduler.py;
# results go to results/Spielberg/rootcause/<run name> in this checkout.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
code=$(cd "$1" && pwd); config=$2; bridge_config=$3; scenario=$4; name=$5; seed=$6; laps=${7:-15}
set +u; source /opt/ros/jazzy/setup.bash; source ~/sim_ws/install/setup.bash; set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
out="$repo/results/Spielberg/rootcause/$name"
mkdir -p "$out"
cd "$code"
PYTHONPATH=~/sim_ws/src/f1tenth_gym_ros:${PYTHONPATH:-} ~/sim_ws/.venv/bin/python -m f1tenth_gym_ros.gym_bridge \
  --ros-args -r __node:=bridge --params-file "$bridge_config" > "$out.bridge.log" 2>&1 &
bridge=$!
trap 'kill -INT $bridge $scheduler 2>/dev/null || true' EXIT
for _ in $(seq 60); do grep -q "simulated sensors on" "$out.bridge.log" && break; sleep 1; done
PYTHONPATH="$code/src:${PYTHONPATH:-}" ~/sim_ws/.venv/bin/python scripts/condition_scheduler.py "$scenario" --log "$out.conditions.jsonl" > "$out.scheduler.log" 2>&1 &
scheduler=$!
PYTHONPATH="$code/src:${PYTHONPATH:-}" ~/sim_ws/.venv/bin/python -c "import sys; from sit_lmpc.cli.racecar import main; sys.argv[0] = 'sit-racecar'; sys.exit(main())" \
  --config "$config" --seed "$seed" --max-laps "$laps" --save-dir "$out" > "$out.log" 2>&1 || true
