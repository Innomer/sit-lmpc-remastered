#!/usr/bin/env bash
# One mid-session change experiment: sensor-enabled lockstep bridge + condition scheduler + one controller.
#   bash scripts/run_change_experiment.sh <controller config> <scenario yaml> <run name> [max laps]
# Every controller gets the same bridge, sensors, scenario and seed. Results: results/Spielberg/<run name>.
set -euo pipefail
cd "$(dirname "$0")/.."
config=$1
scenario=$2
name=$3
laps=${4:-15}
set +u
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
out=results/Spielberg/$name
mkdir -p "$out"
PYTHONPATH=~/sim_ws/src/f1tenth_gym_ros:${PYTHONPATH:-} ~/sim_ws/.venv/bin/python -m f1tenth_gym_ros.gym_bridge \
  --ros-args -r __node:=bridge --params-file ${BRIDGE_CONFIG:-configs/ros-bridge-lockstep-sensors.yaml} > "$out.bridge.log" 2>&1 &
bridge=$!
trap 'kill -INT $bridge $scheduler 2>/dev/null || true' EXIT
for _ in $(seq 60); do grep -q "simulated sensors on" "$out.bridge.log" && break; sleep 1; done
~/sim_ws/.venv/bin/python scripts/condition_scheduler.py "$scenario" --log "$out.conditions.jsonl" > "$out.scheduler.log" 2>&1 &
scheduler=$!
~/sim_ws/.venv/bin/sit-racecar --config "$config" --seed 1 --max-laps "$laps" --save-dir "$out" > "$out.log" 2>&1
grep -v Warn "$out.log" | grep "^\[lap\]\|^\[attempt\].*\(failure\|incomplete\)\|finish\]" | cut -c1-90
