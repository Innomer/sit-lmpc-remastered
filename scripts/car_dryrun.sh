#!/usr/bin/env bash
# Dry run of a car config against the ROS sim (car backend, real time) on the Levine sim map, with a scripted
# operator: at every car_waiting it puts the sim car 2 m behind the start line and presses go; after
# <laps_before_offtrack> laps it teleports the car off the track once (off-track stop and restart test).
#   bash scripts/car_dryrun.sh <config> <name> [laps_before_offtrack] [restarts]
#   bash scripts/car_dryrun.sh configs/car/dryrun-levine-v11.yaml dry-v11 6 9
# Results: results/dryrun/<name>/, logs results/dryrun/<name>.log and <name>.bridge.log. Needs ~/sim_ws/install
# (f1tenth_gym_ros) and ~/sim_ws/.venv.
set -u
cfg=$1
name=$2
laps=${3:-99}
total=${4:-3}
W=$(cd "$(dirname "$0")/.." && pwd)
cd "$W"
set +u
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77} PYTHONPATH=$W/src:${PYTHONPATH:-} XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
out=$W/results/dryrun/$name
mkdir -p "$(dirname "$out")"
PYTHONPATH=~/sim_ws/src/f1tenth_gym_ros:$PYTHONPATH ~/sim_ws/.venv/bin/python -m f1tenth_gym_ros.gym_bridge \
  --ros-args -r __node:=bridge --params-file configs/car/ros-bridge-dryrun-levine.yaml > "$out.bridge.log" 2>&1 &
bridge=$!
for _ in $(seq 60); do grep -q "simulated sensors on" "$out.bridge.log" && break; sleep 1; done
~/sim_ws/.venv/bin/python -m sit_lmpc.cli.racecar --config "$cfg" --save-dir "$out" --headless > "$out.log" 2>&1 &
racer=$!
pose() { ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: map}, pose: {pose: {position: {x: $1, y: $2}, orientation: {z: $3, w: $4}}}}" > /dev/null 2>&1; }
go() { ros2 topic pub --once /sit_lmpc/go std_msgs/msg/Empty > /dev/null 2>&1; }
count() { c=$(grep -c "$1" "$out/events.jsonl" 2>/dev/null); echo "${c:-0}"; }
teleported=0
for k in $(seq "$total"); do
  for _ in $(seq 600); do [ "$(count car_waiting)" -ge "$k" ] && break; kill -0 $racer 2>/dev/null || break; sleep 1; done
  kill -0 $racer 2>/dev/null || { echo "[dry] racer ended"; break; }
  [ "$(count car_waiting)" -ge "$k" ] || { echo "[dry] timeout waiting for car_waiting #$k"; break; }
  done_laps=$(count '"event": "lap"')
  echo "[dry] car_waiting #$k after $done_laps laps: behind the line, go"
  sleep 2; pose -2.0 -0.07 0.0037 1.0; sleep 2; go
  if [ "$teleported" -eq 0 ] && [ "$done_laps" -ge "$laps" ]; then
    for _ in $(seq 30); do [ "$(count car_launch)" -ge "$k" ] && break; sleep 1; done
    sleep 6; echo "[dry] teleport off the track"; pose 5.0 3.0 0.0037 1.0
    teleported=1
  fi
done
sleep 5
echo "[dry] Ctrl-C"
kill -INT $racer 2>/dev/null
for _ in $(seq 60); do kill -0 $racer 2>/dev/null || break; sleep 1; done
kill -0 $racer 2>/dev/null && echo "[dry] racer still running" || echo "[dry] racer exited"
kill -INT $bridge 2>/dev/null; sleep 2; kill $bridge 2>/dev/null
grep -E "^\[(lap|attempt|car_stop|car_lap_stop|imu_leveled|imu_level_skipped|finish|error|control_timing_summary)\]|Traceback" "$out.log" | cut -c1-160
