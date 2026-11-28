# Car runbook: SIT-LMPC with the v11 world model on the F1TENTH car

Branch `irl-world-model`. Every step below is a command to type, in order. Steps 1–5 need no driving; steps 6–10 drive
the car. Stop at any step whose check fails and fix that first.

Car: Jetson Orin Nano Super 8 GB, JetPack 7 (L4T R39, Ubuntu 24.04), ROS 2 Jazzy, Python 3.12, CUDA 13.2.
Track: Levine (`tracks/levine_self_again`). All car configs cap every command at 7 m/s
(`simulation.car.max_speed`), compensate the 0.2 s actuation delay, and use retry mode: after any stop the car waits
for the operator's go at the start line.

**The car stops after every lap.** After crossing the line it slows down along the centreline (3 m/s², about 2.5 s
from 7 m/s), trains the value function and saves the checkpoint while standing (a few seconds), then prints
`car_waiting`. Carry it back 0.3–3 m behind the line, facing forward, and press go: it drives to the line at 1.5 m/s
and starts the next lap. Training while driving paused the control loop for up to 0.6 s in the dry runs, which at
7 m/s ended in the wall; standing still makes that harmless. After an off-track or obstacle stop it is the same:
carry it back, press go (the failed attempt is counted). At startup the run compiles the planner and the value
training (about 20–30 s on the laptop, more on the Jetson) before the first `car_waiting`.

| Config | Planner model | Learned parts | Hand-written rules |
|---|---|---|---|
| `configs/car/levine-bicycle-slow.yaml` | bicycle, 2 m/s cap (first floor laps) | value function | – |
| `configs/car/levine-bicycle.yaml` | bicycle (nominal single-track model) | value function | – |
| `configs/car/levine-v11.yaml` | rssm-v11 world model (`models/rssm-v11`) | world model, value function | – |
| `configs/car/levine-v11-learned.yaml` | v11 + change detection + learned change handling | world model, value with the condition tag | change detection, tag matching for the safe set |

The hand-written speed reaction is in the code but off in every car config (`planner.change_speed: null`).

## 1. Shell setup (every new terminal on the Jetson)

```bash
cd ~/sit-lmpc-remastered
git fetch && git checkout irl-world-model && git pull
source /opt/ros/jazzy/setup.bash
source <car workspace>/install/setup.bash        # the f1tenth_stack / particle_filter workspace
source .venv/bin/activate                         # created by scripts/jetson_setup.sh (first time only)
export ROS_DOMAIN_ID=<the car's domain id>
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35        # leave GPU memory for the ROS stack and the particle filter
```

First time only: `bash scripts/jetson_setup.sh` (creates the venv with ROS Python visible, installs the package and
GPU JAX, and validates the car configs).

In two other terminals (same shell setup), bring up the car. Always the full stack, never the LiDAR alone:

```bash
ros2 launch f1tenth_stack sick_bringup_launch.py   # this car: SICK TiM LiDAR (811 beams over 270 deg)
ros2 launch particle_filter localize_launch.py    # on the Levine map; set the initial pose in RViz
ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
  enable_color:=false enable_depth:=false enable_infra1:=false enable_infra2:=false   # the IMU
```

The VESC on this car has no IMU, so the IMU is the RealSense D435i's (`/camera/camera/imu`, 200 Hz). It needs
librealsense built with the RSUSB backend (the Jetson kernel has no HID sensor modules) and realsense-ros built
against it. Check: `ros2 topic hz /camera/camera/imu` about 200. The car configs remap the camera axes to the car
(`imu_axes: [z, -x, -y]`), level the tilted mount from gravity at the first start (`imu_level`, logged as
`imu_leveled`; keep the car still on flat floor when starting a run), move the accelerations to the centre of gravity
(`imu_position`: 0.346, 0.032 m from the rear axle, from the LiDAR TF and the vision-lab camera offset) and average
each 0.1 s.

## 2. Sensor check: `car_check.py`

Leave the car still for 3 s, then push or drive it slowly through left and right turns, accelerating and braking.
It never sends drive commands.

```bash
python scripts/car_check.py --config configs/car/levine-v11.yaml --seconds 40
```

| It prints | Do |
|---|---|
| topic rates and the largest PF gap | PF gaps must stay under 0.3 s |
| `IMU levelled ... mount tilt` | about 6 deg for the current camera mount |
| suggested `imu_yaw_scale` | should be about +1.0; set `simulation.car.imu_yaw_scale` in `configs/car/levine-bicycle.yaml` |
| IMU acceleration at rest, units, suggested `imu_accel_scale` | z about +9.8 (after the remap and levelling); x, y signs should come out +1 |
| "the IMU reports only zeros" | the IMU topic carries no data: start the RealSense node, or check `imu_topic` |
| VESC speed / PF speed ratio | must be 0.85–1.15 |
| LiDAR layout vs training (1080 beams, ±135°) | under 240° of coverage is a problem |
| v11's 9 inputs vs the training statistics | anything marked "far off" means wrong units, sign or frame |
| car's s / ey on the track, joystick buttons | set `go_button` |

All car configs inherit these settings from `configs/car/levine-bicycle.yaml`, so set them there once. Run the check
again until it prints `OK`.

## 3. Build the planner's track from your SLAM map

No new mapping: this turns the SLAM map the particle filter already uses into the planner's track files
(`tracks/levine_self_again/`: map image in the PF frame, centreline, wall distances). The copy in the repository was
made without the map (walls assumed 0.4 m away).

```bash
python scripts/make_track.py --raceline configs/racelines/levine_self_again_raceline.csv \
  --map <your SLAM map>.yaml --name levine_self_again --plot
```

| It prints / saves | Do |
|---|---|
| refuses: "only N% of the raceline lies in free space" | wrong map or frame; use the map the PF localizes on |
| wall distance min / 5th percentile / median | measure the narrowest corridor spot with a tape; the printed minimum should agree within a few cm |
| suggested `planner.half_width` | put it into `configs/car/levine-bicycle.yaml` (`planner.half_width`; the planner subtracts `footprint_buffer` 0.10 m) |
| `tracks/levine_self_again/levine_self_again_check.png` | the map with the measured walls. Orange circles: much narrower than nearby (often a blob of SLAM noise in the corridor). Red circles: no wall within 1 m outside a corner (a gap: glass, a door, missed scans, or a real side corridor) |

A noisy SLAM map is fine as it is: fuzzy or grey walls only make the measured width slightly smaller, and the planner
uses one half-width (the 5th percentile) for the whole track. Only if the printed widths disagree with the tape, or
the plot shows red circles where the wall is solid: paint those gaps black and corridor blobs white in the map image
and rerun.

## 4. Jetson planning-time check

```bash
bash scripts/jetson_check.sh
```

It checks that JAX sees the GPU, then times a full control step (v11's filter step + planner) for every config in
`configs/car/`. A config with p95 over 90 ms is retimed with 512 and 256 candidates and with a 10 × 0.15 s horizon;
the script prints the override to add to that config (for example `planner: samples: 512`). Results:
`results/jetson-check.json`. Do not drive v11 until its p95 is under 90 ms.

## 5. Bench test (car on blocks, wheels free)

```bash
python -m sit_lmpc.cli.racecar --config configs/car/levine-bicycle-slow.yaml
```

1. The run prints `car_waiting`. Put the car (on its blocks) 0.3–3 m behind the start line, facing forward.
2. Hold the deadman, press go (or `ros2 topic pub --once /sit_lmpc/go std_msgs/msg/Empty`). The wheels turn and steer.
3. Ctrl-C: the wheels must stop at once, and the run must save.
4. Restart, press go, release the deadman: the wheels stop (the car's mux).
5. Restart, lift the car off the blocks onto the floor away from the line and press go: the start is rejected
   (`car_start_rejected`, "reposition the car").

## 6. First floor laps (2 m/s)

```bash
python -m sit_lmpc.cli.racecar --config configs/car/levine-bicycle-slow.yaml
```

Car behind the start line, deadman held, go. It drives to the line, then pure-pursuit warm-up laps, then LMPPI laps,
stopping after every lap (carry it back, go).
Check in the log: `control_step_timing` (planning p95 under ~90 ms, overruns and watchdog stops near 0), `car_lap_stop`
after each lap, and lap times.
If it leaves the track or sees an obstacle it stops and waits; put it back behind the line and press go.

## 7. Bicycle baseline: 5 warm-up + 10 learned laps, with bags

```bash
python -m sit_lmpc.cli.racecar --config configs/car/levine-bicycle.yaml
```

15 laps: 5 warm-up, 10 learned. The run records its own bag (`simulation.car.record_bag: true`) in
`results/levine_self_again/car-levine-bicycle-<time>/bag`; check that the log prints `car_bag_recording`. To record by
hand instead (for example a joystick drive, or if the run's recording failed), in another terminal with the same shell
setup:

```bash
ros2 bag record -o bags/levine-$(date +%Y%m%d-%H%M%S) \
  /pf/pose/odom /odom /camera/camera/imu /sensors/core /scan /drive /joy /tf /tf_static
```

Ctrl-C stops the recording. These are the topics in `simulation.car.bag_topics`; `scripts/bag_check.py` needs at
least the pose, wheel speed, IMU, `/scan` and `/drive`. After a battery swap, resume from the last checkpoint:
`python -m sit_lmpc.cli.racecar --config configs/car/levine-bicycle.yaml --resume results/levine_self_again/car-levine-bicycle-<time>/checkpoints/trial-<N>`.

## 8. Offline bag check (laptop or Jetson, no driving)

```bash
python scripts/bag_check.py --bag results/levine_self_again/car-levine-bicycle-<time>/bag --config configs/car/levine-v11.yaml
```

It runs v11 along the recorded drive with the commands that were actually sent, and prints v11's and the bicycle
model's prediction error at 0.5, 1.0 and 1.5 s on what was measured (position against the PF, yaw rate against the
IMU, speed against the wheel speed). Stops and the carries back to the line are skipped. Use it to decide whether v11 is ready to drive: its position error should be
no worse than the bicycle model's. If it is far worse, rerun step 2 (input signs/units) before driving v11.

## 9. v11 run

```bash
python -m sit_lmpc.cli.racecar --config configs/car/levine-v11.yaml
```

Same procedure as step 7: 15 laps, bag recorded. Then the same with the learned change handling:

```bash
python -m sit_lmpc.cli.racecar --config configs/car/levine-v11-learned.yaml
```

With no change it must behave like v11 (no detections: `detected` events in `events.jsonl` are false detections).

## 10. Change test (mid-session)

Use `configs/car/levine-v11-learned.yaml` (and the bicycle config for comparison).

1. Drive 8 laps (5 warm-up + 3 learned).
2. After lap 8, when the car has stopped and prints `car_waiting`, press Ctrl-C (the checkpoint is saved during the
   stop).
3. Make the change: either tape 0.5–1 kg to the chassis over the rear axle, or lay a slippery sheet (plastic or
   cardboard, taped down flat) on one corner.
4. Resume from the lap-8 checkpoint so the safe set and value function carry over:

```bash
python -m sit_lmpc.cli.racecar --config configs/car/levine-v11-learned.yaml \
  --resume results/levine_self_again/car-levine-v11-learned-<time>/checkpoints/trial-8
```

5. Drive 7 more laps. Record: crashes or stops, lap times, the first `detected` event (time and stretch), which
   stretches are marked `changed`, and laps until lap times settle.

## Laptop dry run (no car)

The same car code path against the ROS sim, on the Levine sim map, with simulated IMU, wheel speed, PF pose and LiDAR.
Scripted (places the car behind the line and presses go at every stop, one off-track teleport after N laps):
`bash scripts/car_dryrun.sh configs/car/dryrun-levine-v11.yaml dry-v11 6 9`. By hand:

```bash
ros2 run f1tenth_gym_ros gym_bridge --ros-args -r __node:=bridge --params-file configs/car/ros-bridge-dryrun-levine.yaml
python -m sit_lmpc.cli.racecar --config configs/car/dryrun-levine-v11.yaml
# at every car_waiting: put the sim car behind the line, then go
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {position: {x: -2.0, y: -0.07}, orientation: {z: 0.0037, w: 1.0}}}}"
ros2 topic pub --once /sit_lmpc/go std_msgs/msg/Empty
```

The dry-run configs turn the delay compensation off (the sim has no actuation delay) and use the sim car's steering
limit. In the sim, `bag_check.py` always favours the bicycle model (it is the simulator's own model); results of the
dry runs are in `results/dryrun/README.md`.

## If something goes wrong

| Symptom | Likely cause | Do |
|---|---|---|
| `stale_odometry` stop | PF dropped out | check the particle filter, CPU/GPU memory (`XLA_PYTHON_CLIENT_MEM_FRACTION`) |
| many watchdog stops | planning too slow | step 4 cuts; check `control_step_timing` |
| `stop_out_of_bounds` / `stop_obstacle` | left the track or saw something while slowing down after a lap | carry it back, go; if it repeats at the same spot, lower `stop_decel` |
| long wait before the first `car_waiting` | startup compile of the planner and value training | normal; wait for `planner_precompile` in the log |
| v11 drives badly from the start | input units/signs wrong | step 2 again; step 8 on a bag |
| `car_start_rejected` | car not 0.3–3 m behind the line, off the track, or facing the wrong way | reposition |
| stops in narrow spots | track half-width too large/small for that spot | step 3 values; report the spot |
