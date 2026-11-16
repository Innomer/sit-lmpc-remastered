# SIT-LMPC on the real car (branch `irl`)

Status 2026-11-14: code ready, not yet run on the car. Dry-run on the ROS sim in real time (Spielberg): launch, pure pursuit warm-up lap (68.9 s at a 4 m/s cap, 10 Hz with 0 overruns), Ctrl-C stop, out-of-bounds stop and start rejection all work.

## What runs

`sit-racecar --config configs/car-levine-known.yaml` with `simulation.backend: car` uses
`src/sit_lmpc/adapters/ros_car.py` (`RosCar`) instead of the sim bridge. The runner, planners and learning are unchanged.

| Input / output | Topic (config `simulation.car`) | Use |
|---|---|---|
| Pose, heading | `/pf/pose/odom` (`odom_topic`) | x, y (shifted from the rear axle to the CoG by lr), yaw |
| Speed | `/odom` (`speed_topic`, VESC wheel odometry) | v. `""` uses the PF odom twist |
| Yaw rate | `/sensors/imu/raw` (`imu_topic`), gyro z × `imu_yaw_scale` | r. `""` uses the PF odom twist |
| Steering angle | none (not measured) | commanded angle through the servo rate limit `sv_max` |
| Slip | none | kinematic β = atan(lr / L · tan δ) |
| LiDAR | `/scan` (`scan_topic`) | stop if anything is closer than `stop_distance` within ±`scan_half_angle` ahead |
| Go | joystick `go_button` on `/joy`, or `std_msgs/Empty` on `/sit_lmpc/go` | start / restart a lap |
| Drive | `/drive` AckermannDriveStamped (speed, steering_angle) | through the car's ackermann mux, so the deadman still applies |

Control: 10 Hz on the wall clock. The planner's (steering rate, acceleration) becomes steering target δ + rate·0.1 s
(clipped to ±0.34 rad) and speed target v + a·`speed_horizon` (clipped to 0…`max_speed`). Delay compensation
(`simulation.ros.actuation_delay: 0.2`, the MPC lab value) rolls the state forward through the queued commands.
During pure pursuit warm-up laps, pure pursuit's own steering and speed targets are sent at the odometry rate.

Safety, all inside the adapter:
- Watchdog thread: no drive command for `command_timeout` (0.3 s) → speed 0. This covers planner stalls, JIT compiles,
  checkpoint writes and crashes.
- |ey| > `planner.half_width`, a LiDAR return inside `stop_distance`, or PF odometry older than `odom_timeout` → stop at once.
- Ctrl-C or SIGTERM → stop, then save. `max_speed` caps every command.
- After a stop, the car waits for the operator: put it 0.3–`start_zone` m behind the start line, facing forward, and press go.
  The start is rejected (log `car_start_rejected`) if it is outside that zone, off the track or facing the wrong way.
  The first step then drives the car over the line at `initial_speed` along the centerline before the planner takes over.
- The known model (`sit_lmpc_mode`) ends the run on a failure. The BNN configs reset and wait for go.
  Resume works after a battery swap: `sit-racecar --config <cfg> --resume <run>/checkpoints/trial-N` (it restarts at the line).

Logs per run (in `results/<track>/<exp>-<time>/`): the usual run files, `control_timing.json`/`.npz` (planning
time, overruns, watchdog stops), `car_log.npz` (per command: PF pose, speed, IMU and PF yaw rate, δ estimate,
commands, closest LiDAR) and `bag/` (rosbag of `simulation.car.bag_topics`, including the full VESC IMU, for the
offline slip estimator).

## What happens during a run

1. Start-up: load the config, track and model (the BNN from `models/`), compile the planners and the hull
   variants, wait for all car topics, then wait for go with the car behind the start line.
2. Go: the car follows the centerline over the start line at `initial_speed`.
3. Warm-up laps (5 = `safe_set_size` + 1): pure pursuit on the raceline, at most `planner.warmup.max_speed`.
   Each finished lap is stored. After each lap the value function (a normalizing flow over lap states) is
   retrained in the background.
4. LMPPI laps: 10 Hz sampling MPC. It samples 1024 control sequences over a 1 s horizon through the model (known ST,
   or BNN samples), scores them with the value function, keeps them inside the safe set (convex hull of the
   newest 4 laps) and within the track band, and applies the first action. Each new lap replaces the oldest
   one in the safe set, so laps should get faster.
5. BNN runs only: after each LMPPI lap, the BNN posterior is updated with that lap's data (the network is
   trained offline on the laptop; on the car only the posterior changes). Warm-up laps are not used for this.
6. Failure (off the band, obstacle): the car stops. The known model ends the run there. The BNN configs wait for go
   and retry from the start line.
7. End: after `max_laps` (15) or Ctrl-C the car stops and the run directory holds laps, events, timing, `car_log.npz`
   and the bag. A checkpoint after every lap allows `--resume`.

## Track

The configs use `tracks/levine_self_again/`. It was built from `levine_self_again_raceline.csv` alone, because the matching
map is only on the car (it does not line up with `levine_self.pgm` here). It uses the raceline as the centerline with 0.4 m
assumed on each side, so `planner.half_width` is 0.25. Rebuild it from the car's PF map before driving:

```bash
python scripts/make_track.py --raceline configs/racelines/levine_self_again_raceline.csv \
  --map <particle_filter>/maps/levine_self_again.yaml --name levine_self_again
```

The script refuses a map that does not contain the raceline. It prints the wall distances and a suggested
`planner.half_width` (5th-percentile wall distance minus the 0.155 m car half width); put that into
`configs/car-levine-known.yaml`. On the old `levine_self` map the same script gives 0.53 m. s = 0 is the raceline's first point.

## Python and ROS

The package needs Python 3.12+ (JAX 0.11), and the planner imports `rclpy` directly. That works where ROS 2 is Jazzy
(Python 3.12), as on the laptop. If the Jetson runs Humble (Python 3.10), `scripts/jetson_setup.sh` stops and says so.
Then either run inside a ROS 2 Jazzy container (`--runtime nvidia --network host`, same `ROS_DOMAIN_ID`; the car
stack stays on Humble), or install Jazzy. LMPPI needs the GPU: on the laptop CPU it took ~500 ms per step against a 100 ms period.

## Steps on the car

1. `git clone -b irl <repo> && cd sit-lmpc-remastered`, source ROS and the car workspace, then `bash scripts/jetson_setup.sh`.
   It reports the JAX device and a matmul time, and validates the car configs.
2. Bring up the car stack (VESC, ackermann mux, joystick) and the particle filter on the Levine map.
3. `python scripts/car_check.py --config configs/car-levine-known.yaml --seconds 30`, while you push or drive the car
   through left and right turns. It never sends drive commands. It checks topic rates and PF gaps, suggests
   `imu_yaw_scale` (sign and units against the PF heading), compares VESC and PF speed, shows the car's s/ey on the
   track, the closest LiDAR return and which joystick buttons work. Set `imu_yaw_scale` and `go_button` from it.
4. Rebuild the track from the PF map (above).
5. First drive: car on blocks or held, run `sit-racecar --config configs/car-levine-known.yaml`. Hold the deadman,
   press go and watch the wheels; Ctrl-C must stop them. Then on the floor with `simulation.car.max_speed: 1.5` and
   `initial_speed: 1.0` for the first warm-up laps.
6. Known-model baseline: pure pursuit warm-up laps (`planner.warmup.max_speed`), then LMPPI. Check `control_step_timing`
   events: planning p95 must stay under ~80 ms, and overruns and watchdog stops near 0.
7. BNN: `configs/car-levine-bnn-nochance.yaml` (model `models/vehicle-bnn-f1tenth-wide-l96`), then
   `configs/car-levine-bnn.yaml` if time allows. In sim the BNN took 63–99 ms per step on the laptop GPU. If the Jetson is
   slower, lower `planner.samples` or `planner.dynamics.particles`.

Dry run on the laptop (no car): start the bridge with `configs/ros-bridge-realtime.yaml`. Place the sim car behind the
start line with `/initialpose`, run `sit-racecar --config configs/car-sim-dryrun.yaml` and publish `/sit_lmpc/go`.

## Known gaps

- `speed_horizon` (0.2 s), `launch_lookahead`, the pure pursuit `latency` (0.2 s) and the steering estimate are untested on
  the car. Compare `car_log.npz` against the PF trajectory after the first laps.
- Between laps the runner may block (value retraining on the first LMPPI entry, checkpoint writes). The watchdog then
  stops the car, which restarts from low speed. Count `watchdog_stops`, and raise `output.checkpoint_every_trials` if it
  happens often.
- The vehicle model is the F1TENTH gym's (mass, tyres). The known-model baseline on the car therefore runs with model
  mismatch, and that is part of what the BNN should fix.
- Planning-time numbers for the Jetson do not exist yet.
