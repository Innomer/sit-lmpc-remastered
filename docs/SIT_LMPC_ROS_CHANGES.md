# SIT-LMPC on the ROS 2 F1TENTH sim: what changed

Companion to `PROJECT_BRIEF.md`. Status as of 2026-11-14. The full decision log, measurements and
open questions are in `src/sit-lmpc-remastered/NOTES.md`.

## Outcome

The known-dynamics SIT-LMPC baseline (analytic single-track model, no dynamics learning) now runs
in the ROS 2 simulator (`f1tenth_gym_ros`) at F1TENTH scale on Spielberg. It drives 5 warm-up laps,
then learns lap by lap. Baseline run (`results/Spielberg/known25-fixed`, figures in
`results/figures/known25-fixed/`): 25 of 25 laps completed, with no failures, wall contacts or
track-limit violations.

| | lap time | mean speed |
|---|---|---|
| Warm-up laps 1–5 (MPPI tracking) | 135.2 s | 2.50 m/s |
| LMPPI laps 1–5 | 166.5 → 145.2 → 134.5 → 115.6 → 105.4 s | 2.0 → 3.2 m/s |
| LMPPI laps 6–10 | 97.7 → 91.3 → 85.8 → 82.7 → 78.9 s | 3.5 → 4.3 m/s |
| LMPPI laps 11–20 | 77.0 → … → **70.9 s** (converging around 71 s) | 4.4 → 4.8 m/s |

The best lap was 48% faster than warm-up. Time per control step on the laptop GPU (RTX 3070):
LMPPI median 19.5 ms, 99th percentile 37 ms, with 0.15% of steps over the 100 ms control period
(the moments the value model retrains). On CPU it was about 460 ms. With per-step diagnostic
recording switched on, steps cost 55–115 ms, so leave recording off when measuring timing.

Earlier runs, including a 34% improvement that ended in a wall contact, had a track bug that made
the car brake at every finish line. They are superseded; see "Track bug" below.

**Learned-dynamics (BNN) baseline** (`results/Spielberg/bnn25`, figures in `results/figures/bnn25/`):
SIT-LMPC with a Gaussian BNN trained here on an F1TENTH-scale population (friction 0.4–1.2, mass
×0.8–1.3, tire stiffness ×0.85–1.15), between-lap belief updates and chance constraints, as in the
authors' BNN example. 25 of 25 laps, no failures or violations. Warm-up 135.2 s, then
144.3 → 129.0 → 121.6 → 114.3 → … → **110.1 s** (19% faster). It starts level with the known model,
then falls behind and levels off; the known model reaches 70.9 s. The authors' recorded runs have
the same shape: their BNN trails the known model by up to ~40% and catches up only after ~300 laps.
The model predicts the car to within a millimetre over 1 s, but its uncertainty becomes overconfident
on the faster laps, so many belief updates are held back. Planning takes about 100 ms per step
(64 sampled cars per plan), against 20 ms for the known model.

**Ablation: BNN without chance constraints** (`results/Spielberg/bnn-nochance25`): 25 of 25 laps,
best **75.6 s** (44% faster than warm-up), close to the known model's 70.9 s; planning median 64 ms.
The chance constraints, not the BNN itself, cause most of the slow learning.

**Fast warm start (sim only)**: 5 warm-up laps driven by the lab 5 pure pursuit (`submission` tag)
on its Spielberg raceline at 60% of its speed profile, about 38.0 s per lap (its full-speed 24 s
laps were too fast for 10 Hz LMPPI to take over). Figures in `results/figures/all-baselines-compare/`
and `results/figures/fast-start-compare/`.

| Fast start | LMPPI laps | best LMPPI lap | how it ended |
|---|---|---|---|
| Known model (`known-ppwarm25`) | 17 | 38.9 s | spun in the first corner |
| BNN + chance (`bnn-ppwarm25`) | 5 | 39.8 s | conservative fallback braked to a stop |
| BNN, no chance (`bnn-nochance-ppwarm25`) | 7 | 38.3 s | braked to a stop on track |

None beat the 38.0 s warm-up, and all drifted slower lap over lap (the safe set keeps only the
newest 4 laps, so the faster warm-up laps drop out). The BNN never fits this fast, high-slip regime
well: its training population barely covers slip angles above 0.3 rad, which these laps reach
routinely, so its belief stays close to the prior.

## How it fits together

The planner, safe set, value learning and lap bookkeeping are the authors' code, unchanged. The
only seam they talk to the simulator through is a small "simulator" object (`reset`, `step`,
`close`). We added a second implementation of that object that talks ROS instead of running the
bundled gym in-process:

```text
sit-racecar (runner: laps, value training, checkpoints — unchanged)
   │  step(action) / reset(state)
   ▼
RosSimulator (new)  ── /drive ──►  f1tenth_gym_ros bridge (sim)
   ▲                ◄─ /ego_racecar/odom, /collision ──
   └─ builds the 7-dim state + Frenet coordinates for the planner
```

Selecting it is one config switch: `simulation.backend: ros`.

## Changes, by area

**ROS adapter** (`sit-lmpc-remastered/src/sit_lmpc/adapters/ros_f1tenth.py`, new)
- Builds the 7-dim state from odometry: position (shifted from the rear axle to the centre of
  gravity), heading, speed, yaw rate, and slip angle (from the odometry's lateral velocity).
- Steering angle is not published by the sim, so it is estimated by replaying the commanded
  steering rate through the sim's own steering limits. This is exact in the sim; the real car will
  need an identified actuator model.
- Lockstep stepping: the bridge only advances when it receives a command, so one control step is
  exactly 0.1 s of sim time no matter how long planning takes. Lap times are therefore exact, and
  runs can go faster than real time.
- Reset: the sim can only place the car at rest, so the adapter places it slightly behind the
  requested pose and accelerates it to the requested speed.
- Measures and saves planning time per control step (`control_timing.json/.npz` in each run).

**Runner and config** (small hooks only)
- `simulation.backend` and a `simulation.ros` settings block (topics, drive mode, stepping).
- Vehicle parameters come from the ROS gym's F1TENTH set (3.74 kg, 0.33 m wheelbase) instead of the
  full-size car.
- Resume from a checkpoint now works with the ROS sim: the restored state is checked against the
  checkpoint with a physical tolerance instead of an exact match.
- `simulation.steering_limit` (off by default): caps the steering angle for both the planner's model
  and the commands sent. The real car needs 0.34 rad (the sim allows 0.419; the sim baseline never
  used more than 0.29).
- `simulation.ros.actuation_delay` (off by default): compensates the car's ~0.2 s command delay, as
  in the MPC lab. Each step, the state is predicted forward over the delay using the commands
  already sent, and the planner plans from that. Not testable in the sim (no delay there); to be
  tuned on the car.
- `planner.warmup.controller: pure_pursuit` (sim only; default `mppi`): warm-up laps driven by a port
  of the lab 5 pure pursuit (`src/sit_lmpc/planners/pure_pursuit.py`; same speed profile, lookahead,
  corner cut and steering law), following `planner.warmup.path` with `speed_scale`. It runs at the
  sim's 100 Hz: the adapter calls it every 0.01 s inside each 0.1 s step (`substep_policy`), because
  at 10 Hz it spins at these speeds. Needs lockstep and one physics step per message.
- Pure-pursuit warm-up laps are not given to the BNN belief update (their commands change within
  each 0.1 s step, so the step-averaged labels are wrong); they still seed the safe set and value
  function.

**Simulator (`src/f1tenth_gym_ros`)**

Only the ROS bridge node changed; the gym physics package inside it (`f1tenth_gym_ros/f1tenth_gym`)
is untouched. With the new parameters left at their defaults, the bridge behaves exactly as before,
so the other labs (pure pursuit, gap follow, MPC, teleop) are unaffected.

`f1tenth_gym_ros/gym_bridge.py` (+50 / −13 lines):

| Change | What it does | Why |
|---|---|---|
| New parameter `control_mode` (default `speed_steering_angle`) | `accel_steering_speed` switches the gym's control inputs to direct acceleration (`ACCL`) and steering rate (`STEERING_SPEED`), read from `drive.acceleration` and `drive.steering_angle_velocity`. The default keeps reading `drive.speed` / `drive.steering_angle` through the gym's P controllers. Invalid values stop the bridge with an error. | SIT-LMPC plans steering rate and acceleration. Applying them directly gives the planner the exact plant it models, instead of going through speed/angle P loops. |
| New helper `_drive_action()` | One place that turns a drive message into the gym's two inputs for the selected mode, with clipping to the vehicle limits (±3.2 rad/s, ±9.51 m/s² in accel mode; ±0.419 rad steering in the default mode). Used for the ego car and for opponents. | Replaces the duplicated speed/steering code in the ego and opponent callbacks. |
| New parameter `substeps_per_drive` (default `1`) | Synchronous mode only: each ego drive message advances this many 0.01 s physics steps before odometry is published once. | One message per 0.1 s control step instead of ten, to cut ROS round trips in batch runs. |
| Wheel drawing in `_publish_wheel_transforms()` | In accel mode, the front-wheel TF uses the simulated steering angle from the gym state. The default mode still draws the commanded angle. | In accel mode the stored "steer" is a rate (up to ±2.6), so the wheels in Foxglove looked like they were flickering violently. |

`config/sim.yaml`: documents `control_mode` and `substeps_per_drive` at their defaults. (Other
diffs in this file — `map_path: maps/levine_obs` and the commented start poses — are earlier local
edits, not part of this work.)

Notes:
- `control_mode` applies to every car in the sim (ego and opponents).
- Keyboard teleop assumes the default mode; in accel mode its ±0.3 steering would be read as a rate.
- The installed copy in `install/` was not rebuilt. Until
  `colcon build --packages-select f1tenth_gym_ros` is run, `ros2 run`/`ros2 launch` use the old
  bridge; our runs start it from source instead (see "How to run").

Simulator settings for SIT-LMPC runs (parameter files in `sit-lmpc-remastered/configs/`; your
`sim.yaml` is not used for these runs):

| File | Key settings |
|---|---|
| `ros-bridge-lockstep.yaml` | `map_path: maps/Spielberg`, start pose (0, 0, −2.879) on the finish line, `vehicle_params: f1tenth`, `control_mode: accel_steering_speed`, `async_mode: False` (step only on each drive message), `use_sim_time_bridge: True` (sim-time stamps and `/clock`), `lidar_enabled: False` (not needed; faster), `kb_teleop: False` |
| `ros-bridge-lockstep-fast.yaml` | same, plus `substeps_per_drive: 10` (pair with `configs/f1tenth-ros-known-fast.yaml`) |

Our test runs used `ROS_DOMAIN_ID=77`, so they never touched a sim running on the default domain 0.

**Tuning for 1/10 scale** (`configs/f1tenth-ros-known.yaml`)
- Map: Spielberg, which ships identically in both repos and is already F1TENTH-sized (1.1 m
  half-width).
- Warm-up speed 2.5 m/s; track half-width 1.1 m with a 0.25 m safety buffer.
- Steering-rate limit 1.0 rad/s and acceleration 3 m/s² (were 3.2 and 5). The track needs at most
  0.44 rad/s, and the larger limit made the warm-up steering jitter up to 0.38 rad.
- Smaller planner sampling noise. The full-size setting made the first learned lap stall to a
  stop, because most sampled plans ended far outside anything the warm-up had seen.

**Environment**
- Everything runs in `~/sim_ws/.venv` (Python 3.12, which ROS Jazzy needs). The package's
  Python 3.13 requirement and an exact JAX pin were relaxed; the missing libraries were installed.
- Laptop GPU fixed: the NVIDIA kernel module was missing for the current kernel, and the GPU
  crashed when waking from idle power-saving. Runtime power management is now disabled via
  `/etc/modprobe.d/nvidia-runtimepm.conf`.

**Tools** (`sit-lmpc-remastered/scripts/`)
- `presentation_figures.py`: lap times, track map coloured by speed, speed and lateral-offset
  profiles, planning time (CPU vs GPU), warm-up steering before/after, plus CSV tables.
- `analyze_ros_run.py`: quick lap table and timing summary for a run folder.
- `ros_viz_launch.py`: Foxglove, map and car model for an already running bridge.

## Track bug (fixed)

Every learned lap used to brake to about 0.7 m/s at the finish line. The cause was a one-line bug
in the authors' track code (`Track.get_s_search_range`): it wrapped its search window by the
number of *resampled* centerline entries (3434) instead of the number of spline points (865). JAX
silently clamps out-of-range indices, so to the planner the track past the finish line collapsed
to a single point and crossing earned no progress. The authors' own map (`custom1`) has no
separate centerline file, so both counts match there and it never showed; Spielberg is the first
map loaded from a centerline file. After the fix, learned laps cross at a median 3.9 m/s and the
baseline improved from 89 s (with a crash) to 70.9 s (no failures).

## Every file touched

`src/sit-lmpc-remastered` (existing files: +96 / −12 lines; the rest are new):

| File | Change |
|---|---|
| `src/sit_lmpc/adapters/ros_f1tenth.py` | **New.** `RosSimulator` (state building, steering estimate, lockstep/realtime stepping, reset with run-up, timing log) and F1TENTH vehicle parameters |
| `src/sit_lmpc/adapters/__init__.py` | `vehicle_parameters_for()`: F1TENTH parameters for the ROS backend, full-size for the bundled gym |
| `src/sit_lmpc/experiments/runner.py` | Uses `vehicle_parameters_for()`; builds `RosSimulator` when `backend: ros`; resume check uses the simulator's tolerance and logs the mismatch |
| `src/sit_lmpc/config/schema.py` | `RosConfig` and `simulation.backend`/`simulation.ros`; validation (substep multiples, drive modes, no render window with ROS); resume overrides for `simulation.ros` |
| `src/sit_lmpc/config/default.yaml` (`configs/default.yaml` is a symlink to it) | Documents the new settings |
| `src/sit_lmpc/track/track.py` | One-line fix: the Frenet search window wraps by the spline's point count (see "Track bug") |
| `pyproject.toml`, `vendor/dyna-dist/pyproject.toml` | Python `>=3.12` (was 3.13 only), `jax>=0.11.1,<0.12` (was exactly 0.11.2) |
| `configs/f1tenth-ros-known.yaml` | **New.** The F1TENTH baseline config |
| `configs/f1tenth-ros-known-fast.yaml` | **New.** Fast batch variant |
| `configs/ros-bridge-lockstep.yaml`, `configs/ros-bridge-lockstep-fast.yaml` | **New.** Simulator parameter files |
| `scripts/presentation_figures.py`, `scripts/analyze_ros_run.py`, `scripts/ros_viz_launch.py` | **New.** Figures, run summary, Foxglove launch |
| `NOTES.md` | **New.** Decision log, measurements, open questions |
| `patches/f1tenth_gym_ros-sit-lmpc.patch` | **New.** The simulator bridge changes (`control_mode`, `substeps_per_drive`, wheel drawing) as a patch against upstream f1tenth_gym_ros `1fef653`; apply with `git apply` in `src/f1tenth_gym_ros` |
| `models/vehicle-bnn-f1tenth-l96/` | **New.** The trained F1TENTH BNN (748 KB); `sit-import-bnn --source models/vehicle-bnn-f1tenth-l96 --destination artifacts/vehicle-bnn-f1tenth-l96` |
| `configs/collect-vehicle-f1tenth.yaml`, `configs/train-dynamics-f1tenth.yaml`, `configs/f1tenth-ros-bnn-fast.yaml` | **New.** BNN data collection, training and SIT-LMPC configs |
| `configs/f1tenth-ros-bnn-nochance-fast.yaml` | **New.** Ablation: BNN without chance constraints |
| `src/sit_lmpc/planners/pure_pursuit.py` | **New.** Lab 5 pure pursuit as a warm-up controller |
| `configs/collect-vehicle-f1tenth-wide.yaml`, `configs/train-dynamics-f1tenth-wide.yaml` | **New.** Wider BNN population (speeds to 20 m/s) for the fast start |
| `configs/f1tenth-ros-{known,bnn,bnn-nochance}-ppwarm.yaml`, `configs/racelines/spielberg_pp_lab5.csv` | **New.** Fast-start configs and the lab 5 raceline |
| `vendor/dyna-dist/src/dyna_dist/{config.py, data/vehicle_collection.py, adapters/gym_st.py}` | F1TENTH profile for data collection; physics labels from f1tenth_gym_ros when the bundled gym is absent |
| `docs/SIT_LMPC_ROS_CHANGES.md`, `docs/PROJECT_BRIEF.md` | Copies of this document and the project brief |
| `results/` | Run outputs (git-ignored) |

`src/f1tenth_gym_ros`: `f1tenth_gym_ros/gym_bridge.py`, `config/sim.yaml` (see above).

Outside the repos: Python packages installed into `~/sim_ws/.venv` (flax, distrax, optax,
orbax-checkpoint, tfp-nightly, yamldataclassconfig, matplotlib, `jax[cuda13]`, plus `dyna-dist` and
`sit-lmpc-remastered` as editable installs); NVIDIA module packages for the current kernel and
`/etc/modprobe.d/nvidia-runtimepm.conf` (GPU fixes, applied with sudo by you).

## How to run

Every terminal:

```bash
source /opt/ros/jazzy/setup.bash && source ~/sim_ws/install/setup.bash && source ~/sim_ws/.venv/bin/activate
export ROS_DOMAIN_ID=77          # keeps these runs separate from a sim on the default domain
cd ~/sim_ws/src/sit-lmpc-remastered
```

Then:

```bash
# 1) simulator bridge (lockstep, accel mode), from source until f1tenth_gym_ros is rebuilt
PYTHONPATH=~/sim_ws/src/f1tenth_gym_ros:$PYTHONPATH python -c "from f1tenth_gym_ros.gym_bridge import main; main()" \
    --ros-args -r __node:=bridge --params-file configs/ros-bridge-lockstep.yaml
#    (after `colcon build --packages-select f1tenth_gym_ros`:
#     ros2 run f1tenth_gym_ros gym_bridge --ros-args -r __node:=bridge --params-file configs/ros-bridge-lockstep.yaml)

# 2) Foxglove at ws://localhost:8765 (map, car model, browser tab)
ros2 launch scripts/ros_viz_launch.py map:=Spielberg

# 3) controller: 5 warm-up laps, then SIT-LMPC
sit-racecar --config configs/f1tenth-ros-known.yaml --max-laps 25 --save-dir results/Spielberg/<name>

# 4) figures and tables
python scripts/presentation_figures.py results/Spielberg/<name> --out results/figures/<name>
```

- Resume: `sit-racecar --config configs/f1tenth-ros-known.yaml --resume results/Spielberg/<name>/checkpoints/trial-N --max-laps M`
- Skip warm-up by reusing another run's warm-up: set `output.quick_load: true` and
  `output.init_path: results/Spielberg/<run>/init.npz`.
- Fast batch runs (no Foxglove): bridge with `configs/ros-bridge-lockstep-fast.yaml`, controller
  with `configs/f1tenth-ros-known-fast.yaml` (both must agree on 10 substeps per message).
- After a reboot, check `nvidia-smi` first; run logs must not contain `CUDA_ERROR_NO_DEVICE`.

## Known issues and next steps

- **Track limit uses the car's centre only:** at large heading angles the nose can reach the wall
  while the centre is within limits (this caused the one wall contact before the track fix). It
  has not recurred since the fix, but a heading-aware limit would be safer, especially on the car.
- **BNN at high speed:** the population data barely covers high slip, so the BNN cannot learn the
  fast-start laps. Fix would be collecting data with a controller that holds high slip (not started).
- **Fast start ends in stops/spins:** the newest-4-laps safe set lets lap times creep up after a
  fast warm-up, and every fast-start run ended in an unforced stop or spin.
- **Real car:** steering limit is 0.34 rad (sim 0.42); steering and slip angle need estimators
  (no direct measurement); real-time (non-lockstep) operation is untested.
- **Our contributions** (in-lap belief updates, condition-aware safe set, mid-session changes) are
  not started; the design leaves room for them at the per-step data collection and at the
  safe-set update.
