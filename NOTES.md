# SIT-LMPC Remastered on the ROS 2 F1TENTH sim — working notes

Team 8, ESE 6150. Started 2026-11-13. See PROJECT_BRIEF.md (workspace root) for context.

## 1. How the existing code fits together (as read on 2026-11-13)

Control loop, one step (`experiments/runner.py` `_run`, lines ~1093–1555):

1. `safe_set.ready` false → `MPPI.update(state, initial_speed)` tracks the raceline waypoints at a
   constant target speed (warm-up). Reward is `-|xy - ref xy|`; the raceline's own speed column
   is ignored.
2. `safe_set.ready` true (`safe_set_size` successful laps stored) → `LMPPI.update(state+noise,
   frenet, value_snapshot, safe_set, ...)`: samples 1024 action sequences, rolls them through the
   predictor, converts to Frenet, scores `-time + value(terminal) - lambda_hull*hull_dist -
   lambda_boundary*|ey|>w`, picks the smallest feasible lambda pair, returns the first action.
3. `simulator.step(normalized_action, control_noise)` → `(cartesian, frenet), control, terminated,
   truncated, info`. The runner stores the transition in `AttemptCollector`.
4. Failure test: `|ey| > planner.half_width` or `info["collision"]`. In `sit_lmpc_mode` (our
   baseline) the first physical failure ends the experiment.
5. Lap end (collector's Frenet progress crosses track length): `safe_set.add_lap`, value trainer
   restarted in a worker thread, `publish_belief()` (BNN posterior update, between laps only),
   periodic `write_checkpoint`.

Ownership:

- **Runner owns**: attempt/lap bookkeeping (own Frenet-progress lap detection, not the sim's),
  value training (`ValueTrainer`, NF/MLP, background thread), belief updates
  (`GaussianBNNLearner.observe/update` at attempt boundaries), checkpoints/resume, events,
  diagnostics, keyboard.
- **Planners own** (`planners/mppi.py`, `planners/lmppi.py`): sampling, rollouts through `model`
  (`STModel` analytic ST or `gaussian_bnn` predictor, same `predictor_step/rollout` interface),
  Frenet conversion via `Track`, hull + boundary penalties, lambda selection, chance constraints.
- **SafeSet** (`planners/safe_set.py`): per-lap Frenet/Cartesian records with next-lap extension,
  convex hull selection in a local s window.
- **Dynamics** (`dynamics/st.py`, `physics.py`): 7-state ST `(x, y, delta, v, yaw, yaw_rate,
  beta)`, control `(steering_speed, acceleration)`; normalized actions in [-1,1] scaled by
  `simulation.action_scale`. BNN (`dynamics/gaussian_bnn.py`): fixed neural features, Gaussian
  `z` (96-dim) updated in closed form.
- **Simulator seam** (`adapters/f1tenth.py` `F1TenthSimulator`): the ONLY place the runner
  touches the bundled gym. Interface used by the runner: `reset(state) -> (cart, frenet), info`,
  `step(norm_action, noise)`, `snapshot()`, `restore()`, `render()`, `capture_render()`,
  `display_observation()`, `window_open()`, `bind_keyboard()`, `close()`. The runner uses
  `hasattr` guards for `snapshot/restore/capture_render/bind_keyboard`.
- `build_vehicle_dynamics` → `vehicle_parameters()` pulls **fullscale** params from the bundled
  gym's `F110Env.fullscale_vehicle_params()`.

## 2. Everything that assumes full-car scale

| Where | Value | F1TENTH equivalent |
|---|---|---|
| `adapters/f1tenth.py` `vehicle_parameters` | fullscale (m=1226 kg, lf .884, lr 1.509, sv_max .4, a_max 11.5, v_max 45.8) | ROS gym `F1TENTH_VEHICLE_PARAMETERS`: m 3.74, lf .15875, lr .17145, s_max .4189, sv_max 3.2, a_max 9.51, v_switch 7.319, v_max 20, mu 1.0489, C_Sf 4.718, C_Sr 5.4562, I .04712, h .074 |
| `simulation.action_scale` | [0.4 rad/s, 3.5 m/s²] | start [3.2, 5.0] (sv_max; gentler accel), tune |
| `simulation.initial_speed` | 10 m/s | ~2.0–2.5 m/s |
| `simulation.friction` / `planner.friction` | 0.5 / 0.813 | ROS sim mu = 1.0489 (known dynamics → planner.friction 1.0489) |
| `planner.half_width` / `boundary_buffer` | 4.0 / 0.03 m | Spielberg F1TENTH: 1.1 / ~0.25 (CoG point boundary; car half-width 0.155) |
| `planner.safe_set_window_margin` | 5.0 m (predicted_window only, unused with `exact`) | — |
| `planner.search_points` | 400 waypoints | Spielberg raceline spacing 0.2 m → 80 m window, fine |
| `simulation.map` | custom1 (382 m, w4) | **Spielberg already is the F1TENTH map**: `track/maps/Spielberg` is byte-identical to `f1tenth_gym_ros/maps/Spielberg*` (centerline widths 1.1 m, raceline v 4.5–8 m/s, 338 m) |
| `simulation.timestep` / `integrator_timestep` | 0.1 / 0.02 | keep 0.1; ROS bridge integrates at 0.01 |
| BNN `examples/racecar/models/vehicle-bnn-v2-l96` | `baseline_profile: fullscale`, friction 0.65 population | **not usable at 1/10 scale**; retrain with dyna-dist on an F1TENTH ST population (later) |
| `planner.belief.residual_scales` | per-state residual scales (model-bank path, disabled) | n/a for now |
| `physics.py` low-speed switch `v < 0.5` | same in ROS gym | fine |
| runner `cartesian[3] < 0.1 → speed_too_low` | assumes car spawns moving | ROS reset spawns at rest → adapter must launch to `initial_speed` before returning from `reset()` |

## 3. ROS sim facts that matter (f1tenth_gym_ros, Jazzy, Python 3.12)

- Bridge publishes `/ego_racecar/odom` (pose = **base_link = rear axle**, twist.linear.x = v cos β,
  linear.y = v sin β, angular.z = yaw rate), `/ego_racecar/collision` (Bool), `/scan`,
  `lap_count`, `lap_time`. It does **not** publish steering angle or slip angle directly, but
  slip is recoverable in sim as `atan2(vy, vx)` and speed as `hypot(vx, vy)`.
- Drive input is `AckermannDriveStamped` `steering_angle` + `speed`, run through the gym's P
  controllers (`pid_steer` kp = 10·sv_max/(s_max−s_min) ≈ 38 /s, `pid_accl` kp = 10·a_max/v_max
  ≈ 4.8 /s). SIT-LMPC commands `(steering_speed, acceleration)`. The gym itself supports
  `ACCL` / `STEERING_SPEED` modes and the Ackermann message has `steering_angle_velocity` and
  `acceleration` fields, so a bridge `control_mode` parameter is a ~15-line change.
- `async_mode: False` steps the sim exactly one 0.01 s substep per `/drive` message and
  publishes odom in the same callback → deterministic lockstep; controller compute time does
  not affect the plant. `use_sim_time_bridge` stamps with sim time.
- Sim pose is CoG internally; SIT-LMPC's ST state is also CoG → convert odom base_link → CoG
  with `lr` (x + lr cos ψ, y + lr sin ψ).
- JAX in `~/sim_ws/.venv` sees **CPU only** (`nvidia-smi` cannot reach the driver on this
  laptop). Planning time per step on CPU must be measured.
- The bundled `vendor/f1tenth_gym` and the ROS gym share the package name `f1tenth_gym`; the
  venv already has the ROS one. Only `adapters/f1tenth.py` imports it, so the ROS path must not
  import the bundled gym.
- `sit-lmpc-remastered` pins Python 3.13; rclpy on Jazzy is built for 3.12. Nothing in
  `src/sit_lmpc` uses 3.13-only syntax (grep), so the plan is to run it on the 3.12 venv.

## 4. Design decisions (2026-11-13)

1. **Drop-in simulator behind the existing seam; runner untouched in logic.** `RosSimulator`
   (`src/sit_lmpc/adapters/ros_f1tenth.py`) implements the same `reset/step/render/close/
   window_open/display_observation` interface as `F1TenthSimulator`. The runner's loop *is* the
   ROS node's loop: `step()` publishes a drive and blocks (`rclpy.spin_once`) until odometry has
   advanced one control period. Lap/attempt bookkeeping, value training, belief updates and
   checkpoints are the runner's unchanged code. Runner changes: parameter lookup goes through
   `adapters.vehicle_parameters_for(simulation, ...)`, and one `if simulation.backend == "ros"`
   at simulator construction. Config: `simulation.backend: gym|ros` and a `simulation.ros` block.
2. **Drive interface: both modes, adapter-selectable** (`simulation.ros.drive_mode`).
   - `accel_steering_speed` (default, same plant as the paper): the bridge got a `control_mode`
     parameter (`f1tenth_gym_ros/gym_bridge.py`, ~20 lines) that switches the gym to
     `ACCL`/`STEERING_SPEED` and reads `drive.acceleration` / `drive.steering_angle_velocity`.
     Default bridge behaviour is unchanged.
   - `speed_steering_angle` (real-car path): the adapter inverts the bridge's P controllers,
     `steering_angle = delta_hat + sv/kp_steer`, `speed = v + a/kp_accl(v, a)`.
3. **Map: Spielberg first** (already F1TENTH scale in both repos). Levine is a follow-up: needs a
   TUM-format raceline for `sit_lmpc.track` and a tighter constant `half_width`.
4. **Python: everything in `~/sim_ws/.venv` (3.12).** `requires-python` relaxed to `>=3.12` and
   `jax` pin to `>=0.11.1` in `pyproject.toml` and `vendor/dyna-dist/pyproject.toml`; deps
   installed with pip, `dyna-dist` and `sit-lmpc-remastered` as editable `--no-deps`. The bundled
   `vendor/f1tenth_gym` is *not* installed; `f1tenth_gym` resolves to the ROS gym (only the ROS
   adapter imports it, for `F1TENTH_VEHICLE_PARAMETERS`).
5. **Stepping: lockstep first** (`simulation.ros.stepping: lockstep`; bridge `async_mode False`,
   `use_sim_time_bridge True`, params in `configs/ros-bridge-lockstep.yaml`). One control step =
   10 drive messages, each followed by a wait for the sim-time-stamped odometry to advance 0.01 s.
   Lap times are exact sim time, independent of planning time. `realtime` mode publishes once per
   period and waits for the period to elapse (for the compute-budget test later).
6. **State estimation in the adapter** (7-dim CoG ST state from `/ego_racecar/odom`):
   `x, y` = base_link + `lr`·(cos ψ, sin ψ); ψ unwrapped across steps (the gym's yaw is
   continuous); `v = hypot(vx, vy)`, `β = atan2(vy, vx)` (0 below 0.05 m/s), `ψ̇ = angular.z`;
   **steering angle δ is not published**, so the adapter replicates the plant's actuator: the
   commanded rate (direct, or `clip(kp(δ_cmd − δ), ±sv_max)` in angle mode) integrated with RK4 at
   the bridge timestep with the gym's `steering_constraint` at each stage. Frenet via
   `Track.cartesian_to_frenet` (centerline, same as the gym's `frenet_pose`).
7. **Reset = `/initialpose` + launch.** The bridge spawns at rest and the runner aborts below
   0.1 m/s, so `reset()` accelerates straight ahead (acceleration `action_scale[1]`, or speed
   target in angle mode) until `initial_speed` is reached, then returns the observation.
8. **Timing.** The adapter measures planning time (wall time between returning an observation and
   receiving the next command) and simulator step time per control step; emits
   `control_step_timing` every `timing_report_steps` steps and writes `control_timing.json/npz`
   to the run directory on close.
9. **Not implemented / known gaps.** Resume from checkpoint on the ROS backend (the runner compares
   the restored Frenet state exactly; the launch makes it differ). `snapshot/restore`
   unimplemented (hasattr-guarded). Steering estimate drifts in `realtime` mode if the bridge
   applied a command for a different number of substeps than assumed.

## 5. Files touched

- `f1tenth_gym_ros/f1tenth_gym_ros/gym_bridge.py`: `control_mode` param, `_drive_action()`.
  `config/sim.yaml` documents it (default unchanged). The installed copy in `install/` is a copy,
  not a symlink → `colcon build --packages-select f1tenth_gym_ros` is needed before `ros2 run`
  uses the new mode (tests here ran the bridge from source).
- `sit-lmpc-remastered`: `src/sit_lmpc/adapters/ros_f1tenth.py` (new), `adapters/__init__.py`
  (`vehicle_parameters_for`), `experiments/runner.py` (2 small hooks), `config/schema.py`
  (`RosConfig`, `SimulationConfig.backend/ros`, validation), both `default.yaml` (docs),
  `configs/f1tenth-ros-known.yaml`, `configs/ros-bridge-lockstep.yaml`, `pyproject.toml` pins,
  `vendor/dyna-dist/pyproject.toml` pins.

## Run recipe

```bash
source /opt/ros/jazzy/setup.bash && source ~/sim_ws/install/setup.bash && source ~/sim_ws/.venv/bin/activate
cd ~/sim_ws/src/sit-lmpc-remastered
# terminal 1 (after rebuilding f1tenth_gym_ros; or run the bridge from source with PYTHONPATH=~/sim_ws/src/f1tenth_gym_ros)
ros2 run f1tenth_gym_ros gym_bridge --ros-args -r __node:=bridge --params-file configs/ros-bridge-lockstep.yaml
# terminal 2
sit-racecar --config configs/f1tenth-ros-known.yaml
```

## 6. Results

All runs: laptop CPU only (JAX `CpuDevice`), bridge in lockstep from source on `ROS_DOMAIN_ID=77`,
`configs/f1tenth-ros-known.yaml`, seed 1.

- **Smoke (40 steps)**: reset + launch to 2.5 m/s in 0.5 s sim; state consistent (displacement
  speed matches odom speed; δ estimate within ±0.38 rad; β ≤ 0.08 rad at 2.5 m/s).
- **One warm-up lap**: 134.8 s lap, 1348 steps, mean speed 2.51 m/s, max |ey| 0.929 m, 0
  out-of-bounds steps. Planning (MPPI warm-up, 1024 samples × H=10): **p50 26 ms, p95 29 ms**
  per control step; first step 1.2 s (JIT). Lockstep simulator round trip (10 drive/odom
  exchanges): p50 24 ms. Wall time 87 s for 135 s of sim time.
- **Full run 1 (10 trials, CPU), warm-up phase**: trials 1–5 MPPI, all 134.8–135.1 s at
  2.51 m/s, 0 out-of-bounds steps. Value model retrained after each lap (NF, 300 epochs; loss
  −2.22 → −3.94 over generations 1–4). LMPPI started at step 6746 (after 5 successful laps;
  `safe_set_size 4` plus the "previous lap" needed for the extension), hull 446 facets.
  Planning per control step: MPPI p50 26 ms / p95 ~30 ms (spikes to 80–260 ms while the value
  trainer thread runs); **LMPPI p50 546 ms / p95 640 ms** on CPU, first step 8.9 s JIT.
  LMPPI is 5–6× over the 0.1 s control period on this CPU → GPU required for real-time; lockstep
  keeps the learning result valid regardless. Stopped by hand at step ~7700 (first LMPPI lap ~45%
  done, 2.1 m/s, |ey| 0.41, no violations) so the machine could reboot for the GPU driver.
  Results kept in `results/Spielberg/cpu-full1-20261113` and `cpu-lap1-20261113`;
  `scripts/analyze_ros_run.py <run dir>` prints the lap table and timing.
- **Full run 2 (GPU, 2026-11-13 afternoon)**: warm-up trials 1–5 = 134.9, 135.1, 135.0, 135.0,
  134.8 s (identical to CPU, as expected in lockstep with the same seed). Planning per step:
  MPPI p50 5–8 ms; **LMPPI p50 41 ms / p95 59 ms / max 79 ms** (first step 12.7 s JIT) — fits the
  0.1 s period on the RTX 3070. Sim round trip p50 13–30 ms. Run data lost with the session
  scratchpad (runs now write to `results/` in the repo).
- **First LMPPI lap failed to finish (both runs)**: GPU attempt 6 ended `speed_too_low` after
  2062 steps at s = 263/343 m; mean speed 1.25 m/s, max |ey| 0.84 m, 0 out-of-bounds steps, hull
  violation max 0.154. The car gradually slowed to 0.04 m/s. In SIT mode the runner treats
  `speed_too_low` as a stop, not a failed attempt, so the run ended with 5 successful laps.
  Diagnosis from the last 50 recorded steps:
  - chosen plans end at ~2.4 m/s, yet the applied first accelerations were small or negative
    (−1.9 … +1.6 m/s²) while the car was at 0.3–0.9 m/s;
  - learned value is **anti-correlated with terminal speed** (corr ≈ −0.5; "faster = more time
    to go") — out-of-distribution extrapolation, since all warm-up laps were at a constant 2.5 m/s;
  - ~0% of the 1024 sampled terminal states fall inside the hull (relaxation 0.03); hull distance
    is lowest for terminal speeds near the warm-up speed and grows with |ey|;
  - acceleration sample spread is large at this scale: `sample_std 0.25 × 5 m/s²` = 1.25 m/s² per
    step at 2.5 m/s (full-scale preset: 0.875 m/s² at 10 m/s), so samples end between 1.1 and
    3.4 m/s.
- **Diagnosis via quick-load** (`results/logs/quickload-diag.yaml` loads the CPU run's
  `init.npz` and starts LMPPI at step 0; 450 steps, 400-step insight recording):
  - one-step model error ≈ 0 (|Δv| 0.00, |Δr| 0.02 rad/s): the ST predictor matches the ROS sim
    through `accel_steering_speed`, so model mismatch is ruled out;
  - **MPPI procrastination**: at 0.75 m/s the shifted-in plan starts with +1.9 m/s², the
    re-optimized first action is −0.7 m/s²; plans end +0.55 m/s above current speed but the
    applied change averages −0.02 m/s per step. Terminal-only hull/value costs let "accelerate
    later" score the same as "accelerate now", and the value's spread across samples (~20 s for
    terminals metres apart) is noise after normalization.
  - **Cause: full-scale MPPI noise.** `sample_std 0.25` × `action_scale [3.2, 5.0]` = 0.08 rad
    of steering per step (19% of the F1TENTH range, vs 1% in the full-scale preset) and
    1.25 m/s² at 2.5 m/s, so samples end far outside the 4-lap data where the NF value
    extrapolates wildly.
  - `sample_std [0.0625, 0.1]` (same per-step noise as `action_scale [0.8, 2.0]`; quick-load
    refuses a changed `action_scale`): 45 s progress 92.4 m vs 89.6 m, min speed 1.33 vs
    0.59 m/s; still ~18% below warm-up pace (112 m). Now the default in
    `configs/f1tenth-ros-known.yaml`.
- **First successful LMPPI session** (`results/Spielberg/lmppi6`, GPU, quick-load from the CPU
  run's 4-lap warm-up, `sample_std [0.0625, 0.1]`, 6 LMPPI laps, seed 1):

  | lap | time (s) | mean v (m/s) | max v | max \|ey\| (m) | out-of-bounds | contacts | hull dist max |
  |---|---|---|---|---|---|---|---|
  | warm-up (MPPI) | 134.8–135.1 | 2.51 | ~2.6 | 0.93 | 0 | 0 | — |
  | LMPPI 1 | 147.0 | 2.30 | 3.23 | 0.849 | 0 | 0 | 0.41 |
  | LMPPI 2 | 155.7 | 2.17 | 3.26 | 0.850 | 0 | 0 | 0.39 |
  | LMPPI 3 | 124.6 | 2.71 | 3.48 | 0.851 | 0 | 0 | 0.80 |
  | LMPPI 4 | 125.0 | 2.71 | 3.48 | 0.846 | 0 | 0 | 0.37 |
  | LMPPI 5 | 107.3 | 3.15 | 4.15 | 0.849 | 0 | 0 | 0.40 |
  | LMPPI 6 | **100.9** | 3.35 | 3.95 | 0.848 | 0 | 0 | 0.57 |

  Dip-then-improve, like the authors' full-scale example (first LMPPI laps 7% slower, faster from
  LMPPI lap 4). Lap 6 is 25% faster than warm-up. Max |ey| sits at the planner limit
  (1.1 − 0.25 = 0.85 m) every lap: the boundary is the active constraint.
  Planning per control step (7604 LMPPI steps, RTX 3070): **p50 55 ms, p95 74 ms, p99 98 ms**,
  max 514 ms; 58 steps (0.8%) over the 100 ms period — spikes coincide with value retraining
  after each lap, which shares the GPU. Status `lap_limit`, 6/6 successful, 0 failures.
  **Done criteria met for the known-dynamics baseline** (warm-up + LMPPI laps in the ROS sim at
  F1TENTH scale, lap times dropping, no track violations).
- Raceline offset from the centerline: max 0.925 m, p95 0.825 m, so warm-up laps run close to
  the LMPPI controller boundary (1.1 − 0.25 = 0.85 m). Kept `boundary_buffer 0.25`: at
  |ey| = 0.9 the car's body edge is ~4 cm from the wall.

### GPU (diagnosed 2026-11-13)

`nvidia-smi` cannot reach the driver because **kernel 7.0.0-38 has no NVIDIA module**: the
prebuilt signed module package `linux-modules-nvidia-595-open-generic-hwe-24.04` is installed at
7.0.0-31 and was not upgraded with the kernel; `/lib/modules/7.0.0-38-generic` has no `nvidia*`,
no DKMS is installed, Secure Boot is enabled (so only Canonical-signed prebuilt modules load
without MOK enrollment). The OS boots from a USB drive (`sda4` → `/`, `sda3` → `/boot/efi`); this
does not change the fix, it only means the GRUB entry for 7.0.0-31 is a working fallback (its
module is present). Fix: `sudo apt install linux-modules-nvidia-595-open-7.0.0-38-generic`
(candidate 7.0.0-38.38~24.04.4+1) and reboot, or
`sudo apt install --only-upgrade linux-modules-nvidia-595-open-generic-hwe-24.04`. Then install
CUDA JAX in the venv: `pip install "jax[cuda13]==0.11.1"`.

**Fixed 2026-11-13 13:24.** The apt install (after `apt update`) also brought kernel 7.0.0-43 and
`linux-modules-nvidia-595-open-generic-hwe-24.04 7.0.0-43`, so after the reboot: driver
595.99.02, CUDA 13.2, `nvidia.ko` present for the running kernel, `jax.devices()` →
`CudaDevice(id=0)`. If a future kernel update breaks this again, check that
`dpkg -l | grep linux-modules-nvidia` matches `uname -r`.

### Resume on the ROS backend (2026-11-13)

- `RosSimulator.reset()` now starts the car a run-up distance (`v²/2a`, or the P-controller
  equivalent in angle mode) behind the requested pose, so the launch ends at the requested pose
  and speed; it logs `ros_reset` with requested vs reached state.
- The runner's restore check reads `getattr(simulator, "restore_tolerance", 1e-5)` (gym keeps
  the exact check; ROS uses 0.25) and logs `resume_state_restored` with the mismatch.
- `load_config` overrides now merge a `simulation.ros` dict into `RosConfig` (resume uses that
  path).
- Test: resume from `lmppi6/checkpoints/trial-000006` with `--config configs/f1tenth-ros-known.yaml`:
  reached state within 1.6 mm / 0.01 m/s of the checkpoint, max Frenet mismatch 0.022. Re-run
  after the GPU fix (`results/Spielberg/resume-test`): **LMPPI lap 7 = 93.8 s** (3.61 m/s, max
  |ey| 0.849, 0 out-of-bounds), continuing the improvement from lap 6 (100.9 s). Planning p50
  56 ms / p95 92 ms (first step 15.7 s JIT). Resume verified.
- Resume needs `--config` pointing at a ROS config: unmarked YAML settings override the
  checkpoint, and the packaged default has `backend: gym`.

### GPU crash after idle (2026-11-13 16:58)

The dGPU runtime-suspended after the 6-lap run (`NVreg_DynamicPowerManagement=2`, PCI runtime PM
`auto`, Ubuntu's `nvidia-persistenced` runs with persistence mode off) and failed to wake:
`GFW_BOOT` timeout → Xid 119 (GSP RPC timeout) ×3 → Xid 154 "GPU Reset Required"; JAX silently
fell back to CPU. Fix: `options nvidia NVreg_DynamicPowerManagement=0x00` in
`/etc/modprobe.d/nvidia-runtimepm.conf` — **this exact filename**: modprobe applies files in
filename order and the last option wins, so a differently named file (first attempt:
`nvidia-no-runtime-pm.conf`) loses to Ubuntu's `/lib/modprobe.d/nvidia-runtimepm.conf` (0x02);
the same name in `/etc` replaces it. Then `update-initramfs -u`. For the current boot without a
reboot: `echo on > /sys/bus/pci/devices/0000:01:00.0/power/control` and `nvidia-smi -pm 1`.
Check `modprobe -c | grep DynamicPowerManagement` shows only 0x00, and check run logs for
`CUDA_ERROR_NO_DEVICE` before trusting timing numbers.

### Warm-up steering flicker (2026-11-13)

Seen in Foxglove. Two causes:
- **Bridge drawing (my bug):** in `accel_steering_speed` mode `ego_steer` holds the steering
  *rate*, and the wheel TFs were drawn from it. Fixed: in that mode the bridge draws the wheels
  from the gym's steering-angle state (default mode unchanged).
- **Real jitter:** `planners/mppi.py` (warm-up) samples with unit std in normalized units and
  ignores `sample_std`, so with `action_scale` steering 3.2 rad/s the warm-up steering swung
  0.08 rad per step and reached 0.378 rad (sim limit 0.419; real car 0.34) on a lap that needs
  0.147 rad. Spielberg's raceline needs ≤ 0.44 rad/s of steering rate even at 8 m/s
  (`|dδ/ds|` max 0.055 rad/m). New `action_scale [1.0, 3.0]` (and `sample_std [0.2, 0.167]` to
  keep LMPPI noise at 0.2 rad/s / 0.5 m/s²): over 300 warm-up steps, steering change per step
  0.021 rad (was 0.080), max |δ| 0.170 rad (was 0.378), same tracking. Remaining small
  per-step dither is inherent to unit-std warm-up sampling; removing it needs a planner change.

Also fixed: the reset run-up fell 1.7 cm short (launch stops at target − 0.02 m/s), so at the
start line the car began at s = 343.30 and the runner logged a **1-step "successful lap"** into
the safe set. Run-up is now sized for the stop speed and lands 5 cm past the requested pose.

Real car: steering limit is **0.34 rad** (sim 0.419). The planner takes `s_min/s_max` from the
ROS gym's F1TENTH parameters; the car will need an override.

### Faster-than-real-time runs

Measured on the fresh run (Foxglove on, `trace_steps 50`): warm-up 1.7–1.9× real time, LMPPI only
0.6–0.95×. Per 0.1 s control step: warm-up 10 ms planning + 44 ms sim round trip; LMPPI 74 ms +
49 ms. The round trip is 10 sequential drive→step→odom exchanges; it was 13–30 ms in runs without
Foxglove (its relay subscribes to TF/odom published 10× per step). `trace_steps > 0` makes LMPPI
copy its full debug state off the GPU every step (inflates planning time); value retraining
shares the GPU at lap ends.

Added (untested at time of writing): bridge `substeps_per_drive` (sync mode; default 1),
adapter `simulation.ros.bridge_substeps` (must match; error if the bridge advances a different
amount per message; launch caps its last acceleration to land on the target speed),
`configs/f1tenth-ros-known-fast.yaml` (10 substeps, `trace_steps 0`) and
`configs/ros-bridge-lockstep-fast.yaml`. Use for batch runs without Foxglove; keep the normal
configs when watching or recording diagnostics.

Gotcha: `configs/default.yaml` is a symlink to `src/sit_lmpc/config/default.yaml`; edit one.

### Fresh end-to-end run (`results/Spielberg/fresh25`, 2026-11-13)

Fresh warm-up with `action_scale [1.0, 3.0]`, `sample_std [0.2, 0.167]`, GPU, Foxglove on.
Warm-up 135.1–135.3 s. LMPPI laps: 168.5, 142.7, 122.3, 118.0, 108.3, 102.6, 97.9, 92.6, 92.0,
**89.0 s** (34% faster than warm-up), all 0 out-of-bounds, max |ey| 0.85 (the planner limit).
Attempt 16: **collision** 5 s after the line → run ended (SIT stops at the first failure). The
previous lap crossed the line at 0.60 m/s (finish-line braking); the car then crawled at
0.5–0.8 m/s while its heading swung −0.26 → +0.45 rad; CoG |ey| stayed at the 0.85 m limit but
the front-left body corner reached 1.117 m > 1.1 m. Two causes: the slow lap-boundary regime, and a
boundary constraint on the CoG only (0.25 m buffer covers the half-width only at small heading
error). Figures: `results/figures/fresh25/`.

**Planning time and GPU throttling.** LMPPI planning was 55–61 ms for LMPPI laps 1–2, then a flat
~113 ms from lap 3 on; the CPU-only sim round trip stayed at 42–52 ms. Resuming both the lmppi6
(previously 55 ms) and fresh25 checkpoints afterwards gave 120 and 117 ms, so it is not the learned
state. `nvidia-smi` under load: P5, 1290/2100 MHz, 26 W, **SW power cap + SW thermal slowdown
active** (~30 min). The laptop GPU throttles under sustained load; planning time tracks the GPU
clock. Report 55 ms (cool) and ~115 ms (sustained) for this laptop; the Jetson number is what
matters for the car.

### Steering limit and delay compensation (car prep, 2026-11-13)

- `simulation.steering_limit` (rad, default null): overrides the vehicle's s_min/s_max in
  `adapters.vehicle_parameters_for`, so the planner model and the ROS adapter both use it. The
  adapter also bounds each commanded steering rate so the estimated angle stays within the limit
  over the message duration. Car: 0.34. The sim baseline never needed it (max |δ| 0.289 LMPPI,
  0.170 warm-up < 0.34). Check with a 0.15 limit (`results/Spielberg/limit-check`, 300 LMPPI steps):
  max |δ| 0.121, no violation (limit not reached on that stretch).
- `simulation.ros.actuation_delay` (s, default 0 = off; multiple of `timestep`): the MPC lab's
  pure-delay model. The adapter keeps the last `delay/timestep` applied controls and the steering
  estimate history; each step it uses the plant's steering angle from `delay` ago, rolls the
  measured state forward through the ST model with the in-flight controls, and returns that
  predicted state (so safe-set/value data pair x(t+d) with u(t)). Reset: the launch counts
  in-flight accelerations, then holds for `delay` before handing over; run-up includes v·delay.
  Per the user: **no sim validation; tune on the car** (the bridge stays without delay emulation).
  Crash-only smoke with 0.2 s in the (delay-free) sim (`results/Spielberg/delay-smoke`): runs; the
  launch hands over at 1.91 instead of 2.51 m/s because it counts commands the sim already applied —
  expected without a real delay; check the launch first on the car.

### Presentation figures

`scripts/presentation_figures.py <run> --out results/figures/<name>` writes PNG + SVG and CSV
tables: `lap_times`, `track_speed` (track map colored by speed, warm-up lap vs best LMPPI lap),
`speed_profile`, `lateral_offset` (with planner limit and track edge), `value_training`,
`planning_time` (`--timing "LABEL=RUN" ...` compares runs; runs stopped before close fall back to
the 100-step timing summaries, marked *), `warmup_steering` (`--steering "LABEL=RUN" ...`), plus
`laps.csv` and `planning_time.csv`. Colors: warm-up = blue, SIT-LMPC = orange in every figure;
speed = single-hue blue ramp. Palette = the dataviz reference's first three categorical slots
(documented as passing all-pairs CVD checks in light mode; `node` is not installed here to rerun
the validator). Test output: `results/figures/test-lmppi6`.

### BNN baseline: F1TENTH population model (2026-11-13)

Pipeline: (1) collect paired transitions for a population of ST cars that all replay the same
action tapes (dyna-dist `collect_vehicle`); (2) train the Gaussian BNN offline (mean net + basis net
× 96-dim Gaussian z, residual on the analytic ST at a baseline friction); (3) import into
`artifacts/`; (4) SIT-LMPC with `backend: gaussian_bnn`, between-lap tempered posterior updates of
z, chance constraints (boundary 0.05, terminal hull 0.9, planning-sample estimator), as in the
authors' BNN example. Decisions with the user: chance constraints ON; population ranges below.

dyna-dist changes (vendored copy): `config.py` allows `profile: f1tenth` for collection;
`data/vehicle_collection.py` uses `config.profile` (was hard-coded `fullscale` twice);
`adapters/gym_st.py` falls back to f1tenth_gym_ros's ST + `rk4_integration` (`SubstepRK4`, same
sub-stepping as the bundled gym's `RK4Integrator`) when the bundled gym is not installed.

`configs/collect-vehicle-f1tenth.yaml`: μ 0.4–1.2 (sim car 1.0489), front/rear cornering ×0.85–1.15,
mass ×0.8–1.3, 24 random + 9 anchor instances (train/validation/test split by dyna-dist), speeds
1–8 m/s in 0.5 steps, 2 tapes × 12 s per speed, lateral-accel envelope 6 m/s², speed target ±1 m/s,
yaw-rate validity limit 8 rad/s. Result `data/f1tenth-vehicle` (22 s, 22 MB): 33 instances, 990
trajectories, 118,685 valid transitions, 1 trajectory stopped at the envelope; endpoint spread after
12 s grows from 0.2 m (1 m/s) to 60 m (8 m/s).

**Finding: mass barely matters in the ST sim.** One-at-a-time at 8 m/s, same actions, 12 s:
friction low/high → +11.5 / −4.6 m lateral; front stiffness +13.2 / −10.7 m; rear −10.2 / +9.1 m;
**mass +0.07 / −0.01 m**. CommonRoad ST tire forces scale with μ·m·g, so mass cancels in the
lateral/slip dynamics and acceleration is commanded directly; mass only acts through m/I in yaw,
and inertia is fixed in this population. For the mid-session "added weight" test in sim, use
`inertia_policy: proportional` or test it on the real car; grip (μ, cornering stiffness) changes
are the meaningful sim knobs.

Training: `configs/train-dynamics-f1tenth.yaml` = the authors' recipe (20,000 steps, latent 96,
[128,128]) with `baseline_profile: f1tenth`, `baseline_friction: 0.8` (population middle). Output
`runs/dynamics-f1tenth`. ~1000 steps/min on the laptop GPU; validation NLL per component 24.9 →
−2.68 by step 4000, train ≈ validation.

SIT-LMPC config: `configs/f1tenth-ros-bnn-fast.yaml` (extends the fast known config;
`sit_lmpc_mode: false`, so failed attempts are reset and counted rather than ending the run).
Imported model: `runs/dynamics-f1tenth/best_refit` (best validation NLL −3.70 at step 19,200, prior
EM-refit on the 25 training instances; the authors' shipped model is also a population refit) →
`artifacts/vehicle-bnn-f1tenth-l96`.

### BNN baseline result (`results/Spielberg/bnn25`) — OFFICIAL

Fast mode, seed 1, fresh warm-up, BNN + between-lap tempered posterior + chance constraints.
25/25 successful, 0 failures, 0 contacts, 0 out-of-bounds steps, LMPPI max |ey| 0.844.
- Warm-up 135.1–135.3 s (identical to known).
- LMPPI: 144.3, 139.6, 129.0, 124.2, 121.6, 118.0, 117.9, 114.3, 114.0, 112.8, 112.6, 112.2,
  111.8, 111.0, 111.2, 110.8, 111.2, 110.3, 110.4, **110.1 s** (19% faster than warm-up; last 5
  mean 110.6). Known dynamics at the same point: 70.9 s.
- BNN/known lap-time ratio: 0.87 (LMPPI lap 1), 0.96 (3), 1.15 (5), 1.29 (7), 1.43 (10), ~1.55
  (17–20). Same shape as the authors' recorded full-scale runs (0.98, 1.02, 1.05, 1.09, 1.16, peak
  ~1.39 near lap 80, back to 1.01 by lap ~295), compressed into fewer laps because F1TENTH learning
  is faster per lap. 20 learned laps cover only the widening part of the gap; the authors needed
  ~300 laps for the BNN to catch up.
- Belief: 25 updates, 12 published, 13 held (`held_no_calibrated_candidate`), mostly laps 9–17;
  generations 10–12 were accepted only at evidence weight (likelihood power) 0.1. Held-out 1 s
  position error 0.1–0.9 mm; coverage 86–100% (target 95%) — accurate mean, overconfident
  uncertainty on faster laps.
- Chance fallbacks: 32 steps in total (0–5 per lap, < 0.4%); authors' run 9.4%.
- Constraint weights: the same smallest feasible combination (hull 3.75, boundary 1.25, chance 2,
  chance 2) is chosen on nearly every step; the chance terms also enter every sampled plan's cost
  at weight 2, which is the leading hypothesis for the slower expansion.
- Planning per step: BNN warm-up p50 37.7 ms; **BNN LMPPI p50 98.5 ms, p95 102.5, p99 132.6 ms,
  21% of steps over 100 ms** (64 sampled cars × candidate checks; GPU at the throttled 1290 MHz).
  Known LMPPI: 19.5 ms. Lap times unaffected (lockstep).
- Figures: `results/figures/bnn25/` (compare_lap_times, bnn_vs_known_ratio, bnn_belief,
  chance_fallbacks, plus the per-run set).
### Ablation: BNN without chance constraints (`results/Spielberg/bnn-nochance25`)

`configs/f1tenth-ros-bnn-nochance-fast.yaml` (= BNN baseline with `probabilistic_safety.enabled:
false`), seed 1, fresh warm-up. 25/25, 0 failures, 0 contacts, 0 out-of-bounds, LMPPI max |ey| 0.850.
- Warm-up identical to the BNN baseline (135.1–135.3 s; same belief history through generation 4).
- LMPPI: 170.3, 149.6, 114.1, 114.1, 98.4, 99.7, 91.1, 88.6, 84.5, 81.0, 79.5, 77.9, 76.9, 76.1,
  75.8, 75.7, 75.8, 76.3, 76.0, **75.6 s** (44% faster than warm-up; last 5 mean 75.9). Known: 70.9;
  BNN + chance: 110.1.
- **Conclusion: the chance constraints, not the BNN predictor, cause the slow learning of the BNN
  baseline.** Without them the BNN tracks the known model within 2–4 s per lap and converges ~5 s
  (7%) behind it; the residual gap is consistent with the learned model's growing error on faster
  laps (held-out 1 s error 0.2 → 1–2.5 mm).
- Belief: 18 published, 7 held; many published at evidence weight 0.1; coverage dipped to 67–77% on
  some faster laps (overconfident uncertainty, unused by the planner without chance constraints).
- Planning (LMPPI ran alone; warm-up windows contaminated by parallel tests): **p50 64.2 ms, p95
  74.3, p99 85.2 ms, 0.4% > 100 ms** vs 98.5 ms / 21% with chance constraints and 19.5 ms known.
- Figures: `results/figures/bnn-nochance25/` (incl. three-way `compare_lap_times`).
- Caveat for the report: no run came close to failing, so this does not measure the safety benefit
  of chance constraints, only their cost.

### Fast warm start with the lab 5 pure pursuit (sim only, 2026-11-14)

Plan (user): after the BNN no-chance ablation, 20 LMPPI laps from a fast pure-pursuit warm start
with (a) known model, (b) BNN + chance, (c) BNN without chance; then the IRL branch.

- Controller: `src/sit_lmpc/planners/pure_pursuit.py`, a port of the ESE 6150 lab 5 submission
  (`ese-6150-lab-5-pure-pursuit-group-8`, tag `submission` = 4e5e5ce, `src/pure_pursuit_node.cpp`):
  the same speed profile (±3-point curvature, max curvature over the next 0.7 m, lateral cap,
  understeer steering-margin cap, braking then forward passes over 2 laps), lookahead
  base + gain·v with corner-cut shortening, 25 ms latency projection, dynamic steering
  (steady-state cornering model + yaw-rate feedback). Defaults = the submission's Spielberg
  parameters (max speed 25, lateral 22, brake 7, forward 9, lookahead 0.6 + 0.16·v in [0.7, 5],
  corner cut 0.12, understeer 0.0028, latency 0.025, yaw-rate gain 0.12). Raceline: the
  submission's `waypoints/spielberg_raceline.csv` → `configs/racelines/spielberg_pp_lab5.csv`.
- **Rate matters.** Offline, with the sim's ST physics from the start line: the port completes a lap
  in 24.7 s at 100 Hz (submission measured 24.35 s), but spins within 2–7 s at 50, 20 or 10 Hz. My
  first simplified pure pursuit at 10 Hz also crashed in the ROS sim (twice). So the warm-up runs at
  the bridge's 100 Hz inside each 0.1 s SIT-LMPC step: the runner sets `simulator.substep_policy`
  to `PurePursuit.substep` during warm-up laps; the adapter queries it before each 0.01 s bridge step
  with the current measured state, converting the targets with the sim's own actuator loops
  (steering 10·sv_max/(s_max−s_min) ≈ 38/s capped at 3.2 rad/s; speed 10·a_max/v_max up,
  10·a_max/|v_min| braking). The step's recorded control is the mean of the ten commands (exact
  end-of-step steering angle and speed; only the intra-step path differs — relevant only for BNN
  belief data from warm-up laps). LMPPI is unchanged at 10 Hz. Validation requires backend ros,
  lockstep and `bridge_substeps 1` (normal bridge).
- Why SIT-LMPC stays at 10 Hz: horizon is 10 steps = 1 s; faster rates need more steps (cost
  scales), the BNN predicts 0.1 s increments (retrain needed), and both baselines/authors use 10 Hz.
- ROS sim check (`results/Spielberg/ppwarm-test4`): warm-up laps **24.8 s** (from the 5 m/s launch)
  and **24.0 s**, 0 contacts, max |ey| 0.65 m, top speed 20 m/s (sim v_max), slip up to 0.65 rad.
- Configs: `f1tenth-ros-known-ppwarm.yaml`, `f1tenth-ros-bnn-ppwarm.yaml`,
  `f1tenth-ros-bnn-nochance-ppwarm.yaml`: `action_scale [3.2, 9.51]` (sim actuator limits, as the
  submission had), `sample_std [0.0625, 0.0526]` (same physical LMPPI noise as the baselines),
  `initial_speed 5`, normal lockstep bridge (`configs/ros-bridge-lockstep.yaml`).
- Wide BNN for these runs: the original model covers 1–8 m/s, ~6 m/s². New
  `configs/collect-vehicle-f1tenth-wide.yaml`: same population, speeds 1–20 m/s, excitation
  16 m/s² (22 m/s² made the mid-population μ = 0.8 nominal car leave the ST validity envelope and
  aborted collection), slip limit 1.2, yaw-rate limit 10. `data/f1tenth-vehicle-wide`: 1320
  trajectories, 154,488 transitions, 62 truncated at the envelope; 31% of training transitions
  above 14 m/s, but only 2.6% with |slip| > 0.3 rad while the warm-up laps reach p95 0.49 rad —
  high-slip data is sparse. Training: `configs/train-dynamics-f1tenth-wide.yaml` →
  `runs/dynamics-f1tenth-wide` → `artifacts/vehicle-bnn-f1tenth-wide-l96`.
- **Known model from the 24 s warm start fails at the LMPPI switch (2 attempts, 5/5 warm-up laps
  clean each time).** Both crashed within ~2 s of LMPPI taking over at 18 m/s:
  (1) `sample_std [0.0625, 0.0526]` (baseline-sized noise; `results/Spielberg/known-ppwarm25-std-small`):
  weaved 0 → −0.46 m on the straight and braked only −0.4 to −2.6 m/s² where the warm-up braked
  ~−7, entered the first corner at 16.9 vs 13–15 m/s, slip 0.68 → 1.74 rad, wall at step 20;
  (2) `sample_std [0.25, 0.25]` (authors' default; `results/Spielberg/known-ppwarm25`): braked
  hard (−7 to −7.7) but the steering rate flipped sign almost every step (±1.5–2.6 rad/s), steering
  swung ±0.2–0.42 rad, slip oscillated with growing amplitude (±0.3, 0.59, 1.8 rad), spun at
  s ≈ 25 m, wall at step 17. Interpretation: at 15–20 m/s this low-inertia ST car needs faster
  control than 10 Hz (the same limit as pure pursuit: clean at 100 Hz, spins at ≤ 50 Hz).
  Not tuned further; options for the user: run SIT-LMPC at 20 Hz (horizon 20; BNN would need
  retraining at 0.05 s), use a slower warm start (`speed_scale`) that 10 Hz can take over from,
  or report the finding and drop the fast series. **User chose the slower warm start.**
- The two failures point at the noise channels separately: at 18 m/s cornering needs ~0.03 rad of
  steering, so 0.8 rad/s steering noise held for 0.1 s overshoots; braking needs ~−7 m/s², which
  0.5 m/s² acceleration noise cannot find. Split noise `sample_std [0.0625, 0.25]` = 0.2 rad/s
  steering, 2.4 m/s² acceleration.
- **`speed_scale 0.6`** (profile 37.2 s, 3.7–15 m/s) with split noise
  (`results/Spielberg/ppwarm-scale06`, 7 laps): warm-up 38.4, 38.1, 38.0, 38.1, 38.0 s (max 14.7 m/s,
  slip ≤ 0.14); LMPPI 39.0, 39.4 s; 0 contacts; LMPPI slip ≤ 0.31. Made the fast-start default in
  all three configs. Series runs sequentially via `results/logs/run-ppwarm-series.sh`
  (known → BNN + chance → BNN no chance), results in `results/Spielberg/*-ppwarm25`.
- **Known model, fast start (`results/Spielberg/known-ppwarm25`)**: warm-up 38.4, 38.1, 38.0, 38.1,
  38.0 s; LMPPI 39.1, 38.9, 39.3, 39.2, 39.0, 38.9, 39.0, 39.0, 38.9, 38.9, 39.5, 39.2, 39.5, 39.2,
  39.8, 39.8, 40.6 s, then a spin in the first corner (attempt 23: entry at 12.6 m/s, heading error
  −0.27 → −1.6 rad, slip 2.1) → run ended (SIT). Never beat the warm-up (best LMPPI 38.9 vs 38.0)
  and drifted slower over the last 7 laps, consistent with the newest-4-laps safe set dropping the
  faster warm-up laps. All three fast-start failures were in that first corner (hard braking into
  a turn) — the weak point for 10 Hz control on this track.
- **BNN + chance, fast start, first attempt (`results/Spielberg/bnn-ppwarm25-warmupbelief`) —
  confounded.** LMPPI 40.3, 39.9, 40.9, 42.1, 42.4 s, then braked repeatedly and stopped at
  s ≈ 110 m (`speed_too_low`, no violation). Belief held-out 1 s error 17–167 mm (slow runs:
  0.1–2.5 mm), calibration only accepted evidence weight 0.0001 (≈ prior), so the 64 sampled cars
  disagreed widely and the chance constraints grew ever more cautious. Cause: the 100 Hz warm-up
  laps are recorded with step-mean controls; at 9–15 m/s pure pursuit's commands change a lot
  within 0.1 s, so those labels do not explain the transitions. **Fix:** the runner no longer passes
  pure-pursuit warm-up laps (all steps `mppi`) to the belief learner (`dynamics_observation_skipped`
  event); they still seed the safe set and value function. Both BNN fast-start runs rerun with it
  (`results/logs/run-bnn-ppwarm-series.sh`); the first BNN no-chance attempt was stopped after 3
  warm-up laps.
- **BNN + chance, fast start (`results/Spielberg/bnn-ppwarm25`), with the belief fix**: warm-up
  38.4, 38.1, 38.0, 38.1, 38.0 s; LMPPI 40.0, 39.8, 41.3, 41.6, 42.0 s, then attempt 11 stopped after
  49 steps (`speed_too_low`, on track, s ≈ 33.6 m). 186 `no_feasible_candidate` fallbacks from the
  first LMPPI step: P_violate(hull) = 1 for every sample (limit 0.9), so the conservative fallback
  braked. Belief: held-out 1 s error 42 → 60 → 44 → 28 → 32 mm, ~230 of ~400 transitions per lap
  accepted, evidence weight pinned at 0.0001 throughout. LMPPI planning p50 99 ms, 40% of steps
  over 100 ms.
- **BNN no chance, fast start (`results/Spielberg/bnn-nochance-ppwarm25`)**: warm-up as above;
  LMPPI 38.3, 38.6, 38.7, 38.8, 38.9, 39.1, 39.1 s, then attempt 13 braked on track from ~12 m/s to
  a stop at s ≈ 109 m (`speed_too_low`, no fallbacks — chance off; ey ≈ 0, slip ≈ 0). Belief
  improved: error 58 → 79 → 15 → 45 (held) → 34 (held) → 9.7 → 33 mm, evidence weight rose to
  0.001 from gen 3; two updates held for calibration. LMPPI planning p50 63 ms, 4.8% over 100 ms.
- **Fast-start summary**: no fast-start run beat the 38.0 s warm-up. Best LMPPI: BNN no chance
  38.3 s, known 38.9 s, BNN + chance 39.8 s; the BNN no-chance run beat the known model on its
  first laps but ended sooner (7 vs 17 LMPPI laps). Every run drifted slower lap over lap (newest-4
  safe set ratchet) and every run ended in an unforced stop or spin. Chance constraints make
  things worse here, as in the slow-start ablation, and more sharply: the hull-chance check is
  infeasible for all samples. Root cause for the BNN: the wide population covers |slip| > 0.3 in
  only 2.6% of its data, against a lap p95 of 0.49, so the posterior stays near the prior and the
  64 sampled cars spread wide. Candidate fix (not started, needs a decision): population tapes
  driven closed-loop at high slip.
- Figures: `results/figures/{known,bnn,bnn-nochance}-ppwarm25/`, `results/figures/fast-start-compare/`
  (lap times, planning time for the three fast runs) and `results/figures/all-baselines-compare/compare_panels.png`
  (slow vs fast start side by side, same colour per model; new `--compare-right` option in
  `presentation_figures.py`; the single-panel `--compare` only has three colours).
- Concurrency note: the no-chance ablation, the known fast-start run (second sim, ROS domain 78)
  and the wide-model training ran at the same time; lap results are unaffected (lockstep), but
  their per-step timing is contaminated — re-measure planning time separately.

### Known-dynamics baseline, after the track fix (`results/Spielberg/known25-fixed`) — OFFICIAL

Fast mode (`configs/f1tenth-ros-known-fast.yaml` + `ros-bridge-lockstep-fast.yaml`, no Foxglove,
`trace_steps 0`), seed 1, fresh warm-up. 25/25 successful, 0 failures, 0 contacts, 0
out-of-bounds steps; max |ey| 0.947 (warm-up raceline), LMPPI 0.85.
- Warm-up 135.1–135.3 s (mean 135.2).
- LMPPI: 166.5, 145.2, 134.5, 115.6, 105.4, 97.7, 91.3, 85.8, 82.7, 78.9, 77.0, 74.8, 73.9, 72.8,
  72.1, 72.0, 71.4, 71.3, 71.1, **70.9 s** (48% faster than warm-up; last 5 mean 71.3 s).
- Finish-line speeds: median 3.87 m/s, min 1.85 (first LMPPI lap); before the fix ~0.7.
- Planning per control step: **LMPPI p50 19.5 ms, p95 26.5, p99 36.8 ms**, 0.15% > 100 ms
  (value retraining, first JIT 4.4 s); warm-up p50 4.8 ms. Sim round trip p50 ~13.5 ms.
  The earlier 55–115 ms figures were mostly `trace_steps > 0` (per-step device_get of 1024
  sampled rollouts) plus GPU throttling.
- Wall time: warm-up 5.4× real time.
- Figures: `results/figures/known25-fixed/`.
All earlier LMPPI results (lmppi6, fresh25, resume-test) ran with the track bug and are superseded.

### Root cause of the finish-line braking: track search-window bug (fixed 2026-11-13)

Trace across a crossing (`results/Spielberg/finish-trace`, resume fresh25 trial 10, 250 insight
steps): the chosen plan's terminal s was pinned at exactly L = 343.32 for the last ~1.5 m; the car
braked from 3.1 m/s (22 steps before the line) to 0.75 m/s. Positions past the line all projected
to s = L, so crossing earned no progress.

Cause: `Track.get_s_search_range` wrapped its index window with `% self.centerline.length`.
`Raceline.length = len(ss)`, and `from_centerline_file` resamples ss every 0.1 m (Spielberg: 3434
entries) while the spline used for projection keeps the raw points (865 rows, closed). Near the
line the window became 861…1260; JAX clamps out-of-range gathers to the last row (the finish
point), so the track past the line was a single point. The numpy projection already used
`len(spline.s) − 1`. custom1 has no centerline file (counts 3823 = 3823), so the authors' runs were
unaffected; Spielberg is the first centerline-file map.
Fix (one line, `track/track.py`): wrap by `len(self.centerline.spline.s) - 1`. Offline check: the
1024 recorded terminal positions now project within 0.1 mm of the numpy reference; custom1
unchanged; track tests pass.

Earlier hypotheses below (slow extension ratchet) describe the downstream effect: once laps cross
slowly, the stored lap starts are slow too.

### Finding: LMPPI brakes at the finish line every lap

Every LMPPI lap slows to 0.5–0.8 m/s within ~10 m of the finish and starts the next lap at that
speed (warm-up laps cross at 2.4–2.5 m/s); see `results/figures/test-lmppi6/speed_profile.png`.
Costs roughly 5–8 s per lap. Hypothesis (unverified): near s = L the terminal hull is built from
the newest lap's start (the next-lap extension records), so once one lap crosses slowly (LMPPI
lap 1: 0.69 m/s) the next one's terminal targets at the finish are slow too, and it reinforces.
To investigate: hull/value at s ≈ L, `extension_seconds`, and whether the authors' full-scale
example shows the same dip.

Evidence (lmppi6 checkpoint trial 6 safe set): each record spans s = 0 … ~350 m, i.e. the
extension reaches only 7–8 m past L (`extension_seconds 2 × horizon 10 × dt 0.1 × vmax`). Mean
speed in the records: 2.5–3.2 m/s in [L−20, L), but 1.9–2.2 m/s in [L, L+10) because the extension
is the next lap's (slow) start. With ~3.5 m/s and a 1 s horizon, terminal states near the line
must sit in a hull built from those slow starts → braking → the next extension is slow too
(ratchet). Not yet explained: why LMPPI lap 1 braked (0.69 m/s) when its extension came from
warm-up starts at 2.4–2.5 m/s. Wall-clock pauses at lap ends are only 0.2–0.5 s (10 s once, the
LMPPI JIT) and freeze sim time in lockstep, so they are not the cause.
Next: after the 25-lap run, a short run with full insight recording across a lap boundary to see
which term (hull vs value) forces the braking.

## 7. Status and open questions (2026-11-14)

Later status (world model, closed-loop sim runs, car code): see `docs/PROGRESS_LOG.md`.

Sim baselines on Spielberg, all done:

| run | warm-up | best LMPPI lap | LMPPI laps | planning p50 |
|---|---|---|---|---|
| known, slow start (`known25-fixed`) | 135.2 s | 70.9 s | 20/20 | 19.5 ms |
| BNN + chance, slow (`bnn25`) | 135.2 s | 110.1 s | 20/20 | 98.5 ms |
| BNN no chance, slow (`bnn-nochance25`) | 135.2 s | 75.6 s | 20/20 | 64 ms |
| known, fast start (`known-ppwarm25`) | 38.0 s | 38.9 s | 17, then spin | 19.7 ms |
| BNN + chance, fast (`bnn-ppwarm25`) | 38.0 s | 39.8 s | 5, then fallback stop | 99 ms |
| BNN no chance, fast (`bnn-nochance-ppwarm25`) | 38.0 s | 38.3 s | 7, then stop | 63 ms |

Open questions:
- BNN in the fast/high-slip regime: population data covers |slip| > 0.3 in 2.6% of samples (laps
  p95 0.49). Closed-loop high-slip excitation tapes would be the fix. **Decision (2026-11-14):
  report the fast-start BNN result as is; no recollection.**
- Fast start: why LMPPI brakes to a stop on track (no fallback, no violation) in the no-chance run;
  and the newest-4-laps safe set ratchets lap times up after a fast warm-up (relevant to the
  condition-aware safe set).
- Chance constraints: hull chance limit 0.9 is infeasible for every sample at speed; tuning the
  limits or the sample count is untested.
- Resume on the ROS backend relies on the 0.25 m restore tolerance (launch run-up).
- Planning time: BNN steps exceed 100 ms often (40% with chance at speed); matters for the Jetson.
- Real car slip: kinematic β; VESC IMU gyro for yaw rate; log the full IMU on every run and build a
  slip estimator offline, enabled only if it clearly beats kinematic β (decided 2026-11-14).
- Real car: steering angle has no sensor (estimate from commanded rate; identify actuator delay,
  `actuation_delay` ≈ 0.2 s); slip from PF odom twist or β = 0; steering_limit 0.34 rad; Jetson
  GPU budget unknown.
- Levine map in sim still to do.

Next steps (agreed roadmap): the IRL branch (Levine lobby,
particle-filter odom, VESC IMU yaw rate, steering limit 0.34, delay compensation, Jetson setup).

### Hull recompile fix (2026-11-22)

The safe-set hull code compiled small GPU programs during control steps whenever the hull's facet count changed
(fixed in "Safe set: stop hull recompiles while driving"). It only cost wall-clock planning time. All ros-sim
baselines and experiments before the fix ran on the lockstep bridge, where the simulator waits for the planner, so
their driving results (lap times, crashes, trajectories) stay valid. Planning-time numbers recorded before the fix
(p95, steps over 100 ms) include the recompiles and overstate the cost, most of all with the 15-step horizon.
