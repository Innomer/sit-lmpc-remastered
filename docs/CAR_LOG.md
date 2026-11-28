# Car log: SIT-LMPC with the v11 world model on the F1TENTH car

Branch `irl-world-model`. Car: Jetson Orin Nano Super 8 GB, JetPack 7, ROS 2 Jazzy. Track: Levine hallway loop
(`levine_self_again`, 62.6 m). Steps to run everything: `docs/CAR_RUNBOOK.md`.

## 2026-11-27/28: first day on the car

### 1. Before the car: dry runs of the car code in the ROS sim

The car configs ran in real time against the ROS sim on the Levine sim map (`scripts/car_dryrun.sh`), the same code
path as on the car. Every bug below only shows in real time (in the lockstep sim the sim waits for the planner):

| Found | What happened | Fix |
|---|---|---|
| a stop after moving ended the whole run | no retry possible | it is a failed attempt; the car waits for go |
| restart launched at the failed lap's line speed | 7 m/s from standstill within 2 m | restarts drive to the line at 1.5 m/s (`retry_speed`) |
| the lap checkpoint waited for value training | control loop frozen 2.8 s at 7 m/s, car into the wall | see the last row |
| first learned step compiled the planner | 26 s without commands while driving | planner and value training compiled at startup |
| stale obstacle flag | false stop after a restart | cleared when a start is accepted |
| training in the same process | 0.26-0.64 s control-loop pauses; spins at 7 m/s | **the car stops after every lap**, trains and saves while standing, then waits for go |
| Ctrl-C inside a ROS callback | run ended as `error` | Ctrl-C is raised after the ROS spin returns |

Final dry run: warm-up and first learned lap clean, 0 watchdog stops, planning max 62 ms (laptop).

### 2. Car setup: sensors and software

| Item | Finding | Result |
|---|---|---|
| Python packages | Ubuntu's `coverage` and `matplotlib` (built for NumPy 1) broke imports in the venv | newer copies installed in the venv |
| LiDAR | SICK TiM5xx: 811 beams over 270 deg at 15 Hz (training layout: 1080 beams) | resampled to 1080 beams |
| VESC IMU | the VESC sends only zeros (no IMU on this board); `/imu` belongs to the SICK driver and is never filled | not usable |
| RealSense D435i IMU | IMU disabled at first: the Jetson kernel has no HID sensor modules | librealsense built with the RSUSB backend + realsense-ros from source: `/camera/camera/imu` at 200 Hz |
| IMU in the car code | camera axes differ from the car's; camera tilted; IMU 0.35 m ahead of the rear axle | axis remap `[z, -x, -y]`; automatic levelling from gravity at start (1.4 deg on the floor); accelerations moved to the centre of gravity using the camera offset from the vision lab (`DEFAULT_C_L`) and the LiDAR TF; 0.1 s averaging; best-effort QoS (the RealSense publishes best effort) |
| `car_check.py` | IMU signs and units right (x, y slopes positive; z 9.8 at rest); VESC/PF speed ratio 1.001 | `imu_yaw_scale` 1.0, `imu_accel_scale` [1, 1] |
| Particle filter CPU | numpy's OpenBLAS kept one busy-waiting thread per core: **388% CPU**, all 6 cores at 100%, PF gaps up to 0.37 s | `OPENBLAS_NUM_THREADS=1`: **58% CPU**, same 47 Hz, gaps under 35 ms |

### 3. Track from the SLAM map

`scripts/make_track.py` on the PF map: 62.64 m, wall distance from the centreline median 0.76 m, 5th percentile
0.62 m, so `planner.half_width: 0.62` (the old guess was 0.40). The right straight (y = 9-13.5 m) has SLAM noise:
the measured centreline wobbles and steps about 0.15 m there (13 narrow spots, one at 0.00 m). More smoothing
(`--smooth 2.5`) made it worse (5th percentile 0.48 m: corners get rounded), so the default was kept. The start line
is in the middle of the bottom-left corner; the car drives counterclockwise on the map.

### 4. First car run: bicycle model, 2 m/s cap

`configs/car/levine-bicycle-slow.yaml`, full stack, bag recorded. Results: `results/car/car-levine-bicycle-slow-20261128-011526/`.

| | Result |
|---|---|
| 5 warm-up laps (pure pursuit) | all clean, 32.2-32.5 s at 2.0 m/s |
| stop after each lap, training while standing, restart on go | worked every time; 4 value generations trained on the Jetson |
| watchdog stops | 0 |
| learned laps | both failed within 0.7 s: the planner took p95 243 ms, max 327 ms per step with the full stack, and with the control step and its waits a new command reached the car only every **0.6-0.75 s**, so the car drove 0.4-0.5 m per decision on old steering (heading error grew to 0.45-0.55 rad, then a "stopped" failure and a LiDAR obstacle stop) |

### 5. v11 on real data (offline bag check)

`scripts/bag_check.py` on the 5 warm-up laps (349 starts over 500 s): v11 vs the bicycle model, predicted with the
commands that were actually sent, compared with what was measured.

| Ahead | Position error median / p90: v11 | bicycle model | v11 vs bicycle |
|---|---|---|---|
| 0.5 s | 0.14 / 0.25 m | 0.15 / 0.27 m | about equal |
| 1.0 s | 0.26 / 0.49 m | 0.38 / 0.71 m | **31% smaller** |
| 1.5 s | 0.49 / 0.96 m | 0.74 / 1.57 m | **34% smaller** (p90: 39%) |

Yaw rate: v11 better at 0.5 s (0.45 vs 0.74 rad/s), equal later. Speed: the bicycle model is better (0.01 vs
0.05 m/s). Slow, steady laps only. The first real evidence that the world model beats the physics model on the car,
and that the RealSense IMU path works.

### 6. Planning time on the Jetson

Budget: 100 ms per control step (p95 under 90 ms). `scripts/jetson_check.py` (made-up centreline laps, no driving):

| Setting | Bicycle p50 / p95 | v11 p50 / p95 |
|---|---|---|
| first check (PF at 388% CPU) | 204 / 232 ms | 430 / 446 ms |
| 512 / 256 samples, or a 10-step horizon | 181-186 / 208-214 ms | 425-428 / 446-459 ms |
| full stack up, PF fixed | 155 / 188 ms | - |
| + XLA command buffers | 160 / 195 ms | - |
| + 4 value samples instead of 20 | 168 / 201 ms | - |
| XLA autotuning off | GPU matmul 7x slower (0.98 to 6.9 ms): do not use | - |

Fewer samples and the shorter horizon barely help, so the GPU math is not what limits the Jetson.
`scripts/profile_step.py` on the Jetson (bicycle, 160 ms per step) shows where the time goes: **each small JAX
operation run from Python costs about 1.3 ms on the Jetson** (about 0.1 ms on the laptop). About 49 of them per step
cost 63 ms. The biggest blocks: track-limit check with the car's corners 25 ms, candidate safety totals 18 ms,
reduced-speed and previous-plan candidates 20 ms, validity counts 7 ms, reading results back from the GPU 17 ms,
Python in the step itself 13 ms.

**Fix in progress ("folding"):** those operations are put into a few compiled functions, so each step makes a handful
of GPU calls instead of about 50, with the same math. Branch `planner-jit` (sim) and the car test branch
`irl-planner-jit`. Bicycle: the chosen plans are exactly identical to the original code over 40 steps. v11 cannot
be checked bit for bit: the original code run twice already differs (0.0002 at step 0, growing to 1.1 through the
planner's choices; GPU sums are not bit-exact run to run), and folded vs original differs by the same amount.
Jetson, bicycle, stack on (`profile_step.py`): **160 ms -> 112 ms median, p95 192 -> 127 ms** (max 135 ms). Loose
operations 63 -> 16 ms. Still over the budget; what is left: waiting for GPU results 50 ms, Python in the step 22 ms,
remaining loose operations and slices about 28 ms, hull build 6.5 ms, random-key splits 5.3 ms.

With 512 samples (instead of 1024) the GPU wait dropped from 50 to 35 ms: now that the per-operation overhead is
gone, the sample count matters. Second folding round (both random keys in one compiled split, the state read to
the CPU once per step; bicycle plans again exactly identical): loose operations 16 -> 7.8 ms. **Bicycle, 512
samples: 160 -> 80 ms median, p95 192 -> 96 ms** on the Jetson with the full stack. Left per step: GPU results
30 ms, Python in the step 21 ms, hull build 5.6 ms, hull check 4.4 ms, 4 loose operations 7.8 ms.

**v11** (same folds, plus the filter step folded to one upload and one read: 30 -> 24 ms, outputs identical): 430 ->
172 ms median at 512 samples, 165 ms at 256. Its LiDAR clearance check (`latent_costs`, 2 calls) costs 42 ms on the
Jetson whatever the number of plans (clearance off: 117 ms), and v11's other compiled calls also cost a fixed amount
each: the time is per call, not per plan. On the laptop the same call returns in about 1 ms and finishes in 2.3 ms
even for 8 plans. XLA command buffers and full loop unrolling did not help. Next: compute the clearance once per step
on the 256 final candidates only (needs a sim check), then reduce v11's per-call cost.

Also learned: compiling the planner takes several minutes on the Jetson at every start; with
`JAX_COMPILATION_CACHE_DIR` set, only the first start compiles.

### 7. Sim checks of the speed-ups (2026-11-28)

**Clearance on the final candidates only** (`planner.dynamics.rssm_sample_clearance: false`, branch `planner-jit`
`16ef7b41`): v11's LiDAR clearance cost is no longer computed for the sampled plans, only for the final candidates
(one model call less per step). The final pick still checks clearance on every candidate. Jetson, v11, 256 samples:
165 -> 143 ms median, p95 197 -> 165 ms (clearance 42 -> 21 ms).

| Levine sim map, retry from standstill, 7 m/s cap, 20 attempts | Plain v11 | Clearance on candidates only |
|---|---|---|
| crashes | 12 (9 walls, 3 off track) | 13 |
| crash spots | s = 39.5 (x6), 59-60 (x4), 8 (x2) | s = 39.5 (x8), 57-60 (x3), 8, 36 |
| clearance changed the final pick | 0.3% of steps | 0.2% |
| laps with a wall closer than 0.2 m | 10 of 15 | 6 of 15 |
| planning p50 / p95 (laptop) | 42.3 / 51.1 ms | 38.5 / 48.3 ms |

No measurable change in behaviour; the crash spots are entry 45's. v11 crashes in about 60% of learned attempts on
this map in both variants: that, not only the planning time, is open before v11 drives the car.

**Spielberg, unexplained:** plain v11 in retry mode (config identical to entry 45's capped run, which crashed 2 of
25 times) crashed on every learned attempt today (3 of 3 on the original code, 3 of 3 and 20 of 20 on the folded
branch), always after about 4 s at s = 31-35 m with a spin (heading error -1.0 rad, slip 0.49). Entry 45's lap 6
passed the same corner with heading error -0.75 rad and slip 0.25 and recovered. Code, model files, packages and the
sim bridge are unchanged since that run, and the laptop was plugged in. The corner is at the edge of what v11
handles; why today's runs fall on the other side is not known.

**Integrator substep** 0.02 -> 0.05 s (Jetson): no gain (v11 168 ms vs 172 ms; bicycle update 73 vs 77 ms); reverted.

### 8. v11 speed-ups, round 3 (2026-11-28)

On `planner-jit` `cf25d105`, merged into `irl-planner-jit`; the clearance safety stays on for every final candidate.

| Change | Check | Jetson, v11, 256 samples, p50 / p95 |
|---|---|---|
| (before) clearance on final candidates only | Levine sim: same crashes as plain v11 | 143 / 165 ms |
| candidate prediction and clearance from one rollout call (they ran the same rollout twice) | trajectories and clearance identical (max difference 0.0) | 126 / 167 ms |
| safety-cost terms summed inside compiled code | v11 on the CPU, 40 planner steps: plans, choices and summaries identical to the original code | (in the line above) |
| filter step on the CPU (`planner.dynamics.rssm_filter_device: cpu`), its state handed to the GPU once per step | outputs within 3.3e-6 of the GPU version | **114 / 145 ms** |

v11 on the Jetson: 430 -> 114 ms median, 446 -> 145 ms p95; still over the budget. Left per step: v11's two compiled
calls about 54 ms (a fixed cost per call on the Orin), GPU results 20 ms, filter 18 ms, 8 loose operations 14.5 ms,
hull 5 ms. Note: on the CPU backend the bicycle's plans now differ from the original by one rounding step at step 28
(then the planner amplifies it); on the GPU, which the car uses, the bicycle was bit-identical after both fold rounds.

### Open

- Bicycle learned laps on the car with the folded planner (`irl-planner-jit`, 512 samples, 80 / 96 ms).
- v11 is still about 165 ms per step: compute the clearance once per step on the final candidates only (sim check
  first), then cut v11's fixed per-call cost.
- Then v11 and v11 + learned change handling on the car, and the change test (weight or slippery patch, runbook step 10).

## Appendix: Jetson measurements (2026-11-27/28, as measured)

### Sensors (`scripts/car_check.py`, 40 s, car pushed by hand)

| Topic | Rate, first check (VESC IMU) | Rate, second check (RealSense IMU) |
|---|---|---|
| `/pf/pose/odom` | 44.0 Hz | 39.6 Hz |
| `/odom` (VESC) | 49.0 Hz | 49.6 Hz |
| IMU | `/sensors/imu/raw` 49.0 Hz, all zeros | `/camera/camera/imu` 191.3 Hz |
| `/scan` (SICK TiM5xx) | 14.6 Hz | 14.0 Hz |
| `/joy` | 71.2 Hz | 65.2 Hz |

- LiDAR: 811 beams over 270 deg (-135 to +135 deg), angle step 0.333 deg, scan time 0.0667 s, range max 100 m.
- VESC speed / PF speed median ratio: 1.001.
- RealSense IMU at rest (camera frame): acceleration (-0.108, -9.728, 1.030) m/s^2, gyro (0.001, 0.006, 0.007) rad/s; `ros2 topic hz`: 199.99 Hz.
- After remap and levelling (car frame, at rest): acceleration (0.000, 0.000, 9.779) m/s^2; gyro z bias 0.011 rad/s. Mount tilt measured: 6.9 deg on the stand, 1.4 deg on the floor (the camera itself is nearly level).
- IMU vs PF: x slope +1.02 (correlation 0.53), y slope +0.59 (correlation 0.40), suggested yaw scale 1.19 with 0.94 rad/s scatter; kept at 1.0 (references from the PF are noisy at walking speed).
- v11 inputs while pushing (mean / std; training in brackets): ax 0.02 / 1.03 (0.63 / 2.64), ay 0.30 / 1.17 (0.28 / 5.00), gyro z 0.14 / 0.75 (0.06 / 1.08), wheel speed 1.48 / 0.90 (5.39 / 2.87), PF dx/dt 1.38 / 0.93 (5.34 / 2.92), PF dy/dt -0.11 / 0.37 (-0.05 / 0.86), PF dyaw/dt 0.01 / 0.08 (0.06 / 1.04).

### Particle filter and system load

| | Before (`OPENBLAS_NUM_THREADS` unset) | After (`OPENBLAS_NUM_THREADS=1`) |
|---|---|---|
| PF CPU | 388% (23 threads; 5 OpenBLAS workers with ~17 min CPU each in 25 min) | 58% (18 threads) |
| PF rate | ~47 Hz | 46-48 Hz |
| largest PF gap | 0.367 s (at subscription start; 0.079 s steady) | 0.033 s |
| all 6 CPU cores, stack only | 84-100% | (headroom freed) |
| GPU load (GR3D), stack only | 0-14% | - |
| RAM | 3.2 GB of 7.5 GB with the stack; 6.3 GB during a planner compile | - |

PF settings: 10000 particles, `rmgpu` (GPU ray marching), every 18th beam, max range 10 m.

### Planning time (`scripts/jetson_check.py`, synthetic centreline laps; p50 / p95 / max)

| Run | Stack | XLA flags | GPU matmul 1024x1024 | Bicycle | v11 | v11-learned |
|---|---|---|---|---|---|---|
| 1 | off | none | 0.97 ms | 203.6 / 232.4 / 403.7 ms | 429.7 / 446.0 / 466.3 ms | 431.1 / 446.3 / 479.4 ms |
| 1, 512 samples | off | none | | 183.4 / 214.3 / 327.5 ms | 425.0 / 446.4 / 471.1 ms | 425.3 / 455.4 / 466.1 ms |
| 1, 256 samples | off | none | | 181.2 / 211.9 / 227.4 ms | 425.9 / 454.9 / 468.4 ms | 425.9 / 455.5 / 475.9 ms |
| 1, 10-step horizon | off | none | | 186.3 / 207.6 / 217.5 ms | 427.9 / 459.3 / 474.4 ms | 426.2 / 454.5 / 478.4 ms |
| 2 | on, PF 388% | autotune off + command buffers | 8.59 ms | 269.5 / 322.4 / 476.5 ms | | |
| 3 | on, PF fixed | autotune off + command buffers | 6.92 ms | (not finished) | | |
| 4 | on, PF fixed | none | 0.98 ms | 155.1 / 187.6 / 380.7 ms | | |
| 5 | on, PF fixed | command buffers only | 0.81 ms | 160.0 / 195.2 / 325.6 ms | | |
| 6 | on, PF fixed | none; 4 value samples | 0.96 ms | 167.7 / 200.9 / 326.6 ms | | |
| 7 (folded planner, `irl-planner-jit`) | on, PF fixed | none | - | 114.9 / 129.9 / 135.1 ms | | |
| 8 (folded, 512 samples) | on, PF fixed | none | - | 97.1 / 118.1 / 139.3 ms | | |
| 9 (folded round 2, 512 samples) | on, PF fixed | none | - | **79.6 / 96.3 / 101.4 ms** | | |
| 10 (folded round 2, 512 samples) | on, PF fixed | none | - | | 178.8 / 225.3 / 256.8 ms (filter 29.6 ms) | |
| 11 (+ folded filter step, 512 samples) | on, PF fixed | none | - | | 171.8 / 203.7 / 215.6 ms (filter 23.9 ms) | |
| 12 (same, + command buffers) | on, PF fixed | command buffers | - | | 167.7 / 196.9 / 239.3 ms | |
| 13 (same, 256 samples) | on, PF fixed | none | - | | 164.6 / 196.7 / 215.7 ms | |
| 14 (same, 256 samples, clearance off; timing only) | on, PF fixed | none | - | | 116.6 / 155.6 / 162.9 ms | |
| 15 (same, 256 samples, clearance on final candidates only) | on, PF fixed | none | - | | 143.4 / 165.4 / 172.5 ms (filter 23.4 ms, clearance 20.8 ms) | |
| 16 (same + prediction and clearance in one rollout call) | on, PF fixed | none | - | | 126.2 / 166.9 / 179.4 ms (filter 23.8 ms) | |
| 17 (same + filter step on the CPU) | on, PF fixed | none | - | | **114.2 / 145.2 / 148.7 ms** (filter 17.7 ms, p95 23.6) | |

Laptop for comparison (same check, on battery): bicycle 21 / 25 ms, v11 36 / 43 ms (filter step 3.9 ms of it).

### Where the time goes on the Jetson (`scripts/profile_step.py`, bicycle, stack on, 60 steps)

Planner step median 159.7 ms, p95 192.0 ms.

| Function | ms per step | calls per step |
|---|---|---|
| `update` (whole step) | 161.9 | 1 |
| loose JAX ops (`apply_primitive`) | 63.5 | 49 |
| array indexing (`__getitem__`) | 51.3 | 18 |
| `boundary_view` / `with_footprint` | 24.8 | 1 |
| `total_safety` | 17.9 | 1 |
| reading GPU results (`_value`) | 17.4 | 18.6 |
| `.at[].set` scatters | 16.4 | 3 |
| `with_reduced_speed` | 14.0 | 1 |
| Python in `update` itself | 13.2 | 1 |
| `safety_step_costs` | 12.1 | 1 |
| `sample_validity` | 7.2 | 1 |
| `with_previous_plan` | 5.8 | 1 |

(Rows overlap: the indexing and loose-op rows are part of the named functions.)

### Car run (`levine-bicycle-slow`, 2026-11-28 01:15)

- Startup: value precompile 17.0 s, planner precompile 151.9 s, then the hull precompile (238 variants); before the compile cache, starts took several minutes; with `JAX_COMPILATION_CACHE_DIR` the next start was quick.
- Warm-up laps: 32.3, 32.4, 32.4, 32.2, 32.5 s, all at 2.0 m/s; pure-pursuit step 0.7 ms (p50), up to 9.8 ms.
- Value training while standing: losses -1.19, -2.03, -2.26, -3.10 (generations 1-4).
- Control loop over the run: 1701 steps, 21 overruns, 0 watchdog stops; control step p50 98.7 ms, p95 128.9 ms.
- Learned steps (last window): planner p95 243.0 ms, max 327.4 ms; car step p50 94.7 ms, p95 140.1 ms; commands reached the car every 0.6-0.75 s; overruns went from 10 to 20 within 13 learned steps.
- Starts: rejected at `behind` 54.14 m and 62.61 m (car past the line), accepted at 0.80-1.29 m; on the stand the launch timed out after 10 s (expected: it cannot reach the line).

### Track (`scripts/make_track.py` on the PF map)

| Smoothing | Length | Points | Wall distance min / 5th pct / median | Narrow spots | Gaps (no wall within 1 m) |
|---|---|---|---|---|---|
| default (1 m) | 62.64 m | 627 | 0.00 / 0.62 / 0.76 m | 13 | 36 |
| 2.5 m | 61.56 m | 616 | 0.00 / 0.48 / 0.76 m | 36 | 27 |

Start line: (-6.48, 3.42) in the map frame, heading -61 deg (bottom-left corner).

### Jetson profiles after folding (`scripts/profile_step.py`, stack on)

| | Bicycle, fold 1, 1024 | Bicycle, fold 1, 512 | Bicycle, fold 1+2, 512 | v11, fold 1+2, 512 | v11, + filter fold, 512 | v11, 256, clearance off |
|---|---|---|---|---|---|---|
| step p50 / p95 | 114.9 / 129.9 ms | 97.1 / 118.1 ms | 79.6 / 96.3 ms | 178.8 / 225.3 ms | 171.8 / 203.7 ms | 116.6 / 155.6 ms |
| `update` median | 112.1 ms | 94.5 ms | 77.3 ms | 145.3 ms | 143.7 ms | 90.4 ms |
| v11 filter step median | - | - | - | 29.6 ms | 23.9 ms | 23.4 ms |
| Python in `update` itself | 22.0 ms | 21.9 ms | 21.3 ms | 54.2 ms | 52.6 ms | 50.1 ms |
| waiting for GPU results | 49.8 ms | 35.3 ms | 29.7 ms | 27.5 ms | 25.3 ms | 23.8 ms |
| loose JAX ops | 16.0 ms (10) | 15.8 ms (10) | 7.8 ms (4) | 21.1 ms (11) | 12.9 ms (8) | 13.4 ms (8) |
| v11 clearance check | - | - | - | 43.2 ms (2 calls) | 42.1 ms (2 calls) | off |
| hull build | 6.6 ms | 5.9 ms | 5.6 ms | 5.6 ms | 5.1 ms | 4.9 ms |
| random-key splits | 5.3 ms | 4.9 ms | (folded) | - | - | - |

## Appendix: code changes (2026-11-27/28)

| Change | Branch, commit | Effect |
|---|---|---|
| retry restart speed 1.5 m/s; checkpoint waits moved off the control loop; obstacle flag cleared at start; planner and value training precompiled at startup; stop after every lap | `irl-world-model` `9d9f5ad1` | dry-run bugs fixed (section 1) |
| RealSense IMU: axis remap, automatic levelling, centre-of-gravity correction, 0.1 s averaging; Ctrl-C raised after the ROS spin; `scripts/car_dryrun.sh` | `bb3d490d` | IMU usable for v11 |
| IMU and LiDAR subscriptions with best-effort QoS | `21cdfc6f` | RealSense IMU messages arrive |
| `jetson_check.py`: timed steps kept away from the finish line | `e279adf5` | the check crashed on the short real track |
| `scripts/profile_step.py` | `c1e0fdcc` | per-function time on the Jetson |
| fold 1: track-limit check, step summary, validity count, candidate safety totals, reduced-speed and previous-plan candidates, plan shift and terminal-state slices as compiled calls | `planner-jit` `cdea813f` | bicycle 160 -> 112 ms on the Jetson; plans identical |
| fold 2: both random keys in one compiled split; the state read to the CPU once per step | `planner-jit` `501a9611` | loose ops 16 -> 7.8 ms; plans identical |
| v11 filter step: one upload, one read; LiDAR log features computed on the CPU | `planner-jit` `e4fce382` | filter 30 -> 24 ms; outputs identical |
| all of the above for the car | `irl-planner-jit` (merges `1a8a79b6`, `2ce7014c`) | car test branch |
| runbook: SICK bringup, RealSense IMU launch, `OPENBLAS_NUM_THREADS=1`, `ros2 bag record`, stop after every lap | `irl-world-model` | `docs/CAR_RUNBOOK.md` |

Config settings found or changed on the car (not committed; local on the Jetson): `planner.half_width: 0.62`
(from the track), `planner.samples: 512`, `planner.safe_set_size` (warm-up laps = this + 1), `simulation.car.command_timeout: 0.5`
for the slow test; environment: `OPENBLAS_NUM_THREADS=1`, `JAX_COMPILATION_CACHE_DIR`, no `XLA_FLAGS`.

## Appendix: diagnostic tests on the laptop

| Test | Result | Conclusion |
|---|---|---|
| first profile of the planner check | counted the check's own setup (fake laps, value training) as planner time: 62 + 23 loose ops "per step" | wrong; a profile limited to the planner step showed the real picture |
| eager-op trace inside the step (bicycle) | 12 loose ops and 15 GPU reads per step (indexing ops not caught by that trace; about 50 in total) | the step was already mostly compiled |
| value samples 20 / 4 / 1 (laptop) | 20.7 / 13.8 / 15.3 ms | a third of the step on the laptop; no change on the Jetson (GPU not the bottleneck there) |
| fold 1 on the laptop | 24 ms before and after | the laptop is GPU-limited, so the saved CPU work was hidden |
| equality: original vs folded, 40 steps | bicycle identical (max difference 0.0); v11 differs, but the original run twice differs just as much (0.0002 at step 0, up to 1.1) | v11 on the GPU is not bit-exact run to run |
| speed passed to the candidate assembly as a CPU number (float64, then float32) | plans no longer identical (1e-5 at step 3, growing) | kept the GPU value `state[3]` |
| v11 filter, 30 steps, original vs folded, on the CPU | identical (max difference 0.0) | filter fold exact |
| v11 clearance call vs plan count | 512 / 64 / 8 plans: returns in 1.04 / 1.46 / 1.05 ms, done in 3.02 / 3.30 / 2.33 ms | fixed cost per call, not per plan |
| passing v11's model to a compiled call | 0.02 ms (same or a new object each step); flatten 66 arrays 0.009 ms | not the cause |
| XLA full loop unrolling | clearance 3.01 -> 3.84 ms, bicycle rollout 1.23 -> 1.68 ms | slower; not used |
| XLA autotuning off (Jetson) | GPU matmul 0.98 -> 6.9 ms | never use for runs |

