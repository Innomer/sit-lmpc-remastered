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

### Open

- Jetson timing of the folded planner; then learned laps on the car (bicycle, then v11, then v11 + learned change handling).
- v11's extra time on the Jetson (430 vs 204 ms before the PF fix) still needs its own profile.
- The change test (weight or slippery patch mid-session, runbook step 10).

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
