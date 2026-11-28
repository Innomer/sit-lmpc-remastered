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
| learned laps | both failed within 0.7 s: planner steps took **0.6-0.75 s** with the full stack, so the car drove 0.4-0.5 m per decision on old steering (heading error grew to 0.45-0.55 rad, then a "stopped" failure and a LiDAR obstacle stop) |

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
*Pending: Jetson timing of `irl-planner-jit` (expected: about 70-90 ms less per step).*

Also learned: compiling the planner takes several minutes on the Jetson at every start; with
`JAX_COMPILATION_CACHE_DIR` set, only the first start compiles.

### Open

- Jetson timing of the folded planner; then learned laps on the car (bicycle, then v11, then v11 + learned change handling).
- v11's extra time on the Jetson (430 vs 204 ms before the PF fix) still needs its own profile.
- The change test (weight or slippery patch mid-session, runbook step 10).
