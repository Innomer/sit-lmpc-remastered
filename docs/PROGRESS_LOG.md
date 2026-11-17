# Progress log: SIT-LMPC at F1TENTH scale, real car, world model

ESE 6150 Team 8. A running history of what was built, tried and found, one line per item.
Details are in `NOTES.md`, `docs/CAR_DEPLOYMENT.md` and `results/baselines/README.md`.
Last updated 2026-11-16.

## 1. Methods in play

| Method | Dynamics model | What adapts online | Status |
|---|---|---|---|
| SIT-LMPC, known model | F1TENTH single-track (bicycle) with the sim car's exact parameters | safe set, value function (per lap) | sim baselines done; car runs pending |
| TAs' BNN (remastered SIT-LMPC) | learned network, mean + 96 basis functions, Gaussian over z (96) | + z belief, closed form, between laps | sim baselines done; retrained on the world dataset |
| TAs' BNN + chance constraints | same | same, plus P(off track) ≤ 0.05, P(end outside safe set) ≤ 0.9 | sim baselines done |
| Ours: physics-grounded world model | bicycle + learned residual b + B·z (z 16) | + z belief every 0.1 s from a sensor encoder; condition-aware safe set | data and transition trained; encoder not yet built |

## 2. Sim setup (ROS 2 F1TENTH sim, Spielberg)

| Date | Item | Outcome |
|---|---|---|
| 11-13 | `simulation.backend: ros` (lockstep bridge on domain 77) | runner unchanged; same interface as the standalone gym |
| 11-13 | GPU JAX on the laptop (RTX 3070 Laptop) | LMPPI ~20 ms per step vs ~500 ms on CPU |
| 11-13 | Track bug fixed (search range wrapped by resampled count) | car no longer brakes at every finish line |
| 11-13 | Steering limit override, 0.2 s delay compensation | car-ready options; off in sim baselines |
| 11-13 | BNN retrained at F1TENTH scale (dyna-dist, narrow and wide 1–20 m/s models) | `models/vehicle-bnn-f1tenth-l96`, `-wide-l96` |
| 11-14 | Pure-pursuit warm-up (lab 5 submission, 100 Hz inside each 0.1 s step) | fast start: 38.0 s warm-up laps |

## 3. Frozen sim baselines (TAs' code, seed 1, 25 attempts)

| Run | Warm-up | Best LMPPI lap | LMPPI laps | Planning p50 | One-line result |
|---|---|---|---|---|---|
| known, slow start | MPPI 135.2 s | 70.9 s | 20/20 | 19.5 ms | improves strongly from a slow warm-up |
| BNN + chance, slow | MPPI 135.2 s | 110.1 s | 20/20 | 98.5 ms | conservative; chance constraints slow it |
| BNN no chance, slow | MPPI 135.2 s | 75.6 s | 20/20 | 64 ms | close to the known model |
| known, fast start | PP 38.0 s | 38.9 s | 17, then spin | 19.7 ms | never beats the warm-up, then spins |
| BNN + chance, fast | PP 38.0 s | 39.8 s | 5, then stop | 99 ms | slows lap by lap, then stops |
| BNN no chance, fast | PP 38.0 s | 38.3 s | 7, then stop | 63 ms | same |

Frozen copies: `results/baselines/` (configs, seeds, commit, summaries, figures).

## 4. Fast-start diagnosis (known model, 2026-11-16)

| Run | Change from the fast baseline | Result | Finding |
|---|---|---|---|
| 1 | warm-up at speed_scale 0.5 (45.5 s laps) | LMPPI 47.6, 47.2 s, then stopped in the hairpin | stops are not BNN-specific |
| (analysis) | — | hairpin min speed 3.2 → 1.2 → 1.9 → 0 m/s over laps | newest-4-laps safe set ratchets down |
| 2 | keep all laps in the safe set | 39.1 → 37.4 s, beats the warm-up from lap 18; spin at attempt 24 | ratchet fixed |
| (analysis) | replay crash lap through planner model and gym | identical predictions; yaw rate oscillates with growing amplitude | spin is planner-induced fishtail, not model mismatch; safe set ignores yaw rate and slip |
| 3 | + 1.5 s horizon | 37.8 → 35.2 s; spin at attempt 16 | faster laps, spin remains |
| 4 | + 10× action-rate penalty | **25/25, best 35.0 s, no failures** | working fast known-model baseline |
| (timing) | — | p50 19.5 → 94 ms (all laps) → ~132 ms (run 4) | all-laps safe set grows planning cost; not usable on the car as is |
| 5 | BNN no chance (old wide model), run-4 fixes | stopped after 7 attempts (38.0, 38.3 s LMPPI) | too slow (~600 ms/step); replaced by the retrained model |
| 6 | BNN no chance, **retrained world-data model**, run-4 fixes, 15 attempts | 14/15, 37.2 → best 35.7 s; crash on attempt 15 (start bend) | close to the known model (35.0 s); fishtail not fully gone; planning p50 574 ms |
| 7 | BNN + chance, retrained model, run-4 fixes, 15 attempts | stopped on the first LMPPI lap | P(end outside safe set) = 1 for every candidate: prior-width belief (no updates from PP warm-up laps) × 1.5 s horizon |
| 8 | BNN + chance, retrained model, 1 s horizon | **15/15, no failures**, 39.4–41.1 s | works but conservative (151 fallbacks); slower than the warm-up, as expected for chance constraints with a wide belief |

## 5. Real car (branch `irl`)

| Date | Item | Outcome |
|---|---|---|
| 11-14 | `simulation.backend: car` (`adapters/ros_car.py`) | PF pose, VESC speed, IMU yaw rate in; speed + steering angle out at 10 Hz wall clock |
| 11-14 | Safety | 0.3 s command watchdog; stops on out-of-bounds, LiDAR < 0.25 m, stale odometry, Ctrl-C/SIGTERM; operator go to restart |
| 11-14 | Sim dry run of the car path (real-time bridge) | launch, warm-up lap, 99 ms cadence, Ctrl-C and out-of-bounds stops verified; 4 bugs fixed |
| 11-14 | Track `levine_self_again` from the raceline (±0.4 m stand-in) | rebuild with `scripts/make_track.py --map <PF map>` on the car |
| 11-14 | `scripts/car_check.py` | checks topics, IMU scale/sign, VESC vs PF speed, joystick buttons; never drives |
| 11-16 | f1tenth-gym no longer needed on the car | vehicle parameters inlined (identical) |
| 11-16 | Jetson: Orin Nano Super 8 GB, JetPack 7, ROS Jazzy, Python 3.12, CUDA 13.2 | setup OK; GPU JAX works (2048² matmul 3.5 ms vs 165 ms CPU) |
| 11-16 | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35` | leaves memory for the ROS stack on the shared 8 GB |
| 11-16 | `go_button` may not be a deadman (6 teleop, 8 autonomous) | config check added |
| pending | car_check, track rebuild, bench test, known-model and BNN car runs | waiting for car time |

## 6. World model (ours)

| Date | Item | Outcome |
|---|---|---|
| 11-16 | Plan approved | z 16; PF pose passed through; sim IMU randomized per episode; KL to random-walk z transition; joint stage low LR + z-separation check; car-bag fine-tune with transition frozen |
| 11-16 | Residual mode check | dyna-dist already supports bicycle + b + B·z end to end, including the planner |
| 11-16 | Collection extensions | high-slip tapes (≥ 6 m/s), mid-run change episodes, simulated IMU/wheel speed/PF pose; defaults off, old data reproduces |
| 11-16 | Dataset `data/f1tenth-world` | 236k transitions, 2,310 trajectories, 300 change episodes (158 steps, 142 patches); \|slip\| > 0.3 in 6.3% (was 2.6%) |
| 11-16 | Residual guard | one of 2,182 transitions blocked training; now excluded from normalization if under 1% |
| 11-16 | Transition trained (`runs/residual-world`) | 20k steps, held-out NLL −3.20/component, no overfitting |
| 11-16 | TAs' BNN retrained on the same data (`runs/dynamics-world`) | 20k steps, held-out NLL −3.61/component (different normalization; not comparable) |
| next | Encoder + observation decoder, EKF, offline evaluation | not started |

## 7. Tools

| Tool | Purpose |
|---|---|
| `scripts/presentation_figures.py` | lap-time, planning-time and comparison figures |
| `scripts/make_track.py` | track files from a raceline, with or without the PF map |
| `scripts/car_check.py` | pre-drive input check on the car |
| `scripts/jetson_setup.sh` | Jetson venv, package and GPU JAX |
| `scripts/benchmark_planning.py` + replay backend | planning time per step without sim or ROS (laptop and Jetson); not yet run |

## 8. Decisions

| Date | Decision |
|---|---|
| 11-13 | Sim baselines first (known, then BNN), then the car on a separate branch |
| 11-14 | Fast-start BNN reported as is; no extra high-slip data for the TAs' model at that point |
| 11-14 | Car: kinematic slip, IMU gyro yaw rate, full IMU logged for an offline slip estimator |
| 11-16 | Car track: Levine with `levine_self_again_raceline`, not the lobby |
| 11-16 | Our method is the physics-grounded world model, not the TAs' BNN |
| 11-16 | B and b stay fixed online; fine-tuning them on car data is a later extra |
| 11-16 | Sim/research work on `ros-sim`, car work on `irl` (= `ros-sim` + car) |

## 9. Commits

| Branch | Commit | Content | Pushed |
|---|---|---|---|
| ros-sim | `b4be7fd` | ROS sim backend, known and BNN baselines | yes |
| ros-sim | `9ca8261` | results for known and BNN + chance | yes |
| ros-sim | `cd702d3` | pure-pursuit warm start, ablations, figures | yes |
| ros-sim | `164a7da` | NOTES status | yes |
| ros-sim | `45d9e9e` | world dataset, training configs, diagnosis configs | no |
| ros-sim | `5c6b48c` | dataset fixes, fast-baseline configs | no |
| irl | `9fcd351` | real-car backend | yes |
| irl | `dcc01e2` | car path without f1tenth-gym | yes |
| irl | `910a837` | Jetson CUDA detection, memory note | yes |
| irl | `ea7e5fb` | frozen sim baselines | yes |
| — | uncommitted | replay benchmark, go-button check, this log | — |
