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
| 11-16 | Encoder v1 (`runs/world-encoder`): GRU over IMU, wheel speed, PF pose change, last control → steering, speed, yaw rate, slip + z (16, diagonal) + sensor calibration; decoder; transition frozen; KL warm-up 3k steps | 20k steps in 25 min; validation loss −75 |
| 11-16 | v1 offline eval (test split) | 1 s position error: ours 0.38 m (constant) / 0.29 m (changes) vs bicycle 0.65 / 0.56, TAs' BNN prior 0.50 / 0.51; slip RMSE 0.020 rad vs kinematic 0.18; z → friction R² 0.89, mass R² 0.29 |
| 11-16 | **Flaw found in v1** | ours from the TRUE state with the encoder's z predicts worse (0.58–0.89 m) than the prior z (0.50 m). The encoder was graded only on (its state estimate + its z) together, so z learned to cancel the encoder's own state-estimate errors instead of only describing the car. Breaks "z = conditions" (condition-aware safe set, EKF fallback, the scientific claim). |
| 11-16 | Fix → encoder v2 | extra loss: rollouts from the true state with the encoder's z (sim only), so z only helps by describing the car; check: v2 true-state error must not exceed the prior's |
| 11-16 | EKF baseline | NaN at low speed (jacfwd through the ST low-speed branch) → numerical Jacobian + non-finite guard; still **diverges** on some trajectories (yaw-rate error ~1e10), slip RMSE 0.26 rad; needs a divergence guard + accelerometer before it is a usable baseline |
| 11-16 | v1 re-eval with medians | 1 s position median: TAs' BNN prior 0.20 / 0.21 m vs ours 0.31 / 0.23 m. Ours wins the mean through fewer large misses, not the typical case. Baselines started from the TRUE state, ours from its estimate (unfair to ours); fair comparison from an estimated state still to add |
| 11-16 | Per-step closed-form z baseline (Kalman filter on z, random-walk forgetting = the TAs' Gaussian update every step) + estimated-state baselines; 40-trajectory test subset, v1 encoder | 1 s position mean/median, constant / change: KF fed TRUE states 0.007/0.004 / 0.055/0.015 m (transition is excellent; the bottleneck is state estimation); ours from sensors 0.22/0.19 / 0.34/0.26; BNN prior from the hand-built estimate 0.46/0.41 / 1.09/0.89; KF fed the hand-built estimate 3.2/2.2 / 4.6/3.0, and still worse than no adaptation even with noise ×1e5 (0.48 / 1.59). In the realistic setting the encoder is 2–3× better than all baselines. Caveat: KF on EKF-smoothed states untested (EKF still diverges) |
| 11-16 | EKF divergence guard (reset on Mahalanobis > 200, |r| > 10, v > 25 or covariance trace > 100; clamp slip, speed, steering) | stable; 40 test trajectories: slip 0.041 rad, yaw rate 0.045 rad/s, speed 0.45 m/s, steering 0.030 rad (encoder v1: 0.020 / 0.10 / 0.12 / 0.010) |
| 11-16 | EKF + accelerometer (predicted from bicycle-model rates), with and without accel-bias states, noise 1–4 m/s² | slip 0.062–0.065 rad with accel vs 0.041 without; accel only helps yaw rate slightly. Unmodelled IMU mounting offset dominates. Evaluation uses the EKF's best slip configuration (no accel) |
| 11-16 | robot_localization considered as baseline/fallback | rejected: omnidirectional motion model (no car kinematics), no IMU bias states, no steering estimate (the planner needs it), not installed on the laptop. Bicycle-model EKF stays as classical baseline and car fallback; validate on car bags before trusting it |
| 11-16 | **v2 failed; root cause: KL bug** | the start KL inverted a near-singular prior covariance (condition number ~1e7) in float32 and went negative (−7,978 v1, −27,307 v2); the optimizer exploited it and checkpoint selection (by total loss) picked the exploited final models. v2 encoder: speed RMSE 1.74 m/s, 1 s error 1.70 m. v1 also affected (less) |
| 11-16 | Fix | z whitened (encoder outputs w, z = m₀ + L w, prior N(0, I)); KL from TensorFlow Probability's JAX version (tested library); checkpoint selection by anchor + prediction + true-state prediction only |
| 11-16 | Reference checks (`tests/unit/test_world_model.py`, 6 pass) | KL vs float64; whitened z moments vs prior; rollout vs dyna-dist `_predict_prior`; per-step update with no drift vs dyna-dist `GaussianEvidence`; EKF on noise-free data; loss terms finite, KL ≥ 0 |
| 11-16 | Audit: simulated IMU | vs numerically differentiated body velocity at 1 ms (slip to 1.4 rad): error 0.001 m/s² on ~20 m/s² signals, incl. mounting offset |
| 11-16 | Audit: encoder inputs | noise-free: pose-change features, previous control alignment, gyro, wheel and PF→CoG all exact (errors ~1e-14) |
| 11-16 | Audit: change episodes | 38/39 diverge exactly one step after the recorded change; the 39th is a patch on a straight (grip acts only through cornering in the ST model). Limit: mass and friction only affect cornering, not acceleration or braking |
| 11-17 | v1-fixed / v2-fixed retrained and evaluated (full test split; `results/world-eval-{v1,v2}-fixed`, figures in each `figures/`) | **Flaw fixed:** 1 s median error from the true state with encoder z 0.08 (v1-fixed) / 0.06 m (v2-fixed) vs prior z 0.20 m; the v1 flaw was mostly the KL bug. **Realistic (sensors only):** ours 0.61 m (constant) / 0.45 m (changes) vs best realistic baseline 0.95 / 0.65 m (≈1.5×; the earlier 2–3× came from the buggy model); per-step closed-form on realistic inputs 2.9–6.1 m; oracle 0.01 m. **State RMSE** encoder vs EKF: slip 0.021 vs 0.024 rad, speed 0.36 vs 0.46 m/s, yaw rate 0.125 vs 0.051 rad/s, steering 0.009 vs 0.025 rad (kinematic slip 0.18). **z:** friction R² 0.92, mass 0.29. **Adaptation** in the example patch episode: z follows the drop and recovery but takes ~2–3 s and reads ~0.1 low |
| 11-17 | Findings | bottleneck = state estimate (mainly speed); adaptation slower than claimed (`walk_std` 0.05 likely too small); keep v2-fixed (better z quality, small realistic cost) |
| next | correct the explainer page; condition-aware safe set; planner integration; maybe tune walk_std | — |

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
