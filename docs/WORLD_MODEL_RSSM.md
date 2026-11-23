# Adaptive world-model SIT-LMPC (RSSM): design, changes and status

ESE 6150 Team 8, branch `world-model`. Started 2026-11-21 after the TA asked for a real world model rather than an
adaptive dynamics model. The earlier encoder model (v3, `docs/WORLD_MODEL.md`) stays as a baseline.
Last updated 2026-11-22.

## 1. Summary

| Item | State |
|---|---|
| What it is | A recurrent state-space world model: memory h (GRU) + stochastic latent s, filtered every 0.1 s from LiDAR, IMU, wheel speed and PF pose, with the bicycle model kept inside its prediction |
| Data | 789 on-track episodes, 293k transitions, 3 maps, 1080-beam LiDAR, grip/mass changes and low-grip patches |
| Training | `rssm-v1` unstable after step 9,800 (fixed); `rssm-v2` training, best so far: 1 s position error 0.66 m, slip error 0.019 rad, grip head 0.76 |
| Planner | variant A (full latent rollout, 1024 candidates, 15 × 0.1 s) is the default for sim work; B, C and 10 × 0.15 s will be options |
| Timing | laptop: every variant ~30 ms p95 per step; Jetson not measured yet (`bash scripts/jetson_timing.sh`) |
| Next | finish v2, offline evaluation (step E), planner integration (step F) |

## 2. How it differs from the v3 encoder model

| | v3 encoder model | RSSM world model | Why the change |
|---|---|---|---|
| Inputs | IMU, wheel speed, PF pose, last control | + downsampled LiDAR (64 beams) | a world model should see the world, not only the car |
| Latent | z (16) only describes the car | h (memory) + s = c (8, conditions) + d (24, fast) | separate "what the car is like" (slow) from "what is happening now" (fast) |
| What it predicts | next state | next state **and** what the sensors will read (LiDAR, IMU, wheel) | predicting observations forces the latent to hold real information |
| Prediction | bicycle + b + B·z, z fixed | bicycle + residual(x, u, h, s), with h and s rolled forward | the memory and fast latent can follow the car through a manoeuvre |
| Planning | z fixed over the plan | full latent rollout (A), or two-stage (B) | lets the plan "imagine" sensors, e.g. predicted LiDAR clearance |
| Safety | track limits + safe set | + predicted LiDAR clearance term | catches plans the model expects to bring the car close to a wall |
| Safe set tags | grip read-out per lap / per stretch | latent c per stored state, matched by distance | c is the model's own condition variable |

## 3. Architecture (`src/sit_lmpc/world/rssm.py`)

| Part | What it does | Why it is there |
|---|---|---|
| GRU memory h (128) | carries what the model has seen | slip, steering and conditions are not measured directly; they have to be inferred over time |
| Latent c (8), random-walk prior | slow condition variables | grip and load change rarely; a random walk lets them drift and jump without forgetting |
| Latent d (24), learned prior p(d \| h) | fast driving state | lets the model represent what happens in a manoeuvre without polluting c (the "corner dip" problem of the v3 read-out) |
| Posterior q(s \| h, observation) | updates s from the current sensors | in-lap adaptation every 0.1 s |
| LiDAR encoder (2 conv layers) + sensor MLP | turns 64 beams + 9 sensor features into one vector | small and fast enough for the car |
| Physical state x' = bicycle(x, u) + residual(x, u, h, s) | rolls the 7-dim state forward | keeps physics in the loop so 15-step predictions stay physical |
| h' = GRU(h, s, u, x') | memory update | the memory sees the predicted physical state |
| Anchor head A(h, s) | steering, speed, yaw rate, slip | ties the latent to real quantities; this is the state estimate the planner uses |
| Pose (x, y, heading) | taken from the PF, not learned | learning absolute map position from LiDAR would tie the model to particular maps |
| LiDAR decoder | 64 log-ranges with uncertainty | reconstruction loss, and the clearance safety term in planning |
| Sensor decoder | IMU x/y, gyro, wheel speed with uncertainty | reconstruction loss |
| Condition head c → (grip, mass ratio) | sim-only auxiliary target | forces c to hold the conditions, not driving state |
| Inputs to the networks | steering, speed, yaw rate, slip (normalised), controls | no map position or absolute heading, so the model does not depend on where the car is |
| Bounds | anchor through a sigmoid to physical limits; rollout states clamped | an untrained or confused model must not produce impossible states |

## 4. Training (`src/sit_lmpc/world/rssm_train.py`, `configs/train-rssm.yaml`)

| Loss | What it compares | Why it is there |
|---|---|---|
| Anchor | A(h, s) vs true steering, speed, yaw rate, slip | state estimation |
| Residual (teacher-forced) | bicycle + residual from the true previous state vs true next state | learns the correction to the physics |
| LiDAR NLL | decoded scan vs real scan (log range, Gaussian) | world-model reconstruction |
| Sensor NLL | decoded IMU/gyro/wheel vs real | reconstruction of the car's own sensors |
| KL on d | q(d) vs learned prior p(d \| h) | keeps the fast latent predictable for planning |
| KL on c | q(c_t) vs N(c_{t-1}, var + 0.05²); N(0, I) at the start | random-walk prior for condition changes |
| Free nats (1.0) and KL warm-up (3,000 steps) | — | stop the latent collapsing early |
| Condition head (weight 3) | c → true grip and mass | keeps c about conditions |
| 1 s prediction | open-loop rollout from the filtered state vs true states (10 steps) | the planner uses exactly this rollout |
| 1 s LiDAR prediction (0.5) | decoded scan along the rollout vs real | trains the clearance safety term |
| Consistency (0.5) | anchor of the rolled-forward latent vs rolled-forward physical state | latent and physics must agree over the plan |

| Setting | Value | Why |
|---|---|---|
| Window | 4 s (40 steps), batch 32 | long enough for adaptation and 1 s predictions |
| Optimiser | AdamW 3e-4, warm-up + cosine | standard |
| Gradient clip | 1.0 | see change log (v1 blow-up) |
| Bad-batch guard | skip a batch 20× above the running loss; drop non-finite updates | see change log |
| Bicycle gradients | stopped | its low-speed branch has NaN gradients |
| Checkpoint selection | anchor + residual + prediction + LiDAR on validation | same idea as the v3 selection fix (no KL in the selection) |
| Integrator | 0.02 s (planner's) | training and planning use identical physics |

TensorBoard (`tensorboard --logdir runs`, localhost:6006): loss curves (SCALARS), true-vs-model charts per checkpoint
(CUSTOM SCALARS, runs `diag-<step>`), and figures (IMAGES) with true vs reconstructed vs predicted LiDAR, a top-down
view of scan points and 1 s predicted paths, anchor vs truth and the grip read-out.

## 5. Data (`scripts/collect_track_data.py`, `data/f1tenth-track`, not in git)

| Map | Episodes (train / val / test) | Steps (train / val / test) | Speed caps |
|---|---|---|---|
| Spielberg | 283 / 36 / 35 | 100k / 15k / 15k | 4–10 m/s |
| custom1 | 95 / 12 / 12 | 46k / 6k / 6k | 5–14 m/s |
| Levine sim | 253 / 34 / 29 | 82k / 12k / 11k | 3–7 m/s (150 episodes) + 2–5 m/s (200) |

| Feature | Choice | Why |
|---|---|---|
| Vehicles | grip 0.4–1.2, cornering stiffness ±15%, mass 0.8–1.3 | same ranges as the earlier dataset |
| Changes | 30% none, 30% step change (grip and/or mass), 40% low-grip patches (1–3, 10–40 m, 35–70% of base grip) | in-lap changes are the point of the method |
| Driver | 10 Hz pure pursuit on the centre line, braking-aware speed profile, steering/accel noise | one control per 0.1 s, as the model assumes |
| Sensors | dyna-dist IMU / wheel / PF-pose model, per-step vehicle parameters | already validated; correct accelerations on patches |
| LiDAR | 1080 beams, 270°, bridge gym segment scanner, clean (noise added in training) | same LiDAR model as the closed-loop sim |
| Training LiDAR | 64 beams, closest return per group, log range, 0.05–10 m, noise + 1% dropouts | keeps the nearest wall, small enough for the car |
| Crash | LiDAR clearance < 0.18 m ends the episode | crash data up to the crash is still good driving data |
| Spin tails | cut before slip > 1.0 rad or yaw rate > 8 rad/s (at load time) | see change log (v1 blow-up) |

## 6. Planner variants and timing (laptop, RTX 3070, safe set capped at 4 laps)

| Variant | What | 15 × 0.1 s p50 / p95 | 10 × 0.15 s p50 / p95 |
|---|---|---|---|
| A (default) | full latent rollout of all 1024 candidates | 26 / 32 ms | 16 / 20 ms |
| A, 512 / 256 | fewer candidates | 14 / 22, 11 / 19 ms | 9 / 10, 7 / 10 ms |
| B | cheap screen of 1024, full rollout + LiDAR decode of top 32/64/128 | 27 / 33 ms | 17 / 21 ms |
| C | cheap only (bicycle + residual, h and s held) | 25 / 30 ms | 15 / 19 ms |

The world model adds 1–3 ms; the rest is the SIT-LMPC planner itself. Jetson numbers decide the car variant.

## 7. Change log: what was changed and why

| Date | Change | Because | Effect |
|---|---|---|---|
| 11-21 | Physical state rolled forward inside the rollout (x' = bicycle + residual) instead of only decoded | a decoder-only state has nothing keeping 15-step rollouts physical | physics stays in every prediction |
| 11-21 | Latent split into c (slow) and d (fast) | the v3 grip read-out dipped in every corner: driving and conditions were mixed in one z | c can hold conditions only |
| 11-21 | Condition head on c (user's addition) | stop c absorbing driving state | c has a direct grip/mass target in sim |
| 11-21 | New on-track dataset with LiDAR on 3 maps | the old dataset was driven in empty space | LiDAR sees real walls |
| 11-21 | Two-stage planner (B) designed, A chosen as default | measured cost made A affordable on the laptop | B and C kept as options |
| 11-22 | Hull copied and chunked on the host, not the GPU (planner, all branches) | the planner compiled GPU code while driving: p95 236 ms, 41% of steps over 100 ms | p95 30 ms, none over 100 ms |
| 11-22 | Timing measured instead of estimated | estimate (135 ms+) was far too pessimistic | world model costs 1–3 ms per step |
| 11-22 | Collector: centre line, low noise, braking-aware speed profile, per-map caps, s-hint for the track frame | first drives weaved, overspeeded into corners and crashed in seconds | most crashes are now real grip losses |
| 11-22 | LiDAR scanned in chunks of 16 on the GPU, padded to a fixed length | batching all poses needed 101 GB; each new length recompiled | 600 scans in 50 ms |
| 11-22 | 200 extra Levine episodes at 2–5 m/s | Levine (our real track) crashed 77% of the time and had 11% of the data | Levine 105k steps, crashes ~40% |
| 11-22 | Networks see only steering, speed, yaw rate, slip | first version fed the absolute heading | model independent of where the car points |
| 11-22 | Anchor bounded (sigmoid), rollout states clamped | untrained anchor gave impossible states; rollout exploded (54 m error) | finite losses |
| 11-22 | No gradients through the bicycle model | its low-speed branch gives NaN gradients | training runs |
| 11-22 | Bicycle casts inputs to the parameter dtype | tests (64-bit) failed with a dtype error | 5/5 tests pass |
| 11-22 | TensorBoard: curves, custom-scalar comparisons, figures, top-down LiDAR | user wanted to see training, reconstructions and outputs | localhost:6006 |
| 11-22 | Spin tails cut from the data; clip 10 → 1; non-finite updates dropped; bad-batch guard | `rssm-v1` blew up at step 9,800 (losses ×100) on batches with out-of-range spin states | `rssm-v2` past step 10,500 with no spike, no skipped batch |
| 11-22 | Condition-head weight 1 → 3 | grip head only reached 0.85 in v1 (1.0 = no skill) | v2: 0.76 at step 10,500 and falling |

## 8. Results so far (validation)

| Run | Step | Anchor error: steering / speed / yaw rate / slip | 1 s position | Grip head | Note |
|---|---|---|---|---|---|
| rssm-v1 | 9,000 (best) | 0.016 / 0.30 / 0.12 / 0.025 | 0.69–0.78 m | 0.85 | unstable afterwards |
| rssm-v2 | 10,500 (best so far) | 0.015 / 0.28 / 0.10 / 0.019 | 0.66 m | 0.76 | training |
| v3 encoder (offline eval, different data) | — | steering 0.010, speed 0.44, slip 0.020 | 0.53 m (median, 1 s) | grip R² 0.91 (lap average) | not directly comparable yet; step E compares on the same data |

## 9. Open items

| Item | Plan |
|---|---|
| Grip head still weak (0.76) | if it stalls: looser random walk on c or free nats on c |
| Jetson timing | `bash scripts/jetson_timing.sh` from a `world-model` clone; choose A/B/C |
| Offline evaluation (step E) | vs known model, v3 encoder, EKF; ablations without in-lap update, safe-set tags, LiDAR; two-stage vs full agreement after changes |
| Planner integration (step F) | variants A/B/C and 10 × 0.15 s as config options; LiDAR clearance term; latent-tagged safe set per lap per stretch |
| Map-free dataset | not used yet; could train the residual with LiDAR masked |
