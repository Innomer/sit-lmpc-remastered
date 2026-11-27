# Experiment log: world-model SIT-LMPC

Every model version, fix and test on the `world-model` branch since it was cut from `ros-sim` (2026-11-21), plus the
v3 encoder model from `ros-sim` for reference. Model details are in `docs/WORLD_MODEL_RSSM.md`.

Common setup unless stated otherwise:

- Sim: Spielberg, ROS lockstep bridge (domain 77), single-track gym model, nominal grip 1.049, no grip change.
- Runs: 5 pure-pursuit warm-up laps, then learned (LMPPI) laps. Lap times are from the sim clock.
- Replay check: at every step of a crashed lap, roll 1.5 s ahead from the true state with the controls that were
  actually applied, and compare with what happened (`scripts/replay_v3_on_crash.py`). Bicycle from the true state is
  exact in this sim, so it is the floor.
- Planning time: per control step on the laptop (RTX 3070), learned laps only.
- Verdicts: success = did what it was meant to do; partial = helped but did not solve the problem; failure = no help
  or made it worse.

## Summary

| # | Entry | Branch / commit | Verdict |
|---|---|---|---|
| 0 | v3 encoder model (reference) | `ros-sim` `402708e`, `60ca05b` | reference |
| 1 | Planner hull recompile fix | `world-model` `0a26c2b` (also on `ros-sim`, `irl`) | success |
| 2 | World-model timing tests | `0a26c2b`, `e179dec`, `0eb1261` | success |
| 3 | On-track LiDAR data, first collection | `b8a7eb3`, `e179dec` | success |
| 4 | RSSM v1–v4 (training stability) | `e179dec`, `3755c8b`, `53a9e23` | success (v3) |
| 5 | RSSM v5 + offline evaluation | `411cf6a` | partial |
| 6 | First sim tests (v5) and warm-up capped at 9 m/s | `fa13e62` | failure |
| 7 | Plan A data: fast, closed-loop and hard-manoeuvre episodes | `fa13e62` | partial |
| 8 | RSSM v6 | `fa13e62` | partial |
| 9 | RSSM v7 (spin data, high-slip oversampling) | `38a3f35` | failure |
| 10 | RSSM v8 (latent overshooting fine-tune) | `9e9707a` | failure |
| 11 | LiDAR clearance check | `9e9707a` | partial |
| 12 | RSSM v9 (v8 without batch skipping) | `0eb1261` | failure |
| 13 | Longer v8 runs; warm-up changes (slow, varied) | `0eb1261` | success for the planner, failure for v8 |
| 14 | Planner fix: safety costs outside the lambda grid | `898cd56` | success (code), not yet tested alone |
| 15 | Guardrails: speed cap and slip cost | `898cd56` | not tested alone |
| 16 | Slip-estimate check | `898cd56` | success (diagnosis) |
| 17 | Physics check | `world-model-guarded-attempt` `5f953fd` | failure, removed |
| 18 | Ensemble transition heads | `world-model-guarded-attempt` `5f953fd` | failure, removed |
| 19 | Guarded v8 sim run (14, 15, 17, 18 together) | `5f953fd` | failure |
| 20 | Post-contact frames removed from training and evaluation | `357ac06` | success |
| 21 | Spin-onset data collection | `357ac06` | success |
| 22 | Crash-risk head (soft planner cost) | `1b2224d` | failure in closed loop |
| 23 | Slip cost off by default; bicycle baseline rerun | `4402c6b` | success |
| 24 | v8 lateral error near the edges (diagnosis) | `4402c6b` | done: no consistent bias, large spread |
| 25 | World-model track-limit margin | `4402c6b` | failure |
| 26 | Rollout variants (diagnosis, no retraining) | `dc4ee44` | done: no cheaper fix; estimation error is about half |
| 27 | Starting-state error by component; v8 planning from the true state | `dae8971` | failure: still crashes |
| 28 | Open-space coverage data, rebalanced sampling, rssm-v10 fine-tune | `88f5fd2`, `86fb05c` | failure: training diverged |
| 29 | rssm-v11: from scratch on the full mix with a horizon curriculum and robust rollout losses | `fff687c` | partial: best rollouts so far, just short of the gate |
| 30 | v11 + clearance in closed loop | `1bb6a94` | success: first world-model run with all 10 learned laps |
| 31 | Condition changes mid-session: v11 vs the bicycle model | `1bb6a94` | partial: v11 survives the grip drop, the bicycle model does not; neither handles the patch. Mass scenario uninformative (entry 32) |
| 32 | Does mass matter in this sim; oracle-c test | `995b70e` | done: mass barely matters; a "correct" c does not improve predictions |
| 33 | Grip probe on (h, d), bicycle-error tag, condition-aware safe set | `c03ad4d` | failure: probe misses its target; the safe set did not prevent crashes |
| 34 | Safe set v2 (live change detection) + change-aware value, 3 seeds | `d44c313` | failure: detection works (≈1 s, no false alarms), the reaction does not slow the car |
| 35 | Baseline robustness, run divergence, v11's repeat crash spots, risk head vs speed cap | `7b5a865` | failure: neither option removes the no-change crashes; the speed reaction is built but not run |
| 36 | Root cause: regression or luck? Old code reproduced in worktrees | next commit | done: not a regression; the old clean runs were partly luck and partly crashes that did not end the run |

Main conclusions so far:

1. The RSSM filter is good: state, slip and grip estimates are as good as or better than the v3 encoder.
2. The RSSM's 15-step rollout is the weak part. From the true state it is 0.08–0.39 m off after 1.5 s (median) where
   v3 is 0.02–0.21 m, and it under-predicts slip growth near the limit (v7 spin: −0.20 rad predicted, −0.84 exact).
   No data or loss change so far has fixed this.
3. Every world-model crash happened on one of the first learned laps, either a wall hit after a rollout position
   error or a spin the model did not predict.
4. A flat, slow warm-up breaks SIT-LMPC on its own (the value function has no speed contrast). A warm-up that varies
   speed fixes that: the bicycle-model planner improves from 38.3 s to 36.6 s over 10 learned laps with no crash.

---

## 0. v3 encoder model (reference, `ros-sim`)

| | |
|---|---|
| What | Sensor encoder (IMU, wheel speed, PF pose) gives a latent z; transition = bicycle + residual(x, u, z) with z held over the rollout. Trained on the map-free dataset (`data/f1tenth-world`), no LiDAR. |
| Branch / commit | `ros-sim`, `402708e` (encoder, EKF baseline, offline eval), `60ca05b` (closed-loop results, encoder v3 evaluation) |
| Offline (on-track test split, 1 s ahead, from its own estimate) | median 0.25 m, 90% under 0.59 m (rssm-v5 era eval); 0.22 m median in the later 4,688-prediction eval |
| Replay of the RSSM crashes (1.5 s, from the true state, median) | v5 crash 0.02 m, v6 crash 0.13 m, v7 crash 0.21 m; v7 spin slip −0.49 rad (exact −0.84) |
| Closed loop (`ros-sim`) | reported as "14–15/15 laps", but those runs used `sit_lmpc_mode: false`, which retries after a crash: 3 of the 4 runs had a crash (`world-nochange` attempt 15, `patch-world` attempt 9, `change-friction-world-condition` attempt 12; only `change-friction-world` had none) and carried on. With the newer `sit_lmpc_mode: true` each would have ended at the crash. Spins at 0.72–0.80 rad slip, where the known model does not (corrected in entry 36) |
| Verdict | reference. Better rollouts than any RSSM so far, despite never seeing these tracks. |

## 1. Planner hull recompile fix

| | |
|---|---|
| What changed | Safe-set hulls copied to the GPU with `device_put` and chunked on the host instead of sliced on the GPU |
| Why | With the 15-step horizon, XLA recompiled while driving: p95 236 ms, 41% of steps over 100 ms |
| Commit | `0a26c2b` on `world-model`; own commits on `ros-sim` and `irl` |
| Check | planning p95 236 ms → 30 ms, no step over 100 ms |
| Verdict | success |

## 2. Timing tests

| Test | Commit | Numbers | Verdict |
|---|---|---|---|
| World-model parts, random weights (`scripts/time_world_model.py`) | `0a26c2b`, `e179dec` | full latent rollout of 1024 × 15 steps: 3.7 ms; one filter step 0.1 ms. Whole planner per step, variant A: 26 / 32 ms (p50 / p95); B (top 32–128): 27 / 33 ms; C: 25 / 30 ms; 10 × 0.15 s: 15–17 / 19–21 ms | success: the earlier 135 ms+ estimate was far too pessimistic |
| Jetson timing (`scripts/jetson_timing.sh`) | `e179dec` | not run yet (no Jetson access) | open |
| In the sim, variant A | — | rssm-v5: 38 / 60 ms; v8 + clearance: 49 / 81 ms (GPU free) and 84 / 121 ms (GPU shared with training); bicycle model, same planner: 33 / 49 ms | — |
| Safety checks (`scripts/time_safety_checks.py`, mains power) | `0eb1261`, `5f953fd` | latent rollout of 1024 samples 3.0 ms; clearance 4.0 ms on samples + 2.9 ms on the 256 final candidates; physics check 2.2 + 1.4 ms; ensemble 7.4 + 6.0 ms. On battery everything was ~2.5× slower | success |

## 3. On-track LiDAR data, first collection

| | |
|---|---|
| What | New collector (`scripts/collect_track_data.py`) driving random vehicles (grip 0.4–1.2, stiffness ±15%, mass 0.8–1.3) around Spielberg, custom1 and the Levine sim map with 1080-beam LiDAR, IMU, wheel speed and PF pose. 30% no change, 30% step change, 40% low-grip patches. |
| Why | The map-free dataset has no walls, so LiDAR could not be learned from it |
| Commit | `b8a7eb3`, `e179dec` (Levine top-up options) |
| Fixes during collection | centre line and braking-aware speed profile (the driver weaved and overspeeded into corners); LiDAR scanned in chunks of 16 (batched scan needed 101 GB); fixed-length padding (recompiles); 200 extra Levine episodes at 2–5 m/s (Levine crashed 77% of the time) |
| Result | 789 episodes, 293k transitions; Spielberg capped at 4–10 m/s (max 10.3 m/s) |
| Verdict | success, but the Spielberg speed cap later caused the first sim crash (entry 6) |

## 4. RSSM v1–v4: getting training stable

| Version | Change | Why | Result (validation) |
|---|---|---|---|
| v1 | first full model (GRU 128, c 8 + d 24, anchor, residual, LiDAR/sensor decoders, condition head) | — | blew up at step 9,800 (losses ×100) on batches with spin tails |
| v2 | spin tails cut (slip > 1.0 rad or yaw rate > 8 rad/s); clip 10 → 1; drop non-finite updates; bad-batch guard; condition weight 1 → 3 | v1 blow-up | no spike, but drifted from step ~15,000 |
| **v3** | prediction weight 1 → 0.3; filter feedback detached; decoder noise floor; lr 3e-4 → 1e-4; early stop | v2 drift | stable for 30,000 steps. Best step 26,500: anchor RMSE steering 0.014, speed 0.27, yaw rate 0.086, slip 0.019; 1 s position 0.62 m; grip head 0.80 |
| v4 | looser random walk on c (0.05 → 0.2) | weak grip head | no better than v3, stopped at 18,000 |

Commits `e179dec`, `3755c8b`, `53a9e23`. Verdict: success (v3 is the stable base for later versions).

## 5. RSSM v5 and offline evaluation

| | |
|---|---|
| What changed | speed weighted 3× in the anchor loss, prediction weight 0.3 → 0.6; posterior mean instead of a sample for state estimates |
| Why | offline eval showed the RSSM speed estimate at 0.29 m/s vs 0.18 for v3 |
| Commit | `411cf6a` |
| Check (on-track test split, 2,408 predictions of 1 s) | median 1 s error: bicycle + EKF 0.58 m, v3 0.25 m, rssm-v5 (mean) 0.29 m, bicycle from true state 0.11 m. Posterior mean vs sample: 0.34 → 0.29 m. Speed RMSE 0.25 m/s (v3 0.18) |
| Verdict | partial: 40% better than the bicycle model with an EKF, close to v3, but not better |

## 6. First sim tests (v5) and warm-up capped at 9 m/s

| Run | Setup | Result | Crash type |
|---|---|---|---|
| `rssm-latent-smoke` | v5, normal warm-up | warm-up 38.0–38.4 s; crashed 3.4 s into the first learned lap. Planning 38 / 60 ms | wall hit: warm-up reached 14.7 m/s, Spielberg data only went to 10.3 m/s; speed estimate ~1.1 m/s low, plans weaved |
| `rssm-latent-slow-smoke`, `-slow-logged` | v5, warm-up capped at 9 m/s | warm-up 63.7–63.8 s; first learned lap braked to 3–4 m/s and crashed at the first corner after 9–10 s, both runs | wall hit (first read as a spin; corrected after the replay: the 1.07 rad slip is the impact) |
| `known-slow-smoke` | bicycle model, same capped warm-up | 8/8 laps, no crash, but learned laps got slower: 68.8, 68.3, 83.7 s | — |

Also fixed: no LiDAR reached the model (`lidar_topic: /scan`). Commit `fa13e62`. Verdict: failure. Lesson: the model
must have seen the speeds and manoeuvres the planner will drive.

## 7. Plan A data: fast, closed-loop and hard-manoeuvre episodes

| | |
|---|---|
| What | 232 fast Spielberg episodes (8–14 m/s); 179k steps of the planner's own laps from 35 earlier runs (`scripts/convert_closed_loop_laps.py`); 450 hard throttle/brake/steering episodes (`--excite`) |
| Why | cover the speeds and manoeuvres of the planner |
| Commit | `fa13e62` |
| Result | 88% of the fast episodes crashed during collection; only 1.4% of Spielberg data ended up above 11 m/s. Dataset 612k steps. High slip still rare (under 1% of steps above 0.5 rad) |
| Verdict | partial |

## 8. RSSM v6

| | |
|---|---|
| What changed | retrained on the plan A data (prediction steps 10) |
| Commit | `fa13e62` |
| Validation | best step 27,500: anchor RMSE 0.017 / 0.29 / 0.10 / 0.023; 1 s position 0.72 m; grip head 0.70 |
| Sim run `rssm-latent-v6-smoke` | warm-up 38.0–38.4 s; learned laps 37.4 s and 36.7 s (first learned laps faster than warm-up); crashed early in lap 8 (3.9 s in), ~8 m/s |
| Crash type | wall hit (replay: exact bicycle predicts 0.33 rad, actual 1.11 rad at impact) |
| Replay (1.5 s, true state, median) | v6 model 0.32 m; v3 0.13 m; bicycle 0.00 m |
| Verdict | partial: first clean learned laps, still crashes |

## 9. RSSM v7: spin data and high-slip oversampling

| | |
|---|---|
| What changed | spin cut-off raised to the model limits (slip 1.2 rad, yaw rate 10 rad/s); 15% of each batch drawn from windows with slip > 0.4 rad |
| Why | v6 still under-predicted slip near the limit |
| Commit | `38a3f35` |
| Validation | best step 24,500: anchor RMSE 0.017 / 0.30 / 0.11 / 0.023; 1 s position 0.71 m; grip head 0.74 |
| Offline (4,688 predictions) | median 1 s error 0.29 m (v3 0.22, bicycle + EKF 0.44); slip RMSE 0.018 (v3 0.019) |
| Sim run `rssm-latent-v7-smoke` | one learned lap (38.3 s), then crashed 8.1 s into lap 7 at ~14 m/s |
| Crash type | spin. Predicted slip 1.5 s ahead: v7 −0.18 rad, v3 −0.49, exact −0.84, actual −0.75 |
| Replay (median) | v7 0.52 m; v3 0.21 m; bicycle 0.00 m |
| Verdict | failure: spins still under-predicted |

## 10. RSSM v8: latent overshooting fine-tune

| | |
|---|---|
| What changed | fine-tune from v7: 15-step rollouts with stride 2 from both the filtered state and the true state (new `true_state_prediction` loss), 30% of each batch from the map-free high-slip data, 12,000 steps, lr 5e-5 |
| Why | train the rollout the way the planner uses it; add high-slip examples |
| Commit | `9e9707a` |
| Validation | best step 10,500: 1.5 s position 0.95 m; anchor RMSE 0.022 / 0.31 / 0.11 / 0.023. The bad-batch guard skipped 211 batches |
| Replay (1.5 s, median / last 10 steps) | v5 crash 0.08 / 0.07 m; v6 crash 0.36 / 0.18 m; v7 crash 0.36 / 0.89 m. v7 spin slip −0.23 rad |
| Pass criteria | position close to v3 (0.02–0.21 m) and v7 spin close to −0.84 rad: **both failed** |
| Verdict | failure |

## 11. LiDAR clearance check

| | |
|---|---|
| What | Decode the predicted LiDAR along each candidate's latent rollout; penalise steps where the nearest predicted range is under 0.35 m (weight 1.0) |
| Why | v5 and v6 crashes were wall hits from rollout position error |
| Commit | `9e9707a` (from `898cd56` it counts outside the lambda grid, entry 14) |
| Sim `rssm-latent-v8-clearance` (8 laps) | 8/8 laps, no crash. Warm-up 38.0–38.4 s; learned 37.9, 38.2, 38.2 s (no improvement). Planning 84 / 121 ms (GPU shared with training) |
| Sim `rssm-latent-v8-clearance-long` (20 laps planned) | lap 6: 37.6 s; spun in lap 7 (19.2 s in): braked hard from 11.4 m/s into a corner, steering swinging, yaw rate growing to −5.1 rad/s, slip 0.55 rad. Planning 49 / 81 ms (GPU free) |
| Verdict | partial: first crash-free world-model run, but no improvement and the next run spun. Clearance cannot see a slide building up |

## 12. RSSM v9: v8 without batch skipping

| | |
|---|---|
| What changed | same as v8 with the skip guard off; gradient clipping and non-finite filtering only; log how many high-slip windows were trained on |
| Why | v8 skipped 211 batches, probably the high-slip ones |
| Commit | `0eb1261` |
| Training | 0 skipped. Windows trained on: 143,292 with slip > 0.3 rad and 60,233 with slip > 0.5 rad, of 384,000 (37% and 16%). Best step 10,500, 1.5 s position 0.95 m |
| Replay (median / last 10) | v5 crash 0.08 / 0.07 m; v6 crash 0.36 / 0.20 m; v7 crash 0.39 / 1.14 m. v7 spin slip −0.20 rad |
| Verdict | failure: no better than v8. Showing the model far more slides did not teach it to predict them |

## 13. Longer v8 runs and warm-up changes

| Run | Setup | Laps | Result | Crash type |
|---|---|---|---|---|
| `rssm-latent-v8-clearance-long` | v8 + clearance, normal warm-up (to 14.6 m/s) | 5 + 1 | lap 6 37.6 s, crash in lap 7 | spin (see entry 11) |
| `rssm-latent-v8-slow-long` | warm-up `max_speed` 9 (with speed scale 0.6: a flat 5.6 m/s) | 5 + 2, stopped | warm-up 63.7–63.8 s; learned 75.4, 72.0 s | — |
| `known-varied-long` | bicycle model, warm-up speed scale 1.0, cap 9 m/s (6.2–9.0 m/s around the lap) | 5 + 10 | warm-up 38.2–38.4 s; learned 37.9, 38.8, 37.3, 38.6, 37.8, 37.0, 37.4, 36.9, **36.6**, 36.9 s; no crash; top speed rose to 11.6 m/s; planning 33 / 49 ms | — |
| `rssm-latent-v8-varied-long` | v8 + clearance, same varied warm-up | 5 + 1 | crashed 24.4 s into lap 6 | spin: accelerated from 8.3 to 12.6 m/s out wide (0.8 m off centre), yaw rate swinging ±2–3 rad/s, slip −0.39 then a spin at 14 rad/s yaw rate. Estimated slip one step before: −0.20 vs −0.39 true |

Diagnosis of the slowdown: steering jitter on learned laps (steering-rate sign flips ~50% of steps, ~1.3° per step)
exists in every run but is minor. The cause was the value function. With a flat 5.6 m/s warm-up it never sees two
speeds at the same place; probed offline, it rewarded going slower over the first ~100 m of the lap (3 m/s scored
better than 9 m/s), and each slow lap retrained it further that way.

Verdict: the varied warm-up is a success for the planner (configs `configs/fast-diag/known-varied-warmup.yaml`,
`configs/world/rssm-latent-v8-varied.yaml`). v8 still crashed on its first learned lap, past its 10.3 m/s data range.

## 14. Planner fix: safety costs outside the lambda grid

| | |
|---|---|
| What changed | Safety penalties (clearance, speed cap, slip, and any model checks) are added to the reward on their own, not multiplied by the boundary lambda, and they count in the final candidate ranking (same priority as track-limit violations). Per-lap `safety_summary` in `lap_metrics.json`: for each term, share of steps it hit any candidate, mean share of candidates hit, share of steps it hit the chosen plan |
| Why | The lambda grid contains (0, 0), so one proposal ignored the clearance cost entirely, and the final ranking looked only at track limits and the safe set |
| Commit | `898cd56` (first in `5f953fd`) |
| Checks | unit tests; full suite has the same 137 old failures before and after (old pytest fixture handling, f1tenth adapter import) |
| Verdict | success as code; not yet tested in a run on its own (the only run with it also had entries 17 and 18) |

## 15. Guardrails: speed cap and slip cost

| | |
|---|---|
| What | `planner.safety_speed_cap` (default off; 10 m/s for the world model, just under its 10.3 m/s data) and `planner.safety_slip_limit` 0.35 rad / `safety_slip_weight` 1.0 (default on), both per step on the predicted states |
| Why | keep the planner inside the model's data; penalise predicted slides |
| Commit | `898cd56` |
| Check | only in the guarded run (entry 19): speed cap hit 34% of candidates and the chosen plan on 4% of steps; slip cost hit 14% of candidates, never the chosen plan |
| Verdict | not tested alone. Kept as config options |

## 16. Slip-estimate check

| | |
|---|---|
| What | `scripts/slip_estimate_check.py`: filtered slip estimate vs true slip, binned by true slip, on held-out on-track and map-free windows and on logged crashes |
| Why | find out whether the spin problem is the encoder (estimate) or the transition (prediction) |
| Commit | `898cd56` |
| Result | estimate / true slip: on-track 0.99–1.04 at every level up to 0.6 rad (v8 and v9); map-free 0.92–0.95 above 0.1 rad. Crashes: v7 last step −0.39 true → −0.41 estimated; v8 varied-warm-up crash tracks except one step (−0.39 → −0.20). The earlier "−0.20 vs −0.39" for v7 was the 1.5 s prediction, not the estimate |
| Verdict | success as a diagnosis: no systematic under-read, so no encoder retrain. The transition is the weak part |

## 17. Physics check (removed)

| | |
|---|---|
| What | Each candidate also rolled out with the bicycle model at the RSSM grip estimate; steps with slip > 0.35 rad or outside the track limits penalised (weight 1.0) |
| Why | a physics safety check next to the world model, like the safe set |
| Commit | `world-model-guarded-attempt` `5f953fd` |
| Timing | +3.6 ms per step |
| Result | flagged 91% of candidates at every step in the guarded run. Replaying the planner's sampling at logged states: 85–100% of samples leave the 0.85 m track limit within 1.5 s, with both the bicycle and the world model. The existing boundary term already covers this (scaled and normalised); the physics check added it again at full weight and ×1000 in the ranking, so it decided the final pick almost alone. Its slip part did show the expected gap: at the last state before the crash, bicycle slip > 0.35 rad for 38% of samples, world model 0% |
| Verdict | failure as built; removed from `world-model`. A slip-only version was not tried |

## 18. Ensemble transition heads (removed)

| | |
|---|---|
| What | 5 residual heads on the frozen v8 (encoder, memory and main transition unchanged), each trained on a fixed random half of the windows with a 15-step true-state rollout loss (`sit_lmpc.world.rssm_ensemble`). Planner cost on head disagreement in slip and position along each candidate |
| Why | penalise candidates where the model is unsure |
| Commit | `world-model-guarded-attempt` `5f953fd` |
| Training | unstable: lr 1e-3 / 3e-4 / 1e-4 / 2e-5 all diverged after 250–1,250 steps (gradients through the bicycle model first, then the learning rate, then outlier spin windows; a log loss per window delayed but did not stop it). Best checkpoint step 500 |
| Best checkpoint | 1.5 s position per head 0.50 m on-track (same as v8), 1.31 m map-free, 0.72 m high-slip. Head spread small: slip std 0.004–0.006 rad median, 0.014–0.021 p95; position 0.09–0.20 m median; spread vs error correlation 0.42 on-track, 0.12 on high-slip windows |
| Timing | +13 ms per step |
| Verdict | failure: the heads barely disagree and the disagreement does not track error where it matters. Removed |

## 19. Guarded v8 sim run

| | |
|---|---|
| Setup | `configs/world/rssm-latent-v8-guarded.yaml` (on the attempt branch): v8 + ensemble heads, varied warm-up, clearance, speed cap 10 m/s, slip cost, physics check, ensemble cost |
| Commit | `5f953fd` |
| Laps | warm-up 38.2–38.4 s; crashed 7.5 s into lap 6 |
| Crash type | wall hit: slip only 0.05 rad at contact; the car weaved (yaw rate −4.4 to +1.0 rad/s) out to 0.9 m off centre |
| Max speed / max \|slip\| | 9.7 m/s / 0.78 rad (one short slide at ~4 s) |
| Safety terms (75 steps) | physics: hit 91% of candidates, chosen plan 36% of steps; ensemble: 8% / 11%; clearance: 9% / 9%; speed cap: 34% / 4%; slip: 14% / 0% |
| Planning | 71 / 112 ms (bicycle baseline 33 / 49 ms) |
| Verdict | failure, caused mainly by the physics check (entry 17) |

## 20. Post-contact frames removed

| | |
|---|---|
| What changed | `scripts/mark_contacts.py` finds the first step where the car body outline (rear-axle frame x −0.125..0.455 m, y ±0.155 m, same as the sim's collision body) touches an occupied map cell, and stores it in `data/f1tenth-track/contacts.json`. For closed-loop episodes the sim's own collision flag from the source trajectory is used when it is earlier. `TrackData` ends every episode before its contact step, so training and every evaluation that loads the dataset skip post-contact frames. The crash replays stop before the sim's first collision step |
| Why | Frames after a wall contact are not driving: in the sim the collision response throws the state around (the 1.07 and 1.11 rad "slips" in entry 6/8 were impacts), and in the collector the car keeps driving through the wall |
| Finding | **The data collector lets cars pass through walls.** Its physics has no collision response, and it only ended an episode at LiDAR clearance < 0.18 m. The LiDAR sits mid-car and cannot see the rear 90°, so side and rear contacts went unnoticed. 366 collector episodes touched a wall before they ended (251 Spielberg, 111 Levine, 12 custom1), some driving on inside the wall for hundreds of steps. Median LiDAR range at the detected contact step is 0.22–0.23 m, which matches a side contact |
| Frames removed | **36.7k post-contact frames of 612k (6%)** in the existing data (24.6k Spielberg, 11.7k Levine, 0.4k custom1). Closed-loop episodes: 14 frames in 7 crash episodes |
| Check | body outline vs sim collision flag on the closed-loop episodes: same step or one step apart in 6 of 7 contacts; one contact seen only by the sim flag. Sim crash attempts are flagged 1–2 steps before their end (v5 step 96 of 98, v6 38 of 40, v7 80 of 82) |
| Verdict | success. Models trained before this (v1–v9) saw these frames |

## 21. Spin-onset data collection

| | |
|---|---|
| What | `scripts/collect_spin_onsets.py`: 360 onset groups (120 per map) × 8 rollouts of up to 3 s, appended to `data/f1tenth-track` (2,880 episodes, 72k transitions). Each group is one start near the limit: corner entry above the grip-limited speed, an unknown low-grip patch on a corner, or a steering kick at speed. Random grip 0.4–1.2, stiffness ±15%, mass 0.8–1.3. Rollouts of a group differ by small perturbations (speed 2%, lateral 3 cm, heading 0.01 rad, yaw rate 0.05 rad/s, small control noise). Each group's severity is set by bisection to where an unperturbed rollout flips between crash and recovery. Rollouts end at the first wall contact (body outline) or past slip 1.2 rad / yaw rate 10 rad/s. Whole groups go to one split |
| Why | train on spins and recoveries from near-identical states, so the model can learn what tips a car over |
| Tuning on the way | first version: every rollout crashed, because the collector's pure-pursuit driver cannot hold the limit by itself (weaves, understeers off at 9 m/s). Fixed by a driver at 70% of the true grip limit, 3 s rollouts and the severity search |
| Groups with both outcomes | **251 of 360 (70%)**: corner 85/117, patch 102/120, kick 64/123; Spielberg 73, custom1 99, Levine 79. 111 groups crash even at the mildest setting (mostly fast kicks and Levine's narrow corridor), 17 never crash |
| Outcomes | 36% wall contact, 28% spin, 36% recovery (crash rate 64%) |
| Slip distribution (share of steps) | onset data: \|slip\| > 0.1 rad 48.1%, > 0.3 rad 16.4%, > 0.5 rad 3.0%, > 0.8 rad 0.4%. Existing data (cleaned): 14.0%, 2.0%, 0.5%, 0.1% |
| Speeds | max per rollout: median 9.8 m/s, 90th percentile 14.8 m/s |
| Verdict | success |

## 22. Crash-risk head

| | |
|---|---|
| What | Small network (SIT-LMPC's `RiskNetwork`, 2 × 64, class-balanced cross-entropy) on the frozen v8 filter's (h, s): probability of a crash within 1.5 s (wall contact or spin past slip 1.2 rad / yaw rate 10 rad/s). `sit_lmpc.world.rssm_risk`, `configs/train-rssm-risk.yaml`. Training: the cleaned dataset with the onset rollouts plus five v8 closed-loop runs (clearance, clearance-long, slow-long, varied-long, guarded), 522k steps, 4.7% labelled 1. Held out: the v5, v6 and v7 crash runs. Planner: `planner.dynamics.rssm_risk` / `rssm_risk_weight`, weight × probability at every predicted step, one latent rollout shared with the clearance check (+0.4 ms), not used in the final ranking |
| Why | learn from failures without retraining the world model |
| Commit | `1b2224d`; report in `results/risk-head/report.json` |
| Offline | AUC 0.93 validation, 0.92 test (onset groups 0.80 / 0.85). Alarm threshold 0.78 (5% of safe validation steps above it) |
| Held-out warning time | v7 spin (~14 m/s): first alarm 1.6 s before contact, continuously above for the last 0.4 s, run AUC 0.96. v6 wall hit (~8 m/s): no alarm (peak 0.62), AUC 0.77. v5 slow wall drift (~3 m/s): no alarm (peak 0.58), AUC 0.50 |
| Sim `rssm-latent-v8-risk` (v8 + clearance + risk, weight 1; slip cost still on by default then) | warm-up 38.2–38.4 s; wall hit 10.2 s into lap 6 after riding 0.8–0.9 m off centre for ~5 s with slip near zero. Learned max speed 10.1 m/s, max \|slip\| 1.04 rad (one slide early in the lap). Re-filtered along the driven path the risk was 0.5–0.84 for the last 3 s and above the alarm threshold on 6.9% of learned-lap steps (warm-up 0.5%). Planning 53 / 71 ms |
| Verdict | failure in closed loop. It saw the danger, but near the edge almost every candidate had similar risk, so a soft cost did not change the choice. It also does not recognise low-slip wall drifts. (The per-lap `safety_summary` counts any cost > 0, so for a probability it always reads 100%) |

## 23. Slip cost off by default; bicycle baseline rerun

| | |
|---|---|
| What changed | `planner.safety_slip_weight` default 1.0 → 0.0 (kept as an option) |
| Why | the bicycle baseline on the code with the slip cost on (`known-varied-current`): learned laps 38.8, 38.9, 38.2, 39.0, 39.9 s and a wall hit 12.4 s into lap 11; max \|slip\| on learned laps ~0.3 rad, top speed 10.5 m/s. The old code (`known-varied-long`) gave 38.3 → 36.6 s with no crash |
| Check `known-varied-slipoff` | warm-up 38.2–38.4 s; learned laps 37.9, 39.2, 36.6, 37.4, 36.2, 37.3, 36.7, 36.7, 36.0, 36.1 s; 15/15 laps, no crash. Learned max speed 12.8 m/s, max \|slip\| 0.75 rad, max \|e_y\| 0.81 m. Planning 31 / 47 ms |
| Verdict | success: back to (slightly better than) the old result, so the slip cost alone caused the regression; the safety-cost plumbing with every term at zero leaves the planner unchanged |

## 24. v8 lateral error near the edges

| | |
|---|---|
| What | `scripts/lateral_error_check.py`: on the learned laps of the crash runs (v5 slow-logged, v6, v8 clearance-long, varied-long, guarded, risk), re-filter with v8 and roll out from the planner's start state (true pose, anchor body state) with the applied controls; compare predicted and actual e_y on the planner's track frame at 0.5 / 1.0 / 1.5 s (1,771 starts). Results in `results/lateral-error/` |
| Spread | p95 \|e_y error\| 0.24 m at 0.5 s, 1.08 m at 1.0 s, 2.29 m at 1.5 s. Slow v5 run (~3 m/s): 0.04 / 0.15 / 0.33 m |
| Bias | none consistent. Signed towards the outside of the bend (curvature > 0.05): 0.00 m at 0.5 s (55% outward), −0.02 m at 1.0 s (55%), −0.17 m at 1.5 s (47%). Signed towards the nearer wall it looks outward (+0.12 / +0.40 / +0.53 m) when binned by the actual e_y and inward (down to −1.24 m) when binned by the predicted e_y: both are selection effects of a large, symmetric error, not a bias |
| By edge and speed | by distance from the centre line at the start there is no clear trend (bend subset, 1.5 s: p95 1.33 m for \|e_y\| 0.0–0.3 m, 1.42 m for 0.3–0.5 m, 1.12 m for 0.5–0.7 m; small samples). Speed matters: p95 at 1.0 s is 0.17 m below 4 m/s, 0.61 m at 4–7, 0.95 m at 7–9, 1.14 m at 9–11 and 1.38 m above 11 m/s |
| Validation, near the edge (\|e_y\| ≥ 0.5 m, 10k starts) | p95 per step 0.04, 0.08, 0.12, 0.18, 0.25, 0.32, 0.41, 0.52, 0.64, 0.81, 1.07, 1.22, 1.41, 1.63, 1.80 m (all starts: 0.03 → 1.36 m). A first measurement from the noisy PF pose gave 0.31–0.42 m already at step 1 and was discarded |
| Verdict | the problem is spread, not bias: by 1.0–1.5 s the 95% band is wider than Spielberg's 0.85 m allowed offset |

## 25. World-model track-limit margin

| | |
|---|---|
| What | `planner.boundary_margin`: per-step margin added to \|e_y\| in the track-limit check (plan reward and final ranking). Set to the validation near-edge p95 per step, capped at 0.65 m: 0.04, 0.08, 0.12, 0.18, 0.25, 0.32, 0.41, 0.52, 0.64, then 0.65 m (`configs/world/rssm-latent-v8-margin.yaml`) |
| Why the cap | uncapped, the margin passes the 0.85 m allowed offset at step 10–11 and reaches 1.80 m, so every candidate would be off the track from 1.0 s on |
| Sim `rssm-latent-v8-margin` (v8 + clearance + margin; slip cost and risk head off) | warm-up 38.2–38.4 s; lap 6 41.2 s with a near-spin at s ≈ 217 m (yaw rate 10.5 rad/s, slip −0.64 rad, recovered); lap 7: braked hard (−7.1 m/s²) at s ≈ 103 m from 8 m/s to a standstill, stood still ~4 s, then rolled backwards into the wall at ~2 m/s (contact 18.2 s into the lap). Learned max speed 11.2 m/s, max \|slip\| 0.80 rad before the stall, max \|e_y\| 0.80 m. Planning 52 / 64 ms |
| Crash type | neither a clean spin nor a wall drift: stall, then reversing into the wall. At a standstill v8 is outside its data (episodes are trimmed below 1.2 m/s) |
| Compared with the bicycle baseline (entry 23) | baseline 15/15 laps, 36.0–39.2 s, max \|e_y\| 0.81 m; v8 + margin 1 learned lap (41.2 s), then the stall crash |
| Verdict | failure: a margin as large as the model's error leaves too little room, and the planner stops |

## 26. Rollout variants

| | |
|---|---|
| What | `scripts/rollout_variants_check.py`: the same 1,771 learned-lap starting points as entry 24 (v5 slow-logged 81, v6 764, v8 clearance-long 552, varied-long 228, guarded 59, risk 87), same applied controls, lateral error on the planner's track frame. Results in `results/lateral-error/v8-rollout-variants.json` |
| What the planner does now | `latent_rollout(..., sample=False)`: the fast part d is the prior mean at every step (no sampling) and the slow part c is carried unchanged at its filtered value. So "prior mean for s" and "c fixed, d from the prior mean" are the same rollout as the current one |

| Variant | 0.5 s median / p95 | 1.0 s median / p95 | 1.5 s median / p95 |
|---|---|---|---|
| v8 planner rollout (= prior mean = c fixed + d prior mean), from the planner's start state | 0.09 / 0.24 m | 0.33 / 1.08 m | 0.62 / 2.29 m |
| v8 with d sampled from the prior each step | 0.09 / 0.26 m | 0.37 / 1.52 m | 0.93 / 3.86 m |
| v8 planner rollout from the full true state | 0.04 / 0.19 m | 0.15 / 0.78 m | 0.36 / 1.73 m |
| v3 (encoder z), from the true state | 0.02 / 0.14 m | 0.08 / 0.67 m | 0.18 / 1.63 m |
| Bicycle, sim parameters (= nominal here), from the true state | 0.00 / 0.00 m | 0.00 / 0.00 m | 0.00 / 0.00 m |

| | |
|---|---|
| Findings | (1) No cheaper rollout variant exists: the current one already uses the prior mean with c held; sampling makes it worse. (2) About half of v8's error comes from its starting body state (steering, speed, yaw rate, slip from the anchor): the same rollout from the true state has half the median error. (3) From the true state v8 is still about twice v3's median error, and both have heavy tails at 1.5 s (p95 1.6–1.7 m). (4) The nominal bicycle model is exact in this sim (same parameters, no grip change), so it is a floor, not a fair competitor |
| Verdict | no planner change, no sim run. Next: step 3 (distributional latent) |

## 27. Starting-state error by component; v8 planning from the true state

| | |
|---|---|
| Starting state of the other controllers | Bicycle baseline: the sim's true state. The ROS adapter takes pose, speed, yaw rate and slip from the bridge's ground-truth odometry (body twist) and steering from integrating the commanded steering rate, which is exact in lockstep. v3 closed-loop runs (`belief_update: encoder`, `encoder_state: true`): steering, speed, yaw rate and slip replaced by the v3 encoder's estimate, like v8's runs with the RSSM anchor; pose from odometry in both |
| Estimate error at the 1,771 starts (anchor − truth) | steering RMSE 0.024 rad (mean −0.017, p95 0.043); speed RMSE 0.235 m/s (mean +0.209, p95 0.39); yaw rate RMSE 0.088 rad/s (mean −0.023, p95 0.17); slip RMSE 0.013 rad (p95 0.027) |

| Starting state, v8 planner rollout | 0.5 s median / p95 | 1.0 s median / p95 | 1.5 s median / p95 |
|---|---|---|---|
| all from the anchor (current) | 0.09 / 0.24 m | 0.33 / 1.08 m | 0.62 / 2.29 m |
| true steering only | 0.05 / 0.22 m | 0.18 / 0.85 m | 0.43 / 1.86 m |
| true speed only | 0.07 / 0.23 m | 0.30 / 1.05 m | 0.61 / 2.27 m |
| true yaw rate only | 0.08 / 0.24 m | 0.32 / 1.08 m | 0.62 / 2.30 m |
| true slip only | 0.08 / 0.23 m | 0.33 / 1.05 m | 0.64 / 2.26 m |
| all true | 0.04 / 0.19 m | 0.15 / 0.78 m | 0.36 / 1.73 m |

| | |
|---|---|
| Finding | the steering estimate drives the doubled error: a 0.017 rad bias bends the predicted path over 1.5 s. True steering alone gives most of the gain of the full true state; speed (biased +0.21 m/s), yaw rate and slip barely matter for e_y. In the sim and on the car the commanded steering is known, so it should not be replaced by an estimate |
| Sim `rssm-latent-v8-truestate` (`planner.dynamics.rssm_state: false`: plans from the true sim state, filter still runs for h and s; v8 + clearance, slip cost off, no margin, no risk head, varied warm-up) | warm-up 38.2–38.4 s; crashed 26.1 s into lap 6 at s ≈ 243 m. At 10.5 m/s the car turned in (yaw rate +3.7 rad/s), slid (slip −0.37 rad) and crossed from 0.37 m right to 0.75 m left of centre into the wall: a slide into the wall. Learned max speed 11.4 m/s, max \|slip\| 0.82 rad, max \|e_y\| 0.82 m before contact. Planning 54 / 73 ms |
| Compared with the bicycle baseline (entry 23) | baseline 15/15 laps, 37.9 → 36.0 s, max \|e_y\| 0.81 m, planning 31 / 47 ms |
| Verdict | failure: with a perfect starting state v8 still crashes on its first learned lap, so state estimation alone is not the fix. The transition (rollout) is the remaining problem. Next: step 3 |

## 28. Open-space coverage data, rebalanced sampling, rssm-v10

| | |
|---|---|
| Why | almost all on-track data is gentle driving: every motion variable is sharply peaked at zero (`docs/DATA_DISTRIBUTIONS.md`), so the model learns slides and fast cornering poorly |
| Frame check (before fine-tuning) | the transition's inputs are frame-invariant (steering, speed, yaw rate, slip, commands, h, s; pose integrated by the bicycle model outside the network), but the residual's position correction was added in the world frame without the network knowing the heading. In v8 it is small (≈ 5 mm per step, ≤ 0.08 m over 1.5 s). New option `model.body_frame_residual`: predict it in the car frame and rotate by the heading (used in v10) |
| Coverage collector | `scripts/collect_coverage.py`, `data/f1tenth-coverage`: open space, random vehicles (grip 0.4–1.2, stiffness ±15%, mass 0.8–1.3, 30% with a step change), chains of 0.3–2 s segments (speed/steering targets, or full-throttle/brake bursts with a steering target that stop at a random speed). Before every segment 8 candidates are simulated and the one landing in the least-filled cells of a speed × steering × slip (±0.6 rad) × yaw rate (±8 rad/s) histogram is driven; candidates that lose control score ×0.25. Episodes end at \|slip\| > 1.2 rad or \|yaw rate\| > 10 rad/s. 1,994 episodes, 273k steps (84% end in loss of control). First version piled up at the 15 m/s cap and below 1 m/s (bursts stopped at the limits); fixed, though 7.5% of steps are still above 14.5 m/s |
| Coverage achieved | coverage data: speed and steering close to uniform, slip SD 0.41 rad (current data 0.11). Of 5,245 joint cells reached by either dataset, cells with ≥ 10 steps: current data 25%, coverage data 52%, together 56%. Figures: `docs/figures/data_coverage_compare.png`, `data_merged_before_after.png`, `data_merged_heatmaps.png` |
| Rebalanced track sampling | each track window weighted ∝ 1/√(windows in its (speed, slip, yaw rate) cell at the window midpoint), mean 1, capped at 5× (3.9% of windows at the cap); replaces the 15% high-slip oversampling; validation unweighted. Track share natural → rebalanced: speed > 9 m/s 9.5% → 20.4%, \|slip\| > 0.3 rad 1.8% → 9.1%, > 0.5 rad 0.3% → 1.7%, \|yaw rate\| > 3 rad/s 1.7% → 8.2%. Training mix (50% rebalanced track + 50% coverage): 33.2%, 26.5%, 12.0%, 18.4% |
| rssm-v10 | fine-tune of v8 (`configs/train-rssm-v10.yaml`): 50/50 track/coverage, LiDAR losses only on track windows, car-frame position residual, lr 5e-5 |
| Training | diverged. Best checkpoint step 500 (1.5 s position 1.16 m on track validation, v8 0.95 m); from step 1,000 every loss rose together, the training loss too (466 → 3,155 by step 2,500), early stop at 2,500. At the v8 starting point the coverage windows have ~20× larger losses than track windows (rollout prediction median 282 vs 15, from-true-state 240 vs 6; 17% of them dip below 1 m/s), so half of every batch is far outside what v8 knows and the 15-step rollout losses run away |

| Lateral error, median / p95 (same 1,771 starts) | 0.5 s | 1.0 s | 1.5 s |
|---|---|---|---|
| v8, true state | 0.04 / 0.19 m | 0.15 / 0.78 m | 0.36 / 1.73 m |
| v10 (step 500), true state | 0.05 / 0.25 m | 0.20 / 0.98 m | 0.57 / 2.13 m |
| v3, true state | 0.02 / 0.14 m | 0.08 / 0.67 m | 0.18 / 1.63 m |
| v8, planner start | 0.09 / 0.24 m | 0.33 / 1.08 m | 0.62 / 2.29 m |
| v10, planner start | 0.10 / 0.27 m | 0.40 / 1.07 m | 0.76 / 2.21 m |
| speed > 9 m/s (766), true state, 1.5 s: v8 / v10 / v3 | | | 0.52 / 0.77 / 0.21 m |
| \|slip\| > 0.3 rad in the window (210), true state, 1.5 s: v8 / v10 / v3 | | | 0.69 / 0.64 / 0.45 m |

| | |
|---|---|
| v7 replay (from the true state, now ending before the sim contact at step 80 of 82) | median 1.5 s position error: v10 0.64 m, v8 0.35 m, v3 0.23 m; last 10 starts: v10 2.11 m, v8 0.69 m, v3 0.24 m. The large slips quoted before (−0.75 actual, −0.84 bicycle) fall in the last 2–3 steps at the contact and are outside the cleaned window; at the last clean start (1.5 s ahead, true slip −0.13 rad) v10 predicts −0.10, v8 −0.15, v3 −0.09 |
| Verdict | failure: v10 is worse than v8 almost everywhere (only the high-slip bin is marginally better at 1.5 s) and far from 1.3× v3; no sim run. The coverage data itself looks right; fine-tuning v8 on it at 50% is unstable. Options: train on it gradually (small share first, then more), a robust per-window loss as for the ensemble heads, or train the next model (step 3) on the mix from the start |

## 29. rssm-v11: trained from scratch on the full mix

| | |
|---|---|
| What changed | same architecture (Gaussian latent, GRU 128, c 8 + d 24) with the car-frame position residual, trained from scratch (`configs/train-rssm-v11.yaml`, `results/rssm-v11/`). Data: cleaned on-track data with rebalanced track windows (as v10) plus the open-space coverage data, its share ramped 20% → 50% over the first 30% of training, no LiDAR losses on it. Rollout losses (`prediction`, `true_state_prediction`): Huber (delta 1 on the scaled state error) and a per-window cap (log growth above 10). Horizon curriculum: rollout-loss length ramped 2 → 15 steps over the first half; validation always at 15. 30,000 steps, lr 1e-4, early stop allowed only after step 15,000 |
| Why | v10 diverged when fine-tuning v8 with 50% coverage windows from the first step; the correction to the replay showed the main gap is rollout position accuracy |
| Training | stable on validation throughout: 1.5 s position (filtered start) 2.47 m at step 2,000 → 1.30 m at 14,000 → 1.21 m at 28,000 (best; v8 0.95 m on its own validation). Anchor RMSE steering 0.018, speed 0.28, yaw rate 0.11, slip 0.021 (v8: 0.022 / 0.31 / 0.11 / 0.023). The batch training loss spiked above 100 at 7 of 60 evaluations (steps 13,000–16,000 and 28,500–30,000, up to ~11,000) without moving validation; nothing skipped |

| Lateral error, median / p95 (same 1,771 starts) | 0.5 s | 1.0 s | 1.5 s |
|---|---|---|---|
| v11, true state | 0.03 / 0.10 m | 0.10 / 0.49 m | **0.26 / 1.36 m** |
| v8, true state | 0.04 / 0.19 m | 0.15 / 0.78 m | 0.36 / 1.73 m |
| v3, true state | 0.02 / 0.14 m | 0.08 / 0.67 m | 0.18 / 1.63 m |
| v11, planner start (own estimate) | 0.05 / 0.20 m | 0.19 / 0.79 m | 0.42 / 1.84 m |
| v8, planner start | 0.09 / 0.24 m | 0.33 / 1.08 m | 0.62 / 2.29 m |

| 1.5 s, true state, median / p95 | v11 | v8 | v3 |
|---|---|---|---|
| speed > 9 m/s (766 starts) | 0.41 / 1.49 m | 0.52 / 1.89 m | 0.21 / 1.94 m |
| \|slip\| > 0.3 rad in the window (210 starts) | 0.55 / 1.97 m | 0.69 / 2.08 m | 0.45 / 1.93 m |

| | |
|---|---|
| v7 replay (cleaned, from the true state) | median 1.5 s position error v11 0.36 m (v8 0.35, v3 0.23); last 10 starts v11 0.32 m (v8 0.69, v3 0.24). Predicted slip tracks the truth up to t = 63 but has the wrong sign at the last clean start (true −0.13, v11 +0.13, v3 −0.09) |
| Gate | 1.5 s median from the true state 0.26 m against ~0.23 m (1.3 × v3's 0.18 m): missed by 0.03 m, so no sim run |
| Verdict | partial: the best world model so far. It beats v8 everywhere (true-state 1.5 s median −28%, p95 −21%; from its own estimate −32%) and has a smaller 95th percentile than v3 (1.36 vs 1.63 m), but the median is still 1.44× v3's and the gap is largest above 9 m/s (0.41 vs 0.21 m) |

## 30. v11 + clearance in closed loop

| | |
|---|---|
| Setup | `configs/world/rssm-latent-v11.yaml`: rssm-v11, LiDAR clearance (weight 1), varied warm-up (capped at 9 m/s), slip cost off, no margin, no risk head; same planner settings as the bicycle baseline. Spielberg, no change, 15 laps, nothing else on the GPU. Run `rssm-latent-v11` |
| Laps | warm-up 38.4, 38.3, 38.3, 38.2, 38.3 s; learned 38.9, 39.7, 39.8, 38.7, 39.2, 38.7, 38.0, 38.1, 37.5, 36.9 s |
| Crashes | none: 15/15 laps |
| Learned laps | max \|e_y\| 0.88 m, max speed 11.1 m/s, max \|slip\| 0.63 rad |
| Planning | 54 / 68 ms (bicycle baseline 31 / 47 ms) |
| Compared with the bicycle baseline (entry 23) | baseline learned laps 37.9 → 36.0 s (best 36.0), max speed 12.8 m/s; v11 starts slower (38.9–39.8 s for the first three learned laps) and improves to 36.9 s by lap 15 |
| Verdict | success: the first world-model run to complete all 10 learned laps, still improving at the end. The speed-capped rerun was not needed |

## 31. Condition changes mid-session: v11 vs the bicycle model

| | |
|---|---|
| Setup | 15 laps (5 warm-up + 10 learned), change after 8 finished laps (laps 9–15 under the new condition), same seed and warm-up. v11 as entry 30; bicycle baseline with nominal parameters, not told about the change (`configs/fast-diag/known-varied-warmup.yaml`). Scenarios: `configs/scenarios/spielberg-grip-drop-lap8.yaml` (μ 1.05 → 0.7 everywhere), `spielberg-mass-lap8.yaml` (mass +30%), `spielberg-patch-lap8.yaml` (μ 0.6 on the corner at s = 85–115 m). `scripts/change_report.py`; results and condition plots in `results/change-scenarios/`. Recovery = first lap after the change within 1% of the median of the last three laps |

| Scenario | Controller | Laps 6–8 (before) | Laps 9–15 (after the change) | Crashes | Laps to recover |
|---|---|---|---|---|---|
| grip drop (μ 0.7) | v11 | 39.0, 39.0, 40.0 | 39.8, 39.4, 38.6, 38.2, 38.4, 38.2, 37.8 | none | 4 |
| grip drop (μ 0.7) | bicycle | 38.1, 38.6, 37.6 | crash in lap 9 (20.7 s in) | 1 | – |
| mass +30% (uninformative, entry 32) | v11 | 38.2, 38.9, 39.6 | 39.8, 39.5, 39.2, 38.7, 38.9, 38.1, 38.4 | none | 4 |
| mass +30% (uninformative, entry 32) | bicycle | 38.1, 39.2, 37.5 | 38.0, 36.8, 36.3, 36.8, 36.4, 35.8, 35.5 | none | 6 |
| patch (μ 0.6, one corner) | v11 | 39.9, 40.5, 40.4 | 39.3, 40.1, 39.0, crash in lap 12 (12.4 s in) | 1 | – |
| patch (μ 0.6, one corner) | bicycle | 38.3, 38.8, 37.3 | crash in lap 9 (13.2 s in) | 1 | – |

| | |
|---|---|
| Crash types | all three are spins into the wall at 7–8 m/s: yaw rate swinging to −3.5 to −4.7 rad/s, slip 0.5–0.9 rad, the car sliding across the track. Bicycle grip drop at s ≈ 178 m; both patch crashes on the patch (s ≈ 113–114 m) |
| v11 grip estimate (condition head on c) | no change: 1.01–1.06 per lap (true 1.05). Grip drop: falls within a second of the change but only to 0.96–0.99 per lap (true 0.70). Patch: 0.94 inside the patch vs 1.04 outside after the change (true 0.6 / 1.05; 1.00 on the same stretch before). Mass +30%: the mass read-out stays at 1.02–1.03 (true 1.30) and grip at 1.05–1.06 |
| Verdict | partial. Under a track-wide grip drop the world model keeps the car on the track and recovers in about 4 laps, where the nominal bicycle model spins out on the first lap after the change. With +30% mass the bicycle model is unaffected and faster. A sharp local patch defeats both (v11 survives three passes, then spins). The grip read-out moves in the right direction but far too little, and mass is not detected at all |

## 32. Does mass matter in this sim; oracle-c test

| | |
|---|---|
| Mass in the sim | the bridge applies `mass_ratio` to the mass m only; the yaw inertia I stays fixed. In the single-track model the tyre forces scale with the load, so mass cancels from the slip equation and the commands are accelerations; only the yaw equation keeps an m/I factor |
| Mass check | exact physics, same start states and applied controls (1,280 starts from v11's learned laps), mass ratio 1.0 vs 1.3: position difference median / p95 0.01 / 0.05 m at 0.5 s, 0.03 / 0.17 m at 1.0 s, 0.06 / 0.29 m at 1.5 s; yaw rate differs by up to 0.17 rad/s (p95), slip by 0.015 rad. For scale, grip 1.05 vs 0.70: 0.33 / 2.28 m at 1.5 s; v11's own 1.5 s error 0.26 / 1.36 m |
| Verdict on the mass scenario | uninformative: in this sim +30% mass changes the dynamics far less than the model's own error, so it cannot show adaptation (and v11's mass read-out staying at 1.02 is not evidence of a failure) |
| Oracle-c test | `scripts/oracle_condition_check.py`, results in `results/change-scenarios/oracle-c-*.json`. Oracle c for a grip value = mean filtered c over training steps (track + coverage) whose true grip is within ±0.05 of it, after a 2 s burn-in: for 0.70 from 61,812 steps (grip read-out 0.67), for 1.05 from 181,432 steps (read-out 1.05). On the grip-drop run (`change-grip-drop-v11`), 1.5 s rollouts from the true state with the applied controls, with v11's own filtered c vs c forced |

| Starts | c used | 1.5 s position median / p95 | 1.5 s e_y median / p95 |
|---|---|---|---|
| after the drop (1,335; true grip 0.70; own c reads 0.98) | own c | 0.37 / 1.29 m | 0.26 / 1.43 m |
| | oracle c for 0.70 (reads 0.67) | 0.55 / 1.66 m | 0.52 / 2.06 m |
| | c for 1.05 (wrong grip) | 0.42 / 1.46 m | 0.30 / 1.68 m |
| before the drop (583; true 1.05; own c reads 1.04) | own c | 0.38 / 1.39 m | 0.29 / 1.63 m |
| | oracle c for 1.05 | 0.31 / 1.29 m | 0.24 / 1.40 m |

| | |
|---|---|
| Findings | (1) v11's own predictions after the drop are as accurate as before it (e_y median 0.26 vs 0.29 m), although its grip read-out barely moves: the adaptation happens in the memory h and the fast latent d, not in c. (2) Forcing c to the value that reads the true grip makes predictions worse (0.26 → 0.52 m); the wrong-grip c is closer to the model's own (0.30 m). So the transition does not use c as a physical grip variable in the way the read-out suggests, and an average "grip 0.7" c is not consistent with the run's h |
| Verdict | a better grip estimate in c would not improve predictions, so the condition-estimate fix (looser prior on c, higher read-out weight, grip-change episodes, v12) was not done. Open question for the condition-aware safe set: tag laps with something the model actually uses (h, d, or predicted-vs-actual dynamics), not the c read-out |

## 33. Grip probe, fallback tag and the condition-aware safe set

Note from entry 32: c did not become a grip variable; v11's adaptation lives in the memory h and the fast latent d.
So the tag for the safe set is taken from (h, d), or from something model-agnostic, not from the c read-out.

| | |
|---|---|
| Grip probe | `sit_lmpc.world.rssm_grip` (`configs/train-grip-probe.yaml`, report in `results/condition-tags/`): SIT-LMPC's MLP (2 × 64) on frozen v11's (h, d), c left out, regressing the true friction; trained on the track + coverage train splits (625k steps), 10 epochs. Validation (88k steps): MAE 0.076, RMSE 0.117, R² 0.77, 74% of steps within 0.1 (track 71%, coverage 84%) |
| Probe on the replays (`scripts/condition_tag_check.py`, filtered with a 0.5 s time constant; plots in `results/condition-tags/`) | before the changes: 1.03 against 1.05, within 0.1 98% of the time. Grip drop (true 0.70): mean 0.92 after the drop, within 0.1 only 5% of the time and never for 2 s in a row. Patch (true 0.60): mean 0.91 inside the patch, lowest 0.70–0.74 per pass, never within 0.1. Better than the c read-out (0.98) but far from the target (within 0.1 within ~2 s), so the safe set uses the fallback tag |
| Fallback tag | the nominal bicycle model's 1.0 s position error ending at each step (from the true state 10 steps earlier with the applied controls; model-agnostic). Before the changes median 0.003 m (p95 0.18 m, the warm-up's 100 Hz pure-pursuit controls); after the grip drop median 0.20 m (p95 1.37 m); after the patch appears 0.24 m inside it vs 0.00004 m outside. In this sim the nominal model is exact at nominal grip, so this tag is close to an ideal detector here; on the car it would include the model's normal error |
| Condition-aware safe set | `planner.condition_tag: bicycle_error` (also `probe`, `readout`), `condition_mode: state`, `condition_threshold: 0.08` m, `condition_min_laps: 2`, `condition_fallback: newest` (`configs/world/rssm-latent-v11-condition.yaml`; off by default). Every stored state carries the tag (2 s centred average); per hull window each lap's tag is the median of its states there; keep laps within the threshold of the newest lap's value on that stretch, or of the current value (median of the last 2 s) when that is worse; if fewer than 2 laps match, the 2 newest. Per-step selection logged to `safe_set_trace.npy`, tags to `condition_tag.npy`; `scripts/safe_set_report.py`, plots in `results/condition-safe-set/` |

| Run (v11 + clearance, change after lap 8) | Laps 6–8 | After the change | Crash | Safe set on learned laps (laps kept / in window, fallback share) |
|---|---|---|---|---|
| patch, safe set off (entry 31) | 39.9, 40.5, 40.4 | 39.3, 40.1, 39.0, crash in lap 12 | spin on the patch | – |
| patch, safe set on | 38.2, 39.3, 39.3 | 40.1, crash in lap 10 (13.7 s in, s = 114 m) | spin on the patch: yaw rate −4.4 rad/s, slip 0.58 rad | 3.8/4 → 6.4/8; fallback 0–13% per lap |
| grip drop, safe set off (entry 31) | 39.0, 39.0, 40.0 | 39.8, 39.4, 38.6, 38.2, 38.4, 38.2, 37.8 | none | – |
| grip drop, safe set on | 39.6, 39.8, 39.2 | crash in lap 9 (3.6 s in, s = 32 m) | drift into the outer wall at low slip (0.11 rad), e_y 0.83 m | 3.8/4 → 6.2/7; fallback 0–12% per lap, none in lap 9 |

| | |
|---|---|
| What the filter did | patch: it recognised the patch stretch (reference tag 0.25–0.30 m) and kept only the 2–4 laps that had driven it, as designed, but the car still spun there. Grip drop: the live tag stayed at 0.07–0.09 m, just inside the threshold of the pre-change laps' value, so it kept 4–7 of 7 laps (all from before the drop) and changed little; the crash is a first-lap drift of the kind the model makes without the filter. Before the changes it already dropped some laps on some stretches (fallback 7–13% on laps 6 and 8), so the runs diverge from the "off" runs from lap 6 on |
| Verdict | failure: in one run per scenario the condition-aware safe set did not prevent the patch spin and the grip-drop run crashed where the "off" run did not. The filter itself behaved as designed on the patch; with one run each, the grip-drop crash cannot be blamed on it (it barely changed the selection). The safe set changes which terminal states are allowed; it does not fix the model's first-lap errors on a new surface |

## 34. Safe set v2 (live change detection) and change-aware value, 3 seeds

| | |
|---|---|
| What changed | `sit_lmpc.planners.change_monitor.ChangeMonitor` (`planner.change_detection`, off by default; `configs/world/rssm-latent-v11-change.yaml`). Every step the tag is the nominal bicycle model's 1.0 s position error. The track is split into 10 m stretches; each finished lap stores, per stretch, the median tag there. A change is flagged when the median of the last 1 s exceeds 0.05 m and 3× the current stretch's history. On detection every stretch turns conservative at once: the safe set uses the 2 slowest pre-change laps, warm-up first. When the car leaves a stretch it is relaxed if its tag matches its history, otherwise marked changed and kept conservative until 2 laps driven fully after the change exist, then uses those. Value option B (`planner.change_value`): in conservative / new windows the planner's value is the time-to-go of the same laps, interpolated by track position, instead of the learned value function. Per step logged to `change_trace.npy`, events to `change_events.json`; `scripts/change_v2_report.py`, results in `results/change-v2/` |
| Runs | Spielberg, 15 laps (5 warm-up + 10 learned), change after lap 8, seeds 1–3 (`SEED` in `scripts/run_change_experiment.sh`). Bicycle (nominal parameters) and v11 with the safe set off: seed 1 from entry 31, seeds 2–3 new. v11 + safe set v2 + value B: all new. No-change runs with v2 on |

| Scenario | Controller | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| grip drop (μ 0.7) | bicycle | spin, lap 9 (s 178) | spin, lap 9 (s 179) | 40.5, 40.1, 39.2, 39.3, wall lap 13 (s 321) |
| grip drop (μ 0.7) | v11, safe set off | 39.8 → 37.8, no crash, recovered in 4 laps | wall, lap 7 (before the change, s 35) | 39.0, 39.0, spin lap 11 (s 181) |
| grip drop (μ 0.7) | v11, v2 + value B | spin, lap 7 (before the change, s 110) | spin lap 9 (s 40), 3.1 s after detection | spin lap 9 (s 117), 11.5 s after detection |
| patch (μ 0.6) | bicycle | spin, lap 9 (s 114) | 40.4, 41.0, 39.7, spin lap 12 (s 112) | 40.3, 39.5, spin lap 11 (s 113) |
| patch (μ 0.6) | v11, safe set off | 39.3, 40.1, 39.0, spin lap 12 (s 113) | wall, lap 7 (before the change, s 35) | 38.7, spin lap 10 (s 113) |
| patch (μ 0.6) | v11, v2 + value B | spin lap 9 (s 117), 2.9 s after detection | wall lap 9 (s 35, before reaching the patch, not detected) | spin, lap 8 (before the change, s 110) |
| no change | v11, v2 + value B | 40.3, 38.7, 38.3, 37.1, 38.1, 38.0, wall lap 15 (s 34) | 41.5, 40.9, 39.8, 38.9, 38.1, 36.9, 37.1, no crash | spin, lap 8 (s 109) |

| Summary across seeds | Crashed / 3 | Crash types | Crashes before the change | Detection delay | False detections |
|---|---|---|---|---|---|
| grip drop, bicycle | 3 | spin, spin, wall | 0 | – | – |
| grip drop, v11 off | 2 | wall, spin | 1 | – | – |
| grip drop, v11 v2 | 3 | spin ×3 | 1 | 1.1 s, 1.1 s (seed 1 crashed before) | 0 |
| patch, bicycle | 3 | spin ×3 | 0 | – | – |
| patch, v11 off | 3 | spin, wall, spin | 1 | – | – |
| patch, v11 v2 | 3 | spin, wall, spin | 1 | 9.9 s after the change = about 1 s after entering the patch (seed 1); not reached (2, 3) | 0 |
| no change, v11 v2 | 2 | wall, spin | – | – | 0 in all 3 seeds (v2 never activated) |

| | |
|---|---|
| Detection | works: 1.1 s after the grip drop (both seeds that reached it), about 1 s after entering the patch, no false detections in any of the 9 v2 runs (36 completed learned laps, every one before a change or in a no-change run). On a detection every stretch went conservative and the changed stretches were marked within one pass (grip drop: stretches 1–11 changed in seed 3, stretch 2 relaxed; patch: stretches 9–10 changed) |
| Reaction | did not slow the car. After detection the window was conservative and value B was in use on 100% of steps, yet speed stayed the same or rose (seed 2 grip drop 9.3 → 9.7 → 10.3 m/s at the crash; seed 3 8.9 → 9.4 → 9.0; patch seed 1 8.9 → 8.6 → 9.1) and the car spun 2.9–11.5 s after detection. The "slowest stored laps" are the warm-up laps, which drove at the 9 m/s cap (average 9.2 m/s), i.e. no slower than the learned laps, so the conservative terminal set and its time-to-go did not ask for less speed |
| Baseline variance | v11 is less robust than its single seed-1 run suggested: on normal track it crashed before the change in 1 of 3 seeds of each scenario with the safe set off, and in 1–2 of 3 v2 runs (v2 was inactive there, so not caused by it). Crashes cluster at s ≈ 34 m (wall) and s ≈ 109–117 m (spin, the corner the patch sits on). Same-seed runs are not exactly repeatable (the three v2 seed-1 runs share the first 8 laps in principle, but one crashed in lap 7) |
| Verdict | failure. Detection is fast and clean, but reacting by switching to "slower" stored laps cannot work when no stored lap is slower than the current driving at the new grip. A reaction that actually reduces speed is needed: e.g. a speed cap scaled by the detected change, scaling the conservative laps' speeds down, or a conservative warm-up lap kept for this purpose. Option A (value on (state, tag)) was not tried |

## 35. Baseline robustness, run divergence and v11's repeat crash spots

| | |
|---|---|
| Bicycle baseline, no change, 3 seeds (`change-none-bicycle-s1..3`) | seed 1: 38.0, 39.8, 37.2, 37.3, 37.2, 37.0, 37.3, 37.1, 37.1, 37.5 s, no crash. Seed 2: 38.7, 39.2, 38.3, 39.4, 38.6, 38.6, 38.6, 38.7, 38.1, 38.8 s, no crash. Seed 3: 38.4, 38.4, then a spin in lap 8 at s ≈ 110 m, hit point (−75.11, 52.00), slip 0.40 rad at 7.3 m/s. The exact-physics planner crashes at the same corner as v11 |
| Run divergence (same seed) | runs with the same seed are identical through the 5 warm-up laps and separate right after the first learned step: v11 runs more than 1 cm apart 0.3–0.5 s into the first learned lap (0.2–7.6 m apart 10 s later, 16 same-seed pairs); bicycle runs 1.4 s in (1.0–2.6 m 10 s later, 3 pairs). Most likely GPU non-determinism in the value-function training right after the warm-up. Each run from the first learned lap on is effectively an independent sample; "seed" only fixes the warm-up |
| v11's repeat crash spots | from the bridge's collision log: four crashes at exactly (−34.0, −8.3) (s ≈ 35 m, the inside corner of a right-hand kink) and three at (−75.1, 52.0) (s ≈ 110 m), within 5 cm. `scripts/crash_spot_check.py`, `results/crash-spots/` |
| What happens there | at s ≈ 35 m the car is near the centre line (e_y ≤ 0.3 m) at 9–11 m/s, the planner steers in (up to −0.2 rad, after accelerating hard to 11 m/s in one case), the rear steps out (yaw rate to −3.2…−5.8 rad/s, slip 0.3–0.6 rad) and the car rotates into the inside wall within ~0.5 s (heading error up to −1.2 rad) |
| Model error, speed or line? | not model error: rolled out from the true state with the applied controls, v11 predicts the slide almost exactly (slip 0.24–0.38 vs actual 0.27–0.41 rad, yaw rate −3.0…−5.6 vs −3.2…−5.8 rad/s, position error 0.04–0.25 m up to 1.1 s ahead). The crash approaches are somewhat faster than clean passes over the same stretch (9.3–10.6 vs 9.0 m/s mean). The planner chooses a sharp turn-in that it correctly predicts will slide: nothing penalises the slide (slip cost off) and the track-limit check only looks at the reference point, not the rotated car body |
| v11 risk head | `configs/train-rssm-v11-risk.yaml`, report in `results/risk-head-v11/`: retrained on v11's latent, adding 10 v11 closed-loop runs; AUC 0.94 validation, 0.92 test; held-out crash runs AUC 0.95 (s ≈ 110 m spin, patch spin), 0.72 (s ≈ 34 m wall) |

| No-change runs, 3 seeds | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| (a) risk head, weight 5, counted in the final ranking (`rssm_risk_rank`) | crash in lap 6, 46 s into the lap, stalled and rolled backwards (slip −3 rad) | crash in lap 6 | crash in lap 6, 55 s into the lap, at 9.7 m/s |
| (b) speed cap 10 m/s | 40.2, 40.8, 40.7, 40.9, 40.3 s, crash in lap 11 at s ≈ 35 m | 40.0, 40.4, 40.5, 40.2 s, crash in lap 10 at s ≈ 174 m (yaw rate −2.8 rad/s, slip 0.33 rad at 8.2 m/s) | 40.5, 41.1, 41.4, 40.8, 39.7 s, crash in lap 11 at s ≈ 35 m |

| | |
|---|---|
| Findings | (a) makes things worse: the risk cost was active on 100% of the chosen plans, the planner crawled (lap mean speed 5.3–5.7 m/s, laps of 46–55 s) and crashed in the first learned lap in all three runs. (b) costs about 2 s per lap (40–41 s) and still crashed in all three, twice at the same s ≈ 35 m kink: the snap happens at 8–9 m/s there, so speed is not the trigger |
| Verdict | failure: neither gets the no-change crashes to 0/3, so step 3 (the speed reaction on a detected change) was not run. The speed reaction is built (`planner.change_speed`, `change_speed_factor`, `change_speed_step`; per-stretch caps in `ChangeMonitor`, unit-tested) but untested in closed loop. Likely next fixes for the base crashes: a cost on predicted slip / yaw-rate overshoot near walls, or a track-limit check on the car body's corners instead of its reference point |

## 36. Root cause: regression or luck?

### Which code produced the old results

| Old result | Run | Code | Config |
|---|---|---|---|
| v3 "14–15/15 laps" (3 of 4 runs had a crash, see below) | `world-nochange` (00:40 EST 11-17), `change-friction-world`, `patch-world`, `change-friction-world-condition` (11-17) | `ros-sim` at `d47d98b` / `b93d4d7` (identical `src/` and configs apart from one scenario file) | `results/Spielberg/world-nochange/config.yaml` on `ros-sim`: `belief_update: encoder` (v3), `sit_lmpc_mode: false`, warm-up `speed_scale` 0.6, `max_speed` 25 |
| bicycle 38.3 → 36.6 s | `known-varied-long` (11-23) | `world-model` at `9e9707a` | `configs/fast-diag/known-varied-warmup.yaml` (saved as `configs/rootcause/bicycle-known-varied-long.yaml`) |

**The v3 runs were not clean.** `sit_lmpc_mode: false` means "allow learning and retry": a crash is one failed attempt and the run goes on. `world-nochange` crashed in attempt 15, `patch-world` in attempt 9 (then 6 more laps), `change-friction-world-condition` in attempt 12; only `change-friction-world` had no crash. Every newer run uses `sit_lmpc_mode: true`, which ends the run at the first crash, so the same behaviour now looks like "crashed in lap N".

### Changes since those commits (planner, runner, safe set, config)

| Change | Commit | On by default? | Affects the bicycle baseline? |
|---|---|---|---|
| Hull copied with `device_put` and chunked on the host (no recompiles while driving) | `0a26c2b` (also `ros-sim` `99dff48`) | yes | numerically the same; only speed |
| World model in the planner (RSSM backend, variants, `apply_rssm`) | `fa13e62` | only with `backend: rssm` | no |
| Per-state condition tags, state-mode condition filter | `3fd7e56` / `9e9707a` | no (`condition_threshold` 0) | no |
| LiDAR clearance check | `9e9707a` | no (weight 0; RSSM only) | no |
| Safety costs outside the lambda grid, counted in the final pick; speed cap and slip cost | `898cd56` | slip cost on from `898cd56` to `4402c6b`, off since; speed cap off | only while the slip cost was on (`known-varied-current`); at HEAD all safety costs are zero, so the reward and the final ranking are unchanged |
| Crash-risk head (soft or ranked) | `1b2224d`, `7b5a865` | no | no |
| Track-limit margin | `4402c6b` | no (`boundary_margin` null) | no |
| Grip probe, bicycle-error tag, newest-lap fallback | `c03ad4d` | no | no |
| Change monitor, conservative safe set, value option B, per-stretch speed caps | `d44c313`, `7b5a865` | no (`change_detection` false, `change_speed` null) | no |
| `SEED` option in the launcher | `d44c313` | – | no |
| Warm-up: `speed_scale` 1.0, `max_speed` 9 (varied warm-up) | config, from `0eb1261` | used by all newer runs | config, not code; v3's old runs used scale 0.6 / cap 25 |
| Steering source, action-rate penalty, horizon | none: the saved run configs agree on every other planner setting (horizon 15, 1024 samples, action-rate penalty, lambda grid, safe-set size, etc.) | – | – |

### Reproduction (no change, 15 laps, Spielberg)

Old code in detached git worktrees (`~/sim_ws/wt-old-v3` at `b93d4d7`, `~/sim_ws/wt-old-bicycle` at `9e9707a`), each run with that checkout's `src/` and scripts and the original saved run config (`scripts/run_with_code.sh`); the sim bridge (`f1tenth_gym_ros`) is unchanged since 11-16. The old v3 run's first lap reproduces the original exactly (38.4 s, 8.894 m/s). Results in `results/Spielberg/rootcause/` (`*-r`; the first attempt failed to start because of a launcher bug and is left there) and `results/rootcause/report.json` (`scripts/rootcause_report.py`).

| Code | Model | Seed | Crashes (attempt, s, bridge hit point) | Learned laps (s) |
|---|---|---|---|---|
| old `b93d4d7` | v3 | 1 | attempt 6, s 34.3, (−33.86, −8.24) | 37.6, 36.8, 37.6, 37.5, 36.8, 36.6, 36.2, 36.3, 36.0 |
| old `b93d4d7` | v3 | 2 | attempt 10, s 34.9, (−33.96, −8.28) | 38.3, 36.9, 37.4, 37.8, 37.8, 37.0, 36.5, 36.5, 35.7 |
| old `b93d4d7` | v3 | 3 | attempt 7, s 109.1, (−75.06, 51.84) | 37.2, 37.2, 37.6, 38.3, 37.0, 37.8, 37.5, 35.6, 36.4 |
| HEAD | v3 | 1 | attempt 6, s 14.5, (−14.88, −3.07) | 37.5, 37.6, 37.0, 37.5, 36.2, 36.9, 36.0, 35.3, 35.9 |
| HEAD | v3 | 2 | attempts 11 and 12, s 34.1, (−33.77, −8.25) twice; ended "speed too low" | 36.5, 38.8, 36.9, 37.0, 36.7 |
| HEAD | v3 | 3 | attempt 6, s 34.9, (−33.96, −8.27) | 38.1, 38.5, 37.1, 36.9, 37.2, 36.5, 36.4, 36.7, 35.8 |
| old `9e9707a` | bicycle | 1 | none | 37.7, 39.1, 37.3, 39.4, 38.4, 37.3, 37.3, 37.3, 37.3, 37.4 |
| old `9e9707a` | bicycle | 2 | none | 37.7, 38.2, 37.5, 38.9, 38.3, 38.7, 38.0, 37.7, 38.1, 38.4 |
| old `9e9707a` | bicycle | 3 | none | 38.1, 38.0, 37.8, 39.8, 38.6, 38.7, 38.3, 38.6, 38.1, 38.3 |
| old `9e9707a` | bicycle | original (`known-varied-long`) | none | 37.9 → 36.6 |
| HEAD | bicycle | 1 (`change-none-bicycle-s1`) | none | 38.0, 39.8, 37.2, 37.3, 37.2, 37.0, 37.3, 37.1, 37.1, 37.5 |
| HEAD | bicycle | 2 | none | 38.7, 39.2, 38.3, 39.4, 38.6, 38.6, 38.6, 38.7, 38.1, 38.8 |
| HEAD | bicycle | 3 | attempt 8, s 109.5, (−75.11, 52.00) | 38.4, 38.4 |
| `4402c6b` (= HEAD for the bicycle) | bicycle | 1 (`known-varied-slipoff`) | none | 37.9 → 36.0 |

| Crash rate | Runs | Runs with a crash | Crashes per learned attempt |
|---|---|---|---|
| v3, old code | 3 | 3 | 3 / 30 (10%) |
| v3, HEAD | 3 | 3 | 5 / 28 (18%) |
| bicycle, old code | 4 | 0 | 0 / 40 |
| bicycle, HEAD-equivalent | 4 | 1 | 1 / 33 (3%) |

### Interpretation

| | |
|---|---|
| v3 | **not a regression.** The old code crashes in every run too, at the same two spots as v11 and the bicycle model: the inside corner at s ≈ 34 m (−34.0, −8.3) and the corner at s ≈ 109 m (−75.1, 52.0). HEAD shows a few more crashes (5 vs 3 in three runs each), which with these numbers is within run-to-run variation; with `sit_lmpc_mode: false` the old runs' "14/15 laps" simply hid them |
| bicycle | **no evidence of a regression.** 0 crashes in 4 old-code runs and 1 in 4 HEAD-equivalent runs, at the s ≈ 109 m corner. At HEAD every added feature is off or zero for the bicycle baseline, so the code path is the same as the old one up to floating-point order; one crash in four runs cannot be separated from chance (and same-seed runs diverge right after the warm-up, entry 35) |
| What it means | the earlier "clean" results were partly luck (single runs, not reproducible after the warm-up) and, for v3, partly crashes that did not end the run. The two crash spots are a property of this track and this planner (SIT-LMPC sampling, the value function and the warm-up line), shared by the bicycle model, v3 and v11; the world-model crashes there are not caused by the world model |
| Not done | no bisect (no regression found); no fix. More seeds of the bicycle baseline (old vs HEAD) would tighten the bicycle comparison |

## 37. Body-footprint track limits; change scenarios with the footprint

### What changed

Before this, the planner checked track limits on the centre point of the rear axle only (`e_y` of the state). The body sticks out about 0.3 m on each side and 0.28 m to the front, more when the car is at an angle to the track, so plans that kept the centre point inside could still put a corner of the body on the wall.

- `footprint_limits` (default **on**, all controllers): the planner takes the four body corners from (x, y, yaw) and uses the worst corner's offset in place of `e_y` for every track-limit check: the sampling cost and the final pick (same priority as before: boundary × 1000). Corner offset: `e_y + dx·sin(ψ_e) + dy·cos(ψ_e)`. Size from the vehicle parameters (0.58 × 0.31 m), or `footprint_length` / `footprint_width`; `footprint_offset` −0.0065 m (body centre relative to the state point). The margin option (entry 26) is applied on top. Code: `footprint_corners` / `with_footprint` in `planners/lmppi.py`; unit test in `tests/unit/test_rssm.py`.
- `safety_heading_limit` (default **off**): a safety cost on |heading relative to the track| above the limit (ramps from 0 to `safety_heading_weight` over 0.2 rad), added with the other safety costs outside the λ grid. Built and tested; not needed in the runs below (v3 and v11 did not crash at s ≈ 34 m or s ≈ 109 m with the footprint on).
- v3 (`gaussian_bnn`) keeps its parameters as a 2-D array, so the size lookup fell back wrongly at first (shape error at the first learned step); fixed by using 0.58 × 0.31 m when the parameters are not one flat vector.

### No-change runs, 5 seeds, 15 laps (5 warm-up + 10 learned)

Same warm-up as entry 36 (varied warm-up, scale 1.0, cap 9 m/s), LiDAR bridge, Spielberg. Bicycle and v11 (+ clearance) use `sit_lmpc_mode: true` (stop at the first crash); v3 needs `false` (retry after a crash), so its crash count is per learned attempt. Body |e_y| = worst body corner; heading = heading relative to the track; both over learned laps up to contact. Planning time on learned steps. Results `results/footprint/no-change.json` (`scripts/footprint_report.py`).

| Run | Crash | Max body \|e_y\| (p99) m | Max \|heading\| (p99) rad | Planning p50 / p95 ms | Learned laps (s) |
|---|---|---|---|---|---|
| bicycle s1 | no | 0.81 (0.72) | 1.23 (0.68) | 41.0 / 55.5 | 38.4 39.4 37.4 38.5 38.8 38.1 37.7 37.7 37.7 38.1 |
| bicycle s2 | no | 0.84 (0.74) | 1.18 (0.69) | 40.9 / 57.4 | 38.3 38.9 38.2 38.9 38.0 38.6 38.9 38.0 38.1 39.0 |
| bicycle s3 | no | 0.82 (0.73) | 1.18 (0.70) | 41.0 / 55.7 | 38.0 39.0 38.0 39.3 38.5 38.2 38.1 37.7 37.8 37.8 |
| bicycle s4 | no | 0.84 (0.74) | 1.17 (0.69) | 41.2 / 59.3 | 39.9 40.4 42.4 42.9 41.5 38.6 38.5 38.4 38.0 37.8 |
| bicycle s5 | no | 0.81 (0.73) | 1.11 (0.68) | 41.5 / 59.5 | 39.6 41.2 41.9 42.8 40.8 39.3 39.0 38.2 38.0 38.1 |
| v11 s1 | no | 0.92 (0.79) | 1.39 (0.77) | 56.7 / 70.9 | 38.4 39.8 39.7 40.0 39.4 38.5 38.7 38.0 38.9 37.9 |
| v11 s2 | no | 0.95 (0.82) | 1.38 (0.74) | 56.7 / 71.6 | 37.0 37.8 36.9 37.8 37.5 36.9 36.3 38.0 36.4 36.7 |
| v11 s3 | no | 0.92 (0.77) | 1.31 (0.77) | 57.4 / 72.5 | 40.7 40.2 39.9 39.4 39.0 38.2 38.1 37.7 37.6 38.0 |
| v11 s4 | no | 0.87 (0.77) | 1.38 (0.75) | 56.8 / 72.1 | 39.7 41.0 41.5 41.1 40.1 38.6 38.1 36.9 36.8 36.1 |
| v11 s5 | no | 0.90 (0.79) | 1.46 (0.73) | 56.9 / 71.9 | 39.7 40.3 40.9 40.4 40.0 38.5 38.6 37.8 38.4 38.0 |
| v3 s1 | no | 1.03 (0.74) | 0.93 (0.59) | 137.9 / 178.2 | 37.7 38.4 38.3 38.2 37.0 36.9 38.1 38.4 38.2 36.8 |
| v3 s2 | **wall, s = 95.5 m**, attempt 6, hit (−65.70, 42.39); run went on | 0.94 (0.76) | 1.21 (0.61) | 142.9 / 179.8 | 37.3 38.2 37.3 37.4 37.3 38.0 37.1 37.4 38.3 |
| v3 s3 | no | 0.90 (0.77) | 0.90 (0.60) | 145.3 / 183.4 | 38.3 38.0 38.3 37.9 37.9 37.8 36.7 37.8 37.3 36.6 |
| v3 s4 | no | 1.12 (0.79) | 1.22 (0.62) | 143.5 / 184.2 | 37.6 38.5 37.9 37.6 36.6 36.9 37.1 36.3 35.1 36.3 |
| v3 s5 | no | 0.90 (0.77) | 1.09 (0.62) | 144.1 / 184.1 | 37.9 38.2 38.0 37.2 37.3 36.6 37.3 36.1 36.5 36.7 |

| Crashes per learned attempt | Without footprint (entry 36) | With footprint |
|---|---|---|
| bicycle | 0/40 old code, 1/33 (3%) HEAD | **0/50** |
| v3 | 3/30 (10%) old code, 5/28 (18%) HEAD | **1/50 (2%)**, at s = 95.5 m (a spot not seen before) |
| v11 + clearance | crashed at s ≈ 109 m in earlier runs (entry 35) | **0/50** |

The two known crash spots (s ≈ 34 m and s ≈ 109 m) are gone for all three models. Body p99 |e_y| for the bicycle dropped from about 0.88 m to 0.72–0.74 m. Lap times are in the same range as before (learned laps 35–43 s, ending at 36–39 s), so the check costs no clear lap time.

**Planning time.** p50 / p95 on learned steps, before (entry 36 runs) → with footprint: bicycle 33–34 / 47–57 → 41–42 / 56–60 ms (+8 ms); v11 54–57 / 68–73 → 57 / 71–72 ms (+2 ms); v3 133–138 / 173–177 → 138–145 / 178–184 ms (+6 ms). This is more than the few extra array operations should cost: `with_footprint` runs as separate GPU calls outside the compiled rollout cost. Moving it inside the jitted function should bring it close to zero (not done). v3 was already over the 100 ms control step before this; the lockstep sim waits for it, but on the car it would not.

**Invalid runs left in `results/Spielberg/`:** `fp-none-v3-s*-r` (seed passed as the lap count by mistake, runs stopped after "seed" laps); the first `fp-none-v3-s*` attempts (shape error above). The valid v3 runs are `fp-none-v3-s*-r2`.

### Change scenarios, 3 seeds, 15 laps, change after lap 8

Only run after v11 had no crashes in the no-change runs. Grip drop: μ 1.05 → 0.7 on the whole track. Patch: μ 0.6 on s = 85–115 m. Controllers: bicycle + footprint; v11 + footprint, no reaction; v11 + footprint + full reaction (`configs/world/rssm-latent-v11-react.yaml`: change detection, conservative safe set, value option B, per-stretch speed caps at 0.8 of the pre-change speed, raised by 0.05 per clean lap). Results `results/footprint/change.json` (`scripts/fp_change_report.py`).

"Laps to recover" = first lap after the change within 1% of the median of the last three laps. Detection delay for the grip drop is from the start of lap 9 (the change starts there); for the patch it is from the car entering the patch (s = 85 m), since the patch is 11 s into the lap. Speed: mean over 2 s before and 2 s after the first detection, and on the same stretch one lap earlier.

| Scenario | Controller | Seed | Crash | Lap times after the change (s) | Laps to recover | Detection delay | Speed before → after detection (prev. lap same stretch) m/s | False detections |
|---|---|---|---|---|---|---|---|---|
| grip drop | bicycle | 1 | no | 40.2 41.8 40.5 40.2 39.7 39.3 39.1 | 6 | – | – | – |
| grip drop | bicycle | 2 | no | 39.6 39.5 40.0 39.3 39.4 40.0 40.4 | 1 | – | – | – |
| grip drop | bicycle | 3 | no | 40.0 39.7 40.0 39.5 39.4 38.9 39.4 | 2 | – | – | – |
| grip drop | v11 | 1 | no | 39.2 38.9 38.5 38.8 39.2 38.6 39.5 | 1 | – | – | – |
| grip drop | v11 | 2 | no | 38.5 38.8 39.1 39.8 39.5 39.7 38.8 | 1 | – | – | – |
| grip drop | v11 | 3 | no | 39.5 39.3 39.2 38.5 37.7 38.0 37.7 | 5 | – | – | – |
| grip drop | v11 + reaction | 1 | no | 66.9 69.0 67.9 66.7 54.6 49.8 48.7 | 6 | 3.0 s | 8.65 → 6.06 (7.89) | 0 |
| grip drop | v11 + reaction | 2 | no | 66.0 68.2 67.4 65.4 52.9 48.7 47.5 | 6 | 1.9 s | 9.50 → 6.36 (9.72) | 0 |
| grip drop | v11 + reaction | 3 | no | 66.7 70.9 68.8 67.6 52.8 50.0 47.5 | 6 | 3.7 s | 8.09 → 6.26 (7.85) | 0 |
| patch | bicycle | 1 | no | 39.2 39.4 38.7 39.4 39.3 38.6 38.9 | 1 | – | – | – |
| patch | bicycle | 2 | no | 40.8 40.1 39.1 38.2 38.2 37.8 37.9 | 4 | – | – | – |
| patch | bicycle | 3 | no | 38.6 38.6 39.7 39.5 39.6 39.6 38.5 | 1 | – | – | – |
| patch | v11 | 1 | no | 40.1 39.4 38.2 38.3 38.4 38.1 37.8 | 3 | – | – | – |
| patch | v11 | 2 | no | 39.1 49.4 38.4 40.7 38.7 40.5 38.7 | 3 | – | – | – |
| patch | v11 | 3 | no | 39.8 39.0 38.8 38.3 38.4 39.0 38.4 | 4 | – | – | – |
| patch | v11 + reaction | 1 | no | 61.8 70.3 70.7 69.3 67.9 67.3 67.5 | (1)* | 0.5 s | 8.55 → 5.13 (8.33) | 0 |
| patch | v11 + reaction | 2 | no | 60.7 69.4 67.8 69.8 67.3 66.7 67.0 | (1)* | 0.4 s | 8.21 → 5.37 (8.55) | 0 |
| patch | v11 + reaction | 3 | no | 61.5 71.1 71.2 70.2 69.3 69.2 68.8 | (1)* | 0.8 s | 8.73 → 5.64 (7.92) | 0 |

\* The metric reads "1" only because the laps never get faster again; the car stays at about 68 s per lap to the end.

| Scenario | Controller | Crashes | Mean lap after change (s) | Laps to recover | Detection delay | False detections |
|---|---|---|---|---|---|---|
| grip drop | bicycle + footprint | 0/3 | 39.8 | 6, 1, 2 | – | – |
| grip drop | v11 + footprint | 0/3 | 38.9 | 1, 1, 5 | – | – |
| grip drop | v11 + footprint + reaction | 0/3 | 60.2 | 6, 6, 6 | 1.9–3.7 s | 0 |
| patch | bicycle + footprint | 0/3 | 39.0 | 1, 4, 1 | – | – |
| patch | v11 + footprint | 0/3 | 39.4 | 3, 3, 4 | – | – |
| patch | v11 + footprint + reaction | 0/3 | 67.9 | never | 0.4–0.8 s after entering the patch | 0 |

### What the change runs show

- **No crashes in any of the 18 runs.** With the footprint on, v11 handles both changes with no reaction at all and is slightly faster than the bicycle model after the grip drop (38.9 vs 39.8 s).
- **Detection works:** every change was found (2–4 s into the grip drop, under 1 s after entering the patch), and no run had a detection before the change.
- **The speed reaction is far too strong.** It costs 20–30 s per lap and buys nothing here, since the runs without it did not crash. Two causes, from `change_trace.npy` / `change_events.json`:
  1. **The whole track gets capped, not the changed part.** Each "detected again" (most laps) puts every stretch that is not marked "changed" back into conservative mode and caps it again. In the patch runs 99–100% of the lap is capped from lap 10 on, though only 3 of the 35 stretches (the patch) are marked changed. The patch runs never recover; the grip-drop runs start to recover only once all stretches are marked changed (lap 13 on) and the caps rise by 0.05 per lap.
  2. **The car drives about 2 m/s under the cap.** Mean cap ≈ 7.0 m/s, mean speed ≈ 5.0 m/s on capped stretches. The cap cost itself is zero below the cap, so something else slows the car further: most likely the safety cost's × 1000 weight in the final pick (any sampled step above the cap in the 1.5 s horizon ranks the plan last, so the planner keeps a wide gap), with the conservative safe set and value B on top. Not tested.
- **Bug found and fixed:** the 6 reaction runs first stopped at the end of lap 9 (`KeyError: 'stretch_cap'`). The end-of-lap safety summary looked up each safety term in every step, but the speed-cap term only exists after a detection. Fix in `experiments/runner.py` (`safety_summary`, missing term = 0); planner unchanged. The failed runs are kept as `fp-*-v11-react-s*-keyerror`; patch seed 3 already ran with the fix.

### Not done

Compile the footprint check into the rollout cost (planning time). Fix the reaction: limit re-detection to the stretch where it fires instead of the whole track, and test caps with the safety weight lowered to separate the two slowdown causes. The heading option was not run (not needed).

## 38. Compiled footprint check; speed-reaction fix attempt; harsher change scenarios

### 1. Footprint check inside the compiled planner functions

Entry 37's footprint check ran as separate GPU calls before the sampling cost and the final pick. Now the corner offsets and the margin are passed in as one argument (`boundary`, built once in `LMPPI.__init__`) and applied inside the jitted `propose*` / `finish_candidates` (`boundary_view` in `planners/lmppi.py`); same maths, no extra pass. One 8-lap run each (5 warm-up + 3 learned), seed 1, no change (`fpjit-*`, `results/footprint/jit-timing.json`):

| Controller | No footprint (entry 36) p50 / p95 ms | Footprint, separate pass (entry 37) | Footprint, compiled |
|---|---|---|---|
| bicycle | 33–34 / 47–57 | 41–42 / 56–60 | **33.3 / 55.1** |
| v11 | 54–57 / 68–73 | 57 / 71–72 | **54.0 / 68.9** |
| v3 | 133–138 / 173–177 | 138–145 / 178–184 | **123.1 / 159.6** |

The check now costs no measurable planning time. The v3 timing run spun at s = 121.8 m in its first learned lap (1 crash in 3 learned attempts); a short run, so not counted against entry 37's 1/50.

### 2. Speed reaction: one fix attempt

**Cap scope** (`change_cap_scope: local`, the new default; `track` = the old behaviour). A cap is set only on a stretch whose live tag shows a change: the stretch where the detection fires, and each stretch marked "changed" when the car leaves it. A repeat detection caps only its own stretch and does not touch stretches that have relaxed. The whole track is capped (except stretches relaxed since the change) once more than `change_global_share` (0.5) of the stretches driven since the change look changed and at least `change_global_min` (8, i.e. 80 m) have been driven. With fewer, the patch looked track-wide: the first 3 stretches driven after a patch detection are the patch itself (3/3 changed); caught in a unit test before any run. `ChangeMonitor(scope=..., global_share=..., global_min=...)`; test `test_change_monitor_local_caps`.

**Check runs** (v11 + reaction, local caps, μ 0.7 grip drop and μ 0.6 patch, seed 1, `rx-check-*`): **both spun in lap 9**, the first lap after the change, at s ≈ 113 m (the corner where entry 34's runs spun). Every footprint run of these scenarios survived in entry 37, including the old track-wide reaction (6/6) and v11 without reaction (6/6).
- Grip drop: the track-wide cap fired after 80 m (s ≈ 100 m), 1.3 s before the corner, with the car at 10.5 m/s, faster than before the change: the conservative safe set and value B do not slow the car (as in entry 34). The old reaction capped every stretch at the detection itself, which is what slowed it down in time.
- Patch: caps landed on stretches 9 and 10 as they were judged; stretch 11, where the car spun, was not capped yet.

**Why the car drives about 2 m/s under the cap.** New option `change_breakdown` logs, every learned step, the final-pick cost terms of every candidate (`candidate_breakdown.npz`; `scripts/breakdown_report.py`). Diagnostic run with the old track-wide caps, μ 0.7, 12 laps (`rx-diag-track-grip-drop-s1`), 2730 capped steps:

| | Mean |
|---|---|
| cap at the car's position | 7.10 m/s |
| actual speed | 5.03 m/s |
| chosen plan, mean planned speed | 5.73 m/s |
| fastest candidate inside the track limits, mean planned speed | 6.66 m/s |
| chosen plan, peak speed minus cap | −0.77 m/s |
| fastest candidate, peak speed minus cap | +0.81 m/s |
| chosen plan = fastest feasible candidate | 0.5% of steps |
| the cap term alone makes the fastest candidate lose | 95% of the steps where they differ |
| hull (safe-set) term: chosen / fastest | 1.43 / 0.30 (the fastest plan is better on the safe set) |
| footprint, clearance terms | 0 for both |

The 2.1 m/s gap splits into: about 0.8 m/s from the final pick (any cap overshoot counted ×1000, like a wall hit, so the chosen plan's peak stays 0.8 m/s under the cap), about 0.6 m/s from a plan's mean speed being below its peak, and about 0.7 m/s from the car not reaching the speed the model plans on the low-grip track. The first part is a weighting artifact. Fix: `change_cap_rank_weight` (default 1.0) counts the cap cost at weight 1 in the final pick, on the scale of the safe-set violation, instead of ×1000. **It did not help:** the same run with the fix (`rx-diag-track-rank1-grip-drop-s1`) drove 5.19 m/s under a 7.17 m/s cap (laps 67.6, 68.1, 66.4, 65.6 s vs 68.0, 69.9, 69.3, 68.2 s), chosen = fastest on 0.8% of steps. The fastest candidate's cap cost (≈ 4.4, summed over the 15 horizon steps) still outweighs its safe-set advantage (≈ 1.1). And even the fastest candidate averages 6.8 m/s, under the cap, so the per-step cap cost in the sampling (up to 1 per step against a value scaled to [0, 1]) already keeps every candidate slow. Both are weighting problems; not fixed further (one attempt).

### 3. Harsher change scenarios, 3 seeds, change after lap 8, 15 laps, footprint on

Grip drop μ 1.05 → 0.55 on the whole track (`spielberg-grip-drop055-lap8.yaml`); patch μ 0.45 on s = 85–115 m (`spielberg-patch045-lap8.yaml`). Controllers: bicycle; v11 without reaction; v11 + fixed reaction (local caps, cap rank weight 1, everything else as entry 37). Results `results/footprint/harsh-change.json` (`scripts/harsh_change_report.py`); run names `hc-*`.

Near-miss measures, per lap: minimum body clearance to the wall = distance from the body outline (0.58 × 0.31 m about the centre of gravity) to the nearest occupied pixel of the sim's map (same file as the bridge; 5.8 cm pixels; states are 0.1 s apart, so at a logged contact it reads about 0.1–0.2 m), and max |slip angle|. Slip is already 0.4–0.9 rad on normal laps here, so it does not separate controllers well; clearance does (about 0.3 m on normal laps). Detection delay: from the start of lap 9 (grip drop) or from entering the patch.

| Scenario | Controller | Seed | Crash (type, lap, s, hit point) | Lap times after the change (s) | Min clearance per lap after the change (m) | Max \|slip\| per lap (rad) | Detection delay | False det. | Speed 2 s before → after detection (m/s) |
|---|---|---|---|---|---|---|---|---|---|
| grip 0.55 | bicycle | 1 | spin, lap 9, s 178.1, (−12.20, 43.43) | – | 0.08 | 0.74 | – | – | – |
| grip 0.55 | bicycle | 2 | spin, lap 9, s 112.1, (−75.02, 54.07) | – | 0.06 | 0.85 | – | – | – |
| grip 0.55 | bicycle | 3 | spin, lap 9, s 176.9, (−11.46, 45.06) | – | 0.06 | 1.03 | – | – | – |
| grip 0.55 | v11 | 1 | spin, lap 9, s 176.7, (−11.54, 45.00) | – | 0.00 | 0.89 | – | – | – |
| grip 0.55 | v11 | 2 | spin, lap 11, s 179.4, (−12.84, 42.39) | 39.7, 40.0 | 0.08, 0.06, 0.06 | 0.68, 0.71, 0.54 | – | – | – |
| grip 0.55 | v11 | 3 | spin, lap 11, s 179.7, (−12.88, 42.40) | 40.2, 40.0 | 0.21, 0.13, 0.12 | 0.92, 0.86, 0.94 | – | – | – |
| grip 0.55 | v11 + reaction | 1 | spin, lap 14, s 117.4, (−69.51, 54.69) | 60.8, 68.1, 60.9, 54.4, 48.2 | 0.08, 0.29, 0.25, 0.29, 0.17, 0.06 | 0.64, 0.31, 0.39, 0.71, 0.38, 0.41 | 2.1 s | 0 | 7.56 → 8.90 |
| grip 0.55 | v11 + reaction | 2 | spin, lap 9, s 37.9, (−36.80, −7.26) | – | 0.30 | 0.96 | 1.7 s | 0 | 9.07 → 8.48 |
| grip 0.55 | v11 + reaction | 3 | spin, lap 15, s 38.6, (−37.25, −6.55) | 61.9, 69.8, 63.8, 56.2, 48.8, 44.2 | 0.08, 0.29, 0.29, 0.25, 0.12, 0.13, 0.37 | 0.68, 0.30, 0.51, 0.41, 0.35, 0.53, 0.54 | 0.8 s | 0 | 8.60 → 9.85 |
| patch 0.45 | bicycle | 1 | spin, lap 9, s 110.9, (−76.18, 53.71) | – | 0.13 | 0.96 | – | – | – |
| patch 0.45 | bicycle | 2 | spin, lap 9, s 113.0, (−74.17, 54.20) | – | 0.13 | 0.90 | – | – | – |
| patch 0.45 | bicycle | 3 | spin, lap 9, s 111.7, (−75.06, 54.08) | – | 0.29 | 1.04 | – | – | – |
| patch 0.45 | v11 | 1 | spin, lap 9, s 112.9, (−73.54, 54.28) | – | 0.12 | 0.78 | – | – | – |
| patch 0.45 | v11 | 2 | spin, lap 9, s 112.0, (−74.82, 54.07) | – | 0.25 | 0.90 | – | – | – |
| patch 0.45 | v11 | 3 | spin, lap 9, s 112.6, (−73.97, 54.19) | – | 0.33 | 0.81 | – | – | – |
| patch 0.45 | v11 + reaction | 1 | spin, lap 9, s 112.7, (−74.01, 54.18) | – | 0.35 | 1.00 | 0.6 s | 0 | 8.63 → 7.23 |
| patch 0.45 | v11 + reaction | 2 | **no** | 37.9, 43.3, 41.0, 39.9, 39.2, 38.0, 37.9 | 0.06, 0.31, 0.25, 0.25, 0.12, 0.06, 0.06 | 0.75, 0.60, 0.62, 0.59, 0.56, 0.62, 0.82 | 0.5 s | 0 | 8.71 → 8.35 |
| patch 0.45 | v11 + reaction | 3 | spin, lap 15, s 112.1, (−74.62, 54.09) | 38.7, 43.1, 42.5, 40.5, 39.7, 38.3 | 0.06, 0.29, 0.29, 0.23, 0.06, 0.12, 0.42 | 0.63, 0.51, 0.45, 0.68, 0.45, 0.61, 0.69 | 0.6 s | 0 | 8.81 → 8.06 |

Lap times before the change were 38–40 s in every run. Clearance lists include the lap that ended in the crash. "Laps to recover" is not given: no run without a crash got back to a steady lap time apart from patch reaction seed 2 (back to 37.9 s by the last laps).

| Scenario | Controller | Crashed / 3 | Laps driven after the change, before a crash | Crashes per lap after the change | Laps under 0.2 m clearance | Lap-time cost after the change | Detection delay | False det. |
|---|---|---|---|---|---|---|---|---|
| grip 0.55 | bicycle | 3 | 0, 0, 0 | 3/3 | 3/3 | – | – | – |
| grip 0.55 | v11 | 3 | 0, 2, 2 | 3/7 | 6/7 | – | – | – |
| grip 0.55 | v11 + reaction | 3 | 5, 0, 6 | 3/14 | 6/14 | +6 to +32 s (60–70 s, falling to 44–48 s) | 0.8–2.1 s | 0 |
| patch 0.45 | bicycle | 3 | 0, 0, 0 | 3/3 | 2/3 | – | – | – |
| patch 0.45 | v11 | 3 | 0, 0, 0 | 3/3 | 1/3 | – | – | – |
| patch 0.45 | v11 + reaction | 2 | 0, 7, 6 | 2/15 | 7/15 | +0 to +5 s (38–43 s) | 0.5–0.6 s | 0 |

### What this shows

- **The harsher changes are beyond every controller without a reaction.** All 12 bicycle and v11 runs spun, 10 of them in the first lap after the change, at the two fast corners (s ≈ 111–113 m and s ≈ 177–180 m).
- **The reaction does not prevent crashes, but it delays them a lot.** With it, the car drove 5–7 laps after the change in 4 of 6 runs (one patch run finished all 15 laps) where no run without it got past 2. Crashes per lap driven after the change: 3/14 vs 3/7 (grip), 2/15 vs 3/3 (patch). Near-misses (laps under 0.2 m clearance) fall from 6/7 to 6/14 (grip); for the patch they cannot be compared, since the runs without reaction crashed in their first lap.
- **The crashes that remain are in the first lap after the change, or once the caps have been raised again.** In the first lap the local caps arrive after the car has passed the stretch (the same weakness as the check runs; the grip-drop speed even rose after detection in seeds 1 and 3, 7.6 → 8.9 and 8.6 → 9.9 m/s). The late crashes (laps 14–15) came after the caps had risen by 0.05 per lap for 5–6 laps, so the car was back near its old speed on a track with 0.55 grip.
- **The reaction costs too much on the grip drop and is about right on the patch.** Grip drop: 60–70 s laps at first, then 44–48 s, more than ~3 s/lap; part of that is the under-cap weighting above. Patch: 0–5 s per lap.
- **Default:** the reaction stays off by default (`change_speed: null`, `change_detection: false`, as before). By the rule set for this test it would have been switched off: on the grip drop it costs more than 3 s/lap, and whether it reduces near-misses cannot be cleanly shown, since the runs without it crashed too early to have comparable laps. It does make the car survive longer on both harsh scenarios, so the code stays available. `change_cap_scope: local` and `change_cap_rank_weight: 1.0` are the defaults whenever it is switched on.

### Not done

The under-cap weighting is not fixed (one attempt; the cap cost would need a deadband in the final pick and a weight in the sampling that is comparable to the value term). Caps on stretches ahead of the car at the first detection (the first-lap crashes) and capping the cap rise below the pre-change speed when the grip stays low (the late crashes) were not tried. The track-wide (old) reaction was not run on the harsher scenarios.

## 39. Evidence-based speed reaction

### What changed

The speed reaction was rebuilt from what entry 38 showed (`planner.change_cap_scope: evidence`, now the default scope whenever `change_speed` is set; the reaction itself is still off by default).

1. **On detection:** every stretch is capped at once, at 0.8 × that stretch's pre-change speed (the `probe` mode still works as before).
2. **Relax per stretch:** a cap only changes after the car has driven that stretch with the cap in place (the pass must start after the cap was set). If the stretch's change tag (bicycle 1.0 s position error) matches its pre-change history, the cap is removed. If it still looks changed, the cap stays.
3. **Raise on evidence:** on a stretch that still looks changed, the cap goes up by 0.05 × the pre-change speed only when v11's own 1.0 s prediction error on that stretch, over the latest pass, is within its pre-change range (the largest per-lap median of the last laps before detection). At 1.0 the cap is removed. The +0.05-per-lap timer is gone.
4. **Repeat detections** cap only their own stretch. They never re-cap a stretch that has already relaxed.
5. **Softer cap:** the cap term in the sampling cost has its own weight (`change_cap_sample_weight`). The final-pick weight stays at 1 (`change_cap_rank_weight`).

v11's own error is computed in the runner: the RSSM state from 10 steps earlier is rolled forward with the controls actually applied, and the predicted position is compared with where the car actually is. Each run saves it as `model_error_tag.npy`. Unit test: `test_change_monitor_evidence_caps`.

Report scripts: `scripts/ev_report.py` (output `results/footprint/evidence-reaction.json`) and `scripts/breakdown_report.py` (output `results/footprint/evidence-breakdown.json`).

### Finding: v11's own error does not show the change

On normal laps v11's 1.0 s error has a median of about 0.47 m; the bicycle error is near 0. After the change, at the lower capped speed, v11's error drops to 0.2–0.3 m. So it is always "within the pre-change range", and every capped stretch gets its cap raised on every pass. In practice rule 3 acts like a per-pass timer: about 4 passes from 0.8 to cap removed. Meanwhile the bicycle tag still shows the change (median 0.1–0.3 m vs about 0 before). The rule was kept as specified.

### Check runs (sample weight 0.1)

| Run | Result | Laps after the change (s) | Speed / cap on capped steps | Laps < 0.2 m clearance | Detection delay |
|---|---|---|---|---|---|
| grip drop μ 0.55 (`ev-check-grip055-s1`) | no crash, 15 laps | 56.5, 51.3, 49.8, 46.6, 43.6, 41.4, 38.8 | 6.93 / 7.66 (0.73 under) | 3/7 | 0.8 s |
| patch μ 0.45 (`ev-check-patch045-s1`) | spin, lap 13, s = 113.6 | 52.5, 46.5, 42.1, 41.5 | 6.19 / 7.10 (0.90 under) | 3/5 | 0.5 s |

Both missed the 0.5 m/s target. The cost log shows why: the chosen plan was the fastest feasible plan on only 1% of capped steps, and the cap term alone made the faster plan lose 93–96% of the time. The sample weight was therefore lowered once, to 0.03, for the seed runs (`configs/world/rssm-latent-v11-react.yaml`).

### Seed runs (sample weight 0.03)

The bicycle and v11-without-reaction runs are reused from entry 38 (`hc-*`): same configs, and their code path did not change.

**Grip drop, μ 0.55, from lap 9**

| Controller | Seed | Crash | Laps after change | Laps < 0.2 m | Speed / cap | Detection delay |
|---|---|---|---|---|---|---|
| bicycle | 1 | spin, lap 9, s = 178.1 | 0 | 1/1 | – | – |
| bicycle | 2 | spin, lap 9, s = 112.1 | 0 | 1/1 | – | – |
| bicycle | 3 | spin, lap 9, s = 176.9 | 0 | 1/1 | – | – |
| v11 | 1 | spin, lap 9, s = 176.7 | 0 | 1/1 | – | – |
| v11 | 2 | spin, lap 11, s = 179.4 | 2 (39.7, 40.0) | 3/3 | – | – |
| v11 | 3 | spin, lap 11, s = 179.7 | 2 (40.2, 40.0) | 2/3 | – | – |
| v11 + reaction | 1 | spin, lap 9, s = 39.0 | 0 | 0/1 | 9.28 / 6.98 | 1.1 s |
| v11 + reaction | 2 | spin, lap 9, s = 41.0 | 0 | 1/1 | 9.62 / 7.13 | 1.6 s |
| v11 + reaction | 3 | spin, lap 9, s = 37.4 | 0 | 0/1 | 9.19 / 6.76 | 0.7 s |

**Patch, μ 0.45 at s = 85–115 m, from lap 9**

| Controller | Seed | Crash | Laps after change | Laps < 0.2 m | Speed / cap | Detection delay |
|---|---|---|---|---|---|---|
| bicycle | 1–3 | spin, lap 9, s = 110.9 / 113.0 / 111.7 | 0 / 0 / 0 | 2/3 | – | – |
| v11 | 1–3 | spin, lap 9, s = 112.9 / 112.0 / 112.6 | 0 / 0 / 0 | 1/3 | – | – |
| v11 + reaction | 1 | spin, lap 14, s = 115.1 | 5 (49.0, 39.9, 40.9, 40.3, 39.8) | 3/6 | 7.08 / 7.19 | 0.8 s |
| v11 + reaction | 2 | spin, lap 9, s = 111.6 | 0 | 0/1 | 9.47 / 6.73 | 0.7 s |
| v11 + reaction | 3 | no crash, 15 laps | 7 (51.0, 44.1, 42.3, 42.2, 41.3, 39.8, 38.9) | 3/7 | 6.61 / 6.90 | 0.6 s |

**No change, v11 + reaction:** 3/3 finished 15 laps, no false detections, no caps, 0/45 laps under 0.2 m. Mean of laps 6–15: 39.18 / 39.03 / 38.55 s, against 38.93 / 37.13 / 38.88 s for v11 without reaction (`fp-none-v11-s1..3`). That is +0.6 s on average, inside the seed spread. The reaction does not hurt normal laps.

There were no false detections in any run.

### When caps were raised or removed (surviving runs)

- **Patch, seed 3:**
  - lap 9: all stretches capped; 12–34 removed (tag matches history);
  - lap 10: 0–8 removed;
  - laps 10–12: 9, 10, 11 and 12 raised once per pass (12 was re-capped by a repeat detection in lap 10; 13's repeat detections were removed straight away);
  - lap 13: 9, 10 and 12 removed (model error in range);
  - lap 14: lowest clearance of the run (0.0 m, no contact logged).
- **Patch, seed 1:** same pattern. 10 was removed in lap 12 and 12 in lap 13; the spin came in lap 14 at s = 115, with 9 still capped at 8.49 and two laps under 0.06 m just before.
- **Patch, check run:** 10 removed in lap 12 and 9 in lap 13; spin in lap 13 at s = 113.6.
- **Grip drop, check run:**
  - lap 9: 6 stretches removed (tag matches history), the rest raised to 0.85;
  - every pass after that: raised again;
  - by laps 12–14: all removed.
  - The near-misses came after removal: 0.058 / 0.082 / 0.058 m in laps 13–15, against 0.21–0.42 m while capped.

Rule 4 held in every run: repeat detections capped only their own stretch.

### What the runs show

1. **0.03 is too soft.** In 4 of the 6 harsh runs the car never got under the cap. In the three grip-drop runs and patch seed 2 it was still 2.3–2.7 m/s over the cap 2–4 s after detection, and it spun at the next fast corner (s ≈ 37–41 m on the grip drop, inside the patch on seed 2). The cost log agrees: in those runs the chosen plan's peak speed was 2.9–3.3 m/s over the cap, and the cap term almost never decided the final pick (0–26% of steps). At 0.1 (the check runs) the car braked straight after detection (9.0 → 6.5 m/s on the grip drop) and survived. Where the car did get under the cap at 0.03 (patch seeds 1 and 3), it drove 0.11 and 0.29 m/s under it, so the 0.5 m/s target is met. But the target conflicts with braking hard at the moment of detection.
2. **The raise rule removes the caps too soon.** v11's error falls when the car slows, so it reads "in range" on every pass. The patch crashes (lap 13–14 at s = 113–115) and the grip-drop near-misses all come after caps were removed on stretches whose bicycle tag still showed the change.
3. **"Driven under the cap" only checks timing.** It checks that the pass started after the cap was set, not that the car was actually under it. In grip-drop seed 1 the caps on stretches 1–2 were raised within 1.2 s of detection while the car was still at 9.3 m/s against a 7.0 cap.
4. **Relaxing unchanged stretches works.** On the patch, 32 of 35 caps were removed within 1–2 laps because the tag matched history. Normal laps are not affected.

### Default

The reaction stays off by default (`change_speed: null`). `change_cap_scope: evidence`, `change_cap_sample_weight: 1.0` and `change_cap_rank_weight: 1.0` are the defaults when it is switched on. `rssm-latent-v11-react.yaml` keeps the 0.03 used in these runs.

### Not done

These were not tried:
- a strong sampling weight right after detection with a softer one later;
- raising caps based on the bicycle tag, or on v11's error compared at the same speed;
- requiring the pass speed to be under the cap before a pass counts.

## 40. Speed reaction: two-phase cap weight, speed-matched raising, counted passes

### What changed

Three fixes to the evidence reaction from entry 39 (`configs/world/rssm-latent-v11-react.yaml`):

1. **Two-phase cap weight.**
   - Every capped stretch starts in a strong phase: sampling weight `change_cap_strong_weight` (0.1), and the cap counts in the final pick (`change_cap_rank_weight` 1).
   - Once the car's speed is at or under the cap on the current pass of that stretch, the stretch switches to a soft phase: sampling weight `change_cap_soft_weight` (0.03), and the cap no longer counts in the final pick.
   - Each new pass starts strong again (see the check runs for why).
   - The phase on the car's stretch is logged each step (`change_trace.npy` column 11: 0 no cap, 1 strong, 2 soft), and an "under cap" event is logged when a stretch goes soft.
2. **Speed-matched cap raising.** A still-changed stretch's cap goes up only after a counted pass on which all of these hold:
   - v11's 1.0 s error (median over the pass) is no higher than the largest per-lap median of its pre-change error on the same stretch, using only steps whose speed was within ±0.5 m/s of the pass's mean speed (`change_raise_band`; at least 3 such steps per lap);
   - the minimum body-to-wall clearance on the pass is at least 0.2 m (`change_min_clearance`). It is computed live from the sim map with the same measure as the report scripts, and agrees with them within one map pixel;
   - the max |slip angle| on the pass is no higher than the stretch's pre-change max.

   The raise is +0.3 m/s per qualifying pass (`change_raise_step`) and never goes above the pre-change speed; reaching it removes the cap. Each refusal is logged as "cap kept" with its reasons.
3. **Counted passes.** A pass on a capped stretch counts toward relaxing or raising only if the speed was at or under the cap on at least 90% of its steps (`change_pass_share`). Otherwise a "pass not counted" event is logged.

Code: `change_monitor.py` (phases, matched limit, slip history, clearance helpers), `lmppi.py` (per-stretch cap weights in sampling and final pick), `runner.py` (live clearance and slip passed to the monitor, phase in the trace), schema and defaults. Unit test: `test_change_monitor_evidence_caps` (rewritten). Report: `scripts/ev_report.py --prefix ev2` (output `results/footprint/evidence2-reaction.json`; the check runs alone are in `evidence2-check.json`).

### Check runs

The first check runs latched the soft phase: a stretch went soft the first time the car was under its cap and stayed soft for good. Both spun in lap 10 (grip drop at s = 114.4, patch at s = 113.0). In lap 10 the car came into the still-capped stretches 8–10 from an uncapped stretch at 9–10.5 m/s against caps of 6–7 m/s. A soft cap (0.03, not in the final pick) cannot brake that. These runs are kept as `ev2-check-*-s1-latch`.

Fix: each new pass starts strong. So while the car is still on the stretch before, the cap ahead is strong and the car brakes before entering.

| Check run (per-pass phase) | Result | Laps after change (s) | Strong phase speed / cap | Soft phase speed / cap | Detection → under cap |
|---|---|---|---|---|---|
| grip drop μ 0.55 | no crash, 15 laps | 54.4, 51.7, 54.4, 52.3, 52.7, 49.9, 51.5 | 9.22 / 7.17 (185 steps) | 6.53 / 7.10 (3415 steps) | 0.3 s |
| patch μ 0.45 | no crash, 15 laps | 50.6, 44.0, 41.6, 42.2, 41.6, 41.5, 40.6 | 9.99 / 6.86 (86 steps) | 6.22 / 6.93 (911 steps) | 2.6 s |

The strong-phase steps are mostly the car braking into a capped stretch, so it is over the cap by definition there.

Cap timelines:
- **Grip drop check:** stretch 1 was raised in laps 11, 13 and 15 (7.47 → 7.77 → 8.07). Every other changed stretch kept its cap all run, mostly with "no pre-change data at this speed" (22–25 per lap), with a few "slip above pre-change range" and "clearance under limit".
- **Patch check:** stretches 12–34 were removed in lap 9 and 0–7 in lap 10 (tag matches history). Stretch 10 was raised in each of laps 10–14 (6.42 → 7.62). Stretches 9 and 11 kept their caps.

### Seed runs

Bicycle and v11 without reaction are reused from entry 38 (`hc-*`).

**Grip drop, μ 0.55, from lap 9**

| Controller | Seed | Crash | Laps after change, lap times (s) | Laps < 0.2 m | Strong speed / cap | Soft speed / cap | Detection → under cap | Detection delay |
|---|---|---|---|---|---|---|---|---|
| bicycle | 1–3 | spin lap 9 (s = 178.1 / 112.1 / 176.9) | 0 / 0 / 0 | 3/3 | – | – | – | – |
| v11 | 1 | spin lap 9, s = 176.7 | 0 | 1/1 | – | – | – | – |
| v11 | 2 | spin lap 11, s = 179.4 | 2: 39.7, 40.0 | 3/3 | – | – | – | – |
| v11 | 3 | spin lap 11, s = 179.7 | 2: 40.2, 40.0 | 2/3 | – | – | – | – |
| v11 + reaction | 1 | no | 7: 55.5, 50.4, 54.9, 54.1, 52.5, 51.0, 51.7 | 3/7 | 9.38 / 6.91 | 6.53 / 7.10 | 0.2 s | 1.1 s |
| v11 + reaction | 2 | no | 7: 55.3, 53.4, 55.1, 54.4, 51.6, 51.1, 51.2 | 3/7 | 9.31 / 7.08 | 6.51 / 7.08 | 0.1 s | 1.2 s |
| v11 + reaction | 3 | spin lap 9, s = 38.5 | 0 | 0/1 | 9.43 / 6.90 | – | never | 1.6 s |

**Patch, μ 0.45 at s = 85–115 m, from lap 9**

| Controller | Seed | Crash | Laps after change, lap times (s) | Laps < 0.2 m | Strong speed / cap | Soft speed / cap | Detection → under cap | Detection delay |
|---|---|---|---|---|---|---|---|---|
| bicycle | 1–3 | spin lap 9 (s = 110.9 / 113.0 / 111.7) | 0 / 0 / 0 | 2/3 | – | – | – | – |
| v11 | 1–3 | spin lap 9 (s = 112.9 / 112.0 / 112.6) | 0 / 0 / 0 | 1/3 | – | – | – | – |
| v11 + reaction | 1 | no | 7: 51.3, 43.9, 42.9, 42.6, 42.2, 41.7, 41.0 | 5/7 | 9.40 / 7.17 | 6.29 / 6.97 | 0.4 s | 0.4 s |
| v11 + reaction | 2 | spin lap 14, s = 113.5 | 5: 49.0, 43.9, 40.7, 41.2, 41.1 | 3/6 | 9.57 / 7.17 | 6.42 / 7.06 | 0.4 s | 0.6 s |
| v11 + reaction | 3 | no | 7: 50.9, 43.7, 41.8, 43.2, 42.3, 43.3, 40.5 | 4/7 | 10.02 / 6.75 | 6.20 / 6.79 | 0.9 s | 0.5 s |

**No change, v11 + reaction**

| Seed | Crash | Laps 6–15 mean | Detections |
|---|---|---|---|
| 1 | no | 38.52 s | 0 |
| 2 | spin lap 10, s = 108.9 | 38.45 s (laps 6–9) | 1, during the spin |
| 3 | no | 38.78 s | 0 |

v11 without reaction (`fp-none-v11-s1..3`): 38.93 / 37.13 / 38.88 s.

The seed 2 spin is not caused by the reaction. Speed fell from 8.5 to 0.7 m/s over steps 3560–3590 with no cap anywhere. The one detection came at step 3599, from the spin itself. So this is a plain v11 spin on a normal lap that happened to occur in a run with the reaction switched on. The fp-none-v11 seeds had no such spin.

**False detections:** none before the change in any harsh run. The only detection in a no-change run is the one triggered by the spin above.

**Planning time** (no change): p50 / p95 about 70 / 91 ms with the react config, against 53 / 70 ms for v11. Entry 39's react runs show the same (72 / 92 ms), so the cost comes from what the react config turns on (per-candidate cost logging `change_breakdown`, v11's own error replay), not from this entry's changes. It was not split further.

### Summary

| Scenario | Controller | Crashes | Laps after change | Laps < 0.2 m |
|---|---|---|---|---|
| grip drop μ 0.55 | bicycle | 3/3 | 0, 0, 0 | 3/3 |
| | v11 | 3/3 | 0, 2, 2 | 6/7 |
| | v11 + reaction | 1/3 | 7, 7, 0 | 6/15 |
| patch μ 0.45 | bicycle | 3/3 | 0, 0, 0 | 2/3 |
| | v11 | 3/3 | 0, 0, 0 | 1/3 |
| | v11 + reaction | 1/3 | 7, 5, 7 | 12/20 |
| no change | v11 + reaction | 1/3 (v11 spin, reaction inactive) | – | 0/40 |

With the check runs included, 6 of 8 harsh runs with the reaction finished all 15 laps, against 0 of 6 without it.

### Cap timelines (seed runs)

- **Grip drop, seed 1:**
  - lap 9: detection in stretch 1, every stretch capped;
  - tag matches removed 8 stretches in lap 9 and 3 each in laps 11, 13 and 15;
  - one raise in the whole run: stretch 31 in lap 13 (to 7.5);
  - everything else stayed capped. Each lap had 19–28 "no pre-change data at this speed", 0–2 "slip above pre-change range", and in laps 13–14 one "clearance under limit".
- **Grip drop, seed 2:**
  - tag matches removed 4 stretches in lap 9, then 1 per lap;
  - stretch 28 was raised in laps 9–13 (7.5 → 8.7) and removed in lap 14 (pre-change speed reached);
  - stretch 6 was raised in laps 11–15 (7.28 → 8.48); stretches 8, 9 and 12 once each in laps 13–14;
  - the rest kept their caps (18–29 "no pre-change data" per lap).
- **Grip drop, seed 3:** detection at 1.6 s. The car never got under the cap and spun 2.8 s later at s = 38.5. In those 28 steps every candidate was over the cap: the fastest feasible plan averaged 9.7 m/s against a 6.9 cap, and the chosen plan peaked 3.3 m/s over it. The cap term never decided the final pick. Strong weight 0.1 was not enough to pull the samples down in time. This is the same failure as entry 39 at 0.03, now in 1 of 3 runs instead of 3 of 3.
- **Patch, seeds 1 and 3:**
  - stretches 12–34 were removed in lap 9 and 0–8 in lap 10 (tag matches history);
  - repeat detections capped only their own stretch (12, 13, 8);
  - stretches 9, 10 and 11 kept their caps for the whole run: 1–2 "no pre-change data at this speed" per lap, a few "slip above pre-change range", and 2–9 passes per lap not counted.
- **Patch, seed 2:** same, except that in lap 12 stretch 11 (s = 110–120 m, holding the last 5 m of the patch) relaxed with a tag of 0.99 m and lost its cap. The spin came in lap 14 at s = 113.5, inside stretch 11, at about 6 m/s (the old cap there was 5.95).

### Bug found: post-change laps leak into the tag history

A stretch that is capped but not yet judged (CONSERVATIVE, not marked changed) keeps adding each lap's tag to its "pre-change" history. With the 90% rule, the car can drive such a stretch for several laps without a counted pass. On patch seed 2, stretch 11's history gained 0.84, 1.17 and 1.03 m in laps 9–11 (0.0–0.56 m before), so its limit rose to about 2.5 m and a 0.99 m tag "matched history". Patch seed 3 has the same leak (stretch 11 history up to 1.3 m) but no counted pass happened to come before the end of the run.

The fix is one line in `finish_lap`: skip history for stretches that are capped and still waiting to be judged. It is not applied; all runs here used the code as described above.

### What the runs show

1. **The reaction now keeps the car alive in most harsh runs:** 4 of 6 seed runs and both check runs finished 15 laps, against 0 of 6 for v11 and the bicycle.
2. **Soft phase:** the car drove 0.56–0.68 m/s under the cap on capped stretches. That is close to, but not inside, the 0.5 m/s target.
3. **The speed-matched raise almost never fires on the grip drop.** At 0.8× speed the car is slower than it ever was on most stretches before the change, so there are no pre-change samples to compare with. Caps stay at 0.8× and laps stay at 50–55 s, against 38 s before the change. On the patch, the stretches inside the patch (9–11) never got raised either. Where a match existed, v11's error was again below its pre-change level (0.18–0.34 m against 0.28–0.67 m). So speed alone does not explain why v11's error drops after the change.
4. **Near-misses are frequent:** 6 of 15 and 12 of 20 laps after the change came under 0.2 m clearance, with caps in place. The caps keep speed down but do not keep the car away from the wall.
5. **Two crashes, two causes:**
   - strong weight 0.1 sometimes cannot brake from 9.4 to 6.9 m/s in the 2–3 s after a late detection;
   - the history leak removed a cap inside the patch.

### Default

The reaction stays off by default (`change_speed: null`). By the rule set for this test it beats v11 without reaction on survival (6 of 8 against 0 of 6). It was kept off for these reasons:
- one of the two crashes comes from the history bug above;
- the other is the no-braking failure, which is still open;
- grip-drop laps cost 13–17 s each with no way back up;
- the react config adds about 17 ms of planning time (p95 about 91 ms, close to the 100 ms control period).

`change_detection` itself needs the RSSM backend, so the plain default config could not switch it on anyway; only the v11 change configs could.

### Not done

- the history fix;
- a stronger or longer strong phase for late detections (for example the final-pick weight ×1000 until under the cap);
- a fallback for raising when no pre-change data exists at the capped speed (for example the bicycle tag);
- splitting the planning-time cost between cost logging and the error replay.
