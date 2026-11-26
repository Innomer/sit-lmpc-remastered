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
| 35 | Baseline robustness, run divergence, v11's repeat crash spots, risk head vs speed cap | next commit | failure: neither option removes the no-change crashes; the speed reaction is built but not run |

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
| Closed loop (`ros-sim`) | completes laps; spins about once per run at 0.72–0.80 rad slip, where the known model does not |
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
