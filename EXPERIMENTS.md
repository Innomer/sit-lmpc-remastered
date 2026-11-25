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
| 26 | Rollout variants (diagnosis, no retraining) | next commit | done: no cheaper fix; estimation error is about half |

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
