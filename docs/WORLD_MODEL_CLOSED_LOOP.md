# World-model SIT-LMPC: closed-loop status

ESE 6150 Team 8, branch `world-model`, updated 2026-11-28. Model details: `docs/WORLD_MODEL_RSSM.md`. Every run and
number: `EXPERIMENTS.md` (entry numbers below).

## Summary

The RSSM world model (rssm-v11) drives the SIT-LMPC planner in the ROS lockstep sim on Spielberg. Its prediction
step is bicycle physics plus a learned correction conditioned on a recurrent latent built from raw sensors (LiDAR,
IMU, wheel speed, particle-filter pose); the latent is rolled forward along each plan, and the same model predicts
the LiDAR scan for a clearance check.

| Condition | State |
|---|---|
| No change | 15/15 laps, best 36.9 s (bicycle model 36.0 s) (entry 30). After the planner structure fixes of entry 42 the boxed-in stop at s ≈ 107 m is gone (no-change checks 15/15 for v11 and the bicycle model) |
| Moderate grip drop (μ 1.05 → 0.7) | v11 survives and recovers in about 4 laps where the bicycle model spins (entry 31) |
| Harsh grip drop (μ 0.55), low-grip patch (μ 0.45) | without extra handling both v11 and the bicycle model crash on the first laps after the change (entry 38). A hand-written speed reaction helps but needs many hand-set parts (entries 39–41) |
| Retry mode (crash → restart, 25 attempts), learned change handling | entries 42–43 |

## System

| Part | Setting | Learned or hand-written |
|---|---|---|
| World model | rssm-v11: GRU memory 128, latent 8 (conditions) + 24 (fast), 523k parameters, trained offline on 684k on-track steps and 273k open-space steps | learned |
| Prediction | bicycle model + learned correction R(state, control, h, s); h and s advanced along the plan | bicycle fixed, rest learned |
| Planner | LMPPI: 1,024 plans of 15 × 0.1 s, 256 λ candidates + last step's plan, final pick by tiers (track limit + safe set, then safety terms) | hand-written structure |
| Value | normalizing flow on the stored laps' time-to-go, retrained after every lap | learned online |
| Safe set | convex hull of stored states from completed laps, body-corner-consistent | data + hand-written rules |
| LiDAR clearance check | cost when the predicted nearest range is under 0.35 m; changes the chosen plan on about 0.6% of steps (entry 42) | learned prediction, hand-set margin |
| Change detection | bicycle 1.0 s position-error tag per 10 m stretch | hand-written rule |
| Speed reaction | caps, phases, braking plans, cap raising | hand-written rule (change configs only) |
| Learned change handling | tag-matched safe set, tag-conditioned value after a detection | learned value + hand-written matching (`configs/retry/v11-learned.yaml`) |

## Offline (entry 29)

1.5 s lateral error on 1,771 held-out starts, median / p95: v11 from the true state 0.26 / 1.36 m (v8 0.36 / 1.73,
v3 0.18 / 1.63); from its own estimate 0.42 / 1.84 m. Above 9 m/s the median is about twice v3's.

## Closed-loop history

| Stage | What happened | Entry |
|---|---|---|
| rssm-v5 to v7 | crashed within 1–3 learned laps; the model had not seen the planner's speeds and manoeuvres, and its 1.5 s rollouts were 0.14–0.52 m off where the v3 model was 0.02–0.21 m off | 6–9 |
| rssm-v8, guardrails | latent overshooting, LiDAR clearance check, speed cap, slip cost, crash-risk head; still crashed | 10–27 |
| rssm-v11 | first world-model run with all 10 learned laps; survives a moderate grip drop | 29–31 |
| Change detection, safe set v2 | detection from the bicycle-error tag works (delay 0.4–1.6 s, no false detections); conservative safe set alone does not slow the car enough | 33–36 |
| Body-footprint track limits | on by default | 37 |
| Hand-written speed reaction | survival up (6/8 harsh runs in entry 40), lap times down on the grip drop, crashes from a history bug and from soft-phase overshoot | 38–41 |
| Boxed-in stop root cause | safe set was a soft cost in the final pick and last step's plan was not a candidate; plans reaching the safe set went through the wall | 41 |
| LMPC structure fixes, retry mode | entries 42–43 | 42–43 |

## What is still open

| Item | Note |
|---|---|
| Harsh changes without hand-written rules | the learned change handling is under test (entry 43) |
| Rollout accuracy at speed | median 1.5 s error above 9 m/s about twice v3's |
| Grip read-out | the condition head on c is too weak to use as a tag |
| Learning from crashes | crash attempts are stored but no between-lap learner uses them |
| Real car | not run with the world model yet |
