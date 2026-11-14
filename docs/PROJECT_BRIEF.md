# Adaptive World-Model LMPC for F1TENTH: Project Overview

ESE 6150 (Roboracer / F1TENTH), UPenn, Fall 2026, Team 8. This document covers concepts only. It contains no implementation instructions.

## Goal

We want the car to learn how it really moves, including slip, grip, and lag, while it drives. It should use that knowledge to get faster every lap without crashing. If the car or the floor changes mid-session, for example from added weight or a slippery patch, it should notice and adapt by itself.

## The base method: SIT-LMPC

SIT-LMPC is a learning MPC for repeated tasks such as racing laps. It is described in Zang, Amine et al., RA-L 2026, arXiv 2602.16187. Our TAs wrote it, and the code is in `sit-lmpc-remastered`.

- **Iterative improvement.** Every completed lap is stored. The controller uses past laps to drive the next lap faster.
- **Planner.** The planner is sampling-based MPC (MPPI). It samples many control sequences, rolls each one out through a dynamics model over a short horizon, scores them, and blends the best ones.
- **Safe set.** The safe set holds states from previous successful laps (a convex hull). The end of every plan must land inside it. This makes the car aim for places it has already proven it can recover from.
- **Value function.** A learned estimate of time-to-finish from any state, trained from completed laps. It acts as the terminal cost, so a short-horizon plan still optimizes the whole lap.
- **Constraints.** Track limits and the safe set are enforced through adaptive penalty weights. The controller picks the least-penalized feasible option.
- **Warm-up.** The first few laps simply track a reference line slowly. This seeds the safe set and the value function.

## What the remastered code adds

- **Two dynamics options** in the same planner:
  - a fixed analytic single-track (bicycle) model
  - a learned Bayesian model
- **Learned model structure.** A neural network gives a mean prediction plus basis functions. A small Gaussian vector `z` (96-dim) weights those basis functions.
  - The network is pretrained offline across many cars with different friction and mass.
  - Online, only the distribution over `z` is updated, in closed form, from measured transitions.
  - This is a "which car am I?" belief.
- **Belief updates happen only between laps.** No update happens during a lap.
- **Chance constraints.** Sample several plausible dynamics models from the belief, then require that most of them stay on track and end in the safe set.
- **Their sim result.** With the learned model, the controller converges to nearly the same lap time as with the true model, but learns more slowly.

## What is missing, which is our contribution

1. **In-lap adaptation.** Update the dynamics belief every control step, so that a change is picked up within the same lap rather than after it. Because only a Gaussian over `z` is learned, this can be a cheap recursive update with forgetting, like a Kalman filter or recursive least squares. No network retraining is needed.
2. **Condition-aware safe set.** Today, old laps stay "safe" even after conditions change. We will tag each stored state with the belief under which it was recorded. When planning, we will trust only states recorded under similar conditions, and fall back to the most recent laps if none match.
3. **Mid-session changes as the main test.** Change friction or mass partway through a session, or add a low-friction section of track. Then measure how quickly each controller recovers.
4. **F1TENTH scale and ROS 2.** The current code is a standalone Python sim at full-car scale and has no ROS. We need it running at 1/10 scale through the ROS 2 F1TENTH simulator, and later on the real car (Jetson Orin).

## Overall architecture

```text
f1tenth_gym (ROS 2 sim, later the real car)
   │  odometry / pose / IMU
   ▼
State estimation → 7-dim single-track state
   (x, y, steering angle, speed, yaw, yaw rate, slip angle)
   │
   ▼
Dynamics belief (learned model + belief over z)  ◄── updated every step (ours)
   │
   ▼
SIT-LMPC planner (MPPI + safe set + value function)
   │  safe set filtered by condition similarity (ours)
   ▼
Steering / acceleration commands → sim or car
   │
   └── at lap end: store the lap in the safe set, retrain the value function, update the belief
```

## How we'll judge it

- **Baselines:**
  - SIT-LMPC with the fixed analytic model
  - SIT-LMPC with the learned model and between-lap updates only (the TAs' current code)
- **Ablations:** our method without in-lap updates, and our method without condition tags on the safe set.
- **Metrics:**
  - lap time per lap
  - laps needed to recover after a change
  - track-limit violations
  - top stable speed
  - multi-step prediction error of the model
- **Milestones:**
  - *Minimum:* the learned model predicts better than the bicycle model, and SIT-LMPC with it completes clean laps in sim.
  - *Target:* it recovers faster after a change, in sim and on the car, with no more violations.
  - *Stretch:* it recovers within a single lap, and reaches a higher top stable speed.

## Known hard parts

- **State estimation.** Slip angle and steering angle are not measured directly, and they must be estimated for the 7-dim state.
- **Real-time budget.** Planning with sampled models must fit in the control period on a Jetson.
- **Value function after a change.** The value function was learned under the old conditions, so it is over-optimistic until new laps arrive.
- **Uncertainty calibration.** The learned uncertainty is not guaranteed to be calibrated. The safety we can show is empirical, not a proof.