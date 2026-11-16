# Frozen sim baselines (TAs' SIT-LMPC code, F1TENTH scale, ROS 2 sim)

Frozen 2026-11-16. These are the reference runs for every later comparison. Do not rerun them in
place. New baselines on the mid-session change scenarios go in new folders.

All runs: Spielberg, `f1tenth_gym_ros` bridge in lockstep on the laptop GPU, seed 1, 25 attempts.
Each run used the uncommitted code that went into the commit listed under "Code".

| Run | Controller | Warm-up | Best LMPPI lap | LMPPI laps | Planning p50 | Config | Code |
|---|---|---|---|---|---|---|---|
| `known25-fixed` | known ST model | MPPI, 135.2 s | 70.9 s | 20/20 | 19.5 ms | `configs/f1tenth-ros-known-fast.yaml` | `b4be7fd` |
| `bnn25` | BNN + chance constraints | MPPI, 135.2 s | 110.1 s | 20/20 | 98.5 ms | `configs/f1tenth-ros-bnn-fast.yaml` | `b4be7fd` |
| `bnn-nochance25` | BNN, no chance constraints | MPPI, 135.2 s | 75.6 s | 20/20 | 64 ms | `configs/f1tenth-ros-bnn-nochance-fast.yaml` | `cd702d3` |
| `known-ppwarm25` | known ST model | pure pursuit, 38.0 s | 38.9 s | 17, then spin | 19.7 ms | `configs/f1tenth-ros-known-ppwarm.yaml` | `cd702d3` |
| `bnn-ppwarm25` | BNN + chance constraints | pure pursuit, 38.0 s | 39.8 s | 5, then fallback stop | 99 ms | `configs/f1tenth-ros-bnn-ppwarm.yaml` | `cd702d3` |
| `bnn-nochance-ppwarm25` | BNN, no chance constraints | pure pursuit, 38.0 s | 38.3 s | 7, then stop | 63 ms | `configs/f1tenth-ros-bnn-nochance-ppwarm.yaml` | `cd702d3` |

Bridge parameters: `configs/ros-bridge-lockstep-fast.yaml` for the MPPI warm-up runs, and
`configs/ros-bridge-lockstep.yaml` for the pure-pursuit runs.

BNN models are byte-identical to the tracked copies:
- slow-start runs: `models/vehicle-bnn-f1tenth-l96`;
- fast-start runs: `models/vehicle-bnn-f1tenth-wide-l96`.

The run configs name them under `artifacts/`.

Each folder keeps:
- the resolved `config.yaml`, `metadata.json`, `summary.json` and `lap_metrics.json`;
- `control_timing.json`, `events.jsonl` and `laps.npz`;
- the value and dynamics-posterior metadata (BNN runs);
- the run's figures.

Checkpoints, trajectories, `init.npz` and model weights stay in the untracked `results/Spielberg/<run>`
folders. `index.json` lists the same facts per run.

To reproduce, check out the listed commit and start the bridge with the listed bridge file. Then run
`sit-racecar --config <config> --seed 1 --max-laps 25 --save-dir results/Spielberg/<new name>`.
The NOTES.md run recipe has the full commands.
