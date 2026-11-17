# SIT-LMPC Remastered

Refactored racecar SIT-LMPC with online value learning, a Gaussian BNN dynamics distribution, calibrated Bayesian belief updates, and dynamics-based chance constraints.

## ESE 6150 Team 8 fork: world-model SIT-LMPC at F1TENTH scale

This fork runs SIT-LMPC on the ROS 2 F1TENTH simulator and the real F1TENTH car, and adds our method: a
physics-grounded world model that adapts to changing conditions (grip, load) **during** a lap.

| Branch | Content |
|---|---|
| `ros-sim` | ROS sim backend, baselines, world model, all sim results and videos |
| `irl` | `ros-sim` + the real-car backend (`docs/CAR_DEPLOYMENT.md`), Jetson setup |

**Our method.** Next state = bicycle model + learned correction b + learned directions B·z, with z (16 numbers)
describing the car's current conditions. A GRU encoder reads the IMU, wheel speed, particle-filter pose and last
command every 0.1 s and outputs z (with its uncertainty) and the unmeasured state (steering, speed, yaw rate, slip).
The planner is the TAs' LMPPI, unchanged. The safe set is condition-aware: each lap is tagged with its grip estimate
and only laps with similar grip are used. Details: [docs/WORLD_MODEL.md](docs/WORLD_MODEL.md).

| Method | Car model | What adapts while driving |
|---|---|---|
| Known model | exact sim physics | safe set, value function (per lap) |
| TAs' BNN | learned network + z (96), Gaussian belief | + z, after each lap |
| Ours | bicycle + b + B·z (16) | + z and state every 0.1 s from sensors; grip-aware safe set |

**Status (2026-11-17).**

| Item | Result |
|---|---|
| Fast-start sim baselines (Spielberg, 38.0 s warm-up) | known 25/25, best 35.0 s; TAs' BNN 14/15, best 35.7 s; BNN + chance 15/15, 39.4 s |
| Encoder v3, offline (held-out sim cars) | 1 s position error from sensors 0.53 m vs 0.95 m for the best baseline; slip RMSE 0.020 rad; grip from z R² 0.91 |
| Ours in the loop, no change | 14/15, best 35.7 s; one spin |
| Ours in the loop, grip 1.05 → 0.75 after lap 7 | 14/15, best 35.8 s; grip guess 0.75 on the first slippery lap; one spin |
| Baselines on the grip drop, real-car runs | not yet run |

Results: [results/world-closed-loop](results/world-closed-loop/README.md) (with videos in `results/Spielberg/*.mp4`),
[results/world-eval-v3](results/world-eval-v3). Full history, including mistakes and the runs they invalidated:
[docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md). Older, wrong evaluations (`results/world-eval`, `-v1`, `-v2`) are kept
on purpose.

**Run (sim).** Needs ROS 2 Jazzy, the `f1tenth_gym_ros` bridge with `patches/f1tenth_gym_ros-sit-lmpc.patch`, and a
trained encoder in `runs/` (not in git).

```bash
# world-model training and evaluation
python -m sit_lmpc.world.train --config configs/train-world-encoder-v2.yaml --output runs/world-encoder-v3
python -m sit_lmpc.world.evaluate --encoder runs/world-encoder-v3 --out results/world-eval-v3
python -m sit_lmpc.world.readout --encoder runs/world-encoder-v3
# one mid-session change experiment (bridge + condition scheduler + controller, ROS domain 77)
bash scripts/run_change_experiment.sh configs/world/world-nochance-condition.yaml \
  configs/scenarios/spielberg-friction-drop.yaml change-friction-world-condition 15
# top-down video of a run
python scripts/render_run_video.py results/Spielberg/change-friction-world-condition --speedup 2
```

The rest of this README is the upstream (TAs') documentation.

## Main findings and capabilities

- The known-dynamics preset retains MPPI warm-up, convex-hull safe sets, NF/MLP value learning and the smallest-feasible-lambda selection behavior.
- The BNN preset uses the mean of coherent sampled-function predictions for nominal MPPI prediction, with one fixed function per scenario throughout each horizon.
- Attempt-boundary learning updates the latent Gaussian posterior using measured applied controls and successful/failed trajectories. Tempering uses separate calibration/test windows to avoid accepting an unjustifiably concentrated posterior.
- Chance constraints estimate boundary and terminal-hull violation probabilities under sampled dynamics. Thresholds and the probability estimator are configurable; empirical estimates are not physical safety guarantees.
- This repository has its own Python package and uv environment. Walker experiments, source-repository environments and historical results are excluded.

## Validated racecar examples

The [racecar examples](examples/racecar/README.md) contain portable configs, a small pretrained BNN, recorded data and plots from the completed 300-lap comparison. Both controllers completed all laps without a physical failure; best lap times were 13.7 s for known-dynamics SIT-LMPC and 14.0 s for BNN/chance control. The example report also documents chance fallbacks and imperfect held-out coverage.

```bash
uv sync --extra gpu --extra plots --extra dev
uv run python examples/racecar/run_example.py --controller known-dynamics
uv run python examples/racecar/run_example.py --controller bnn-unknown
```

## Run

The simulator uses the bundled snapshot of [`f1tenth/f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym), branch `zirui`, at commit [`1e35dc997988d13e573f1fa94df1a6a20cde3077`](https://github.com/f1tenth/f1tenth_gym/commit/1e35dc997988d13e573f1fa94df1a6a20cde3077) (September 16, 2025). Its 40 vendored files match that commit exactly. `uv sync --extra sim` (or `pip install -e ".[sim]"`) installs this snapshot from `vendor/f1tenth_gym`; no separate Gym checkout or installation is required. The ROS and real-car backends do not need it.

```bash
uv sync --extra gpu --extra plots --extra dev
uv run sit-racecar
```

The default is the known-dynamics SIT-LMPC baseline. Edit `configs/default.yaml`; its target is the packaged, annotated default YAML. Simulator rendering and keyboard controls are independent. To run without the simulator window:

```bash
uv run sit-racecar --headless
uv run sit-racecar --config configs/bnn-chance.yaml --headless
```

The main BNN preset reads `artifacts/vehicle-bnn-v2-l96`. Import the included example model there, or provide your own compatible trained checkpoint:

```bash
uv run sit-import-bnn --source examples/racecar/models/vehicle-bnn-v2-l96
uv run sit-racecar --config configs/bnn-chance.yaml
```

Use `configs/bnn-posterior.yaml` for BNN learning without chance constraints. `--seed 42` selects a reproducible experiment; otherwise a fresh saved master seed is used. `--max-laps` includes physically failed attempts. `--resume` points directly to a self-contained experiment checkpoint and creates a new result folder.

**Table 1. Entry points.**

| Command | Purpose |
|---|---|
| `sit-racecar` | Warm-up, value learning, LMPC and optional BNN/chance constraints |
| `sit-track` | MPPI tracking without LMPC value learning |
| `sit-replay` | Saved experiment trajectories and plots |
| `sit-insight` | Recorded control-step diagnostics |
| `sit-train-value` | Offline NF or MLP value training |
| `sit-import-bnn` | Validate and copy a pretrained population checkpoint |
| `sit-train-dynamics` | Offline population-model training using dyna-dist |

## Code and experiment structure

**Table 2. Modules and saved data.**

| Location | Responsibility |
|---|---|
| `src/sit_lmpc/planners/` | MPPI/LMPPI kernels, safe sets and probabilistic constraints |
| `src/sit_lmpc/dynamics/` | Analytical ST and coherent Gaussian BNN predictors |
| `src/sit_lmpc/training/` | Value training, latent posterior updates and calibration |
| `src/sit_lmpc/data/` | Transitions, complete success/failure attempts and checkpoints |
| `src/sit_lmpc/experiments/` | Runtime and racecar experiment orchestration |
| `src/sit_lmpc/diagnostics/` | Console, keyboard, Qt and matplotlib diagnostics |
| `configs/` | Known-dynamics baseline and BNN presets |
| `examples/racecar/` | Reproducible known/unknown-dynamics examples, compact recorded outcomes and plots |
| `vendor/` | Pinned local source snapshots, with original license notices |
| `artifacts/` | Imported population checkpoints, excluded from Git |
| `results/<map>/<name>-<Eastern timestamp>/` | Resolved configuration, trajectories, models, metrics and checkpoints |

The BNN's pretrained neural features stay fixed during the online latent update. Retraining the population neural network is an explicit offline operation:

```bash
uv run sit-train-dynamics --config configs/train-dynamics.yaml --dataset /path/to/vehicle-study
```

Auxiliary racecar risk-to-go, finite-bank and density modules remain available internally for compatibility; the supplied presets disable their learning/constraints. They are separate from dynamics-based chance constraints.

See the [architecture and belief equations](docs/ARCHITECTURE_AND_DYNAMICS.md), [migration plan](docs/SIT_LMPC_REMASTERED_MIGRATION_PLAN.md), [source provenance](docs/MIGRATION_SOURCE_MANIFEST.json), and [validation results](docs/MIGRATION_VALIDATION.md).

## Development checks

```bash
uv run python scripts/test_project.py
uv run ruff check src tests scripts examples
uv build
```

The test runner uses separate Python processes for unit and integration suites. See [publication validation](docs/EXAMPLES_PUBLICATION_VALIDATION.md) for the tested setup and compiler-isolation note.
