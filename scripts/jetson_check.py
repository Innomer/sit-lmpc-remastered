"""Jetson check: JAX on the GPU, then planning time per car config, with suggested cuts.

    python scripts/jetson_check.py                       # all configs in configs/car/
    python scripts/jetson_check.py --configs configs/car/levine-v11.yaml --steps 150

1. GPU: lists JAX devices, runs a compiled matrix product on the first GPU and fails if there is none.
2. Planning time: builds each config's model and planner exactly as on the car (bicycle model or rssm-v11 with its
   filter step), fills the safe set with laps along the track's centreline (speed profile from the curvature, capped
   at the config's warm-up speed), trains the value function briefly, then times full control steps (filter step +
   planner update) along a lap. Planning cost does not depend on the exact lap data; the safe-set cap bounds the
   hull.
3. If p95 is over --budget (90 ms), the same config is timed with 512 and 256 candidates and with a 10-step horizon
   (0.15 s model steps for v11, so the plan still covers 1.5 s), and the cheapest change that meets the budget is
   printed as a config override.
"""

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def gpu_check():
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    print("JAX", jax.__version__, "devices:", devices)
    gpus = [d for d in devices if d.platform in ("gpu", "cuda")]
    if not gpus:
        raise SystemExit("no GPU visible to JAX: fix the CUDA / jaxlib install before driving (scripts/jetson_setup.sh)")
    x = jax.device_put(jnp.ones((1024, 1024), jnp.float32), gpus[0])
    f = jax.jit(lambda a: (a @ a).sum())
    f(x).block_until_ready()
    start = time.perf_counter()
    for _ in range(20):
        f(x).block_until_ready()
    print(f"GPU ok: 1024x1024 matmul {(time.perf_counter() - start) / 20 * 1000:.2f} ms")


def time_config(config, steps, skip, label):
    import jax.numpy as jnp
    from sit_lmpc.experiments.randomness import ExperimentRandomness
    from sit_lmpc.experiments.runner import build_vehicle_dynamics
    from sit_lmpc.planners.lmppi import LMPPI
    from sit_lmpc.planners.safe_set import SafeSet
    from sit_lmpc.track import Track, map_path
    from sit_lmpc.training.value import ValueTrainer
    from sit_lmpc.planners.warm_laps import synthetic_laps

    planner, simulation = config.planner, config.simulation
    track = Track.from_track_path(map_path(simulation.map))
    _, model = build_vehicle_dynamics(config)
    rssm_filter = None
    if planner.dynamics.backend == "rssm":
        from sit_lmpc.world.rssm_runtime import RSSMFilter
        rssm_filter = RSSMFilter(planner.dynamics.rssm, model.params, model.action_scale, planner.dynamics.rssm_variant,
                                 planner.dynamics.rssm_model_timestep, model.integrator_timestep,
                                 planner.dynamics.rssm_clearance_margin, planner.dynamics.rssm_clearance_weight)
        model = rssm_filter.predictor()
    randomness = ExperimentRandomness(simulation.seed or 0)
    learned = LMPPI(planner, model, track, simulation.seed or 0, config.value.samples, randomness=randomness,
                    value_input_mode=config.value.input_mode)
    frenet_laps, cart_laps = synthetic_laps(track, max(planner.safe_set_cap_laps, 4) + 1,
                                            min(planner.warmup.max_speed, simulation.car.max_speed))
    safe = SafeSet(planner, simulation.timestep, float(track.s_frame_max), simulation.seed or 0, randomness=randomness)
    safe.frenet, safe.cartesian = frenet_laps, cart_laps
    if planner.safe_set_footprint_filter and learned.corners is not None:
        safe.footprint = (learned.corners, planner.boundary_limit)
    with tempfile.TemporaryDirectory() as work:
        trainer = ValueTrainer(replace(config.value, epochs=20), planner.half_width, simulation.seed or 0, work,
                               randomness=randomness.fork("bench_value"),
                               prediction_batch_size=planner.samples * planner.horizon)
        trainer.start(safe.training_data(config.value.input_mode))
        trainer.wait()
        snapshot = trainer.snapshot
    lap_f, lap_c = frenet_laps[-1], cart_laps[-1]
    times = []
    for i in range(min(steps + skip, len(lap_f) - 1)):
        started = time.perf_counter()
        state, frenet = lap_c[i, :7].copy(), lap_f[i, :7].copy()
        if rssm_filter is not None:
            observation = np.array([0.0, state[3] * state[5], state[5], state[3], state[3] * 0.1 / 0.1, 0.0,
                                    state[5], 0.0, 0.0], np.float32)
            scan = np.full(1080, 2.0, np.float32)
            rear = (state[0] - 0.17 * np.cos(state[4]), state[1] - 0.17 * np.sin(state[4]), state[4])
            anchor, _, _ = rssm_filter.step(observation, scan, rear, np.zeros(2))
            learned.model = rssm_filter.predictor()
            if planner.dynamics.rssm_state:
                fields = [3, 5, 6] if planner.dynamics.rssm_command_steering else [2, 3, 5, 6]
                values = np.asarray(anchor)[1:] if planner.dynamics.rssm_command_steering else anchor
                state[fields] = values
                frenet[fields] = values
        learned.update(jnp.asarray(state), frenet, snapshot, safe)
        times.append((time.perf_counter() - started) * 1000)
    measured = np.asarray(times[skip:])
    row = {"config": label, "samples": planner.samples, "horizon": planner.horizon,
           "model_timestep": planner.dynamics.rssm_model_timestep if rssm_filter is not None else simulation.timestep,
           "steps": int(len(measured)), "p50_ms": round(float(np.percentile(measured, 50)), 1),
           "p95_ms": round(float(np.percentile(measured, 95)), 1), "max_ms": round(float(measured.max()), 1)}
    print(f"  {label:42s} samples {row['samples']:4d} horizon {row['horizon']:2d} x {row['model_timestep']:.2f} s: "
          f"p50 {row['p50_ms']:6.1f} ms  p95 {row['p95_ms']:6.1f} ms  max {row['max_ms']:6.1f} ms")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=sorted(str(p) for p in (ROOT / "configs/car").glob("*.yaml")))
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--skip", type=int, default=20, help="first steps excluded (compilation)")
    parser.add_argument("--budget", type=float, default=90.0, help="p95 budget in ms (control period 100 ms)")
    parser.add_argument("--no-gpu-check", action="store_true", help="skip the GPU check (testing on a CPU only)")
    parser.add_argument("--out", default=str(ROOT / "results/jetson-check.json"))
    args = parser.parse_args()
    if not args.no_gpu_check:
        gpu_check()
    from sit_lmpc.config import load_config

    rows = []
    for path in args.configs:
        config = load_config(path)
        name = Path(path).stem
        print(f"== {name}")
        row = time_config(config, args.steps, args.skip, name)
        rows.append(row)
        if row["p95_ms"] <= args.budget:
            print(f"  within budget ({args.budget:.0f} ms)")
            continue
        planner = config.planner
        cuts = [("samples: 512", {"samples": 512}), ("samples: 256", {"samples": 256})]
        if planner.dynamics.backend == "rssm":
            cuts.append(("horizon: 10, dynamics.rssm_model_timestep: 0.15",
                         {"horizon": 10, "dynamics": replace(planner.dynamics, rssm_model_timestep=0.15)}))
        else:
            cuts.append(("horizon: 10", {"horizon": 10}))
        best = None
        for text, change in cuts:
            trial = replace(config, planner=replace(planner, **change))
            cut_row = time_config(trial, args.steps, args.skip, f"{name} ({text})")
            cut_row["change"] = text
            rows.append(cut_row)
            if cut_row["p95_ms"] <= args.budget and best is None:
                best = text
        print(f"  suggested override in {path}: planner: {best}" if best else
              "  no single cut meets the budget; combine samples 256 with the shorter horizon")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
