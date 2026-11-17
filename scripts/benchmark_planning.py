"""Time LMPPI planning per control step without a simulator or ROS: resume a recorded fast-start checkpoint
and replay a recorded lap as the plant. Runs the real runner, so the numbers include its per-step overhead.

    python scripts/benchmark_planning.py --models known bnn-nochance --samples 1024 512 256 --horizons 10 8
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import yaml


def run(checkpoint, replay, samples, horizon, particles, steps, work, skip):
    planner = {"samples": samples, "horizon": horizon, "warmup": {"controller": "mppi"}}
    if particles:
        planner["dynamics"] = {"particles": particles}
    settings = {
        "simulation": {"backend": "replay", "replay_path": str(replay)},
        "planner": planner,
        "diagnostics": {"trace_steps": 0, "summary_plots": False},
    }
    config = work / "bench.yaml"
    config.write_text(yaml.safe_dump(settings))
    output = work / "run"
    shutil.rmtree(output, ignore_errors=True)
    start = json.loads((Path(checkpoint) / "state.json").read_text())["next_step"]
    command = [sys.executable, "-m", "sit_lmpc.cli.racecar", "--resume", str(checkpoint), "--config", str(config),
               "--max-steps", str(start + steps), "--save-dir", str(output), "--headless"]
    result = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ))
    timing = output / "control_timing.npz"
    if result.returncode != 0 or not timing.exists():
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-8:])
        return None, tail
    planning = np.load(timing)["planning"][skip:] * 1000
    return planning, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["known", "bnn-nochance", "bnn-chance"])
    parser.add_argument("--samples", nargs="+", type=int, default=[1024, 512, 256])
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 8])
    parser.add_argument("--particles", nargs="+", type=int, default=[64, 32, 16])
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--skip", type=int, default=20)
    parser.add_argument("--bench", default="bench")
    parser.add_argument("--out", default=None, help="write the table as JSON here")
    args = parser.parse_args()

    bench = Path(args.bench)
    rows = []
    work = Path(tempfile.mkdtemp(prefix="sit-bench-"))
    print(f"{'model':13s} {'samples':>7s} {'horizon':>7s} {'particles':>9s} {'p50 ms':>7s} {'p95 ms':>7s} {'max ms':>7s} {'>100ms':>7s}")
    try:
        for model in args.models:
            checkpoint = bench / "checkpoints" / model
            particle_grid = args.particles if model.startswith("bnn") else [0]
            for samples, horizon, particles in itertools.product(args.samples, args.horizons, particle_grid):
                planning, error = run(checkpoint, bench / "replay-lap.npz", samples, horizon, particles,
                                      args.steps, work, args.skip)
                if planning is None:
                    print(f"{model:13s} {samples:7d} {horizon:7d} {particles:9d}  failed:\n{error}")
                    continue
                row = {"model": model, "samples": samples, "horizon": horizon, "particles": particles,
                       "p50_ms": float(np.percentile(planning, 50)), "p95_ms": float(np.percentile(planning, 95)),
                       "max_ms": float(planning.max()), "over_100ms": float(np.mean(planning > 100))}
                rows.append(row)
                print(f"{model:13s} {samples:7d} {horizon:7d} {particles:9d} {row['p50_ms']:7.1f} {row['p95_ms']:7.1f} "
                      f"{row['max_ms']:7.1f} {row['over_100ms']:7.1%}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
