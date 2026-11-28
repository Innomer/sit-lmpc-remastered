"""Where one control step's time goes, per function, on this machine (planner update and, for v11, the filter step).

    python scripts/profile_step.py --config configs/car/levine-bicycle.yaml
    python scripts/profile_step.py --config configs/car/levine-v11.yaml --steps 80

Same setup as scripts/jetson_check.py (synthetic centreline laps, short value fit). After --skip warm-up steps, it
profiles --steps control steps and prints, per function: total time per step including what it calls, its own time
per step, and calls per step. Run it with the car stack up to see the time under real load.
"""

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/car/levine-bicycle.yaml"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--skip", type=int, default=30)
    parser.add_argument("--top", type=int, default=35)
    args = parser.parse_args()

    import jetson_check
    import sit_lmpc.planners.lmppi as lmppi
    from sit_lmpc.config import load_config

    profile = cProfile.Profile()
    seconds = {"update": [], "filter": []}
    calls = [0]
    original_update = lmppi.LMPPI.update

    def update(self, *a, **k):
        calls[0] += 1
        active = calls[0] > args.skip
        started = time.perf_counter()
        if active:
            profile.enable()
        try:
            return original_update(self, *a, **k)
        finally:
            if active:
                profile.disable()
                seconds["update"].append(time.perf_counter() - started)

    lmppi.LMPPI.update = update
    config = load_config(args.config)
    if config.planner.dynamics.backend == "rssm":
        import sit_lmpc.world.rssm_runtime as runtime
        original_step = runtime.RSSMFilter.step

        def step(self, *a, **k):
            active = calls[0] >= args.skip
            started = time.perf_counter()
            if active:
                profile.enable()
            try:
                return original_step(self, *a, **k)
            finally:
                if active:
                    profile.disable()
                    seconds["filter"].append(time.perf_counter() - started)

        runtime.RSSMFilter.step = step
    jetson_check.time_config(config, args.steps, args.skip, Path(args.config).stem)
    steps = max(len(seconds["update"]), 1)
    for name, values in seconds.items():
        if values:
            print(f"{name}: median {np.median(values) * 1000:.1f} ms, p95 {np.percentile(values, 95) * 1000:.1f} ms "
                  f"over {len(values)} steps")
    stats = pstats.Stats(profile)
    rows = []
    for (path, line, name), (_, count, own, total, _) in stats.stats.items():
        short = path.split("site-packages/")[-1].split("sit_lmpc/")[-1]
        rows.append((total / steps * 1000, own / steps * 1000, count / steps, f"{short}:{line} {name}"))
    print(f"{'ms/step':>9s} {'own ms':>8s} {'calls':>7s}  function")
    for total, own, count, where in sorted(rows, reverse=True)[:args.top]:
        print(f"{total:9.2f} {own:8.2f} {count:7.1f}  {where}")


if __name__ == "__main__":
    main()
