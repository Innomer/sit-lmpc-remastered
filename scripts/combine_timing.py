"""Per-step planner time for each world-model variant: known-model planner (benchmark_planning.py) - bicycle rollout
+ world-model part (time_world_model.py) + one filter step.

    python scripts/combine_timing.py results/wm-timing/laptop-planner-known.json results/wm-timing/laptop-parts.json
"""

import json
import sys


def main():
    planner = {(r["samples"], r["horizon"]): r for r in json.loads(open(sys.argv[1]).read())}
    parts = json.loads(open(sys.argv[2]).read())
    print(f"device: {parts['device']}")
    table = {}
    for r in parts["rows"]:
        if r["part"] == "filter":
            table[r["hidden"], "filter"] = r
        else:
            table[r["hidden"], r["steps"], r["samples"], r["part"]] = r
    rows = []
    for hidden in sorted({k[0] for k in table}, reverse=True):
        filt = table[hidden, "filter"]
        for steps, label in ((15, "15 x 0.10 s"), (10, "10 x 0.15 s")):
            variants = []
            for samples in (1024, 512, 256):
                variants.append((f"A latent, {samples}", samples, [f"latent"]))
            for top in (32, 64, 128):
                variants.append((f"B two-stage 1024, top {top}", 1024, ["cheap", f"stage2_top{top}"]))
            variants.append(("C cheap only, 1024", 1024, ["cheap"]))
            for name, samples, names in variants:
                base = planner.get((samples, steps))
                st = table.get((hidden, steps, samples, "st"))
                found = [table.get((hidden, steps, samples, n)) for n in names]
                if base is None or st is None or None in found:
                    continue
                p50 = base["p50_ms"] - st["p50_ms"] + sum(f["p50_ms"] for f in found) + filt["p50_ms"]
                p95 = base["p95_ms"] - st["p50_ms"] + sum(f["p95_ms"] for f in found) + filt["p95_ms"]
                rows.append((hidden, label, name, p50, p95))
    print(f"{'GRU':>4s}  {'horizon':12s} {'variant':28s} {'p50 ms':>7s} {'p95 ms':>7s}")
    for hidden, label, name, p50, p95 in rows:
        print(f"{hidden:4d}  {label:12s} {name:28s} {p50:7.1f} {p95:7.1f}")


if __name__ == "__main__":
    main()
