import json
import sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
events = [json.loads(line) for line in (run / "events.jsonl").open()]
lmppi_start = None
for e in events:
    if e["event"] in ("ros_launch", "error", "finish", "learned_controller_started", "lap_reset"):
        keep = {k: v for k, v in e.items() if k in ("event", "status", "message", "type", "step", "laps", "elapsed", "seconds", "speed")}
        print(keep)
    if e["event"] == "learned_controller_started":
        lmppi_start = e["step"]
metrics = json.loads((run / "lap_metrics.json").read_text())
print(f"{'trial':>5} {'outcome':>10} {'laptime':>8} {'steps':>6} {'vmean':>5} {'|ey|max':>7} {'oob':>4} {'ssmax':>6} {'bndmax':>6} {'reason':>14}")
for m in metrics:
    lt = m.get("lap_time")
    print(f"{m.get('lap', '?'):>5} {m['outcome']:>10} {(lt if lt is not None else m['duration']):>8.2f} {m['steps']:>6} "
          f"{(m.get('mean_speed') or 0):>5.2f} {(m.get('max_abs_lateral_offset') or 0):>7.3f} "
          f"{m.get('out_of_bounds_steps', 0):>4} {(m.get('safe_set_max') or 0):>6.3f} {(m.get('boundary_max') or 0):>6.3f} "
          f"{m.get('termination_reason', ''):>14}")
timing = run / "control_timing.json"
if timing.exists():
    t = json.loads(timing.read_text())
    p, s = t["planning_seconds"], t["simulator_step_seconds"]
    print(f"planning ms: p50 {1e3*p['p50']:.1f} p95 {1e3*p['p95']:.1f} max {1e3*p['max']:.0f} (n={p['count']}); "
          f"sim step ms: p50 {1e3*s['p50']:.1f} p95 {1e3*s['p95']:.1f}")
    plan = np.load(run / "control_timing.npz")["planning"]
    if lmppi_start is not None and lmppi_start + 2 < len(plan):
        k = lmppi_start
        print(f"planning ms warm-up (steps 1..{k-1}): p50 {1e3*np.percentile(plan[1:k],50):.1f} p95 {1e3*np.percentile(plan[1:k],95):.1f}; "
              f"LMPPI (steps >{k}): p50 {1e3*np.percentile(plan[k+1:],50):.1f} p95 {1e3*np.percentile(plan[k+1:],95):.1f} "
              f"max {1e3*plan[k+1:].max():.0f} (first LMPPI step {1e3*plan[k]:.0f})")
