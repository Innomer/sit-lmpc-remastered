"""Mid-session condition changes for the ROS sim, the same for every controller.

Reads a scenario YAML and, when the car completes the given lap, publishes friction/mass changes to the
bridge (/sim/vehicle_condition) and low-friction patches (/sim/friction_zones). Patches are given as a
stretch of track [s_start, s_end] along the centerline; they become small boxes around the centerline.
Laps are counted from increases of /<namespace>/lap_count, so resets after failures do not restart the count.

    python scripts/condition_scheduler.py configs/scenarios/spielberg-friction-drop.yaml

Scenario:
    map: Spielberg
    events:
      - lap: 8
        condition: {mu: 0.75}
      - lap: 8
        patches: [{s_start: 60.0, s_end: 90.0, mu: 0.6}]
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import yaml


def patch_boxes(track, patch, margin, spacing=0.5):
    boxes = []
    half = patch.get("half_width", 1.5) + margin
    for s in np.arange(patch["s_start"], patch["s_end"], spacing):
        x, y, _ = track.frenet_to_cartesian(float(s), 0.0, 0.0)
        boxes.append({"x_min": float(x - half), "y_min": float(y - half),
                      "x_max": float(x + half), "y_max": float(y + half), "mu": float(patch["mu"])})
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario")
    parser.add_argument("--namespace", default="ego_racecar")
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--log", default=None, help="append applied events as JSON lines here")
    args = parser.parse_args()

    import rclpy
    from std_msgs.msg import Int32, String
    from sit_lmpc.track.resources import map_path
    from sit_lmpc.track.track import Track

    scenario = yaml.safe_load(Path(args.scenario).read_text())
    track = Track.from_track_path(map_path(scenario["map"]))
    events = sorted(scenario["events"], key=lambda event: event["lap"])
    rclpy.init()
    node = rclpy.create_node("condition_scheduler")
    condition_pub = node.create_publisher(String, "/sim/vehicle_condition", 10)
    zones_pub = node.create_publisher(String, "/sim/friction_zones", 10)
    state = {"previous": 0, "laps": 0}
    zones = []

    def apply(event):
        if "condition" in event:
            condition_pub.publish(String(data=json.dumps(event["condition"])))
        if "patches" in event:
            zones.clear()
            for patch in event["patches"]:
                zones.extend(patch_boxes(track, patch, args.margin))
            zones_pub.publish(String(data=json.dumps(zones)))
        record = {"time": time.time(), "lap": state["laps"], "event": event}
        print(f"[condition] after lap {state['laps']}: {event}", flush=True)
        if args.log:
            with open(args.log, "a") as stream:
                stream.write(json.dumps(record) + "\n")

    def on_lap(msg):
        if msg.data > state["previous"]:
            state["laps"] += msg.data - state["previous"]
        state["previous"] = msg.data
        while events and events[0]["lap"] <= state["laps"]:
            apply(events.pop(0))

    node.create_subscription(Int32, f"/{args.namespace}/lap_count", on_lap, 10)
    while condition_pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.2)
    while events and events[0]["lap"] <= 0:
        apply(events.pop(0))
    print(f"[condition] waiting; {len(events)} events by lap", flush=True)
    try:
        while rclpy.ok() and events:
            rclpy.spin_once(node, timeout_sec=0.2)
        time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
