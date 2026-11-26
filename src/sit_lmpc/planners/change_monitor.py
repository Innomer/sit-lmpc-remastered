"""Live change detection for the safe set (safe set v2).

Every step the runner passes the car's track position and a condition tag (the nominal bicycle model's 1.0 s
position error ending at this step). The track is split into stretches. Each finished lap stores, per stretch, the
median tag the car saw there. A change is flagged when the median tag of the last `window` steps exceeds both
`floor` and `ratio` times the current stretch's history (median of its last laps' medians).

On a change every stretch turns conservative at once. When the car leaves a stretch after that, the stretch is
compared with its history: if its tag on this pass matches (below the same floor / ratio test), it returns to normal;
otherwise it is marked changed and stays conservative until `new_laps` laps driven fully after the change are in the
safe set, then uses those ("new"). A new jump on a normal stretch starts the same procedure again.
"""

import numpy as np

NORMAL, CONSERVATIVE, NEW = 0, 1, 2


class ChangeMonitor:
    def __init__(self, track_length, stretch=10.0, window=10, ratio=3.0, floor=0.05, history_laps=5, new_laps=2):
        self.length = float(track_length)
        self.count = int(np.ceil(self.length / stretch))
        self.stretch = self.length / self.count
        self.window, self.ratio, self.floor = int(window), float(ratio), float(floor)
        self.history_laps, self.new_laps = int(history_laps), int(new_laps)
        self.history = [[] for _ in range(self.count)]
        self.lap_values = [[] for _ in range(self.count)]
        self.mode = np.full(self.count, NORMAL)
        self.changed = np.zeros(self.count, bool)
        self.recent = []
        self.lap = 0
        self.active = False
        self.detection_lap = None
        self.pass_values = []
        self.pass_stretch = None
        self.events = []
        self.restarted = set()

    def index(self, s):
        return int((float(s) % self.length) // self.stretch) % self.count

    def reference(self, k):
        values = self.history[k][-self.history_laps:]
        return float(np.median(values)) if values else None

    def limit(self, k):
        ref = self.reference(k)
        return self.floor if ref is None else max(self.floor, self.ratio * ref)

    def observe(self, step, s, tag, learned):
        """One control step: s along the track, the tag ending at this step, whether the planner is driving."""
        k = self.index(s)
        tag = float(tag) if np.isfinite(tag) else 0.0
        self.lap_values[k].append(tag)
        self.recent = (self.recent + [tag])[-self.window:]
        if self.pass_stretch is not None and k != self.pass_stretch:
            self._leave(step, self.pass_stretch)
            self.pass_values = []
        self.pass_stretch = k
        self.pass_values.append(tag)
        if not learned or len(self.recent) < self.window:
            return
        short = float(np.median(self.recent))
        if short > self.limit(k) and self.mode[k] == NORMAL and not self.changed[k]:
            first = not self.active
            self.active = True
            self.detection_lap = self.lap if first else self.detection_lap
            relaxed = self.mode == NORMAL
            self.mode[relaxed & ~self.changed] = CONSERVATIVE
            self.events.append({"step": int(step), "event": "detected" if first else "detected again", "stretch": k,
                                "lap": self.lap, "short_m": short, "limit_m": self.limit(k)})

    def _leave(self, step, k):
        if not self.active or self.mode[k] != CONSERVATIVE or self.changed[k] or not self.pass_values:
            return
        value = float(np.median(self.pass_values))
        if value <= self.limit(k):
            self.mode[k] = NORMAL
            self.events.append({"step": int(step), "event": "relaxed", "stretch": k, "lap": self.lap, "tag_m": value})
        else:
            self.changed[k] = True
            self.events.append({"step": int(step), "event": "changed", "stretch": k, "lap": self.lap, "tag_m": value})

    def finish_lap(self, step, new_condition_laps):
        """At the end of a lap: store this lap's per-stretch medians as history (a changed stretch skips the lap the
        change happened in and then starts a fresh history), and switch changed stretches to the new-condition laps
        once enough exist."""
        for k in range(self.count):
            if self.lap_values[k]:
                value = float(np.median(self.lap_values[k]))
                if self.changed[k]:
                    if self.lap != self.detection_lap:
                        if k not in self.restarted:
                            self.history[k] = []
                            self.restarted.add(k)
                        self.history[k].append(value)
                else:
                    self.history[k].append(value)
            self.lap_values[k] = []
        if self.active and new_condition_laps >= self.new_laps:
            switch = self.changed & (self.mode == CONSERVATIVE)
            if switch.any():
                self.mode[switch] = NEW
                self.events.append({"step": int(step), "event": "new-condition laps in use",
                                    "stretches": np.flatnonzero(switch).tolist(), "lap": self.lap})
        self.lap += 1

    def window_mode(self, lo, hi):
        """Mode for a hull window [lo, hi] in unwrapped s: conservative if any stretch in it is conservative, else
        new if any uses new-condition laps, else normal."""
        if not self.active:
            return NORMAL
        span = np.arange(np.floor(lo / self.stretch), np.floor(hi / self.stretch) + 1).astype(int) % self.count
        modes = self.mode[span]
        if (modes == CONSERVATIVE).any():
            return CONSERVATIVE
        if (modes == NEW).any():
            return NEW
        return NORMAL

    def snapshot(self):
        return {"active": self.active, "conservative": int(np.sum(self.mode == CONSERVATIVE)),
                "new": int(np.sum(self.mode == NEW)), "changed": int(np.sum(self.changed))}
