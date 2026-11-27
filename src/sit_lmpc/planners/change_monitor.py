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
    def __init__(self, track_length, stretch=10.0, window=10, ratio=3.0, floor=0.05, history_laps=5, new_laps=2,
                 speed_factor=None, speed_step=0.05, scope="track", global_share=0.5, global_min=8):
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
        self.speed_factor, self.speed_step = speed_factor, float(speed_step)
        self.speed_history = [[] for _ in range(self.count)]
        self.lap_speeds = [[] for _ in range(self.count)]
        self.caps = np.full(self.count, np.inf)
        self.cap_factor = np.ones(self.count)
        self.clean_new_laps = 0
        self.detection_factor = None
        self.scope, self.global_share, self.global_min = scope, float(global_share), int(global_min)
        self.driven = set()
        self.relaxed_since = set()
        self.global_caps = False
        self.model_history = [[] for _ in range(self.count)]
        self.model_lap_values = [[] for _ in range(self.count)]
        self.model_pass = []
        self.pass_start = 0
        self.cap_step = np.full(self.count, -1)

    def index(self, s):
        return int((float(s) % self.length) // self.stretch) % self.count

    def reference(self, k):
        values = self.history[k][-self.history_laps:]
        return float(np.median(values)) if values else None

    def limit(self, k):
        ref = self.reference(k)
        return self.floor if ref is None else max(self.floor, self.ratio * ref)

    def observe(self, step, s, tag, learned, speed=None, factor=None, model_error=None):
        """One control step: s along the track, the tag ending at this step, whether the planner is driving, the
        car's speed, and (optional) the speed factor to use if a change is detected now."""
        k = self.index(s)
        if speed is not None and np.isfinite(speed):
            self.lap_speeds[k].append(float(speed))
        tag = float(tag) if np.isfinite(tag) else 0.0
        self.lap_values[k].append(tag)
        self.recent = (self.recent + [tag])[-self.window:]
        if self.pass_stretch is not None and k != self.pass_stretch:
            self._leave(step, self.pass_stretch)
            self.pass_values = []
            self.model_pass = []
            self.pass_start = int(step)
        self.pass_stretch = k
        self.pass_values.append(tag)
        if model_error is not None and np.isfinite(model_error):
            self.model_lap_values[k].append(float(model_error))
            self.model_pass.append(float(model_error))
        if not learned or len(self.recent) < self.window:
            return
        short = float(np.median(self.recent))
        if short > self.limit(k) and self.mode[k] == NORMAL and not self.changed[k]:
            first = not self.active
            self.active = True
            self.detection_lap = self.lap if first else self.detection_lap
            if self.scope == "track" or first:
                relaxed = self.mode == NORMAL
                self.mode[relaxed & ~self.changed] = CONSERVATIVE
            else:
                self.mode[k] = CONSERVATIVE
                self.relaxed_since.discard(k)
            if self.speed_factor is not None:
                chosen = float(factor) if factor is not None else float(self.speed_factor)
                self.detection_factor = chosen
                if self.scope == "track":
                    targets = np.flatnonzero(self.mode == CONSERVATIVE)
                elif self.scope == "evidence":
                    targets = np.flatnonzero(self.mode == CONSERVATIVE) if first else [k]
                elif self.global_caps:
                    targets = [j for j in range(self.count) if j not in self.relaxed_since]
                else:
                    targets = [k]
                for j in targets:
                    self._cap(j, chosen)
                    if self.scope == "evidence" and np.isfinite(self.caps[j]):
                        self.cap_step[j] = int(step)
                        self.events.append({"step": int(step), "event": "cap set", "stretch": int(j), "lap": self.lap,
                                            "cap": float(self.caps[j])})
            self.events.append({"step": int(step), "event": "detected" if first else "detected again", "stretch": k,
                                "lap": self.lap, "short_m": short, "limit_m": self.limit(k)})

    def _cap(self, j, factor):
        if self.speed_history[j] and not np.isfinite(self.caps[j]):
            self.cap_factor[j] = factor
            self.caps[j] = factor * float(np.median(self.speed_history[j][-self.history_laps:]))
        elif self.scope == "track" and self.speed_history[j]:
            self.cap_factor[j] = factor
            self.caps[j] = factor * float(np.median(self.speed_history[j][-self.history_laps:]))

    def model_limit(self, k):
        values = self.model_history[k][-self.history_laps:]
        return max(values) if values else None

    def _leave(self, step, k):
        if self.active:
            self.driven.add(k)
        if self.scope == "evidence":
            self._leave_evidence(step, k)
            return
        if not self.active or self.mode[k] != CONSERVATIVE or self.changed[k] or not self.pass_values:
            return
        value = float(np.median(self.pass_values))
        if value <= self.limit(k):
            self.mode[k] = NORMAL
            self.caps[k] = np.inf
            self.cap_factor[k] = 1.0
            self.relaxed_since.add(k)
            self.events.append({"step": int(step), "event": "relaxed", "stretch": k, "lap": self.lap, "tag_m": value})
        else:
            self.changed[k] = True
            self.events.append({"step": int(step), "event": "changed", "stretch": k, "lap": self.lap, "tag_m": value})
            if self.scope == "local" and self.speed_factor is not None:
                self._cap(k, self.detection_factor)
                driven = sorted(self.driven)
                share = float(np.mean(self.changed[driven])) if driven else 0.0
                if not self.global_caps and len(driven) >= self.global_min and share > self.global_share:
                    self.global_caps = True
                    for j in range(self.count):
                        if j not in self.relaxed_since:
                            self._cap(j, self.detection_factor)
                    self.events.append({"step": int(step), "event": "track-wide caps", "lap": self.lap,
                                        "share_changed": share, "driven": len(driven)})

    def _leave_evidence(self, step, k):
        if not self.active or not self.pass_values:
            return
        capped = np.isfinite(self.caps[k])
        if capped and self.pass_start < self.cap_step[k]:
            return
        if self.mode[k] == CONSERVATIVE and not self.changed[k]:
            value = float(np.median(self.pass_values))
            if value <= self.limit(k):
                self.mode[k] = NORMAL
                self.relaxed_since.add(k)
                self.events.append({"step": int(step), "event": "relaxed", "stretch": k, "lap": self.lap,
                                    "tag_m": value})
                if capped:
                    self.caps[k] = np.inf
                    self.cap_factor[k] = 1.0
                    self.events.append({"step": int(step), "event": "cap removed", "stretch": k, "lap": self.lap,
                                        "reason": "tag matches history"})
                return
            self.changed[k] = True
            self.events.append({"step": int(step), "event": "changed", "stretch": k, "lap": self.lap, "tag_m": value})
        if not (self.changed[k] and capped and self.model_pass):
            return
        limit = self.model_limit(k)
        error = float(np.median(self.model_pass))
        if limit is None or error > limit:
            return
        base = self.caps[k] / self.cap_factor[k]
        self.cap_factor[k] = min(1.0, self.cap_factor[k] + self.speed_step)
        self.cap_step[k] = int(step)
        if self.cap_factor[k] >= 1.0:
            self.caps[k] = np.inf
            self.events.append({"step": int(step), "event": "cap removed", "stretch": k, "lap": self.lap,
                                "reason": "model error in pre-change range", "model_error_m": error,
                                "model_limit_m": limit})
        else:
            self.caps[k] = self.cap_factor[k] * base
            self.events.append({"step": int(step), "event": "cap raised", "stretch": k, "lap": self.lap,
                                "cap": float(self.caps[k]), "factor": float(self.cap_factor[k]),
                                "model_error_m": error, "model_limit_m": limit})

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
            if self.lap_speeds[k] and not (self.active and self.lap >= (self.detection_lap or 0)):
                self.speed_history[k].append(float(np.mean(self.lap_speeds[k])))
            self.lap_speeds[k] = []
            if self.model_lap_values[k] and not (self.active and self.lap >= (self.detection_lap or 0)):
                self.model_history[k].append(float(np.median(self.model_lap_values[k])))
            self.model_lap_values[k] = []
        if (self.active and self.speed_factor is not None and self.lap > self.detection_lap
                and self.scope != "evidence"):
            self.clean_new_laps += 1
            capped = np.isfinite(self.caps) & self.changed
            for j in np.flatnonzero(capped):
                base = self.caps[j] / self.cap_factor[j]
                self.cap_factor[j] = min(1.0, self.cap_factor[j] + self.speed_step)
                self.caps[j] = np.inf if self.cap_factor[j] >= 1.0 else self.cap_factor[j] * base
            if capped.any():
                self.events.append({"step": int(step), "event": "caps raised", "lap": self.lap,
                                    "factor_mean": float(self.cap_factor[capped].mean())})
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
                "new": int(np.sum(self.mode == NEW)), "changed": int(np.sum(self.changed)),
                "capped": int(np.sum(np.isfinite(self.caps)))}

    def cap_at(self, s):
        """Speed cap (m/s, inf when none) at each track position s (any shape)."""
        idx = (np.floor(np.asarray(s) / self.stretch).astype(int)) % self.count
        return self.caps[idx]
