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
                 speed_factor=None, speed_step=0.05, scope="track", global_share=0.5, global_min=8,
                 strong_weight=0.1, soft_weight=0.03, raise_band=0.5, raise_step=0.3, min_clearance=0.2,
                 pass_share=0.9, min_matched=3, fit_min_points=20):
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
        self.strong_weight, self.soft_weight = float(strong_weight), float(soft_weight)
        self.raise_band, self.raise_step = float(raise_band), float(raise_step)
        self.min_clearance, self.pass_share, self.min_matched = float(min_clearance), float(pass_share), int(min_matched)
        self.soft = np.zeros(self.count, bool)
        self.base = np.full(self.count, np.inf)
        self.pass_speeds = []
        self.pass_clearance = []
        self.pass_slip = []
        self.lap_pairs = [[] for _ in range(self.count)]
        self.pair_history = [[] for _ in range(self.count)]
        self.lap_slip = [[] for _ in range(self.count)]
        self.slip_history = [[] for _ in range(self.count)]
        self.fit_min_points = int(fit_min_points)
        self.uncounted = np.zeros(self.count, bool)
        self.pre_limit = None
        self.last_pass_tag = np.full(self.count, np.nan)
        self.pending = []
        self.records = np.zeros((0, 3))

    def index(self, s):
        return int((float(s) % self.length) // self.stretch) % self.count

    def reference(self, k):
        values = self.history[k][-self.history_laps:]
        return float(np.median(values)) if values else None

    def limit(self, k):
        ref = self.reference(k)
        return self.floor if ref is None else max(self.floor, self.ratio * ref)

    def observe(self, step, s, tag, learned, speed=None, factor=None, model_error=None, clearance=None, slip=None):
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
            self.soft[self.pass_stretch] = False
            self.pass_values = []
            self.model_pass = []
            self.pass_speeds = []
            self.pass_clearance = []
            self.pass_slip = []
            self.pass_start = int(step)
        self.pass_stretch = k
        self.pass_values.append(tag)
        if model_error is not None and np.isfinite(model_error):
            self.model_lap_values[k].append(float(model_error))
            self.model_pass.append(float(model_error))
            if speed is not None and np.isfinite(speed):
                self.lap_pairs[k].append((float(speed), float(model_error)))
        if speed is not None and np.isfinite(speed):
            self.pass_speeds.append(float(speed))
        if clearance is not None and np.isfinite(clearance):
            self.pass_clearance.append(float(clearance))
        if slip is not None and np.isfinite(slip):
            self.pass_slip.append(abs(float(slip)))
            self.lap_slip[k].append(abs(float(slip)))
        if (self.scope == "evidence" and np.isfinite(self.caps[k]) and not self.soft[k] and speed is not None
                and float(speed) <= self.caps[k]):
            self.soft[k] = True
            self.events.append({"step": int(step), "event": "under cap", "stretch": k, "lap": self.lap,
                                "cap": float(self.caps[k]), "speed": float(speed)})
        if not learned or len(self.recent) < self.window:
            return
        short = float(np.median(self.recent))
        if short > self.limit(k) and self.mode[k] == NORMAL and not self.changed[k]:
            first = not self.active
            if first:
                self.pre_limit = np.asarray([self.limit(j) for j in range(self.count)])
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
                        self.soft[j] = False
                        self.events.append({"step": int(step), "event": "cap set", "stretch": int(j), "lap": self.lap,
                                            "cap": float(self.caps[j])})
            self.events.append({"step": int(step), "event": "detected" if first else "detected again", "stretch": k,
                                "lap": self.lap, "short_m": short, "limit_m": self.limit(k)})

    def _cap(self, j, factor):
        if self.speed_history[j] and not np.isfinite(self.caps[j]):
            self.cap_factor[j] = factor
            self.base[j] = float(np.median(self.speed_history[j][-self.history_laps:]))
            self.caps[j] = factor * self.base[j]
        elif self.scope == "track" and self.speed_history[j]:
            self.cap_factor[j] = factor
            self.caps[j] = factor * float(np.median(self.speed_history[j][-self.history_laps:]))

    def lap_tags(self):
        """Per-stretch median tag of the lap being driven (nan where the lap did not pass)."""
        return np.asarray([float(np.median(v)) if v else np.nan for v in self.lap_values])

    def query_tags(self):
        """Per-stretch tag describing the current condition, for the tag-conditioned value: the latest pass after
        the change where one exists, else the stretch's normal reference."""
        out = np.zeros(self.count)
        for k in range(self.count):
            if np.isfinite(self.last_pass_tag[k]):
                out[k] = self.last_pass_tag[k]
            else:
                ref = self.reference(k)
                out[k] = 0.0 if ref is None else ref
        return out

    def model_limit(self, k):
        values = self.model_history[k][-self.history_laps:]
        return max(values) if values else None

    def matched_limit(self, k, speed):
        medians = []
        for pairs in self.pair_history[k][-self.history_laps:]:
            near = pairs[np.abs(pairs[:, 0] - speed) <= self.raise_band, 1]
            if len(near) >= self.min_matched:
                medians.append(float(np.median(near)))
        return max(medians) if medians else None

    def slip_limit(self, k):
        values = self.slip_history[k][-self.history_laps:]
        return max(values) if values else None

    def phase(self, s):
        k = self.index(s)
        if not np.isfinite(self.caps[k]):
            return 0
        return 2 if self.soft[k] else 1

    def sample_weights(self):
        return np.where(np.isfinite(self.caps), np.where(self.soft, self.soft_weight, self.strong_weight), 0.0)

    def rank_mask(self):
        return (np.isfinite(self.caps) & ~self.soft).astype(float)

    def strong_ahead(self, s, distance, speed):
        first = int(np.floor(float(s) / self.stretch))
        last = int(np.floor((float(s) + max(float(distance), 0.0)) / self.stretch))
        for j in range(first, last + 1):
            k = j % self.count
            if np.isfinite(self.caps[k]) and not self.soft[k] and self.caps[k] < float(speed):
                return True
        return False

    def _leave(self, step, k):
        if self.active:
            self.driven.add(k)
            if self.pass_values:
                self.last_pass_tag[k] = float(np.median(self.pass_values))
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
            self.uncounted[k] = True
            return
        if capped:
            share = float(np.mean(np.asarray(self.pass_speeds) <= self.caps[k])) if self.pass_speeds else 0.0
            if share < self.pass_share:
                self.uncounted[k] = True
                self.events.append({"step": int(step), "event": "pass not counted", "stretch": k, "lap": self.lap,
                                    "cap": float(self.caps[k]), "share_under_cap": share})
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
        if not (self.changed[k] and capped and self.pass_speeds):
            return
        self.pending.append({"stretch": k, "start": int(self.pass_start), "end": int(step), "lap": self.lap,
                             "speed": float(np.mean(self.pass_speeds)),
                             "clearance": min(self.pass_clearance) if self.pass_clearance else None,
                             "slip": max(self.pass_slip) if self.pass_slip else None})

    def error_fit(self, k):
        if not self.pair_history[k]:
            return None
        pairs = np.concatenate(self.pair_history[k])
        if len(pairs) < self.fit_min_points or np.ptp(pairs[:, 0]) < 0.5:
            return None
        return np.polyfit(pairs[:, 0], pairs[:, 1], 1)

    def model_errors(self, steps, s, speeds, errors):
        """v11's 1.0 s errors for the steps of the lap just driven (replayed in one batch at lap end): fill the
        pre-change error history and decide the cap raises of the passes driven so far."""
        rows = np.column_stack([steps, speeds, errors]).astype(float)
        for j in range(len(rows)):
            if np.isfinite(rows[j, 2]):
                k = self.index(s[j])
                self.model_lap_values[k].append(float(rows[j, 2]))
                self.lap_pairs[k].append((float(rows[j, 1]), float(rows[j, 2])))
        self.records = np.concatenate([self.records, rows])[-4000:]
        last = self.records[-1, 0] if len(self.records) else -1
        waiting = []
        for item in self.pending:
            if item["end"] - 1 > last:
                waiting.append(item)
                continue
            inside = (self.records[:, 0] >= item["start"]) & (self.records[:, 0] < item["end"])
            values = self.records[inside, 2]
            values = values[np.isfinite(values)]
            if len(values):
                self._raise(int(last), item, float(np.median(values)))
        self.pending = waiting

    def _raise(self, step, item, error):
        k = item["stretch"]
        if not (self.changed[k] and np.isfinite(self.caps[k])):
            return
        speed, clearance, slip = item["speed"], item["clearance"], item["slip"]
        fit = self.error_fit(k)
        limit = None if fit is None else float(np.polyval(fit, speed))
        slip_limit = self.slip_limit(k)
        why = []
        if limit is None:
            why.append("no pre-change error fit")
        elif error > limit:
            why.append("model error above pre-change fit")
        if clearance is None or clearance < self.min_clearance:
            why.append("clearance under limit")
        if slip is None or slip_limit is None or slip > slip_limit:
            why.append("slip above pre-change range")
        detail = {"stretch": k, "lap": item["lap"], "pass_speed": speed, "model_error_m": error, "model_limit_m": limit,
                  "min_clearance_m": clearance, "max_slip": slip, "slip_limit": slip_limit}
        if why:
            self.events.append(dict({"step": int(step), "event": "cap kept", "cap": float(self.caps[k]),
                                     "reason": ", ".join(why)}, **detail))
            return
        self.cap_step[k] = int(step)
        if self.caps[k] + self.raise_step >= self.base[k]:
            self.caps[k] = np.inf
            self.cap_factor[k] = 1.0
            self.events.append(dict({"step": int(step), "event": "cap removed",
                                     "reason": "clean pass, model error at or below pre-change fit; pre-change speed reached"},
                                    **detail))
        else:
            self.caps[k] = self.caps[k] + self.raise_step
            self.cap_factor[k] = self.caps[k] / self.base[k]
            self.events.append(dict({"step": int(step), "event": "cap raised", "cap": float(self.caps[k]),
                                     "reason": "clean pass, model error at or below pre-change fit"}, **detail))

    def finish_lap(self, step, new_condition_laps):
        """At the end of a lap: store this lap's per-stretch medians as history (a changed stretch skips the lap the
        change happened in and then starts a fresh history), and switch changed stretches to the new-condition laps
        once enough exist."""
        for k in range(self.count):
            if self.lap_values[k] and self.uncounted[k]:
                self.lap_values[k] = []
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
            before = not (self.active and self.lap >= (self.detection_lap or 0))
            if self.model_lap_values[k] and before:
                self.model_history[k].append(float(np.median(self.model_lap_values[k])))
            self.model_lap_values[k] = []
            if self.lap_pairs[k] and before:
                self.pair_history[k].append(np.asarray(self.lap_pairs[k]))
            self.lap_pairs[k] = []
            if self.lap_slip[k] and before:
                self.slip_history[k].append(float(np.max(self.lap_slip[k])))
            self.lap_slip[k] = []
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
        self.uncounted[:] = False
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


BODY_X = (-0.2965, 0.2835)
BODY_Y = 0.155


def wall_distance_field(track):
    from scipy.ndimage import distance_transform_edt
    occupied = np.asarray(track.occupancy_map) == 0
    return distance_transform_edt(~occupied) * float(track.spec.resolution)


def body_clearance(field, spec, cartesian):
    xs = np.linspace(BODY_X[0], BODY_X[1], 9)
    ys = np.linspace(-BODY_Y, BODY_Y, 5)
    bx = np.concatenate([xs, xs, np.full(5, BODY_X[0]), np.full(5, BODY_X[1])])
    by = np.concatenate([np.full(9, -BODY_Y), np.full(9, BODY_Y), ys, ys])
    x = float(cartesian[0])
    y = float(cartesian[1])
    yaw = float(cartesian[4])
    px = x + bx * np.cos(yaw) - by * np.sin(yaw)
    py = y + bx * np.sin(yaw) + by * np.cos(yaw)
    col = np.clip(((px - spec.origin[0]) / spec.resolution).astype(int), 0, field.shape[1] - 1)
    row = np.clip(((py - spec.origin[1]) / spec.resolution).astype(int), 0, field.shape[0] - 1)
    return float(field[row, col].min())
