"""
Record what happened during a session, so the video can be annotated
afterwards and the claims in it can be checked against the data.

Three things are written, for three different jobs:

  <stem>.csv    one row per decoder window. Opens in Excel, good for a
                quick look and for lining the video up with the numbers.
  <stem>.npz    the same rows as arrays, plus the raw EEG and the baseline
                mu power. This is the one to re-analyse from.
  <stem>.json   metadata: the decoder used, its offline accuracy, the
                montage, the band, when the session ran.

The raw EEG matters. Without it the session can only ever be replayed as
summary statistics; with it, the whole thing can be re-decoded later with a
better model, or re-plotted at a different timescale for the video. It is
stored as float32, which at 500Hz and 14 channels is about 2.5MB/minute.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROWS = [
    "t",             # seconds since session start
    "p_left",        # decoder posterior for the LEFT class, this window
    "evidence",      # accumulated evidence, -1..+1
    "command",       # -1 right, 0 none, +1 left
    "task_state",    # reaching / choosing / grasping / ...
    "dwell",         # seconds held inside the deadzone target
    "target_y",      # commanded arm target
    "cue",           # cue label when replaying a recording, else ""
]


class SessionLog:
    def __init__(self, out_dir, ch_names, sfreq, band, meta=None,
                 stem=None, keep_raw=True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stem = stem or datetime.now().strftime("live_%Y%m%d_%H%M%S")
        self.path = self.dir / self.stem
        self.ch_names = list(ch_names)
        self.sfreq = float(sfreq)
        self.band = tuple(band)
        self.keep_raw = keep_raw
        self.t0 = time.time()

        self.rows = []
        self.erd = []          # per-row ERD% for every channel
        self.power = []        # per-row absolute mu power for every channel
        self._raw = []
        self.baseline = None
        self.events = []       # (t, label) - block placed, baseline done, ...

        self.meta = {
            "started": datetime.now().isoformat(timespec="seconds"),
            "channels": self.ch_names,
            "sfreq": self.sfreq,
            "mu_band": list(self.band),
            "columns": ROWS,
        }
        self.meta.update(meta or {})

        self._csv_f = open(self.path.with_suffix(".csv"), "w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(ROWS + [f"erd_{c}" for c in self.ch_names]
                           + [f"mu_{c}" for c in self.ch_names])
        self._write_meta()

    # ------------------------------------------------------------------
    def _write_meta(self):
        with open(self.path.with_suffix(".json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def now(self):
        return time.time() - self.t0

    def add_raw(self, window):
        """Append new EEG samples. Pass only samples not already passed."""
        if self.keep_raw and window is not None and window.size:
            self._raw.append(np.asarray(window, dtype=np.float32))

    def set_baseline(self, baseline_power):
        self.baseline = np.asarray(baseline_power, dtype=np.float32)
        self.event("baseline complete")

    def event(self, label):
        self.events.append((self.now(), str(label)))

    def decision(self, p_left, evidence, command, task_state, dwell,
                 target_y, cue="", erd=None, power=None):
        t = self.now()
        cmd = {"left": 1, "right": -1}.get(command, 0) if isinstance(
            command, str) else int(command or 0)
        row = [round(t, 4), round(float(p_left), 5), round(float(evidence), 5),
               cmd, str(task_state), round(float(dwell), 3),
               round(float(target_y), 5), str(cue or "")]
        n = len(self.ch_names)
        e = list(np.asarray(erd, dtype=float)) if erd is not None else [np.nan] * n
        w = list(np.asarray(power, dtype=float)) if power is not None else [np.nan] * n
        self.rows.append(row)
        self.erd.append(e)
        self.power.append(w)
        self._csv.writerow(row + [f"{v:.3f}" for v in e]
                           + [f"{v:.6g}" for v in w])
        self._csv_f.flush()   # so a crash still leaves a usable file

    # ------------------------------------------------------------------
    def summary(self):
        """Live counters for the session panel."""
        n = len(self.rows)
        dur = max(self.now(), 1e-6)
        return {"decisions": n, "rate": n / dur, "duration": dur}

    def close(self, extra=None):
        self.meta["ended"] = datetime.now().isoformat(timespec="seconds")
        self.meta["duration_s"] = round(self.now(), 2)
        self.meta["n_decisions"] = len(self.rows)
        self.meta.update(extra or {})
        self._write_meta()
        try:
            self._csv_f.close()
        except Exception:
            pass

        payload = {
            "columns": np.array(ROWS, dtype=object),
            "rows": np.array([r[:1] + r[1:3] + [r[3]] + r[5:7]
                              for r in self.rows], dtype=np.float64)
            if self.rows else np.zeros((0, 6)),
            "task_state": np.array([r[4] for r in self.rows], dtype=object),
            "cue": np.array([r[7] for r in self.rows], dtype=object),
            "erd": np.array(self.erd, dtype=np.float32)
            if self.erd else np.zeros((0, len(self.ch_names)), np.float32),
            "mu_power": np.array(self.power, dtype=np.float32)
            if self.power else np.zeros((0, len(self.ch_names)), np.float32),
            "ch_names": np.array(self.ch_names, dtype=object),
            "sfreq": self.sfreq,
            "band": np.array(self.band, dtype=np.float32),
            "events": np.array(self.events, dtype=object)
            if self.events else np.zeros((0, 2), dtype=object),
            "meta": json.dumps(self.meta),
        }
        if self.baseline is not None:
            payload["baseline_power"] = self.baseline
        if self.keep_raw and self._raw:
            payload["raw"] = np.concatenate(self._raw, axis=1)
        np.savez_compressed(self.path.with_suffix(".npz"), **payload)
        return self.path.with_suffix(".npz")
