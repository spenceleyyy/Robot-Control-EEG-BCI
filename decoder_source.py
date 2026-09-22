"""
Where the arm's control signal comes from. Three interchangeable sources,
all returning (evidence, cue) from poll():

  LiveSource      the amp, through LSL, through your decoder
  ReplaySource    your RECORDED session, streamed through the decoder in real
                  time - your actual brain data, no cap needed
  KeyboardSource  arrow keys, for checking the arm works at all

evidence is in [-1, 1]: negative = left imagery, positive = right.
cue is the instructed class during replay ("left"/"right"), else None.
"""

import os
import pickle
import time

import numpy as np

import config
import preproc

SMOOTH = 0.72        # evidence accumulation; matches bci_feedback.py
BIAS_TAU_S = 20.0    # auto-centring time constant


def decoder_path(holdout=False):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "decoder_holdout.pkl" if holdout else "decoder.pkl")


def load_decoder(path):
    if not os.path.exists(path):
        raise RuntimeError(f"No decoder at {os.path.basename(path)}. "
                           f"Run train_decoder first.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


class EvidenceDecoder:
    """Filter state, rolling window, and accumulated evidence.

    Input must already be the decoder's channels, in its order, in volts.
    Samples are pushed in step-sized pieces so a decision happens exactly
    every step_s of signal time, however the data arrives.
    """

    def __init__(self, model, sfreq):
        self.m = model
        self.sfreq = sfreq
        n_ch = len(model["ch_names"])
        self.win_n = int(round(model["window_s"] * sfreq))
        self.step_n = max(1, int(round(model["step_s"] * sfreq)))
        self.filt = preproc.CausalBandpass(sfreq, model["band"], n_ch)
        self.buf = np.zeros((n_ch, self.win_n))
        self.filled = 0
        self.since = 0
        self.evidence = 0.0
        self.bias = 0.0
        self.beta = np.exp(-model["step_s"] / BIAS_TAU_S)
        self.last_p = 0.5
        self.on_decision = None   # optional callback(evidence, p)

    def feed(self, x):
        pos = 0
        while pos < x.shape[1]:
            piece = x[:, pos:pos + (self.step_n - self.since)]
            pos += piece.shape[1]
            self._push(piece)
            self.since += piece.shape[1]
            if self.since >= self.step_n:
                self.since = 0
                if self.filled >= self.win_n:
                    self._decide()

    def _push(self, x):
        x = self.filt(preproc.average_reference(x))
        n = x.shape[1]
        if n >= self.win_n:
            self.buf = x[:, -self.win_n:].copy()
        else:
            self.buf = np.roll(self.buf, -n, axis=1)
            self.buf[:, -n:] = x
        self.filled = min(self.win_n, self.filled + n)

    def _decide(self):
        feats = self.m["csp"].transform(self.buf[None])
        p = float(self.m["lda"].predict_proba(feats)[0, 1])
        d = 2.0 * p - 1.0
        self.bias = self.beta * self.bias + (1 - self.beta) * d
        self.evidence = SMOOTH * self.evidence + (1 - SMOOTH) * (d - self.bias)
        self.last_p = p
        if self.on_decision:
            self.on_decision(self.evidence, p)

    def reset_evidence(self):
        self.evidence = 0.0


# ------------------------------------------------------------------- sources
class LiveSource:
    name = "live EEG"

    def __init__(self, model):
        import lsl_io
        inlet, meta = lsl_io.find_eeg_stream()
        probe = lsl_io.probe_amplitude(inlet, 1.5)
        if probe is None:
            raise RuntimeError("Connected, but no samples are arriving.")
        if abs(meta["sfreq"] - model["sfreq"]) > 1.0:
            raise RuntimeError(f"Decoder trained at {model['sfreq']:.0f} Hz, "
                               f"stream is {meta['sfreq']:.0f} Hz.")
        idx, missing = preproc.channel_indices(meta["ch_names"],
                                               model["ch_names"])
        if missing:
            raise RuntimeError(f"Stream lacks decoder channels: "
                               f"{', '.join(missing)}")
        self.inlet = inlet
        self.idx = idx
        self.scale = lsl_io.resolve_units_scale(probe)
        self.dec = EvidenceDecoder(model, meta["sfreq"])
        self.finished = False

    def poll(self):
        while True:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=4096)
            if not chunk:
                break
            x = np.asarray(chunk, dtype=np.float64).T[self.idx] * self.scale
            self.dec.feed(x)
        return self.dec.evidence, None

    def reset_evidence(self):
        self.dec.reset_evidence()

    def summary(self):
        return []


class ReplaySource:
    """Streams a recorded session through the decoder at real-time speed.

    Use the HOLDOUT decoder for this. A decoder replaying the very trials it
    was trained on will look far better than it is.
    """
    name = "replay"
    WARMUP_S = 12.0

    def __init__(self, model, fif_path=None, start_s=None, speed=1.0,
                 data=None, sfreq=None, cues=None):
        if data is None:
            import mne
            mne.set_log_level("ERROR")
            path = fif_path or model.get("fif_path")
            if not path or not os.path.exists(path):
                raise RuntimeError("Cannot find the recording to replay. "
                                   "Pass --fif data\\mi_..._raw.fif")
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
            idx, missing = preproc.channel_indices(raw.ch_names,
                                                   model["ch_names"])
            if missing:
                raise RuntimeError(f"Recording lacks decoder channels: "
                                   f"{', '.join(missing)}")
            data = raw.get_data()[idx]
            sfreq = float(raw.info["sfreq"])
            cues = [(float(o), d.split("/")[-1])
                    for o, d in zip(raw.annotations.onset,
                                    raw.annotations.description)
                    if d.startswith("mi/cue/")]
            self.path = path
        else:
            self.path = "<array>"
        if abs(sfreq - model["sfreq"]) > 1.0:
            raise RuntimeError("Recording and decoder sampling rates differ.")

        self.data, self.sfreq = data, sfreq
        self.cues = sorted(cues)
        self.speed = speed
        self.dec = EvidenceDecoder(model, sfreq)
        self.dec.on_decision = self._score
        self.agree = []           # (cue_sign, evidence_sign) inside task windows

        if start_s is None:
            start_s = model.get("replay_from")
        if start_s is None:
            start_s = self.cues[0][0] - 3.0 if self.cues else 0.0
        start_n = int(max(0, start_s) * sfreq)

        # settle the filter and the bias on the preceding stretch, instantly
        warm_n = int(max(0, start_s - self.WARMUP_S) * sfreq)
        self.pos = warm_n
        if start_n > warm_n:
            self.dec.feed(self.data[:, warm_n:start_n])
            self.pos = start_n
        self.agree.clear()
        self.dec.reset_evidence()
        self.start_n = start_n
        self.t0 = time.perf_counter()
        self.finished = False

    def _cue_at(self, t):
        for onset, label in reversed(self.cues):
            if onset <= t:
                return label if t < onset + config.T_TASK else None
        return None

    def _score(self, evidence, p):
        t = self.pos / self.sfreq
        for onset, label in reversed(self.cues):
            if onset <= t:
                if onset + 0.5 <= t <= onset + config.T_TASK and evidence != 0:
                    self.agree.append((1 if label == "right" else -1,
                                       1 if evidence > 0 else -1))
                break

    def poll(self):
        due = self.start_n + int((time.perf_counter() - self.t0)
                                 * self.speed * self.sfreq)
        due = min(due, self.data.shape[1])
        while self.pos < due:
            step = min(due, self.pos + self.dec.step_n)
            chunk = self.data[:, self.pos:step]
            self.pos = step          # before feed, so _score sees the right time
            self.dec.feed(chunk)
        if self.pos >= self.data.shape[1]:
            self.finished = True
        return self.dec.evidence, self._cue_at(self.pos / self.sfreq)

    def reset_evidence(self):
        self.dec.reset_evidence()

    def stream_time(self):
        return self.pos / self.sfreq

    def agreement(self):
        if not self.agree:
            return None, 0
        a = np.asarray(self.agree)
        return float(np.mean(a[:, 0] == a[:, 1])), len(a)

    def summary(self):
        acc, n = self.agreement()
        if acc is None:
            return ["No cued imagery was replayed."]
        return [f"Decoded direction matched the cue {acc * 100:.0f}% of the "
                f"time ({n} decisions during imagery periods)."]


class KeyboardSource:
    """Arrow keys, in the MuJoCo window.  <- left   -> right   down = stop"""
    name = "keyboard"
    LEFT, RIGHT, DOWN = 263, 262, 264

    def __init__(self):
        self.direction = 0.0
        self.evidence = 0.0
        self.finished = False

    def key(self, keycode):
        if keycode == self.LEFT:
            self.direction = -0.8
        elif keycode == self.RIGHT:
            self.direction = 0.8
        elif keycode == self.DOWN:
            self.direction = 0.0

    def poll(self):
        self.evidence += 0.15 * (self.direction - self.evidence)
        return self.evidence, None

    def reset_evidence(self):
        self.direction = 0.0
        self.evidence = 0.0

    def summary(self):
        return []
