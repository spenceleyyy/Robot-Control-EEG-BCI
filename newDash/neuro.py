"""
Turning the live EEG into the numbers the dashboard draws.

Everything visual is computed here and nothing is drawn, so the same
calculations feed the live pygame window and the matplotlib preview - and so
the arithmetic can be tested without a cap on anyone's head.

The quantity the whole display is built around is ERD: event-related
desynchronisation. The sensorimotor cortex idles at around 10 Hz, and that
idling rhythm drops in power when the corresponding patch of cortex gets to
work - including when the movement is only imagined. Imagining the left hand
quiets the mu rhythm over the RIGHT hemisphere, and vice versa. Everything
here exists to make that drop visible as it happens.
"""

import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, welch

# ------------------------------------------------------------------ montage
# Where each electrode sits, projected onto a flat disc seen from above.
# +x is the person's right, +y is the front of the head. The unit circle is
# the head outline, so T7/T8 land on the rim at +/-1.
POS_2D = {
    "Fp1": (-0.31, 0.95), "Fpz": (0.0, 1.0), "Fp2": (0.31, 0.95),
    "AF7": (-0.59, 0.81), "AF3": (-0.33, 0.83), "AFz": (0.0, 0.81),
    "AF4": (0.33, 0.83), "AF8": (0.59, 0.81),
    "F7": (-0.81, 0.59), "F5": (-0.67, 0.64), "F3": (-0.50, 0.67),
    "F1": (-0.25, 0.70), "Fz": (0.0, 0.72), "F2": (0.25, 0.70),
    "F4": (0.50, 0.67), "F6": (0.67, 0.64), "F8": (0.81, 0.59),
    "FT9": (-0.95, 0.31), "FT7": (-0.89, 0.31), "FC5": (-0.72, 0.33),
    "FC3": (-0.50, 0.35), "FC1": (-0.25, 0.37), "FCz": (0.0, 0.37),
    "FC2": (0.25, 0.37), "FC4": (0.50, 0.35), "FC6": (0.72, 0.33),
    "FT8": (0.89, 0.31), "FT10": (0.95, 0.31),
    "T9": (-1.05, 0.0), "T7": (-1.0, 0.0), "C5": (-0.75, 0.0),
    "C3": (-0.50, 0.0), "C1": (-0.25, 0.0), "Cz": (0.0, 0.0),
    "C2": (0.25, 0.0), "C4": (0.50, 0.0), "C6": (0.75, 0.0),
    "T8": (1.0, 0.0), "T10": (1.05, 0.0),
    "TP9": (-0.95, -0.31), "TP7": (-0.89, -0.31), "CP5": (-0.72, -0.33),
    "CP3": (-0.50, -0.35), "CP1": (-0.25, -0.37), "CPz": (0.0, -0.37),
    "CP2": (0.25, -0.37), "CP4": (0.50, -0.35), "CP6": (0.72, -0.33),
    "TP8": (0.89, -0.31), "TP10": (0.95, -0.31),
    "P7": (-0.81, -0.59), "P5": (-0.67, -0.64), "P3": (-0.50, -0.67),
    "P1": (-0.25, -0.70), "Pz": (0.0, -0.72), "P2": (0.25, -0.70),
    "P4": (0.50, -0.67), "P6": (0.67, -0.64), "P8": (0.81, -0.59),
    "PO7": (-0.59, -0.81), "PO3": (-0.33, -0.83), "POz": (0.0, -0.81),
    "PO4": (0.33, -0.83), "PO8": (0.59, -0.81),
    "O1": (-0.31, -0.95), "Oz": (0.0, -1.0), "O2": (0.31, -0.95),
}

# Surface Laplacian: subtract the mean of the surrounding electrodes. This is
# what makes a local mu drop visible - it removes the broad activity every
# channel shares through the common reference, which would otherwise swamp it.
LAPLACIAN_NEIGHBOURS = {
    "C3": ["FC3", "C1", "C5", "CP3", "FC1", "FC5", "CP1", "CP5", "Cz"],
    "C4": ["FC4", "C2", "C6", "CP4", "FC2", "FC6", "CP2", "CP6", "Cz"],
    "Cz": ["FCz", "C1", "C2", "CPz", "FC1", "FC2", "CP1", "CP2"],
}


def laplacian_matrix(ch_names, focus=("C3", "Cz", "C4"), n_neighbours=4):
    """A matrix that maps the recorded channels onto Laplacian focus channels.

    Returns (matrix, names, used) where matrix @ data gives one row per focus
    channel. A focus channel without enough neighbours is passed through
    unchanged rather than dropped.
    """
    idx = {c.lower(): i for i, c in enumerate(ch_names)}
    rows, names, used = [], [], {}
    for ch in focus:
        if ch.lower() not in idx:
            continue
        row = np.zeros(len(ch_names))
        row[idx[ch.lower()]] = 1.0
        nb = [c for c in LAPLACIAN_NEIGHBOURS.get(ch, [])
              if c.lower() in idx][:n_neighbours]
        if len(nb) >= 3:
            for c in nb:
                row[idx[c.lower()]] -= 1.0 / len(nb)
        rows.append(row)
        names.append(ch)
        used[ch] = nb
    if not rows:
        return np.eye(len(ch_names)), list(ch_names), {}
    return np.asarray(rows), names, used


# ----------------------------------------------------------------- filtering
class StreamingBandpass:
    """Causal bandpass that keeps its state between chunks, so filtering the
    stream in pieces matches filtering it whole."""

    def __init__(self, sfreq, band, n_ch, order=4):
        nyq = sfreq / 2.0
        hi = min(band[1], nyq * 0.95)
        self.sos = butter(order, [band[0] / nyq, hi / nyq], btype="band",
                          output="sos")
        # Start from rest. scipy's steady-state initial conditions assume an
        # input of magnitude 1, and EEG is microvolts, so they produce a huge
        # opening transient.
        self.zi = np.zeros((self.sos.shape[0], n_ch, 2))

    def __call__(self, x):
        y, self.zi = sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y


class RingBuffer:
    """Fixed-length rolling window of the most recent samples."""

    def __init__(self, n_ch, n_samples):
        self.data = np.zeros((n_ch, n_samples))
        self.filled = 0
        self.total = 0

    def push(self, x):
        n = x.shape[1]
        if n == 0:
            return
        if n >= self.data.shape[1]:
            self.data[:] = x[:, -self.data.shape[1]:]
            self.filled = self.data.shape[1]
        else:
            self.data = np.roll(self.data, -n, axis=1)
            self.data[:, -n:] = x
            self.filled = min(self.data.shape[1], self.filled + n)
        self.total += n

    @property
    def ready(self):
        return self.filled >= self.data.shape[1]

    def latest(self, n_samples):
        """The most recent n_samples, or None if that many are not in yet."""
        n = int(min(n_samples, self.filled))
        if n < 16:
            return None
        return self.data[:, -n:]


# --------------------------------------------------------------- band power
def band_power(x, sfreq, band, nperseg=None):
    """Mean power in a frequency band, per channel."""
    n = x.shape[-1]
    nperseg = int(min(nperseg or n, n))
    if nperseg < 16:
        return np.zeros(x.shape[0])
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, axis=-1)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return np.zeros(x.shape[0])
    return psd[..., m].mean(axis=-1)


def spectrum(x, sfreq, fmax=40.0, nperseg=None):
    """Power spectrum for display, per channel."""
    n = x.shape[-1]
    nperseg = int(min(nperseg or n, n))
    if nperseg < 16:
        return np.array([0.0, 1.0]), np.zeros((x.shape[0], 2))
    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, axis=-1)
    m = freqs <= fmax
    return freqs[m], psd[..., m]


def individual_mu_peak(x, sfreq, search=(7.0, 14.0)):
    """Where this person's mu rhythm actually sits.

    Generic 8-13 Hz dilutes the effect when someone's peak is at 11.5. Worth
    measuring once from a rest period rather than assuming.
    """
    freqs, psd = welch(x, fs=sfreq, nperseg=int(min(x.shape[-1], sfreq * 2)),
                       axis=-1)
    m = (freqs >= search[0]) & (freqs <= search[1])
    if not m.any():
        return None
    mean_psd = psd[..., m].mean(axis=0)
    return float(freqs[m][int(np.argmax(mean_psd))])


# ------------------------------------------------------------------- the ERD
class ErdTracker:
    """Mu power per channel, as a percentage change from a resting baseline.

    Negative means desynchronisation - the rhythm got quieter, that patch of
    cortex got busier. That is the signal the whole interface is about.

    The baseline is captured once at the start, during a short rest period,
    which matches how the offline analysis does it and keeps the numbers
    interpretable. A running baseline would drift along with sustained
    imagery and hide exactly what we want to show.
    """

    def __init__(self, ch_names, sfreq, band=(9.0, 13.0), smooth_s=1.5,
                 history_s=30.0, update_hz=10.0):
        self.ch_names = list(ch_names)
        self.sfreq = sfreq
        self.band = band
        # Smoothing is a time constant in SECONDS, not a per-update weight.
        # Measured on resting data, a single 2s estimate of mu power ranges
        # over 0.39x to 2.69x its own median - so an unsmoothed display
        # swings +/-100% while the participant sits perfectly still, which
        # reads as the BCI detecting things that are not there. Seconds
        # rather than a weight because the decoder's update rate is set by
        # its step size; a fixed weight would silently mean a different
        # amount of smoothing at 4Hz than at 10Hz.
        self.smooth_s = float(smooth_s)
        self._t_last = None
        self.power = np.zeros(len(ch_names))
        self.baseline = None
        self._baseline_acc = []
        self.erd = np.zeros(len(ch_names))
        n_hist = max(8, int(history_s * update_hz))
        self.history = np.full((len(ch_names), n_hist), np.nan)
        self.history_s = history_s

    def collecting_baseline(self):
        return self.baseline is None

    def restart_baseline(self):
        """Throw the reference away and measure it again.

        Worth having on a key: if the participant moved or talked during the
        first rest period, every ERD number afterwards is measured against a
        bad reference, and nothing else in the display will reveal it.
        """
        self.baseline = None
        self._baseline_acc = []
        self.erd = np.zeros(len(self.ch_names))
        self.power = np.zeros(len(self.ch_names))
        self._t_last = None
        self.history[:] = np.nan

    def update(self, window, in_baseline=False, dt=None):
        """window: (n_channels, n_samples) of filtered EEG.

        dt: seconds since the previous update. Pass it when the caller knows
        it; otherwise it is taken from the clock.
        """
        p = band_power(window, self.sfreq, self.band)
        if dt is None:
            now = time.monotonic()
            dt = 0.1 if self._t_last is None else now - self._t_last
            self._t_last = now
        dt = float(np.clip(dt, 1e-3, 5.0))
        alpha = float(np.exp(-dt / max(self.smooth_s, 1e-3)))
        # smooth in the log domain: power is multiplicative, so a 50% drop and
        # a doubling should move the display by the same distance
        lp = np.log(np.maximum(p, 1e-30))
        if self.power.any():
            self.power = np.exp(alpha * np.log(np.maximum(self.power, 1e-30))
                                + (1 - alpha) * lp)
        else:
            self.power = np.exp(lp)

        if in_baseline:
            self._baseline_acc.append(p)
            return self.erd

        if self.baseline is None:
            self.finish_baseline()

        self.erd = 100.0 * (self.power - self.baseline) / self.baseline
        self.history = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = self.erd
        return self.erd

    def finish_baseline(self):
        if self._baseline_acc:
            # median, so one twitch during the rest period cannot set the scale
            self.baseline = np.median(np.asarray(self._baseline_acc), axis=0)
        else:
            self.baseline = np.maximum(self.power, 1e-30)
        self.baseline = np.maximum(self.baseline, 1e-30)

    def value(self, ch):
        try:
            return float(self.erd[self.ch_names.index(ch)])
        except ValueError:
            return 0.0

    def lateralisation(self, left="C3", right="C4"):
        """Positive means the right hemisphere is quieter than the left, which
        is what imagining the LEFT hand should produce."""
        return self.value(left) - self.value(right)


# ------------------------------------------------------------------ topomap
class Topomap:
    """Interpolates a value per electrode onto a disc, for a scalp map.

    The grid and the mask never change, so they are built once; only the
    interpolation weights get applied per frame.
    """

    def __init__(self, ch_names, size=64, expand=1.08):
        self.size = size
        pts, keep = [], []
        for i, c in enumerate(ch_names):
            p = POS_2D.get(c) or POS_2D.get(c.capitalize())
            if p is not None:
                pts.append(p)
                keep.append(i)
        self.idx = np.asarray(keep, dtype=int)
        self.pts = np.asarray(pts) if pts else np.zeros((0, 2))
        self.ok = len(self.pts) >= 4

        g = np.linspace(-expand, expand, size)
        self.gx, self.gy = np.meshgrid(g, g)
        self.mask = (self.gx ** 2 + self.gy ** 2) <= 1.0

        if self.ok:
            # Inverse-distance weights, computed once. Cheap per frame, and
            # unlike griddata it never leaves holes outside the convex hull of
            # the electrodes - which matters with only 14 of them.
            d = np.sqrt((self.gx[..., None] - self.pts[:, 0]) ** 2
                        + (self.gy[..., None] - self.pts[:, 1]) ** 2)
            w = 1.0 / np.maximum(d, 1e-3) ** 2
            self.w = w / w.sum(axis=-1, keepdims=True)

    def grid(self, values):
        """values: one per channel, in the order given at construction."""
        if not self.ok:
            return np.full((self.size, self.size), np.nan)
        v = np.asarray(values, dtype=float)[self.idx]
        out = (self.w * v).sum(axis=-1)
        return np.where(self.mask, out, np.nan)

    def electrode_xy(self, ch_names):
        """Screen-space positions in the same -1..1 frame, for drawing dots."""
        return [(ch_names[i], self.pts[k]) for k, i in enumerate(self.idx)]


# ------------------------------------------------------- quality monitoring
def channel_quality(window, sfreq, line_hz=60.0):
    """Flags channels that are dead or swamped, so a bad electrode shows up on
    screen instead of quietly corrupting the decode."""
    rms = np.sqrt(np.mean(window ** 2, axis=-1))
    mid = band_power(window, sfreq, (25.0, 45.0))
    line = band_power(window, sfreq, (line_hz - 2, line_hz + 2))
    ratio = line / np.maximum(mid, 1e-30)
    med = np.median(ratio) if len(ratio) else 1.0
    out = []
    for i in range(window.shape[0]):
        if rms[i] < 1e-8:
            out.append("dead")
        elif ratio[i] > max(8.0, 4.0 * med):
            out.append("noisy")
        else:
            out.append("ok")
    return out, rms
