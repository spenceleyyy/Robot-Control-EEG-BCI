"""
Signal processing shared by training and online use.

The single most important thing in here is that training and online use the
SAME code path. A decoder trained on zero-phase filtered epochs and then run
on causally filtered live data will underperform for reasons that are very
hard to debug, because nothing looks wrong - it just quietly does worse.

So: one causal filter definition, one referencing rule, one feature
extractor, used by both. CSP is implemented here rather than imported from
MNE so that the online loop does not need MNE at all.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfilt, sosfilt_zi


# ----------------------------------------------------------------- filtering
def design_sos(sfreq, band, order=4):
    nyq = sfreq / 2.0
    return butter(order, [band[0] / nyq, band[1] / nyq],
                  btype="band", output="sos")


class CausalBandpass:
    """Stateful bandpass, so streaming chunk-by-chunk equals filtering the
    whole signal at once."""

    def __init__(self, sfreq, band, n_channels, order=4):
        self.sos = design_sos(sfreq, band, order)
        zi = sosfilt_zi(self.sos)                      # (n_sections, 2)
        self.zi = np.repeat(zi[:, None, :], n_channels, axis=1)
        self.zi = self.zi.copy()

    def reset(self):
        self.zi = np.zeros_like(self.zi)

    def __call__(self, x):
        """x: (n_channels, n_samples) -> filtered, same shape."""
        y, self.zi = sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y


def filter_offline_causal(x, sfreq, band, order=4):
    """Causal filter applied in one pass. Matches the streaming version
    after its start-up transient, which is why training uses this rather
    than a zero-phase filter."""
    return sosfilt(design_sos(sfreq, band, order), x, axis=-1)


# --------------------------------------------------------------- referencing
def average_reference(x):
    """x: (n_channels, n_samples). Common average over the channels given."""
    return x - x.mean(axis=0, keepdims=True)


def channel_indices(stream_names, wanted):
    """Map wanted channel names onto stream positions, case-insensitively.
    Returns (indices, missing)."""
    lower = {n.lower(): i for i, n in enumerate(stream_names)}
    idx = [lower[w.lower()] for w in wanted if w.lower() in lower]
    missing = [w for w in wanted if w.lower() not in lower]
    return idx, missing


# ----------------------------------------------------------------------- CSP
class CSP:
    """Common Spatial Patterns: finds spatial filters whose variance differs
    most between the two classes. Shrinkage-regularised, which matters with
    few trials and few channels.

    Implemented here so the online loop has no MNE dependency.
    """

    def __init__(self, n_components=4, reg=0.05):
        self.n_components = n_components
        self.reg = reg
        self.filters_ = None
        self.classes_ = None

    @staticmethod
    def _norm_cov(x):
        c = np.cov(x)
        tr = np.trace(c)
        return c / tr if tr > 0 else c

    def _shrink(self, c):
        n = c.shape[0]
        return (1 - self.reg) * c + self.reg * (np.trace(c) / n) * np.eye(n)

    def fit(self, X, y):
        """X: (n_trials, n_channels, n_samples), y: (n_trials,)"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("CSP here handles exactly two classes.")

        covs = []
        for c in self.classes_:
            Xc = X[y == c]
            covs.append(self._shrink(np.mean([self._norm_cov(x) for x in Xc],
                                             axis=0)))

        # Generalised eigenproblem: maximise class-1 variance relative to the
        # total. Eigenvalues near 1 and near 0 are the most discriminative.
        w, v = eigh(covs[0], covs[0] + covs[1])
        order = np.argsort(w)
        k = self.n_components
        take = np.concatenate([order[: k // 2], order[-(k - k // 2):]])
        self.filters_ = v[:, take].T
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = np.einsum("fc,tcs->tfs", self.filters_, X)
        var = Z.var(axis=2)
        var = var / np.maximum(var.sum(axis=1, keepdims=True), 1e-20)
        return np.log(np.maximum(var, 1e-20))

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ------------------------------------------------------------------ windowing
def sliding_windows(data, sfreq, win_s, step_s, t_start, t_end, t_zero=0.0):
    """Cut overlapping windows out of one trial.

    Training on short windows that match the online window length is what
    makes the offline number mean something online. Training on one long
    3-second epoch and then classifying 1-second windows live does not.

    data: (n_channels, n_samples) for the whole epoch
    t_zero: index time of sample 0, relative to the cue
    Returns (n_windows, n_channels, win_samples).
    """
    win_n = int(round(win_s * sfreq))
    step_n = max(1, int(round(step_s * sfreq)))
    out = []
    t = t_start
    while t + win_s <= t_end + 1e-9:
        i0 = int(round((t - t_zero) * sfreq))
        i1 = i0 + win_n
        if i0 >= 0 and i1 <= data.shape[1]:
            out.append(data[:, i0:i1])
        t += step_s
    return np.asarray(out) if out else np.empty((0, data.shape[0], win_n))
