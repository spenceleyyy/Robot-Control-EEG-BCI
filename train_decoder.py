"""
Train the online decoder from a recorded session.

    train_decoder.bat                        (newest recording)
    train_decoder.bat data\\mi_xxx_raw.fif

Writes decoder.pkl next to this file, which bci_feedback.py loads.

The cross-validation here is grouped by trial. That matters: windows cut from
the same trial are highly correlated, so a plain random split would let the
classifier see a neighbouring window from the same trial in training and give
you an accuracy that will not survive contact with a live session.
"""

import glob
import os
import pickle
import sys

try:
    import numpy as np
    import mne
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold, cross_val_score
except ImportError as exc:
    _py = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    print(f"\nMissing package: {exc.name}")
    print(f"Use the environment's interpreter:\n    {_py} "
          f"{os.path.basename(__file__)}\n")
    sys.exit(1)

import config
import preproc

mne.set_log_level("ERROR")

WINDOW_S = 1.0      # length of each online decision window
STEP_S = 0.25       # how often a decision is made online
TRAIN_FROM = 0.5    # seconds after cue - skip the visual evoked response
TRAIN_TO = 3.5
N_CSP = 4


def load_windows(fif_path):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])

    use = getattr(config, "USE_CHANNELS", None) or config.CENTRAL_PICKS
    idx, missing = preproc.channel_indices(raw.ch_names, use)
    if len(idx) < 4:
        raise RuntimeError(f"Only {len(idx)} of the configured channels are "
                           f"in this recording.")
    names = [raw.ch_names[i] for i in idx]
    if missing:
        print(f"[warn] not in recording: {', '.join(missing)}")

    data = raw.get_data()[idx]
    data = preproc.average_reference(data)
    data = preproc.filter_offline_causal(data, sfreq, config.CSP_BAND)

    classes = list(config.CLASSES)
    wanted = {f"mi/cue/{c}": i for i, c in enumerate(classes)}
    have = set(raw.annotations.description)
    wanted = {k: v for k, v in wanted.items() if k in have}
    if len(wanted) < 2:
        raise RuntimeError("No motor imagery cues found in this recording.")

    X, y, groups, onsets = [], [], [], []
    trial = 0
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        if desc not in wanted:
            continue
        i0 = int(round((onset + TRAIN_FROM) * sfreq))
        i1 = int(round((onset + TRAIN_TO) * sfreq))
        if i0 < 0 or i1 > data.shape[1]:
            continue
        seg = data[:, i0:i1]
        wins = preproc.sliding_windows(seg, sfreq, WINDOW_S, STEP_S,
                                       0.0, (i1 - i0) / sfreq, t_zero=0.0)
        for w in wins:
            X.append(w)
            y.append(wanted[desc])
            groups.append(trial)
        onsets.append(float(onset))
        trial += 1

    if not X:
        raise RuntimeError("No usable windows extracted.")
    return (np.asarray(X), np.asarray(y), np.asarray(groups),
            names, sfreq, classes, trial, onsets)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train the online decoder.")
    ap.add_argument("fif", nargs="?", help="recording (default: newest)")
    ap.add_argument("--holdout", type=int, default=0,
                    help="leave the last N imagery trials out, for replay")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.fif:
        path = os.path.abspath(args.fif)
    else:
        cand = sorted(glob.glob(os.path.join(here, config.DATA_DIR,
                                             "*_raw.fif")))
        cand = [c for c in cand if "_partial" not in c] or cand
        if not cand:
            print("No recordings in data/. Run a session first.")
            return 1
        path = cand[-1]
    print(f"Training from {os.path.basename(path)}")

    X, y, groups, names, sfreq, classes, n_trials, onsets = load_windows(path)

    n_hold = max(0, min(args.holdout, n_trials - 20))
    if args.holdout and n_hold < args.holdout:
        print(f"[warn] only holding out {n_hold} trials - need at least 20 "
              f"left over to train on.")
    train_mask = groups < (n_trials - n_hold)
    Xtr, ytr, gtr = X[train_mask], y[train_mask], groups[train_mask]
    n_train = n_trials - n_hold

    print(f"{n_train} training trials -> {len(Xtr)} windows of {WINDOW_S}s, "
          f"{X.shape[1]} channels at {sfreq:.0f} Hz")
    if n_hold:
        print(f"{n_hold} trials held out for replay")

    # Grouped CV: whole trials are held out, never individual windows.
    gkf = GroupKFold(n_splits=min(5, n_train))
    scores = []
    for tr, te in gkf.split(Xtr, ytr, gtr):
        c = preproc.CSP(n_components=N_CSP).fit(Xtr[tr], ytr[tr])
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        m.fit(c.transform(Xtr[tr]), ytr[tr])
        scores.append(m.score(c.transform(Xtr[te]), ytr[te]))
    scores = np.asarray(scores)
    print(f"\nWindow-level accuracy, trials held out: "
          f"{scores.mean() * 100:.1f}% (+/- {scores.std() * 100:.1f})")
    print("That is the honest per-decision number. The cursor and the arm")
    print("accumulate several decisions, so they do better than this.")
    if scores.mean() < 0.58:
        print("\n[warn] Close to chance. Live control will be frustrating.")

    csp = preproc.CSP(n_components=N_CSP).fit(Xtr, ytr)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(csp.transform(Xtr), ytr)

    hold_acc = None
    if n_hold:
        Xh, yh = X[~train_mask], y[~train_mask]
        hold_acc = float(lda.score(csp.transform(Xh), yh))
        print(f"Accuracy on the {n_hold} held-out trials: {hold_acc * 100:.1f}%"
              f"  (never seen in training)")

    out = os.path.join(here, "decoder_holdout.pkl" if n_hold else "decoder.pkl")
    with open(out, "wb") as fh:
        pickle.dump({
            "csp": csp, "lda": lda,
            "ch_names": names, "sfreq": sfreq,
            "band": tuple(config.CSP_BAND),
            "window_s": WINDOW_S, "step_s": STEP_S,
            "classes": classes[:2],
            "cv_accuracy": float(scores.mean()),
            "holdout_accuracy": hold_acc,
            "holdout_n": n_hold,
            "replay_from": (onsets[n_train] - 3.0) if n_hold else None,
            "fif_path": path,
            "source": os.path.basename(path),
        }, fh)
    print(f"\nSaved {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
