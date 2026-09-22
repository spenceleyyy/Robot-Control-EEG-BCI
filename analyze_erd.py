"""
Offline analysis of a recorded session.

Answers one question: is there usable event-related desynchronisation over
sensorimotor cortex, and can a classifier separate left from right?

Usage:
    python analyze_erd.py                     # newest file in data/
    python analyze_erd.py data/mi_xxx_raw.fif
"""

import glob
import os
import sys
import warnings

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import mne
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    _venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    _py = (".venv\\Scripts\\python.exe" if os.name == "nt"
           else ".venv/bin/python")
    print(f"\nMissing package: {exc.name}\n")
    if os.path.isdir(_venv):
        print("The environment exists, but you are running a different")
        print("Python. Use the one inside it:\n")
        print(f"    {_py} {os.path.basename(__file__)} "
              f"{' '.join(sys.argv[1:])}\n")
        print("Or just run the analyse script:")
        print("    analyze.bat" if os.name == "nt" else "    ./analyze.sh")
    else:
        print("No .venv here. Run the launcher first "
              "(start.bat / ./start.sh).")
    print()
    sys.exit(1)

import config

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

REPORT = []


def say(line=""):
    print(line)
    REPORT.append(line)


def rule(title=""):
    say()
    say("-" * 66)
    if title:
        say(title)
        say("-" * 66)


# ----------------------------------------------------------------- utilities
def pick_existing(raw, names):
    lower = {c.lower(): c for c in raw.ch_names}
    return [lower[n.lower()] for n in names if n.lower() in lower]


# Neighbours for the surface Laplacian, in order of preference. Whatever your
# cap actually has gets used - a 32ch falls back to the wider ring, which for
# motor imagery is the well-known "large Laplacian" and works fine.
LAPLACIAN_NEIGHBORS = {
    "C3": ["FC3", "C1", "C5", "CP3", "FC1", "FC5", "CP1", "CP5", "Cz"],
    "C4": ["FC4", "C2", "C6", "CP4", "FC2", "FC6", "CP2", "CP6", "Cz"],
    "Cz": ["FCz", "C1", "C2", "CPz", "FC1", "FC2", "CP1", "CP2"],
}


def apply_reference(raw):
    """Recover the hardware reference if asked, then re-reference."""
    name = config.RECOVER_REFERENCE_CHANNEL
    if name:
        if name.lower() in [c.lower() for c in raw.ch_names]:
            say(f"{name} is already a channel - nothing to recover.")
        elif config.REREFERENCE != "average":
            say(f"[warn] Recovering {name} needs REREFERENCE = \"average\". "
                f"Skipping recovery.")
        else:
            raw = mne.add_reference_channels(raw, name, copy=False)
            try:
                raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                                on_missing="ignore", verbose=False)
            except Exception:
                pass
            say(f"Recovered {name}: added back as a flat channel, and the "
                f"average reference below gives it real values.")

    ref = config.REREFERENCE
    if ref == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
        say("Re-referenced to the common average.")
    elif isinstance(ref, (list, tuple)) and len(ref):
        present = pick_existing(raw, list(ref))
        if len(present) == len(ref):
            raw.set_eeg_reference(present, verbose=False)
            say(f"Re-referenced to {', '.join(present)}.")
        else:
            say(f"[warn] Re-reference channels {ref} not all present. "
                f"Leaving the hardware reference in place.")
    else:
        say("Left on the hardware reference.")
    return raw


def hjorth_laplacian(raw):
    """Subtract the mean of surrounding electrodes from each focus channel.

    This is what makes a small mu drop visible: it removes the broad activity
    that every channel shares through the common reference.
    """
    rl = raw.copy()
    data = rl.get_data()
    idx = {c.lower(): i for i, c in enumerate(rl.ch_names)}
    new, applied = {}, []
    for ch, cands in LAPLACIAN_NEIGHBORS.items():
        if ch.lower() not in idx:
            continue
        nb = [idx[c.lower()] for c in cands if c.lower() in idx][:4]
        if len(nb) < 3:
            continue
        # computed from the ORIGINAL data, so channels do not contaminate
        # each other when one is a neighbour of another
        new[idx[ch.lower()]] = data[idx[ch.lower()]] - data[nb].mean(axis=0)
        applied.append(f"{ch} (-{len(nb)} neighbours)")
    if not new:
        return raw, []
    for i, vals in new.items():
        data[i] = vals
    rl._data = data
    return rl, applied


def compute_tfr(epochs, freqs, n_cycles, decim):
    """compute_tfr on new MNE, tfr_multitaper on old."""
    try:
        return epochs.compute_tfr(
            method="multitaper", freqs=freqs, n_cycles=n_cycles,
            time_bandwidth=4.0, return_itc=False, average=True,
            decim=decim, n_jobs=1, verbose=False)
    except AttributeError:
        from mne.time_frequency import tfr_multitaper
        return tfr_multitaper(
            epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
            use_fft=True, return_itc=False, average=True,
            decim=decim, n_jobs=1, verbose=False)


def annotation_span(raw, start_desc, end_desc):
    """Return (tmin, tmax) of a marked resting period, or None."""
    ann = raw.annotations
    starts = [o for o, d in zip(ann.onset, ann.description) if d == start_desc]
    ends = [o for o, d in zip(ann.onset, ann.description) if d == end_desc]
    if not starts:
        return None
    t0 = starts[0]
    t1 = ends[0] if ends else min(t0 + 60.0, raw.times[-1])
    if t1 - t0 < 5.0:
        return None
    return t0 + 2.0, min(t1 - 1.0, raw.times[-1])  # trim edges


# ------------------------------------------------------------ resting alpha
def resting_check(raw, out_prefix):
    """Eyes-open vs eyes-closed PSD. Finds the individual alpha/mu peak and
    confirms the whole recording chain is sane."""
    rule("1. RESTING BASELINE  (is the chain working, where is your mu?)")

    spans = {
        "eyes open": annotation_span(raw, "baseline/eyes_open",
                                     "baseline/eyes_open/end"),
        "eyes closed": annotation_span(raw, "baseline/eyes_closed",
                                       "baseline/eyes_closed/end"),
    }
    if not all(spans.values()):
        say("No usable resting block found (session aborted early?). Skipping.")
        return None

    occ = pick_existing(raw, ["O1", "Oz", "O2", "PO3", "PO4", "POz"])
    cen = pick_existing(raw, config.FOCUS_CHANNELS)
    if not occ:
        occ = cen or pick_existing(raw, config.CENTRAL_PICKS)

    psds = {}
    for label, (t0, t1) in spans.items():
        seg = raw.copy().crop(t0, t1).pick(occ + [c for c in cen if c not in occ])
        spectrum = seg.compute_psd(method="welch", fmin=1, fmax=45,
                                   n_fft=int(raw.info["sfreq"] * 2), verbose=False)
        psds[label] = (spectrum.get_data(), spectrum.freqs, seg.ch_names)

    data_o, freqs, names = psds["eyes open"]
    data_c, _, _ = psds["eyes closed"]

    def band_idx(lo, hi):
        return np.where((freqs >= lo) & (freqs <= hi))[0]

    occ_idx = [names.index(c) for c in occ if c in names]
    a = band_idx(8, 13)
    open_p = np.mean(data_o[occ_idx][:, a])
    closed_p = np.mean(data_c[occ_idx][:, a])
    ratio_db = 10 * np.log10(closed_p / open_p)
    say(f"Posterior alpha, eyes closed vs open: {ratio_db:+.1f} dB")
    if ratio_db > 3:
        say("  -> Clear alpha blocking. Amplifier, electrodes and markers are "
            "all doing their job.")
    elif ratio_db > 1:
        say("  -> Weak but present. Usable; check posterior electrode contact.")
    else:
        say("  !! No alpha increase on eye closure. Something is wrong with the")
        say("     recording itself - impedance, referencing, or channel order.")
        say("     Fix this before interpreting anything below.")

    # individual peak, from central channels where mu lives
    cen_idx = [names.index(c) for c in cen if c in names] or occ_idx
    search = band_idx(6, 15)
    diff = np.log10(data_c[cen_idx][:, search].mean(0)) - \
        np.log10(data_o[cen_idx][:, search].mean(0))
    peak_hz = float(freqs[search][int(np.argmax(diff))])
    mu_band = (max(6.0, peak_hz - 2.0), peak_hz + 2.0)
    say(f"Individual peak near {peak_hz:.1f} Hz "
        f"-> using mu band {mu_band[0]:.1f}-{mu_band[1]:.1f} Hz.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(freqs, data_o[occ_idx].mean(0), label="eyes open", lw=1.6)
    ax.semilogy(freqs, data_c[occ_idx].mean(0), label="eyes closed", lw=1.6)
    ax.axvspan(mu_band[0], mu_band[1], color="orange", alpha=0.15,
               label="mu band used")
    ax.set_xlabel("Hz")
    ax.set_ylabel("power (V^2/Hz)")
    ax.set_title("Resting spectrum, posterior channels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_1_resting_psd.png", dpi=130)
    plt.close(fig)
    return mu_band


# ------------------------------------------------------------------- epochs
def make_epochs(raw, task_tag, classes, quiet=False):
    wanted = {f"{task_tag}/cue/{c}": i + 1 for i, c in enumerate(classes)}
    present = set(raw.annotations.description)
    wanted = {k: v for k, v in wanted.items() if k in present}
    if len(wanted) < 2:
        return None, None
    events, event_id = mne.events_from_annotations(raw, event_id=wanted,
                                                   verbose=False)
    eeg = mne.pick_types(raw.info, eeg=True)
    if len(eeg) == 0:
        return None, None
    ep = mne.Epochs(raw, events, event_id, tmin=-2.5, tmax=5.0,
                    baseline=None, picks=eeg, preload=True,
                    reject_by_annotation=False, verbose=False)
    n_before = len(ep)
    if n_before == 0:
        return None, None

    # Reject on the SENSORIMOTOR channels only, and only across the window
    # that actually feeds the analysis. Judging the whole 7.5s epoch on every
    # channel means a single blink at Fp1 discards an otherwise perfect trial
    # - which rejects essentially everything and rejects nothing useful.
    picks = pick_existing(ep, config.CENTRAL_PICKS) or list(ep.ch_names)
    sub = ep.copy().pick(picks).crop(config.ERD_BASELINE[0],
                                     config.ERD_WINDOW[1])
    d = sub.get_data()
    p2p = d.max(axis=2) - d.min(axis=2)
    bad = np.where(p2p.max(axis=1) > config.REJECT_PEAK_TO_PEAK)[0]

    if len(bad) / n_before > 0.5:
        if not quiet:
            say(f"  [{task_tag}] over half the trials exceed "
                f"{config.REJECT_PEAK_TO_PEAK * 1e6:.0f}uV peak-to-peak on "
                f"sensorimotor channels - keeping all of them.")
            say(f"  [{task_tag}] that is worth knowing: it usually means a "
                f"loose electrode or persistent muscle tension.")
        return ep, {"kept_fraction": 1.0, "n": len(ep), "rejected": True}

    ep_clean = ep.copy().drop(bad, reason="p2p", verbose=False) if len(bad) \
        else ep
    if not quiet and len(bad):
        say(f"  Dropped {len(bad)} of {n_before} trials on amplitude.")
    return ep_clean, {"kept_fraction": len(ep_clean) / n_before,
                      "n": len(ep_clean), "rejected": False}


# ---------------------------------------------------------------------- ERD
def erd_analysis(epochs, classes, mu_band, title, out_path):
    """Time-frequency ERD/ERS maps plus a numeric ERD table."""
    freqs = np.arange(config.TFR_FREQS[0], config.TFR_FREQS[1] + 1, 1.0)
    n_cycles = freqs / 2.0
    decim = max(1, int(epochs.info["sfreq"] // 50))
    focus = pick_existing(epochs, config.FOCUS_CHANNELS)
    if not focus:
        say("  None of C3/Cz/C4 present under standard names - "
            "check your channel labels.")
        return {}

    results, tfrs = {}, {}
    for cls in classes:
        key = [k for k in epochs.event_id if k.endswith("/" + cls)]
        if not key:
            continue
        tfr = compute_tfr(epochs[key[0]], freqs, n_cycles, decim)
        tfr.apply_baseline(config.ERD_BASELINE, mode="percent")
        tfrs[cls] = tfr
        fmask = (tfr.freqs >= mu_band[0]) & (tfr.freqs <= mu_band[1])
        tmask = (tfr.times >= config.ERD_WINDOW[0]) & \
                (tfr.times <= config.ERD_WINDOW[1])
        for ch in focus:
            ci = tfr.ch_names.index(ch)
            val = 100 * float(np.mean(tfr.data[ci][np.ix_(fmask, tmask)]))
            results[(cls, ch)] = val

    if not tfrs:
        return {}

    ncol, nrow = len(tfrs), len(focus)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow),
                             squeeze=False)
    vmax = 60
    for j, (cls, tfr) in enumerate(tfrs.items()):
        for i, ch in enumerate(focus):
            ax = axes[i][j]
            ci = tfr.ch_names.index(ch)
            im = ax.imshow(100 * tfr.data[ci], aspect="auto", origin="lower",
                           extent=[tfr.times[0], tfr.times[-1],
                                   tfr.freqs[0], tfr.freqs[-1]],
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.axvline(0, color="k", lw=1.2)
            ax.axvline(getattr(config, "T_TASK", 4.0), color="k", lw=0.8, ls=":")
            for f_edge in mu_band:
                ax.axhline(f_edge, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_title(f"{cls}  -  {ch}", fontsize=11)
            if i == nrow - 1:
                ax.set_xlabel("s from cue")
            if j == 0:
                ax.set_ylabel("Hz")
    fig.suptitle(f"{title}   (blue = desynchronisation = ERD)", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.7, label="% change from baseline")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return results


def erd_table(results, classes, mu_band, label):
    focus = sorted({ch for (_, ch) in results})
    if not focus:
        return None
    say(f"\n  mu-band ({mu_band[0]:.1f}-{mu_band[1]:.1f} Hz) power change, "
        f"{config.ERD_WINDOW[0]}-{config.ERD_WINDOW[1]}s after cue:")
    say("    " + "class".ljust(10) + "".join(c.rjust(10) for c in focus))
    for cls in classes:
        row = "".join(
            (f"{results[(cls, ch)]:+.0f}%".rjust(10) if (cls, ch) in results
             else "-".rjust(10))
            for ch in focus)
        say("    " + cls.ljust(10) + row)
    say("    (negative = ERD = what you want)")

    best = min((v for v in results.values()), default=0.0)
    lat = None
    contra = None
    needed = [("left", "C4"), ("right", "C3"), ("left", "C3"), ("right", "C4")]
    if all(k in results for k in needed):
        # contralateral: left hand -> C4, right hand -> C3
        contra = (results[("left", "C4")] + results[("right", "C3")]) / 2
        ipsi = (results[("left", "C3")] + results[("right", "C4")]) / 2
        lat = ipsi - contra
        say(f"\n    contralateral mean {contra:+.0f}%, "
            f"ipsilateral mean {ipsi:+.0f}%, lateralisation {lat:+.0f} points")
        if lat > 5 and contra < -10:
            say("    -> Correctly lateralised. This is the signature you need.")
        elif lat > 0 and contra < -5:
            say("    -> Slight lateralisation, weaker than ideal.")
        elif lat > 0:
            say("    -> Sides differ, but the opposite side barely drops. That is")
            say("       not the classic imagery signature.")
        else:
            say("    -> No lateralisation. Both hands look the same to the cap.")
    say(f"\n  Strongest ERD anywhere in {label}: {best:+.0f}%")
    return {"best": best, "lateralisation": lat,
            "contra": contra if lat is not None else None}


def topomap(epochs, classes, mu_band, out_path):
    try:
        pos_ok = epochs.get_montage() is not None
    except Exception:
        pos_ok = False
    if not pos_ok:
        return
    freqs = np.arange(max(4, mu_band[0] - 3), mu_band[1] + 4, 1.0)
    n_cycles = freqs / 2.0
    decim = max(1, int(epochs.info["sfreq"] // 50))
    fig, axes = plt.subplots(1, len(classes),
                             figsize=(4.0 * len(classes), 3.8), squeeze=False)
    drawn = False
    for j, cls in enumerate(classes):
        key = [k for k in epochs.event_id if k.endswith("/" + cls)]
        if not key:
            continue
        tfr = compute_tfr(epochs[key[0]], freqs, n_cycles, decim)
        tfr.apply_baseline(config.ERD_BASELINE, mode="percent")
        fmask = (tfr.freqs >= mu_band[0]) & (tfr.freqs <= mu_band[1])
        tmask = (tfr.times >= config.ERD_WINDOW[0]) & \
                (tfr.times <= config.ERD_WINDOW[1])
        vals = 100 * tfr.data[:, fmask][:, :, tmask].mean(axis=(1, 2))
        try:
            mne.viz.plot_topomap(vals, tfr.info, axes=axes[0][j], show=False,
                                 cmap="RdBu_r", vlim=(-40, 40),
                                 contours=4)
            axes[0][j].set_title(f"{cls}", fontsize=12)
            drawn = True
        except Exception:
            pass
    if drawn:
        fig.suptitle("mu-band ERD topography (blue = desynchronisation)")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ decoding
def decode(raw, task_tag, classes, label):
    """CSP + LDA cross-validation, plus an artifact sanity check."""
    filt = raw.copy().filter(config.CSP_BAND[0], config.CSP_BAND[1],
                             fir_design="firwin", verbose=False)
    ep, _ = make_epochs(filt, task_tag, classes, quiet=True)
    if ep is None or len(ep) < 20:
        say(f"  Not enough {label} trials to run decoding.")
        return None
    ep = ep.copy().crop(config.ERD_WINDOW[0], config.ERD_WINDOW[1])
    y = ep.events[:, -1]
    if len(np.unique(y)) < 2:
        return None

    def cv(picks_names, tag):
        picks = pick_existing(ep, picks_names) if picks_names else ep.ch_names
        if len(picks) < 2:
            return None
        X = ep.copy().pick(picks).get_data()
        n_comp = min(4, max(1, len(picks) - 1))
        pipe = Pipeline([
            ("csp", CSP(n_components=n_comp, reg="ledoit_wolf", log=True,
                        norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy",
                                 error_score="raise")
        say(f"    {tag:<28} {scores.mean() * 100:5.1f}%  "
            f"(+/- {scores.std() * 100:.1f}, n={len(y)}, {len(picks)} ch)")
        return float(scores.mean())

    say(f"\n  CSP + LDA cross-validated accuracy, {label} "
        f"({config.CSP_BAND[0]:.0f}-{config.CSP_BAND[1]:.0f} Hz):")
    all_acc = cv(None, "all channels")
    cen_acc = cv(config.CENTRAL_PICKS, "sensorimotor only")
    fr_acc = cv(config.FRONTAL_PICKS, "frontal only (control)")

    chance = 1.0 / len(np.unique(y))
    say(f"    chance is {chance * 100:.0f}%")
    if fr_acc is not None and cen_acc is not None and fr_acc > cen_acc + 0.05 \
            and fr_acc > 0.7:
        say("    !! Frontal-only beats sensorimotor. You are very likely")
        say("       decoding eye movement or jaw muscle, not motor cortex.")
    return {"all": all_acc, "central": cen_acc, "frontal": fr_acc,
            "chance": chance}


# ------------------------------------------------------------------ verdict
def verdict(me_erd, mi_erd, me_dec, mi_dec, artifacts=False):
    rule("VERDICT")
    ok = lambda v: v is not None

    if ok(me_erd) and me_erd["best"] < -15:
        say("Real movement produced clear ERD. Your recording chain, montage")
        say("and analysis are all working correctly.")
    elif ok(me_erd):
        say("Real movement produced little or no ERD. That points at the setup,")
        say("not at you: impedance, reference, channel labels, or the cap")
        say("sitting too far forward. Fix this before judging imagery.")

    if not ok(mi_erd):
        say("No imagery block to evaluate.")
        return

    # Judge the drop over the OPPOSITE hemisphere, which is what imagery
    # produces. A big change at Cz or on the wrong side is not the signature.
    best = mi_erd["contra"] if mi_erd.get("contra") is not None else mi_erd["best"]
    acc = (mi_dec or {}).get("central") or (mi_dec or {}).get("all")
    say()
    if artifacts:
        say("CAUTION: most trials carried large swings on the motor channels.")
        say("Muscle tension (jaw, neck, straining) or a loose electrode. The")
        say("numbers below are less trustworthy until that is fixed.")
        say()
    if best < -20 and (acc or 0) > 0.70:
        say("GO. Strong imagery ERD and solid separability. Move straight to")
        say("closed-loop control - you have more signal than you need for one")
        say("axis of end-effector velocity.")
    elif best < -10 and (acc or 0) > 0.62:
        say("GO, with practice. The signature is there but modest. This is a")
        say("completely normal first session. Expect it to improve over a few")
        say("sessions of feedback training. Build the closed loop.")
    elif (acc or 0) > 0.62:
        say("MARGINAL. The classifier separates the classes even though the")
        say("averaged ERD is weak, which often means the effect is there but")
        say("in a narrow band or a different spatial pattern. Worth a second")
        say("session before deciding.")
    else:
        say("NOT YET. No convincing imagery signature in this session.")
        say("Before concluding anything, in this order:")
        say("  1. Check the real-movement block above. If that was also flat,")
        say("     it is the setup, full stop.")
        say("  2. Were you doing kinesthetic imagery - the felt sense of the")
        say("     squeeze - or were you picturing your hand? Visual imagery")
        say("     produces almost nothing here.")
        say("  3. Run one more session. First sessions are routinely flat.")
        say("  4. If three good sessions stay flat, you are likely in the")
        say("     15-30% who do not produce usable MI. Switch to SSVEP - same")
        say("     pipeline, different front end, works almost immediately.")


# --------------------------------------------------------------------- main
def analyze(fif_path):
    REPORT.clear()
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    out_prefix = os.path.splitext(fif_path)[0].replace("_raw", "")
    classes = list(config.CLASSES)

    say("=" * 66)
    say(f"  ERD report for {os.path.basename(fif_path)}")
    say("=" * 66)
    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    say(f"{n_eeg} EEG channels at {raw.info['sfreq']:.0f} Hz, "
        f"{raw.times[-1] / 60:.1f} minutes")
    if n_eeg == 0:
        say("No channels were recognised as EEG. Your channel labels do not")
        say("match the 10-20 system - check the montage in BrainVision.")
        return

    raw.filter(1.0, 45.0, fir_design="firwin", verbose=False)
    raw.notch_filter([getattr(config, "LINE_FREQ_HZ", 60.0)], verbose=False)

    use = getattr(config, "USE_CHANNELS", None)
    if use:
        rule("0a. MONTAGE")
        keep = pick_existing(raw, use)
        have = {c.lower() for c in raw.ch_names}
        missing = [c for c in use if c.lower() not in have]
        if len(keep) < 4:
            say(f"Only {len(keep)} of the requested channels are in this "
                f"recording. Using everything instead.")
        else:
            dropped = len(raw.ch_names) - len(keep)
            raw.pick(keep)
            say(f"Using {len(keep)} channels, dropped {dropped} ungelled ones.")
        if missing:
            say(f"[warn] Not in the recording: {', '.join(missing)}")
            if any(c.lower() in ("c3", "cz", "c4") for c in missing):
                say("[warn] A focus channel is missing. That is a problem.")
            if "Oz".lower() in [m.lower() for m in missing]:
                say("       Without Oz the alpha check falls back to central "
                    "channels - weaker, but it still runs.")
            if all(c.lower() in [m.lower() for m in missing]
                   for c in ("Fp1", "Fp2")):
                say("       Without the forehead pair there is no artifact "
                    "control, so a muscle-driven result cannot be caught.")

    rule("0b. REFERENCE")
    raw = apply_reference(raw)
    if config.USE_LAPLACIAN_FOR_ERD:
        raw_erd, lap = hjorth_laplacian(raw)
        if lap:
            say(f"Surface Laplacian for the ERD maps: {', '.join(lap)}")
        else:
            say("[warn] Not enough neighbouring electrodes for a Laplacian - "
                "using the referenced data as is.")
    else:
        raw_erd = raw

    mu_band = resting_check(raw, out_prefix) or (8.0, 13.0)

    rule("2. REAL MOVEMENT  (positive control)")
    me_ep, me_info = make_epochs(raw, "me", classes)
    me_erd = me_dec = None
    if me_ep is not None:
        say(f"  {me_info['n']} clean trials.")
        ep_lap, _ = make_epochs(raw_erd, "me", classes, quiet=True)
        r = erd_analysis(ep_lap or me_ep, classes, mu_band, "Real movement",
                         f"{out_prefix}_2_erd_movement.png")
        me_erd = erd_table(r, classes, mu_band, "real movement")
        me_dec = decode(raw, "me", classes, "real movement")
    else:
        say("  No movement block found.")

    rule("3. MOTOR IMAGERY  (the actual question)")
    mi_ep, mi_info = make_epochs(raw, "mi", classes)
    mi_erd = mi_dec = None
    if mi_ep is not None:
        say(f"  {mi_info['n']} clean trials "
            f"({100 * mi_info['kept_fraction']:.0f}% kept).")
        ep_lap, _ = make_epochs(raw_erd, "mi", classes, quiet=True)
        r = erd_analysis(ep_lap or mi_ep, classes, mu_band, "Motor imagery",
                         f"{out_prefix}_3_erd_imagery.png")
        mi_erd = erd_table(r, classes, mu_band, "imagery")
        topomap(mi_ep, classes, mu_band, f"{out_prefix}_4_topomap.png")
        mi_dec = decode(raw, "mi", classes, "imagery")
    else:
        say("  No imagery block found.")

    verdict(me_erd, mi_erd, me_dec, mi_dec,
            artifacts=bool(mi_info and mi_info.get("rejected")))

    rule()
    figs = sorted(glob.glob(f"{out_prefix}_*.png"))
    say("Figures written:")
    for f in figs:
        say(f"  {f}")
    txt = f"{out_prefix}_report.txt"
    with open(txt, "w") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print(f"\nText report: {txt}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = sorted(glob.glob(os.path.join(here, config.DATA_DIR, "*_raw.fif")))
        if not cand:
            print("No recordings found in data/. Run session.py first.")
            return 1
        path = cand[-1]
        print(f"Using most recent recording: {path}")
    analyze(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
