"""
Live signal quality monitor.

This is NOT an impedance measurement. It cannot be, without Recorder. What it
does instead is infer contact quality from the signal itself, which in
practice catches the same problems:

  - a dead or disconnected electrode shows almost no signal
  - a poorly gelled electrode acts as an antenna and picks up mains hum
  - an unstable electrode drifts wildly at low frequency

Run it with the cap on and gel as you watch. Work on anything red, then start
the session.

    python signal_check.py

Press ENTER to stop.
"""

import os
import sys
import time

import numpy as np
from scipy.signal import welch

import config
import lsl_io

# enable ANSI colours on Windows terminals
os.system("")

GREEN, YELLOW, RED, GREY, DIM, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[90m", "\033[2m", "\033[0m")

WINDOW_S = 2.0
REFRESH_S = 0.7

# These carry the analysis: the three focus channels plus the neighbours the
# Laplacian subtracts. A bad contact on any of them degrades C3 or C4.
PRIORITY = ["C3", "Cz", "C4", "FC1", "FC5", "CP1", "CP5",
            "FC2", "FC6", "CP2", "CP6"]

# Thresholds. Mains and drift are judged RELATIVE to the array median, not in
# absolute terms. Two reasons: EEG is naturally 1/f, so normal low-frequency
# power looks like drift in absolute terms; and reference-free data from the
# direct connector carries common-mode noise on every channel at once. What
# identifies a bad electrode is being an outlier among its peers. Array-wide
# noise is a separate problem (ground/reference) and gets its own warning.
FLAT_UV = 0.5         # below this, nothing is connected
REL_FAIR = 3.0        # times the array median
REL_BAD = 8.0
RMS_BAD_UV = 150.0    # absolute backstop for a wildly swinging channel
GLOBAL_LINE_WARN = 20.0  # median mains ratio that indicates an array-wide issue


def enter_pressed():
    """Non-blocking check for ENTER, on Windows and POSIX alike."""
    try:
        if os.name == "nt":
            import msvcrt
            hit = False
            while msvcrt.kbhit():
                if msvcrt.getch() in (b"\r", b"\n"):
                    hit = True
            return hit
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True
    except Exception:
        pass
    return False


def channel_metrics(buf, sfreq, line_hz):
    """buf: (n_samples, n_channels) -> per-channel quality numbers."""
    x = buf - buf.mean(axis=0, keepdims=True)
    nper = int(min(len(x), sfreq * 1.0))
    freqs, psd = welch(x, fs=sfreq, nperseg=nper, axis=0)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return psd[m].mean(axis=0) if m.any() else np.zeros(psd.shape[1])

    broadband = band(1.0, 45.0) * 44.0
    rms = np.sqrt(np.maximum(broadband, 0))
    mid = np.maximum(band(25.0, 45.0), 1e-24)
    line = band(line_hz - 1.5, line_hz + 1.5) / mid
    drift = band(0.5, 2.0) / mid
    return rms, line, drift


def relative(values):
    """How far each channel sits above the typical channel."""
    med = float(np.median(values))
    if med <= 0 or not np.isfinite(med):
        return np.ones_like(values)
    return values / med


def verdict(rms, line_rel, drift_rel):
    if rms < FLAT_UV:
        return "DEAD", RED
    if rms > RMS_BAD_UV or line_rel > REL_BAD or drift_rel > REL_BAD:
        return "BAD", RED
    if line_rel > REL_FAIR or drift_rel > REL_FAIR:
        return "FAIR", YELLOW
    return "GOOD", GREEN


def render(names, rms, line, drift, sfreq, line_hz, elapsed):
    line_rel, drift_rel = relative(line), relative(drift)
    rows = []
    counts = {"GOOD": 0, "FAIR": 0, "BAD": 0, "DEAD": 0}
    verdicts = []
    prio = [p.lower() for p in PRIORITY]
    for i, n in enumerate(names):
        v, col = verdict(rms[i], line_rel[i], drift_rel[i])
        verdicts.append(v)
        counts[v] += 1
        star = "*" if n.lower() in prio else " "
        rows.append(f"{star}{n:<5}{col}{v:<5}{RESET}"
                    f"{DIM}{rms[i]:6.0f}uV hum{line_rel[i]:4.1f} "
                    f"drift{drift_rel[i]:5.1f}{RESET}")

    ncol = 2
    nrow = -(-len(rows) // ncol)
    out = ["\033[H\033[J"]  # home + clear
    out.append(f"  LIVE SIGNAL QUALITY   {sfreq:.0f} Hz   "
               f"mains {line_hz:.0f} Hz   {elapsed:5.0f}s\n")
    out.append(f"  {DIM}* = load-bearing for the ERD analysis. hum and drift "
               f"are relative to the typical channel.{RESET}")
    out.append(f"  {DIM}With active electrodes, drift is the more reliable "
               f"sign of a bad contact.{RESET}\n")
    for r in range(nrow):
        line_out = "  "
        for c in range(ncol):
            k = r + c * nrow
            if k < len(rows):
                line_out += rows[k].ljust(52)
        out.append(line_out)
    out.append("")
    out.append(f"  {GREEN}{counts['GOOD']} good{RESET}   "
               f"{YELLOW}{counts['FAIR']} fair{RESET}   "
               f"{RED}{counts['BAD']} bad   {counts['DEAD']} dead{RESET}")

    out.append("")
    median_line = float(np.median(line))
    if median_line > GLOBAL_LINE_WARN:
        out.append(f"  {RED}Mains noise across the WHOLE array "
                   f"({median_line:.0f}x). That is not one bad electrode -{RESET}")
        out.append(f"  {RED}check the ground and reference electrodes, and move "
                   f"away from power supplies.{RESET}")

    bad_priority = [names[i] for i, v in enumerate(verdicts)
                    if names[i].lower() in prio and v in ("BAD", "DEAD")]
    if bad_priority:
        out.append(f"  {RED}Work on these before recording: "
                   f"{', '.join(bad_priority)}{RESET}")
    elif counts["BAD"] + counts["DEAD"] > 0:
        out.append(f"  {YELLOW}The channels that matter are fine. The remaining "
                   f"bad ones only affect the average reference.{RESET}")
    elif median_line <= GLOBAL_LINE_WARN:
        out.append(f"  {GREEN}All good. You can start the session.{RESET}")
    out.append(f"\n  {DIM}press ENTER to stop{RESET}")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


def main():
    inlet, meta = lsl_io.find_eeg_stream()
    probe = lsl_io.probe_amplitude(inlet, 1.5)
    if probe is None:
        print("\nNo samples arriving. Is the connector linked to the amp?")
        return 1
    scale_to_uv = 1.0 if lsl_io.resolve_units_scale(probe) == 1e-6 else 1e6

    sfreq = meta["sfreq"]
    all_names = meta["ch_names"]

    use = getattr(config, "USE_CHANNELS", None)
    lower = {n.lower(): i for i, n in enumerate(all_names)}
    if use:
        used_idx = [lower[c.lower()] for c in use if c.lower() in lower]
        missing = [c for c in use if c.lower() not in lower]
        if len(used_idx) < 4:
            print("\n[warn] Too few of the configured channels are in the "
                  "stream. Monitoring all of them instead.")
            used_idx = list(range(len(all_names)))
        elif missing:
            print(f"\n[warn] Not in the stream: {', '.join(missing)}")
    else:
        used_idx = list(range(len(all_names)))
    names = [all_names[i] for i in used_idx]
    n_dry = len(all_names) - len(names)
    if n_dry:
        print(f"Monitoring {len(names)} gelled channels, "
              f"ignoring {n_dry} dry ones.")

    line_hz = getattr(config, "LINE_FREQ_HZ", 60.0)
    keep = int(WINDOW_S * sfreq)
    buf = np.zeros((keep, len(all_names)))
    filled = 0
    t0 = time.time()
    last_draw = 0.0

    print("\nStarting monitor...")
    time.sleep(1.0)
    try:
        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.1, max_samples=2048)
            if chunk:
                arr = np.asarray(chunk, dtype=np.float64)
                n = min(len(arr), keep)
                buf = np.roll(buf, -n, axis=0)
                buf[-n:] = arr[-n:]
                filled = min(keep, filled + n)

            if filled >= keep * 0.75 and time.time() - last_draw > REFRESH_S:
                window = buf[-filled:][:, used_idx] * scale_to_uv
                rms, line, drift = channel_metrics(window, sfreq, line_hz)
                render(names, rms, line, drift, sfreq, line_hz, time.time() - t0)
                last_draw = time.time()

            if enter_pressed():
                break
    except KeyboardInterrupt:
        pass

    print("\nMonitor stopped.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(1)
