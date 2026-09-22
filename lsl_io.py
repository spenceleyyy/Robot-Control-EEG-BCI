"""
Finding the amplifier, recording from it in the background, and writing the
result to disk in a format MNE can read.

Nothing in here touches the screen.
"""

import os
import threading
import time

import numpy as np
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_streams, local_clock

import config


# ------------------------------------------------------------------ discovery
def _channel_labels_from_info(info):
    """Pull channel names out of the LSL stream's XML description."""
    labels = []
    try:
        ch = info.desc().child("channels").child("channel")
        for i in range(info.channel_count()):
            name = ch.child_value("label") or ch.child_value("name")
            labels.append(name if name else f"ch{i + 1}")
            ch = ch.next_sibling()
    except Exception:
        labels = []
    if len(labels) != info.channel_count() or not any(labels):
        labels = [f"ch{i + 1}" for i in range(info.channel_count())]
    return labels


def find_eeg_stream(verbose=True):
    """Resolve an EEG stream, preferring one whose name contains the hint."""
    total = max(5.0, float(config.STREAM_RESOLVE_TIMEOUT))
    waited, streams = 0.0, []
    if verbose:
        print("Looking for an EEG stream on the network...")
        print("(you can start the LSL connector now if it is not running)")
    while waited < total:
        streams = resolve_streams(wait_time=3.0)
        if streams:
            break
        waited += 3.0
        if verbose:
            print(f"  nothing yet, still looking... "
                  f"({int(total - waited)}s left)")

    if not streams:
        raise RuntimeError(
            "No LSL streams found at all.\n\n"
            "  1. Is the LSL connector running AND linked?\n"
            "     Open LSL-actiCHamp, enter your device serial number, click\n"
            "     Link. The button must change to 'Unlink'. Until it does,\n"
            "     nothing is being published.\n\n"
            "  2. Windows Firewall.\n"
            "     LSL finds streams over UDP multicast, which Windows blocks\n"
            "     by default. Allow BOTH the connector and python.exe on\n"
            "     Private networks: Windows Security -> Firewall & network\n"
            "     protection -> Allow an app through firewall.\n\n"
            "  3. Too many network adapters (VPN, Hyper-V, VirtualBox).\n"
            "     This project ships an lsl_api.cfg that points LSL at\n"
            "     localhost directly. Make sure you are running from the\n"
            "     project folder so it gets picked up.\n\n"
            "  4. If you went the Recorder route instead: Recorder must be\n"
            "     in Monitor or Recording mode, not Impedance Check."
        )

    if verbose:
        print("Streams visible on the network:")
        for s in streams:
            print(f"   name={s.name()!r}  type={s.type()!r}  "
                  f"channels={s.channel_count()}  srate={s.nominal_srate()}")

    candidates = [s for s in streams if s.type().upper() == config.STREAM_TYPE.upper()]
    if not candidates:
        candidates = [s for s in streams if s.channel_count() >= 8]
    if config.STREAM_NAME_HINT:
        hinted = [s for s in candidates
                  if config.STREAM_NAME_HINT.lower() in s.name().lower()]
        if hinted:
            candidates = hinted
    if not candidates:
        raise RuntimeError(
            f"Found LSL streams, but none of type {config.STREAM_TYPE!r}. "
            "Set STREAM_NAME_HINT in config.py to the name printed above."
        )

    inlet = StreamInlet(candidates[0], max_buflen=360, recover=True)
    info = inlet.info()
    meta = {
        "name": info.name(),
        "sfreq": float(info.nominal_srate()),
        "n_channels": int(info.channel_count()),
        "ch_names": _channel_labels_from_info(info),
    }

    override = getattr(config, "CHANNEL_NAMES_OVERRIDE", None)
    if override:
        n = meta["n_channels"]
        if len(override) > n:
            raise RuntimeError(
                f"CHANNEL_NAMES_OVERRIDE has {len(override)} names but the "
                f"stream only has {n} channels. Fix config.py before "
                f"recording - otherwise labels land on the wrong electrodes."
            )
        original = list(meta["ch_names"])
        meta["ch_names"] = list(override) + original[len(override):]
        if verbose:
            print(f"Applied CHANNEL_NAMES_OVERRIDE to the first "
                  f"{len(override)} channels.")
            if n > len(override):
                extra = original[len(override):]
                print(f"  {n - len(override)} further channels kept their own "
                      f"names: {', '.join(extra[:8])}"
                      f"{' ...' if len(extra) > 8 else ''}")
                print(f"  If any of those are actually EEG, extend the list.")

    if meta["sfreq"] <= 0:
        raise RuntimeError(
            "Stream reports a sampling rate of 0 (irregular rate). "
            "This script needs a regular-rate EEG stream."
        )
    if verbose:
        print(f"\nConnected to {meta['name']!r}: "
              f"{meta['n_channels']} channels @ {meta['sfreq']:.1f} Hz")
    return inlet, meta


def probe_amplitude(inlet, seconds=2.0):
    """Pull a couple of seconds and report the typical amplitude.

    Used both as a sanity check that data is flowing and to guess whether the
    amp is sending microvolts or volts.
    """
    inlet.flush()
    buf = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        chunk, _ = inlet.pull_chunk(timeout=0.2, max_samples=1024)
        if chunk:
            buf.extend(chunk)
    if not buf:
        return None
    arr = np.asarray(buf, dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        return None
    return {
        "n_samples": arr.shape[0],
        "median_abs": float(np.median(np.abs(arr - np.median(arr, axis=0)))),
        "max_abs": float(np.max(np.abs(arr))),
        "flat_channels": [i for i in range(arr.shape[1])
                          if np.std(arr[:, i]) < 1e-12],
    }


def resolve_units_scale(probe):
    """Decide the factor that converts incoming data to volts."""
    if config.UNITS_SCALE != "auto":
        return float(config.UNITS_SCALE)
    if probe is None or probe["median_abs"] == 0:
        return 1e-6  # BrainVision default
    # Real EEG is ~5-100 uV. In volts that is 5e-6..1e-4.
    return 1e-6 if probe["median_abs"] > 1e-3 else 1.0


# ------------------------------------------------------------------- recorder
class Recorder:
    """Pulls EEG in a background thread and timestamps event markers.

    Marker timestamps and EEG timestamps are both put into the LSL local_clock
    domain, so they line up without any extra synchronisation work.
    """

    def __init__(self, inlet, meta, units_scale, push_lsl_markers=True):
        self.inlet = inlet
        self.meta = meta
        self.units_scale = units_scale
        self._chunks = []
        self._stamps = []
        self.markers = []  # list of (local_clock timestamp, label)
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._time_offset = 0.0

        self.marker_outlet = None
        if push_lsl_markers:
            try:
                mi = StreamInfo("MI_Testbed_Markers", "Markers", 1, 0,
                                "string", "mi_testbed_markers")
                self.marker_outlet = StreamOutlet(mi)
            except Exception:
                self.marker_outlet = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self.inlet.flush()
        try:
            self._time_offset = self.inlet.time_correction(timeout=2.0)
        except Exception:
            self._time_offset = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self):
        last_corr = time.time()
        while not self._stop.is_set():
            try:
                chunk, stamps = self.inlet.pull_chunk(timeout=0.2, max_samples=4096)
            except Exception:
                time.sleep(0.05)
                continue
            if chunk:
                with self._lock:
                    self._chunks.append(np.asarray(chunk, dtype=np.float64))
                    self._stamps.append(np.asarray(stamps, dtype=np.float64))
            if time.time() - last_corr > 5.0:
                try:
                    self._time_offset = self.inlet.time_correction(timeout=0.5)
                except Exception:
                    pass
                last_corr = time.time()

    # -- markers -----------------------------------------------------------
    def mark(self, label):
        ts = local_clock()
        with self._lock:
            self.markers.append((ts, label))
        if self.marker_outlet is not None:
            try:
                self.marker_outlet.push_sample([label], ts)
            except Exception:
                pass

    @property
    def n_samples(self):
        with self._lock:
            return int(sum(c.shape[0] for c in self._chunks))

    # -- saving ------------------------------------------------------------
    def _assemble(self):
        with self._lock:
            if not self._chunks:
                return None, None
            data = np.concatenate(self._chunks, axis=0)
            stamps = np.concatenate(self._stamps, axis=0)
        stamps = stamps + self._time_offset
        return data, stamps

    def save(self, out_dir, basename):
        os.makedirs(out_dir, exist_ok=True)
        data, stamps = self._assemble()
        if data is None or data.shape[0] < 10:
            raise RuntimeError("No EEG data was recorded, nothing to save.")

        npz_path = os.path.join(out_dir, basename + "_raw.npz")
        np.savez_compressed(
            npz_path,
            data=data.astype(np.float32),
            timestamps=stamps,
            ch_names=np.array(self.meta["ch_names"], dtype=object),
            sfreq=self.meta["sfreq"],
            units_scale=self.units_scale,
            marker_ts=np.array([m[0] for m in self.markers], dtype=np.float64),
            marker_label=np.array([m[1] for m in self.markers], dtype=object),
        )

        fif_path = None
        try:
            fif_path = self._save_fif(out_dir, basename, data, stamps)
        except Exception as exc:  # never lose the recording over a format issue
            print(f"[warn] Could not write the .fif file: {exc}")
            print(f"[warn] Raw data is safe in {npz_path}")

        # quality note: how far the sample count drifted from wall clock
        dur_stamps = float(stamps[-1] - stamps[0])
        dur_nominal = (data.shape[0] - 1) / self.meta["sfreq"]
        drift = dur_stamps - dur_nominal
        print(f"Recorded {data.shape[0]} samples "
              f"({dur_nominal:.1f}s nominal, {dur_stamps:.1f}s by timestamp, "
              f"drift {drift:+.2f}s)")
        if abs(drift) > max(1.0, 0.01 * dur_nominal):
            print("[warn] Large clock drift or dropped samples. "
                  "Check the LSL bridge if this persists.")

        return fif_path, npz_path

    def _save_fif(self, out_dir, basename, data, stamps):
        import mne
        from mne.channels import make_standard_montage

        mne.set_log_level("ERROR")
        ch_names = list(self.meta["ch_names"])
        # de-duplicate, MNE requires unique names
        seen = {}
        for i, n in enumerate(ch_names):
            if n in seen:
                seen[n] += 1
                ch_names[i] = f"{n}_{seen[n]}"
            else:
                seen[n] = 0

        montage = make_standard_montage("standard_1005")
        std = {n.lower(): n for n in montage.ch_names}
        ch_types, renamed = [], []
        for n in ch_names:
            key = n.strip().lower()
            if key in std:
                ch_types.append("eeg")
                renamed.append(std[key])  # canonical capitalisation, e.g. "Cz"
            else:
                ch_types.append("misc")
                renamed.append(n)

        info = mne.create_info(renamed, self.meta["sfreq"], ch_types)
        raw = mne.io.RawArray((data * self.units_scale).T, info, verbose=False)
        try:
            raw.set_montage(montage, on_missing="ignore", verbose=False)
        except Exception:
            pass

        t0 = float(stamps[0])
        onsets, descs = [], []
        for ts, label in self.markers:
            onset = ts - t0
            if -1.0 < onset < (data.shape[0] / self.meta["sfreq"]) + 1.0:
                onsets.append(max(0.0, onset))
                descs.append(label)
        if onsets:
            raw.set_annotations(
                mne.Annotations(onset=onsets,
                                duration=[0.0] * len(onsets),
                                description=descs),
                verbose=False,
            )

        fif_path = os.path.join(out_dir, basename + "_raw.fif")
        raw.save(fif_path, overwrite=True, verbose=False)
        n_eeg = sum(1 for t in ch_types if t == "eeg")
        print(f"Saved {fif_path}  ({n_eeg} recognised EEG channels, "
              f"{len(onsets)} markers)")
        return fif_path
