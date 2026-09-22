"""
Pre-flight check. Run this first, before you gel up, to confirm the
amplifier is reaching this machine.

    python check_stream.py
"""

import sys
import numpy as np
import config
import lsl_io


def main():
    inlet, meta = lsl_io.find_eeg_stream()
    probe = lsl_io.probe_amplitude(inlet, 3.0)
    if probe is None:
        print("\nConnected, but no samples arrived in 3 seconds.")
        print("In BrainVision Recorder, switch to Monitor or Recording mode.")
        print("Impedance Check mode does not stream data.")
        return 1

    scale = lsl_io.resolve_units_scale(probe)
    print(f"\n{probe['n_samples']} samples received.")
    print(f"Typical deflection: {probe['median_abs']:.3g} "
          f"-> data looks like {'microvolts' if scale == 1e-6 else 'volts'}")
    print(f"Largest absolute value: {probe['max_abs']:.3g}")

    if probe["flat_channels"]:
        flat = [meta["ch_names"][i] for i in probe["flat_channels"]]
        print(f"\nFlat channels ({len(flat)}): {', '.join(flat)}")
        print("These are disconnected or dead.")

    names = meta["ch_names"]
    print(f"\nChannel labels ({len(names)}):")
    for i in range(0, len(names), 8):
        print("   " + "  ".join(n.ljust(6) for n in names[i:i + 8]))

    use = getattr(config, "USE_CHANNELS", None) or config.FOCUS_CHANNELS
    have_lower = [n.lower() for n in names]
    present = [c for c in use if c.lower() in have_lower]
    missing = [c for c in use if c.lower() not in have_lower]
    print(f"\nMontage you plan to gel ({len(use)} sites):")
    print("   " + "  ".join(present))
    if missing:
        print(f"\nNOT FOUND in the stream: {', '.join(missing)}")
        print("Either the labels are wrong, or CHANNEL_NAMES_OVERRIDE in")
        print("config.py does not match your cap. Fix before gelling up.")
        return 1

    focus_missing = [c for c in config.FOCUS_CHANNELS if c.lower() not in have_lower]
    if focus_missing:
        print(f"\nMissing focus channels: {', '.join(focus_missing)}")
        print("Without C3/Cz/C4 the analysis cannot locate motor cortex.")
        return 1

    print("\nAll good. Run:  python signal_check.py    (then session.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
