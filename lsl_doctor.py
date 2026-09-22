"""
LSL connectivity diagnostic.

Run this when check_stream.py says it cannot find anything. It lists every
LSL stream visible from this machine, not just EEG ones, and reports what is
in the way.

    python lsl_doctor.py
"""

import os
import socket
import sys
import time

try:
    from pylsl import resolve_streams, StreamInlet, StreamOutlet, StreamInfo
except ImportError:
    print("pylsl is not installed. Run the launcher first.")
    sys.exit(1)


def section(title):
    print()
    print("-" * 64)
    print(title)
    print("-" * 64)


def main():
    section("1. Environment")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Hostname    : {socket.gethostname()}")
    print(f"Working dir : {os.getcwd()}")
    cfg = os.path.join(os.getcwd(), "lsl_api.cfg")
    if os.path.exists(cfg):
        print("lsl_api.cfg : found (localhost discovery enabled)")
    else:
        print("lsl_api.cfg : NOT FOUND in the working directory.")
        print("              Run this from the project folder, or discovery")
        print("              may miss a connector on this same machine.")

    section("2. Can LSL talk to itself?")
    print("Publishing a test stream and trying to find it...")
    ok = False
    try:
        info = StreamInfo("LSLDoctorTest", "Test", 1, 100, "float32", "doctor1")
        outlet = StreamOutlet(info)
        time.sleep(1.0)
        found = [s for s in resolve_streams(wait_time=4.0)
                 if s.name() == "LSLDoctorTest"]
        if found:
            print("  PASS - LSL works on this machine.")
            print("  So the problem is the connector, not LSL itself.")
            ok = True
        else:
            print("  FAIL - LSL cannot even find its own stream.")
            print("  This is almost always Windows Firewall blocking UDP.")
            print("  Allow python.exe on Private networks and retry.")
        del outlet
    except Exception as exc:
        print(f"  FAIL - {exc}")

    section("3. Every LSL stream visible right now")
    streams = resolve_streams(wait_time=5.0)
    streams = [s for s in streams if s.name() != "LSLDoctorTest"]
    if not streams:
        print("  None.")
        print()
        if ok:
            print("  LSL is healthy but nothing is publishing. The connector")
            print("  is either not running, or running but not linked.")
            print("  In LSL-actiCHamp the Link button must read 'Unlink'.")
        else:
            print("  Fix the firewall problem in section 2 first.")
    else:
        for s in streams:
            print(f"  name    : {s.name()}")
            print(f"  type    : {s.type()}")
            print(f"  channels: {s.channel_count()}")
            print(f"  rate    : {s.nominal_srate()} Hz")
            print(f"  host    : {s.hostname()}")
            try:
                inlet = StreamInlet(s, max_buflen=10)
                chunk, _ = inlet.pull_chunk(timeout=2.0, max_samples=64)
                if chunk:
                    print(f"  data    : flowing ({len(chunk)} samples pulled)")
                else:
                    print("  data    : CONNECTED BUT NO SAMPLES")
                    print("            The connector is linked but the amp is")
                    print("            not sending. Check power and the cable.")
                del inlet
            except Exception as exc:
                print(f"  data    : could not open inlet ({exc})")
            print()

        eeg = [s for s in streams if s.type().upper() == "EEG"]
        if not eeg:
            print("  No stream is of type 'EEG'. Set STREAM_NAME_HINT in")
            print("  config.py to the name of the right one above.")

    section("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
