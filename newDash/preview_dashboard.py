"""
Render the dashboard to a PNG with synthetic data, so the layout can be
checked without an amplifier, a cap or a robot.

    python preview_dashboard.py              a mid-session frame
    python preview_dashboard.py --state rest

Same panel code the live window runs, so what you see here is what you get.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt

import neuro
import viz
from canvas import MplCanvas, BG

SF = 500.0
MONTAGE = ["C3", "Cz", "C4", "FC1", "FC2", "FC5", "FC6",
           "CP1", "CP2", "CP5", "CP6", "Fp1", "Fp2", "Oz"]
MU = (9.5, 13.5)


def synth_session(total_s, blocks, seed=0):
    """One continuous recording: a rest period, then imagery.

    Generated as a single timeline rather than separate rest and task clips.
    That matters - with independent draws, an UNCHANGED channel can read
    +85% purely because the two clips happened to differ, which makes the
    display look broken when it is not. Real data shares one baseline, so
    the preview has to as well.

    blocks: list of (start_s, end_s, active_side) with side in {"L","R",None}
    """
    r = np.random.default_rng(seed)
    n = int(total_s * SF)
    t = np.arange(n) / SF
    d = r.normal(0, 4e-6, (len(MONTAGE), n))
    mu_sos = butter(4, [MU[0] / (SF / 2), MU[1] / (SF / 2)], btype="band",
                    output="sos")
    weight = {"C3": 1.0, "C4": 1.0, "Cz": .85, "FC1": .7, "FC2": .7,
              "CP1": .7, "CP2": .7, "FC5": .45, "FC6": .45, "CP5": .45,
              "CP6": .45, "Fp1": .08, "Fp2": .08, "Oz": .25}
    side = {"C3": "L", "FC1": "L", "FC5": "L", "CP1": "L", "CP5": "L",
            "C4": "R", "FC2": "R", "FC6": "R", "CP2": "R", "CP6": "R"}

    # one attenuation envelope per hemisphere, ramped so it is not a step
    env = {"L": np.ones(n), "R": np.ones(n)}
    for a, b, act in blocks:
        if act is None:
            continue
        i0, i1 = int(a * SF), int(min(b, total_s) * SF)
        ramp = int(0.4 * SF)
        e = np.ones(i1 - i0)
        k = min(ramp, len(e) // 2)
        if k:
            e[:k] = np.linspace(1, 0.38, k)
            e[k:-k] = 0.38
            e[-k:] = np.linspace(0.38, 1, k)
        else:
            e[:] = 0.38
        env[act][i0:i1] = e

    for i, ch in enumerate(MONTAGE):
        mu = sosfilt(mu_sos, r.normal(size=n))
        mu /= (mu.std() or 1)
        amp = 16e-6 * weight.get(ch, .25)
        sd = side.get(ch)
        d[i] += amp * mu * (env[sd] if sd else 1.0)
        d[i] += 2.5e-6 * np.sin(2 * np.pi * 60 * t)
    return d


def fake_camera(w=480, h=270):
    """Stand-in for the webcam, so the panel can be laid out."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 26 + 12 * np.sin(xx / 60.0)
    img[..., 1] = 30 + 12 * np.cos(yy / 50.0)
    img[..., 2] = 42
    cy, cx = h * 0.52, w * 0.5
    head = ((xx - cx) ** 2 / (w * .13) ** 2 + (yy - cy) ** 2 / (h * .30) ** 2) < 1
    img[head] = (92, 104, 124)
    cap = head & (yy < cy - h * .10)
    img[cap] = (58, 92, 150)
    return img


def fake_arm(w=560, h=620):
    """Stand-in for the MuJoCo render, so the panel can be laid out."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = (22 + 16 * (yy / h)).astype(np.uint8)
    img[..., 1] = (26 + 18 * (yy / h)).astype(np.uint8)
    img[..., 2] = (34 + 22 * (yy / h)).astype(np.uint8)
    img[int(h * .72):] = (40, 46, 58)
    for (x0, y0, x1, y1, t_, col) in [
            (.22, .70, .40, .40, .035, (150, 158, 172)),
            (.40, .40, .62, .32, .030, (150, 158, 172)),
            (.62, .32, .70, .46, .022, (120, 128, 142))]:
        n_ = 200
        for k in range(n_):
            u = k / n_
            px, py = (x0 + (x1 - x0) * u) * w, (y0 + (y1 - y0) * u) * h
            r = t_ * w
            m = (xx - px) ** 2 + (yy - py) ** 2 < r ** 2
            img[m] = col
    for cx, cy, col in [(.40, .78, (200, 90, 80)), (.52, .78, (90, 190, 120)),
                        (.64, .78, (90, 140, 220))]:
        m = (np.abs(xx - cx * w) < .035 * w) & (np.abs(yy - cy * h) < .035 * w)
        img[m] = col
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", choices=["rest", "left", "right", "baseline"],
                    default="left")
    ap.add_argument("--out", default="dashboard_preview.png")
    args = ap.parse_args()

    active = {"left": "R", "right": "L", "rest": None,
              "baseline": None}[args.state]

    lap, lap_names, _ = neuro.laplacian_matrix(MONTAGE)
    tracker = neuro.ErdTracker(MONTAGE, SF, band=MU)
    topo = neuro.Topomap(MONTAGE, size=56)

    # One continuous timeline: 16s of rest to set the baseline, then
    # alternating imagery, ending in whichever state is being previewed.
    win = int(2.0 * SF)
    rest_s, total_s = 16.0, 54.0
    blocks = [(20, 27, "R"), (31, 38, "L"), (42, 54, active)]
    rec = synth_session(total_s, blocks, seed=5)

    n_base = int(rest_s * SF)
    for s in range(0, n_base - win, int(0.1 * SF)):
        tracker.update(rec[:, s:s + win], in_baseline=True, dt=0.1)
    tracker.finish_baseline()
    base_spec = neuro.spectrum(lap @ rec[:, n_base - win:n_base], SF,
                               fmax=40)[1]

    for s in range(n_base, rec.shape[1] - win, int(0.1 * SF)):
        tracker.update(rec[:, s:s + win], dt=0.1)

    recent = rec[:, -win:]
    lap_recent = lap @ rec[:, -int(4 * SF):]
    freqs, psd = neuro.spectrum(lap @ recent, SF, fmax=40)
    qual, _ = neuro.channel_quality(recent, SF)
    qmap = dict(zip(MONTAGE, qual))

    baselining = args.state == "baseline"
    erd3, erd4 = tracker.value("C3"), tracker.value("C4")
    lat = tracker.lateralisation()
    evidence = float(np.clip(lat / 70.0, -1, 1)) * (0 if baselining else 1)
    prob = 0.5 + evidence * 0.42
    command = ("left" if evidence > 0.08 else
               "right" if evidence < -0.08 else None)

    fig = plt.figure(figsize=(viz.W / 100, viz.H / 100), dpi=100,
                     facecolor=tuple(v / 255 for v in BG))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(tuple(v / 255 for v in BG))
    c = MplCanvas(ax, viz.W, viz.H)

    L = viz.LAYOUT
    viz.draw_header(c, L["header"], {
        "source": "live EEG", "sfreq": SF, "n_ch": len(MONTAGE),
        "mu_band": MU, "clock": "04:12"})
    viz.draw_traces(c, L["traces"], lap_recent, lap_names, SF,
                    quality=[qmap.get(n, "ok") for n in lap_names])
    viz.draw_erd(c, L["erd"], erd3, erd4, lat, baselining=baselining)
    hist = tracker.history[[MONTAGE.index(n) for n in lap_names]]
    lat_hist = (tracker.history[MONTAGE.index("C3")]
                - tracker.history[MONTAGE.index("C4")])
    viz.draw_history(c, L["history"], hist, lap_names,
                     lateralisation=lat_hist)
    viz.draw_spectrum(c, L["spectrum"], freqs, psd, lap_names, MU,
                      baseline_psd=base_spec)
    viz.draw_arm(c, L["arm"], fake_arm(), state="choosing",
                 target="holding over block 2")
    viz.draw_decision(c, L["decision"], prob, evidence, command,
                      dwell=0.9, dwell_target=1.5,
                      message="move to choose a block" if baselining else "")
    viz.draw_camera(c, L["camera"], fake_camera())
    viz.draw_topomap(c, L["topomap"], topo.grid(tracker.erd),
                     topo.electrode_xy(MONTAGE), quality=qmap)
    viz.draw_stats(c, L["stats"], {
        "decoder": "mi_20260920 · CSP+LDA",
        "offline accuracy": "75.0%",
        "decisions": "1 004",
        "rate": "4.0 /s",
        "blocks placed": "3 of 4 tries",
        "signal": ("all channels ok", (61, 220, 151)),
        "logging to": "data/live_20261107.npz",
    })

    fig.savefig(args.out, dpi=100,
                facecolor=tuple(v / 255 for v in BG))
    print(f"wrote {args.out}   state={args.state}  "
          f"C3 {erd3:+.0f}%  C4 {erd4:+.0f}%  lat {lat:+.0f}")


if __name__ == "__main__":
    main()
