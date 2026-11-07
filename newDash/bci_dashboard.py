"""
The session dashboard: live EEG, the decode, the arm and the participant,
all in one 1920x1080 window that screen-records cleanly.

    python bci_dashboard.py                    live EEG from LSL
    python bci_dashboard.py --source replay    replay a recorded .fif
    python bci_dashboard.py --source keyboard  arrow keys, for rehearsing
    python bci_dashboard.py --no-camera        skip the webcam

It drives the same decoder and the same task as sim_arm.py. It does not
re-implement either: the decode path is the existing EvidenceDecoder, tapped
rather than replaced, so what the video shows is what the BCI actually did.

Keys:  SPACE restart the trial   B re-take the baseline   F fullscreen
       C cycle camera index      S screenshot             ESC quit
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import neuro                                     # noqa: E402
import viz                                       # noqa: E402
from canvas import PygameCanvas, BG, GOOD, WARN, BAD   # noqa: E402
from camera import Camera                        # noqa: E402
from session_log import SessionLog               # noqa: E402

BASELINE_S = 20.0      # rest at the start, to set the ERD reference
MU_FALLBACK = (9.0, 13.0)


# --------------------------------------------------------------------------
class Tap:
    """Watches the decoder without changing it.

    EvidenceDecoder.feed() is the one place every sample passes through, so
    wrapping it gives the display and the log the exact same data the
    classifier saw. Wrapping rather than copying matters: a second,
    parallel filter chain would drift from the real one, and then the
    picture in the video would not be a picture of the decode.
    """

    def __init__(self, dec):
        self.dec = dec
        self.latest = None
        self.n_windows = 0
        self._inner_feed = dec.feed
        dec.feed = self._feed

        self.decisions = []
        inner_cb = getattr(dec, "on_decision", None)
        if callable(inner_cb):
            self._inner_cb = inner_cb
            dec.on_decision = self._on_decision
        else:
            self._inner_cb = None
            try:
                dec.on_decision = self._on_decision
            except Exception:
                pass

    def _feed(self, x):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 2:
            self.latest = arr
            self.n_windows += 1
        return self._inner_feed(x)

    def _on_decision(self, *a, **k):
        self.decisions.append((time.time(), a, k))
        if self._inner_cb is not None:
            return self._inner_cb(*a, **k)


def build_source(kind, args, cfg):
    import decoder_source as ds

    model = None
    if kind != "keyboard":
        path = getattr(args, "model", None)
        if path:
            model = ds.load_decoder(path)
        else:
            model = ds.load_decoder(ds.decoder_path(getattr(args, "holdout", None)))

    if kind == "live":
        src = ds.LiveSource(model)
    elif kind == "replay":
        fif = args.fif or (model or {}).get("fif_path")
        if not fif:
            raise SystemExit("--source replay needs --fif, or a model that "
                             "records fif_path")
        src = ds.ReplaySource(model, fif, speed=args.speed)
    elif kind == "keyboard":
        src = ds.KeyboardSource()
    else:
        raise SystemExit(f"unknown source {kind}")
    return src, model


def model_channels(model, cfg):
    if model and model.get("ch_names"):
        return list(model["ch_names"])
    return list(getattr(cfg, "USE_CHANNELS", ["C3", "Cz", "C4"]))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["live", "replay", "keyboard"],
                    default="live")
    ap.add_argument("--model", default=None)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--fif", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-raw", action="store_true",
                    help="do not store raw EEG in the .npz (smaller files)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--baseline", type=float, default=BASELINE_S)
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    import pygame
    import config as cfg
    import sim_arm
    import arm_model

    # ---------------------------------------------------------------- setup
    src, model = build_source(args.source, args, cfg)
    dec = getattr(src, "dec", None)
    tap = Tap(dec) if dec is not None else None

    ch_names = model_channels(model, cfg)
    sfreq = float((model or {}).get("sfreq", getattr(cfg, "SFREQ", 500.0)))
    band = tuple((model or {}).get("band", getattr(cfg, "CSP_BAND", (8, 30))))
    # The CSP band is wide (8-30Hz) because the classifier wants both mu and
    # beta. The display wants mu alone, which is where the ERD is legible.
    mu_band = (max(band[0], MU_FALLBACK[0]), min(band[1], MU_FALLBACK[1]))
    if mu_band[1] - mu_band[0] < 2:
        mu_band = MU_FALLBACK

    lap, lap_names, _ = neuro.laplacian_matrix(ch_names)
    tracker = neuro.ErdTracker(ch_names, sfreq, band=mu_band)
    topo = neuro.Topomap(ch_names, size=56)
    ring = neuro.RingBuffer(len(ch_names), int(6 * sfreq))

    cam = None
    if not args.no_camera:
        cam = Camera(args.camera).start()

    log = SessionLog(args.out, ch_names, sfreq, mu_band,
                     keep_raw=not args.no_raw,
                     meta={
                         "source": args.source,
                         "decoder": str((model or {}).get("source", "n/a")),
                         "cv_accuracy": (model or {}).get("cv_accuracy"),
                         "holdout_accuracy": (model or {}).get("holdout_accuracy"),
                         "holdout_n": (model or {}).get("holdout_n"),
                         "csp_band": list(band),
                         "classes": list((model or {}).get("classes", [])),
                     })

    m, d = arm_model.load_model()
    ctrl = sim_arm.ArmController(m, d)
    task = sim_arm.Task(m, d, ctrl)

    import mujoco
    arm_rect = viz.LAYOUT["arm"]
    rw, rh = int(arm_rect[2] - 16), int(arm_rect[3] - 56)
    renderer = mujoco.Renderer(m, height=rh, width=rw)
    cam3d = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam3d)
    cam3d.distance, cam3d.azimuth, cam3d.elevation = 1.35, 135.0, -22.0
    cam3d.lookat[:] = (0.35, 0.0, 0.25)

    pygame.init()
    pygame.display.set_caption("Motor imagery BCI")
    flags = pygame.FULLSCREEN if args.fullscreen else 0
    screen = pygame.display.set_mode((viz.W, viz.H), flags)
    canvas = PygameCanvas(screen)
    clock = pygame.time.Clock()

    # ----------------------------------------------------------- state
    t_start = time.time()
    baselining = True
    acc = f"{100 * (model or {}).get('holdout_accuracy', (model or {}).get('cv_accuracy') or 0):.1f}%"
    placed, tries = 0, 0
    last_cue, message = "", "sit still - measuring your resting rhythm"
    base_psd = None
    arm_img = None
    running = True

    def restart():
        nonlocal message
        if hasattr(task, "reset"):
            task.reset()
        if hasattr(src, "reset_evidence"):
            src.reset_evidence()
        log.event("trial restarted")
        message = ""

    while running:
        dt = clock.tick(60) / 1000.0
        elapsed = time.time() - t_start

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    restart()
                elif ev.key == pygame.K_b:
                    tracker.restart_baseline()
                    baselining, t_start, base_psd = True, time.time(), None
                    log.event("baseline restarted")
                elif ev.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif ev.key == pygame.K_c and cam is not None:
                    cam.stop()
                    cam = Camera((cam.index + 1) % 4).start()
                elif ev.key == pygame.K_s:
                    name = f"frame_{int(time.time())}.png"
                    pygame.image.save(screen, name)
                    log.event(f"screenshot {name}")

        # ------------------------------------------------------ decode
        evidence, cue = src.poll()
        if cue:
            last_cue = cue

        window = tap.latest if tap is not None else None
        new_window = tap is not None and tap.n_windows != getattr(
            main, "_seen", -1)
        if window is not None and new_window:
            main._seen = tap.n_windows
            ring.push(window[:, -int(min(window.shape[1], sfreq)):])
            log.add_raw(window[:, -int(min(window.shape[1], sfreq)):])
            tracker.update(window, in_baseline=baselining)

        if baselining and elapsed >= args.baseline:
            tracker.finish_baseline()
            baselining = False
            log.set_baseline(tracker.baseline)
            rest = ring.latest(int(2 * sfreq))
            if rest is not None:
                base_psd = neuro.spectrum(lap @ rest, sfreq, fmax=40)[1]
            message = ""

        # During the baseline the arm must not move, or the participant will
        # chase it instead of resting.
        drive = 0.0 if baselining else float(evidence or 0.0)
        task.update(drive, cue, dt)
        mujoco.mj_step(m, d)

        state = getattr(task, "state", "")
        if state == "placed" and getattr(task, "_logged", None) is not state:
            placed += 1
            log.event("block placed")
        stats = getattr(task, "stats", None)
        if isinstance(stats, dict):
            placed = stats.get("placed", placed)
            tries = stats.get("tries", tries)

        # ------------------------------------------------------ render
        renderer.update_scene(d, camera=cam3d)
        arm_img = renderer.render()

        recent = ring.latest(int(4 * sfreq))
        screen.fill(BG)
        L = viz.LAYOUT

        viz.draw_header(canvas, L["header"], {
            "source": {"live": "live EEG", "replay": "replay",
                       "keyboard": "keyboard"}[args.source],
            "sfreq": sfreq, "n_ch": len(ch_names), "mu_band": mu_band,
            "clock": time.strftime("%M:%S", time.gmtime(elapsed))})

        qmap = {}
        if recent is not None and recent.shape[1] > sfreq:
            qual, _ = neuro.channel_quality(
                recent, sfreq, line_hz=getattr(cfg, "LINE_FREQ_HZ", 60.0))
            qmap = dict(zip(ch_names, qual))
            viz.draw_traces(canvas, L["traces"], lap @ recent, lap_names,
                            sfreq,
                            quality=[qmap.get(n, "ok") for n in lap_names])
            freqs, psd = neuro.spectrum(lap @ recent[:, -int(2 * sfreq):],
                                        sfreq, fmax=40)
            viz.draw_spectrum(canvas, L["spectrum"], freqs, psd, lap_names,
                              mu_band,
                              baseline_psd=base_psd)
        else:
            viz.panel(canvas, L["traces"], "EEG", "waiting for data")
            viz.panel(canvas, L["spectrum"], "SPECTRUM", "waiting for data")

        erd3, erd4 = tracker.value("C3"), tracker.value("C4")
        lat = tracker.lateralisation()
        viz.draw_erd(canvas, L["erd"], erd3, erd4, lat, baselining=baselining)

        idx = [ch_names.index(n) for n in lap_names if n in ch_names]
        if idx:
            viz.draw_history(
                canvas, L["history"], tracker.history[idx], lap_names,
                lateralisation=(tracker.history[ch_names.index("C3")]
                                - tracker.history[ch_names.index("C4")])
                if "C3" in ch_names and "C4" in ch_names else None)

        viz.draw_arm(canvas, L["arm"], arm_img,
                     state=state, target=getattr(task, "message", ""))
        p_left = float(getattr(dec, "last_p", 0.5) or 0.5)
        command = ("left" if drive > sim_arm.DEADZONE else
                   "right" if drive < -sim_arm.DEADZONE else None)
        viz.draw_decision(canvas, L["decision"], p_left, drive, command,
                          dwell=float(getattr(task, "dwell", 0.0) or 0.0),
                          dwell_target=sim_arm.DWELL_S,
                          message=message or getattr(task, "message", ""))
        if cam is not None:
            viz.draw_camera(canvas, L["camera"], cam.frame(),
                            status=cam.status())
        viz.draw_topomap(canvas, L["topomap"], topo.grid(tracker.erd),
                         topo.electrode_xy(ch_names), quality=qmap)

        bad = [c for c, q in qmap.items() if q != "ok"]
        sig = (("all channels ok", GOOD) if not bad else
               (f"{len(bad)} channel(s): {', '.join(bad[:3])}",
                WARN if len(bad) < 3 else BAD))
        s = log.summary()
        viz.draw_stats(canvas, L["stats"], {
            "decoder": str((model or {}).get("source", args.source)),
            "offline accuracy": acc,
            "decisions": f"{s['decisions']:,}".replace(",", " "),
            "rate": f"{s['rate']:.1f} /s",
            "blocks placed": f"{placed} of {max(tries, placed)} tries",
            "signal": sig,
            "logging to": str(log.path.name) + ".npz",
        })

        pygame.display.flip()

        if not baselining and new_window:
            log.decision(p_left, drive, command, state,
                         float(getattr(task, "dwell", 0.0) or 0.0),
                         float(np.ravel(getattr(task, "target", [0]))[-1]),
                         cue=last_cue,
                         erd=[tracker.value(c) for c in ch_names],
                         power=tracker.power)

    # ------------------------------------------------------------- teardown
    if cam is not None:
        cam.stop()
    path = log.close({"blocks_placed": placed, "tries": tries})
    pygame.quit()
    print(f"session saved to {path}")
    print(f"  and {path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
