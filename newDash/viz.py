"""
The dashboard panels.

Each panel is a function that draws into a Canvas rectangle. Written once,
rendered by pygame during a session and by matplotlib when previewing, so
what gets checked offline is the same code that runs live.

Layout is fixed at 1920x1080 so a screen recording comes out at clean 1080p.
"""

import numpy as np

from canvas import (BG, PANEL, PANEL_HI, BORDER, TEXT, MUTED, DIM,
                    LEFT_HEMI, RIGHT_HEMI, MIDLINE, GOOD, WARN, BAD, ACCENT,
                    mix, erd_colour)

W, H = 1920, 1080

LAYOUT = {
    "header":      (0, 0, W, 56),
    "traces":      (20, 72, 640, 222),
    "erd":         (20, 310, 640, 262),
    "history":     (20, 588, 640, 172),
    "spectrum":    (20, 776, 640, 284),
    "arm":         (676, 72, 624, 700),
    "decision":    (676, 788, 624, 272),
    "camera":      (1316, 72, 584, 330),
    "topomap":     (1316, 418, 584, 360),
    "stats":       (1316, 794, 584, 266),
}

CH_COLOUR = {"C3": LEFT_HEMI, "Cz": MIDLINE, "C4": RIGHT_HEMI}


# --------------------------------------------------------------- chrome
def panel(c, rect, title=None, subtitle=None, fill=PANEL):
    """Rounded card with a heading. Returns the inner content rect."""
    x, y, w, h = rect
    c.rect(x, y, w, h, fill, radius=10)
    c.rect(x, y, w, h, BORDER, radius=10, width=1)
    top = y + 12
    if title:
        c.text(x + 16, top, title.upper(), size=14, colour=MUTED, bold=True)
        if subtitle:
            c.text(x + w - 16, top, subtitle, size=14, colour=DIM, anchor="tr")
        top += 24
    return (x + 16, top + 4, w - 32, y + h - (top + 4) - 14)


def draw_header(c, rect, info):
    x, y, w, h = rect
    c.rect(x, y, w, h, PANEL_HI)
    c.line(x, y + h - 1, x + w, y + h - 1, BORDER, 1)
    c.text(24, y + h // 2, "MOTOR IMAGERY BCI", size=22, colour=TEXT,
           anchor="ml", bold=True)
    c.text(300, y + h // 2 + 1, "imagined hand movement, decoded live",
           size=15, colour=MUTED, anchor="ml")

    chips = [
        (info.get("source", "-"), ACCENT),
        (f"{info.get('sfreq', 0):.0f} Hz", MUTED),
        (f"{info.get('n_ch', 0)} ch", MUTED),
        (f"mu {info.get('mu_band', (9, 13))[0]:.1f}-"
         f"{info.get('mu_band', (9, 13))[1]:.1f} Hz", MUTED),
        (info.get("clock", "00:00"), TEXT),
    ]
    cx = W - 24
    for label, col in reversed(chips):
        tw = max(len(str(label)) * 9 + 24, 64)
        c.rect(cx - tw, y + 14, tw, h - 28, PANEL, radius=6)
        c.text(cx - tw // 2, y + h // 2, label, size=14, colour=col, anchor="mc")
        cx -= tw + 8


# ----------------------------------------------------------- eeg traces
def draw_traces(c, rect, data, names, sfreq, scale_uv=40.0, quality=None):
    """Scrolling EEG, one row per focus channel.

    These are surface-Laplacian channels, not raw electrodes: each is its own
    electrode minus the mean of its four neighbours. That is what makes a
    local rhythm visible instead of whatever the whole head is doing.
    """
    seconds = data.shape[1] / sfreq if data.shape[1] else 0
    ix, iy, iw, ih = panel(c, rect, "EEG  ·  surface Laplacian",
                           f"{seconds:.0f}s window")
    n = max(1, len(names))
    row_h = ih / n
    c.clip(ix, iy, iw, ih)
    for i, name in enumerate(names):
        cy = iy + row_h * (i + 0.5)
        col = CH_COLOUR.get(name, TEXT)
        c.line(ix + 54, cy, ix + iw, cy, (30, 37, 50), 1)
        if data.shape[1] > 2:
            y = data[i] * 1e6 / scale_uv * (row_h * 0.42)
            y = np.clip(y, -row_h * 0.46, row_h * 0.46)
            step = max(1, data.shape[1] // max(1, int(iw - 54)))
            ys = y[::step]
            xs = np.linspace(ix + 54, ix + iw, len(ys))
            c.polyline(list(zip(xs, cy - ys)), col, 2)
        c.text(ix, cy, name, size=17, colour=col, anchor="ml", bold=True)
        if quality and i < len(quality) and quality[i] != "ok":
            c.text(ix + 40, cy, "!", size=16,
                   colour=BAD if quality[i] == "dead" else WARN, anchor="ml")
    c.unclip()
    c.text(ix + iw, iy + ih + 2, f"±{scale_uv:.0f} µV", size=12,
           colour=DIM, anchor="tr")


# ------------------------------------------------------------- the hero
def draw_erd(c, rect, erd_c3, erd_c4, lateralisation, full_scale=70.0,
             baselining=False):
    """Mu power at C3 and C4 as a change from rest.

    This is the panel the whole interface is built around. The sensorimotor
    cortex idles around 10 Hz; that idling gets quieter when the cortex
    underneath goes to work. Imagining the LEFT hand quiets the RIGHT
    hemisphere, so the C4 bar falls - a few hundred milliseconds before any
    movement would have happened, and without any movement at all.
    """
    ix, iy, iw, ih = panel(c, rect, "mu rhythm  ·  change from rest",
                           "baselining…" if baselining else "ERD")

    # explicit vertical budget, so nothing can overlap whatever the values do
    bars_h = ih * 0.56
    label_y = iy + bars_h + 8
    slider_y = iy + ih - 26
    zero_y = iy + bars_h * 0.42
    up_span = zero_y - iy - 2            # room above the line, for ERS
    dn_span = (iy + bars_h) - zero_y - 2  # room below, for ERD

    bar_w = iw * 0.22
    gap = iw * 0.14
    cx_left = ix + iw / 2 - gap / 2 - bar_w / 2
    cx_right = ix + iw / 2 + gap / 2 + bar_w / 2

    c.line(ix, zero_y, ix + iw, zero_y, BORDER, 1)
    c.text(ix, zero_y - 3, "rest", size=11, colour=DIM, anchor="bl")

    for cx, val, name, side, col in (
            (cx_left, erd_c3, "C3", "left brain", LEFT_HEMI),
            (cx_right, erd_c4, "C4", "right brain", RIGHT_HEMI)):
        frac = float(np.clip(val / full_scale, -1.0, 1.0))
        if baselining:
            c.rect(cx - bar_w / 2, zero_y - 3, bar_w, 6, DIM, radius=3)
            c.text(cx, zero_y - 12, "—", size=18, colour=DIM, anchor="bc")
        elif frac < 0:                       # ERD: the rhythm drops
            hgt = -frac * dn_span
            c.rect(cx - bar_w / 2, zero_y, bar_w, max(hgt, 3), col, radius=5)
            ty = min(zero_y + hgt + 4, iy + bars_h - 20)
            c.text(cx, ty, f"{val:+.0f}%", size=19, colour=TEXT, anchor="tc",
                   bold=True)
        else:                                # ERS: idling harder
            hgt = frac * up_span
            c.rect(cx - bar_w / 2, zero_y - hgt, bar_w, max(hgt, 3),
                   mix(col, BAD, 0.6), radius=5)
            ty = max(zero_y - hgt - 4, iy + 16)
            c.text(cx, ty, f"{val:+.0f}%", size=19, colour=TEXT, anchor="bc",
                   bold=True)
        c.text(cx, label_y, name, size=20, colour=col, anchor="tc", bold=True)
        c.text(cx, label_y + 22, side, size=12, colour=DIM, anchor="tc")

    # Which hemisphere is quieter, and so which hand is being imagined.
    sl_x, sl_w = ix + iw * 0.06, iw * 0.88
    c.rect(sl_x, slider_y, sl_w, 6, (30, 37, 50), radius=3)
    c.line(sl_x + sl_w / 2, slider_y - 5, sl_x + sl_w / 2, slider_y + 11,
           DIM, 1)
    if not baselining:
        t = float(np.clip(lateralisation / 80.0, -1, 1))
        px = sl_x + sl_w / 2 + t * sl_w / 2
        col = RIGHT_HEMI if t > 0 else LEFT_HEMI
        c.rect(min(px, sl_x + sl_w / 2), slider_y,
               abs(px - (sl_x + sl_w / 2)), 6, col, radius=3)
        c.circle(px, slider_y + 3, 7, col)
    c.text(sl_x, slider_y + 12, "imagining RIGHT hand", size=11,
           colour=LEFT_HEMI, anchor="tl")
    c.text(sl_x + sl_w, slider_y + 12, "imagining LEFT hand", size=11,
           colour=RIGHT_HEMI, anchor="tr")


def draw_history(c, rect, history, names, full_scale=70.0, lateralisation=None):
    """Recent ERD as a scrolling heat strip, one row per channel.

    The bars say what is happening now; this says what has been happening,
    which is where a sustained imagery period becomes obvious.
    """
    ix, iy, iw, ih = panel(c, rect, "mu over time", "last 30s")
    n = max(1, len(names))
    row_h = ih / n
    cols = history.shape[1]
    if cols < 2:
        return
    # A short moving average along time. Each column is one 1-second window
    # of mu power, which is genuinely jumpy; without this the strip reads as
    # static rather than showing the trend it exists to show.
    k = 7
    sm = np.full_like(history, np.nan, dtype=float)
    for i in range(n):
        row = history[i]
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            filled = np.where(ok, row, np.nanmean(row[ok]))
            kern = np.ones(k) / k
            sm[i] = np.convolve(filled, kern, mode="same")
            sm[i, ~ok] = np.nan
    history = sm
    img = np.zeros((n, cols, 3), dtype=np.uint8)
    for i in range(n):
        for j in range(cols):
            v = history[i, j]
            img[i, j] = (28, 34, 46) if not np.isfinite(v) else \
                erd_colour(float(np.clip(v, -full_scale, full_scale)))
    c.image(ix + 54, iy, iw - 54, ih, img)

    # The heat alone is hard to read as a trend, so trace the lateralisation
    # over the top: above the midline the right hemisphere is the quieter
    # one, which is imagined LEFT hand, and below it the reverse.
    if lateralisation is not None and np.isfinite(lateralisation).sum() > 3:
        lat = np.asarray(lateralisation, dtype=float)
        if len(lat) >= 7:
            kern = np.ones(7) / 7
            good = np.isfinite(lat)
            filled = np.where(good, lat, np.nanmean(lat[good]))
            lat = np.convolve(filled, kern, mode="same")
        xs = np.linspace(ix + 54, ix + iw, len(lat))
        mid = iy + ih / 2
        ys = mid - np.clip(lat / (full_scale * 1.6), -1, 1) * (ih / 2 - 4)
        c.line(ix + 54, mid, ix + iw, mid, (190, 200, 215), 1)
        c.polyline(list(zip(xs, ys)), (248, 250, 253), 2)

    for i, name in enumerate(names):
        c.text(ix, iy + row_h * (i + 0.5), name, size=15,
               colour=CH_COLOUR.get(name, TEXT), anchor="ml", bold=True)
    c.text(ix + 54, iy + ih + 2, "30s ago", size=11, colour=DIM)
    c.text(ix + iw, iy + ih + 2, "now", size=11, colour=DIM, anchor="tr")
    if lateralisation is not None:
        c.text(ix + iw / 2 + 27, iy + ih + 2, "white line = which hand",
               size=11, colour=DIM, anchor="tc")


def draw_spectrum(c, rect, freqs, psd, names, mu_band, baseline_psd=None):
    """Power spectrum, with the mu band shaded.

    The clearest single picture of what the decoder lives on: a bump around
    10 Hz that shrinks when the person imagines moving. The faint line is the
    resting spectrum, so the shrinkage is visible rather than inferred.
    """
    ix, iy, iw, ih = panel(c, rect, "spectrum",
                           f"{mu_band[0]:.1f}–{mu_band[1]:.1f} Hz")
    if len(freqs) < 3:
        return
    fmax = float(freqs[-1])
    lo = ix + (mu_band[0] / fmax) * iw
    hi = ix + (mu_band[1] / fmax) * iw
    c.rect(lo, iy, max(hi - lo, 2), ih, (30, 40, 56))
    c.text((lo + hi) / 2, iy + 2, "mu", size=12, colour=MUTED, anchor="tc")

    # Amplitude (sqrt of power), because power has a dynamic range that
    # leaves everything except the mu peak pressed flat against the axis.
    psd = np.sqrt(np.maximum(psd, 0))
    if baseline_psd is not None:
        baseline_psd = np.sqrt(np.maximum(baseline_psd, 0))
    ref = psd if baseline_psd is None else np.vstack([psd, baseline_psd])
    top = float(np.nanmax(ref)) if np.isfinite(ref).any() else 1.0
    top = max(top, 1e-30) * 1.08

    def ypix(v):
        return iy + ih - (float(np.clip(v, 0, top)) / top) * (ih - 8)

    for hz in (10, 20, 30, 40):
        if hz < fmax:
            gx = ix + (hz / fmax) * iw
            c.line(gx, iy, gx, iy + ih, (28, 34, 46), 1)
            c.text(gx, iy + ih + 2, f"{hz}", size=11, colour=DIM, anchor="tc")
    c.text(ix, iy + ih + 2, "Hz", size=11, colour=DIM)

    xs = ix + (freqs / fmax) * iw
    for i, name in enumerate(names):
        col = CH_COLOUR.get(name, TEXT)
        if baseline_psd is not None and i < len(baseline_psd):
            c.polyline(list(zip(xs, [ypix(v) for v in baseline_psd[i]])),
                       mix(col, PANEL, 0.45), 1)
        if i < len(psd):
            c.polyline(list(zip(xs, [ypix(v) for v in psd[i]])), col, 2)
    c.text(ix + iw, iy, "faint line = resting level", size=11, colour=DIM,
           anchor="tr")


def draw_topomap(c, rect, grid, electrodes, full_scale=70.0, quality=None):
    """Mu-band ERD across the scalp.

    Lateralisation is the whole point and a map shows it instantly: one side
    of the head goes blue while the other does not.
    """
    ix, iy, iw, ih = panel(c, rect, "scalp map", "mu ERD")
    size = int(min(iw, ih) * 0.92)
    cx, cy = ix + iw / 2, iy + ih / 2
    r = size / 2

    if grid is not None and np.isfinite(grid).any():
        g = grid.shape[0]
        img = np.zeros((g, g, 3), dtype=np.uint8)
        for a in range(g):
            for b in range(g):
                v = grid[a, b]
                img[a, b] = PANEL if not np.isfinite(v) else \
                    erd_colour(float(np.clip(v, -full_scale, full_scale)))
        c.image(cx - r, cy - r, 2 * r, 2 * r, img)

    c.circle(cx, cy, r, BORDER, 2)
    c.polyline([(cx - r * 0.13, cy - r * 0.99), (cx, cy - r * 1.12),
                (cx + r * 0.13, cy - r * 0.99)], BORDER, 2)   # nose
    for s in (-1, 1):
        c.polyline([(cx + s * r * 0.99, cy - r * 0.16),
                    (cx + s * r * 1.08, cy),
                    (cx + s * r * 0.99, cy + r * 0.16)], BORDER, 2)  # ears

    for name, (ex, ey) in electrodes:
        px, py = cx + ex * r, cy - ey * r
        bad = quality and quality.get(name) not in (None, "ok")
        c.circle(px, py, 4, BAD if bad else (235, 240, 248))
        if name in ("C3", "C4", "Cz"):
            c.text(px, py - 10, name, size=13,
                   colour=CH_COLOUR.get(name, TEXT), anchor="bc", bold=True)

    bx, bw2 = ix, iw
    for k in range(60):
        t = k / 59.0
        c.rect(bx + t * bw2, iy + ih - 10, bw2 / 59 + 1, 8,
               erd_colour(-full_scale + 2 * full_scale * t))
    c.text(bx, iy + ih - 14, f"−{full_scale:.0f}%", size=11, colour=DIM,
           anchor="bl")
    c.text(bx + bw2, iy + ih - 14, f"+{full_scale:.0f}%", size=11, colour=DIM,
           anchor="br")


# ----------------------------------------------------------- the decode
def draw_decision(c, rect, prob, evidence, command, dwell, dwell_target,
                  neg_class="right", pos_class="left", deadzone=0.08,
                  armed=True, message=""):
    """What the classifier currently believes, and what the arm is doing.

    Two different quantities, deliberately shown apart: the instant
    probability jitters every window, while the accumulated evidence is what
    actually moves anything. Showing only the smooth one would hide how noisy
    the underlying decode really is.

    The axis is named rather than positional, because the obvious guess is
    wrong: positive evidence means the LEFT hand, which is the RIGHT
    hemisphere desynchronising. Labelling the ends "left, right" in reading
    order silently puts each hand over the opposite hemisphere's colour.
    So neg_class sits at the amber (C3, left brain) end and pos_class at
    the blue (C4, right brain) end, matching every other panel.
    """
    ix, iy, iw, ih = panel(c, rect, "decoder",
                           f"{neg_class} vs {pos_class} hand")
    mid = ix + iw / 2

    c.text(ix, iy, "this window", size=12, colour=DIM)
    py = iy + 20
    c.rect(ix, py, iw, 14, (28, 34, 46), radius=7)
    t = float(np.clip(2 * prob - 1, -1, 1))
    px = mid + t * iw / 2
    col = RIGHT_HEMI if t > 0 else LEFT_HEMI
    c.rect(min(px, mid), py, abs(px - mid), 14, mix(col, PANEL, 0.35),
           radius=7)
    c.line(mid, py - 3, mid, py + 17, DIM, 1)
    c.circle(px, py + 7, 8, col)

    ey = py + 48
    c.text(ix, ey - 20, "accumulated evidence", size=12, colour=DIM)
    c.rect(ix, ey, iw, 26, (28, 34, 46), radius=13)
    dz = (deadzone / 1.0) * iw / 2
    c.rect(mid - dz, ey, 2 * dz, 26, (36, 44, 58), radius=13)
    e = float(np.clip(evidence, -1, 1))
    ex = mid + e * iw / 2
    ecol = RIGHT_HEMI if e > 0 else LEFT_HEMI
    c.rect(min(ex, mid), ey, abs(ex - mid), 26, ecol, radius=13)
    c.line(mid, ey - 4, mid, ey + 30, MUTED, 2)
    c.text(ix, ey + 32, f"{neg_class.upper()} HAND", size=13, colour=LEFT_HEMI)
    c.text(ix, ey + 48, "C3 · left brain", size=11, colour=DIM)
    c.text(ix + iw, ey + 32, f"{pos_class.upper()} HAND", size=13,
           colour=RIGHT_HEMI, anchor="tr")
    c.text(ix + iw, ey + 48, "C4 · right brain", size=11, colour=DIM,
           anchor="tr")
    c.text(mid, ey + 32, "no command", size=11, colour=DIM, anchor="tc")

    by = ey + 74
    label = (command or "—").upper()
    lcol = (LEFT_HEMI if command == neg_class else
            RIGHT_HEMI if command == pos_class else MUTED)
    c.text(ix, by + 20, "arm", size=12, colour=DIM, anchor="ml")
    c.text(ix + 44, by + 20, label, size=34, colour=lcol, anchor="ml",
           bold=True)

    if dwell_target > 0:
        rr = 26
        rx, ry = ix + iw - rr - 4, by + 20
        c.circle(rx, ry, rr, (30, 37, 50), 3)
        frac = float(np.clip(dwell / dwell_target, 0, 1))
        for k in range(int(frac * 48)):
            a = -np.pi / 2 + 2 * np.pi * k / 48
            c.circle(rx + rr * np.cos(a), ry + rr * np.sin(a), 2.5, GOOD)
        c.text(rx, ry, f"{dwell:.1f}", size=15, colour=TEXT, anchor="mc",
               bold=True)
        c.text(rx, ry + rr + 6, "hold to grab", size=11, colour=DIM,
               anchor="tc")

    if message:
        c.text(ix, iy + ih - 2, message[:48], size=13,
               colour=MUTED if armed else WARN, anchor="bl")


# ------------------------------------------------------------ the extras
def draw_camera(c, rect, frame, label="participant", status=None):
    ix, iy, iw, ih = panel(c, rect, "camera", status or label)
    if frame is None:
        c.rect(ix, iy, iw, ih, (22, 28, 38), radius=6)
        c.text(ix + iw / 2, iy + ih / 2, status or "no camera", size=16,
               colour=DIM, anchor="mc")
        return
    fh, fw = frame.shape[:2]
    s = min(iw / fw, ih / fh)
    dw, dh = fw * s, fh * s
    c.image(ix + (iw - dw) / 2, iy + (ih - dh) / 2, dw, dh, frame)


def draw_arm(c, rect, frame, state="", target=""):
    ix, iy, iw, ih = panel(c, rect, "robot arm", state)
    if frame is None:
        c.rect(ix, iy, iw, ih, (22, 28, 38), radius=6)
        c.text(ix + iw / 2, iy + ih / 2, "no render", size=16, colour=DIM,
               anchor="mc")
        return
    fh, fw = frame.shape[:2]
    s = min(iw / fw, ih / fh)
    dw, dh = fw * s, fh * s
    c.image(ix + (iw - dw) / 2, iy + (ih - dh) / 2, dw, dh, frame)
    if target:
        c.text(ix + iw / 2, iy + ih - 6, target, size=14, colour=MUTED,
               anchor="bc")


def draw_stats(c, rect, stats):
    ix, iy, iw, ih = panel(c, rect, "session")
    rows = list(stats.items())
    n = max(1, len(rows))
    row_h = min(30, ih / n)
    for i, (k, v) in enumerate(rows):
        y = iy + i * row_h
        col = TEXT
        if isinstance(v, tuple):
            v, col = v
        c.text(ix, y, k, size=14, colour=MUTED)
        c.text(ix + iw, y, str(v), size=15, colour=col, anchor="tr", bold=True)
