"""
A very small drawing abstraction with two backends.

The panels are written once against this, then rendered either by pygame
(live, during a session) or by matplotlib (offline, to a PNG). That second
backend exists so the layout can be looked at and corrected without a cap,
an amplifier and a robot all being present - the same reason the cue figures
in the EMG protocol were previewable.

Coordinates are pixels, origin top-left, y downwards, like a screen.
"""

import numpy as np

# ------------------------------------------------------------------ palette
BG        = (10, 13, 18)
PANEL     = (18, 23, 32)
PANEL_HI  = (26, 33, 45)
BORDER    = (38, 47, 62)
TEXT      = (232, 236, 242)
MUTED     = (125, 136, 153)
DIM       = (72, 82, 97)

# One colour per hemisphere, used everywhere that hemisphere appears, so the
# eye can follow "left brain" across the traces, the bars and the scalp map.
LEFT_HEMI  = (255, 159, 69)    # C3  - amber. Active for RIGHT-hand imagery.
RIGHT_HEMI = (77, 163, 255)    # C4  - blue.  Active for LEFT-hand imagery.
MIDLINE    = (150, 160, 175)   # Cz

GOOD    = (61, 220, 151)
WARN    = (255, 196, 61)
BAD     = (255, 99, 99)
ACCENT  = (167, 139, 250)


def mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def erd_colour(pct):
    """Blue for desynchronisation, red for synchronisation, grey at rest.

    Diverging around zero, because the sign is the meaning: a drop is cortex
    engaging, a rise is cortex idling harder.
    """
    if pct <= 0:
        return mix((90, 100, 115), (90, 200, 255), min(1.0, -pct / 60.0))
    return mix((90, 100, 115), (255, 110, 90), min(1.0, pct / 60.0))


class Canvas:
    """Backend-independent surface."""

    def rect(self, x, y, w, h, colour, radius=0, width=0): ...
    def line(self, x0, y0, x1, y1, colour, width=1): ...
    def polyline(self, pts, colour, width=1): ...
    def circle(self, x, y, r, colour, width=0): ...
    def text(self, x, y, s, size=16, colour=TEXT, anchor="tl", bold=False): ...
    def image(self, x, y, w, h, rgb): ...
    def clip(self, x, y, w, h): ...
    def unclip(self): ...


# ------------------------------------------------------------------ pygame
class PygameCanvas(Canvas):
    def __init__(self, surface, font_name=None):
        import pygame
        self.pg = pygame
        self.s = surface
        self._fonts = {}
        self._name = font_name

    def _font(self, size, bold):
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = self.pg.font.SysFont(self._name, size, bold=bold)
        return self._fonts[key]

    def rect(self, x, y, w, h, colour, radius=0, width=0):
        self.pg.draw.rect(self.s, colour, (int(x), int(y), int(w), int(h)),
                          int(width), border_radius=int(radius))

    def line(self, x0, y0, x1, y1, colour, width=1):
        self.pg.draw.line(self.s, colour, (int(x0), int(y0)),
                          (int(x1), int(y1)), int(width))

    def polyline(self, pts, colour, width=1):
        if len(pts) < 2:
            return
        self.pg.draw.lines(self.s, colour, False,
                           [(int(a), int(b)) for a, b in pts], int(width))

    def circle(self, x, y, r, colour, width=0):
        self.pg.draw.circle(self.s, colour, (int(x), int(y)), int(r), int(width))

    def text(self, x, y, s, size=16, colour=TEXT, anchor="tl", bold=False):
        img = self._font(size, bold).render(str(s), True, colour)
        w, h = img.get_size()
        px = x - w if "r" in anchor[1:] else (x - w // 2 if "c" in anchor[1:] else x)
        py = y - h if anchor[0] == "b" else (y - h // 2 if anchor[0] == "m" else y)
        self.s.blit(img, (int(px), int(py)))
        return w, h

    def image(self, x, y, w, h, rgb):
        arr = np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)))
        surf = self.pg.surfarray.make_surface(arr)
        if surf.get_size() != (int(w), int(h)):
            surf = self.pg.transform.smoothscale(surf, (int(w), int(h)))
        self.s.blit(surf, (int(x), int(y)))

    def clip(self, x, y, w, h):
        self.s.set_clip(self.pg.Rect(int(x), int(y), int(w), int(h)))

    def unclip(self):
        self.s.set_clip(None)


# -------------------------------------------------------------- matplotlib
class MplCanvas(Canvas):
    """Draws into a matplotlib axes in pixel coordinates, for previewing."""

    def __init__(self, ax, width, height):
        self.ax = ax
        self.h = height
        # pygame paints in call order; matplotlib paints by zorder. Hand out a
        # rising zorder per call so the preview matches the live window.
        self._z = 0
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    def _next_z(self):
        self._z += 1
        return self._z

    @staticmethod
    def _c(colour):
        return tuple(v / 255.0 for v in colour[:3])

    def rect(self, x, y, w, h, colour, radius=0, width=0):
        from matplotlib.patches import FancyBboxPatch, Rectangle
        c = self._c(colour)
        if radius:
            p = FancyBboxPatch((x + radius, y + radius), max(w - 2 * radius, 1),
                               max(h - 2 * radius, 1),
                               boxstyle=f"round,pad={radius}",
                               linewidth=width or 0,
                               edgecolor=c if width else "none",
                               facecolor="none" if width else c)
        else:
            p = Rectangle((x, y), w, h, linewidth=width or 0,
                          edgecolor=c if width else "none",
                          facecolor="none" if width else c)
        p.set_zorder(self._next_z())
        self.ax.add_patch(p)

    def line(self, x0, y0, x1, y1, colour, width=1):
        self.ax.plot([x0, x1], [y0, y1], color=self._c(colour),
                     lw=max(width * 0.75, 0.4), solid_capstyle="round",
                     zorder=self._next_z())

    def polyline(self, pts, colour, width=1):
        if len(pts) < 2:
            return
        p = np.asarray(pts, dtype=float)
        self.ax.plot(p[:, 0], p[:, 1], color=self._c(colour),
                     lw=max(width * 0.75, 0.4), solid_capstyle="round",
                     zorder=self._next_z())

    def circle(self, x, y, r, colour, width=0):
        from matplotlib.patches import Circle
        c = self._c(colour)
        self.ax.add_patch(Circle((x, y), r, linewidth=max(width * 0.75, 0.4),
                                 edgecolor=c if width else "none",
                                 facecolor="none" if width else c,
                                 zorder=self._next_z()))

    def text(self, x, y, s, size=16, colour=TEXT, anchor="tl", bold=False):
        ha = {"l": "left", "c": "center", "r": "right"}[
            anchor[1] if len(anchor) > 1 else "l"]
        va = {"t": "top", "m": "center", "b": "bottom"}[anchor[0]]
        self.ax.text(x, y, str(s), color=self._c(colour), fontsize=size * 0.72,
                     ha=ha, va=va, family="DejaVu Sans",
                     zorder=self._next_z(),
                     fontweight="bold" if bold else "normal")
        return (len(str(s)) * size * 0.5, size)

    def image(self, x, y, w, h, rgb):
        self.ax.imshow(np.asarray(rgb, dtype=np.uint8),
                       extent=(x, x + w, y + h, y), zorder=self._next_z(),
                       interpolation="bilinear")

    def clip(self, x, y, w, h):
        pass

    def unclip(self):
        pass
