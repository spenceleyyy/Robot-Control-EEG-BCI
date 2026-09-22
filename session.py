"""
Motor imagery recording session.

Run this with the cap on. It connects to the amplifier, walks you through a
resting baseline, a motor execution block and a motor imagery block, writes
everything to data/, and then runs the ERD analysis.

Keys:  SPACE / ENTER = continue      ESC = abort (data so far is still saved)
"""

import datetime
import os
import random
import sys
import time

import numpy as np
import pygame

import config
import lsl_io


class Aborted(Exception):
    pass


# ----------------------------------------------------------------- rendering
class Screen:
    def __init__(self):
        pygame.init()
        pygame.mouse.set_visible(False)
        if config.FULLSCREEN:
            self.surf = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.surf = pygame.display.set_mode(config.WINDOW_SIZE)
        pygame.display.set_caption("MI Testbed")
        self.w, self.h = self.surf.get_size()
        self.font = pygame.font.SysFont(None, config.FONT_SIZE)
        self.font_big = pygame.font.SysFont(None, config.FONT_SIZE_BIG)
        self.clock = pygame.time.Clock()

    # -- primitives --------------------------------------------------------
    def clear(self):
        self.surf.fill(config.BG_COLOR)

    def flip(self):
        pygame.display.flip()

    def text_block(self, lines, y_start=None, font=None, color=None, center=True):
        font = font or self.font
        color = color or config.FG_COLOR
        line_h = font.get_linesize() + 6
        total = line_h * len(lines)
        y = y_start if y_start is not None else (self.h - total) // 2
        for line in lines:
            img = font.render(line, True, color)
            x = (self.w - img.get_width()) // 2 if center else self.w // 8
            self.surf.blit(img, (x, y))
            y += line_h

    def big_center(self, text, color=None):
        img = self.font_big.render(text, True, color or config.FG_COLOR)
        self.surf.blit(img, ((self.w - img.get_width()) // 2,
                             (self.h - img.get_height()) // 2))

    def fixation(self):
        cx, cy, r, t = self.w // 2, self.h // 2, 28, 5
        pygame.draw.rect(self.surf, config.FIX_COLOR, (cx - r, cy - t // 2, 2 * r, t))
        pygame.draw.rect(self.surf, config.FIX_COLOR, (cx - t // 2, cy - r, t, 2 * r))

    def arrow(self, direction):
        cx, cy = self.w // 2, self.h // 2
        L, H = 150, 70
        if direction == "left":
            pts = [(cx - L, cy), (cx - L // 3, cy - H), (cx - L // 3, cy - H // 3),
                   (cx + L, cy - H // 3), (cx + L, cy + H // 3),
                   (cx - L // 3, cy + H // 3), (cx - L // 3, cy + H)]
        elif direction == "right":
            pts = [(cx + L, cy), (cx + L // 3, cy - H), (cx + L // 3, cy - H // 3),
                   (cx - L, cy - H // 3), (cx - L, cy + H // 3),
                   (cx + L // 3, cy + H // 3), (cx + L // 3, cy + H)]
        else:  # feet / down
            pts = [(cx, cy + L), (cx - H, cy + L // 3), (cx - H // 3, cy + L // 3),
                   (cx - H // 3, cy - L), (cx + H // 3, cy - L),
                   (cx + H // 3, cy + L // 3), (cx + H, cy + L // 3)]
        pygame.draw.polygon(self.surf, config.CUE_COLOR, pts)

    def hint(self, text):
        img = self.font.render(text, True, (110, 110, 110))
        self.surf.blit(img, ((self.w - img.get_width()) // 2, self.h - 90))

    # -- input -------------------------------------------------------------
    def pump(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise Aborted()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                raise Aborted()

    def wait_until(self, t_end):
        while time.perf_counter() < t_end:
            self.pump()
            self.clock.tick(200)

    def wait_key(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise Aborted()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        raise Aborted()
                    if ev.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                        return
            self.clock.tick(120)

    def page(self, lines, hint="press SPACE to continue"):
        self.clear()
        self.text_block(lines)
        self.hint(hint)
        self.flip()
        self.wait_key()

    def countdown(self, seconds, label):
        end = time.perf_counter() + seconds
        last = None
        while True:
            self.pump()
            left = end - time.perf_counter()
            if left <= 0:
                return
            shown = int(left) + 1
            if shown != last:
                self.clear()
                self.text_block([label], y_start=self.h // 2 - 140)
                self.big_center(str(shown))
                self.flip()
                last = shown
            self.clock.tick(60)


# ------------------------------------------------------------------ paradigm
def build_trials(classes, per_class, block_size):
    """Pseudo-random order: shuffle within blocks so classes stay balanced."""
    trials = []
    n_blocks = max(1, (per_class * len(classes)) // max(1, block_size))
    per_block = max(1, per_class // n_blocks)
    for _ in range(n_blocks):
        chunk = classes * per_block
        random.shuffle(chunk)
        trials.extend(chunk)
    # top up any remainder so counts come out exact
    for c in classes:
        while trials.count(c) < per_class:
            trials.append(c)
    return trials


def run_trial(scr, rec, cls, task_tag):
    """One cued trial. Markers are pushed at the moment of the screen flip."""
    # fixation
    scr.clear()
    scr.fixation()
    scr.flip()
    rec.mark(f"{task_tag}/fixation")
    t0 = time.perf_counter()
    scr.wait_until(t0 + config.T_FIXATION)

    # cue onset  -- this is the epoch anchor, t = 0
    scr.clear()
    scr.fixation()
    scr.arrow(cls)
    scr.flip()
    rec.mark(f"{task_tag}/cue/{cls}")
    t_cue = time.perf_counter()
    scr.wait_until(t_cue + config.T_CUE_VISIBLE)

    # cue removed, task continues while the cross is up
    scr.clear()
    scr.fixation()
    scr.flip()
    scr.wait_until(t_cue + config.T_TASK)

    # inter-trial interval, blink here
    scr.clear()
    scr.hint("relax  -  blink now")
    scr.flip()
    rec.mark(f"{task_tag}/end")
    iti = random.uniform(config.T_ITI_MIN, config.T_ITI_MAX)
    scr.wait_until(time.perf_counter() + iti)


def run_block(scr, rec, trials, task_tag, block_label):
    done = 0
    per_run = config.MI_TRIALS_PER_RUN
    for i, cls in enumerate(trials):
        if i > 0 and i % per_run == 0:
            scr.page([
                f"{block_label}: {i} of {len(trials)} trials done.",
                "",
                "Take a breather. Move around, blink, relax your jaw.",
                "Stay seated so the cap does not shift.",
            ])
            scr.countdown(3, "starting again")
        run_trial(scr, rec, cls, task_tag)
        done += 1
    return done


def run_baseline(scr, rec):
    scr.page([
        "RESTING BASELINE",
        "",
        "First, two minutes of rest. Nothing to do.",
        "This finds your personal alpha frequency and proves",
        "the recording chain is actually working.",
        "",
        "Sit still. Relax your jaw and shoulders.",
    ])

    scr.page([
        "EYES OPEN  -  1 minute",
        "",
        "Look at the cross. Let your mind wander.",
        "Try not to move or clench.",
    ])
    scr.clear()
    scr.fixation()
    scr.flip()
    rec.mark("baseline/eyes_open")
    scr.wait_until(time.perf_counter() + config.BASELINE_EYES_OPEN_S)
    rec.mark("baseline/eyes_open/end")

    scr.page([
        "EYES CLOSED  -  1 minute",
        "",
        "When you press SPACE, close your eyes and keep them closed.",
        "A beep is not used - you will hear nothing.",
        "Open them again when you feel the minute is up,",
        "or just wait for the screen to change.",
    ], hint="press SPACE, then close your eyes")
    scr.clear()
    scr.text_block(["eyes closed"], color=(60, 60, 60))
    scr.flip()
    rec.mark("baseline/eyes_closed")
    scr.wait_until(time.perf_counter() + config.BASELINE_EYES_CLOSED_S)
    rec.mark("baseline/eyes_closed/end")

    scr.clear()
    scr.big_center("open your eyes")
    scr.flip()
    scr.wait_until(time.perf_counter() + 3.0)


def run_execution(scr, rec):
    scr.page([
        "BLOCK 1  -  REAL MOVEMENT",
        "",
        "This block you actually move. It is the control condition:",
        "real movement produces a much larger signal than imagery,",
        "so if nothing shows up here, the setup is at fault.",
        "",
        f"{config.ME_TRIALS_PER_CLASS * len(config.CLASSES)} trials, about 3 minutes.",
    ])
    scr.page([
        "WHAT TO DO",
        "",
        "Arrow points LEFT   ->  squeeze your LEFT hand, open, squeeze...",
        "Arrow points RIGHT  ->  squeeze your RIGHT hand, open, squeeze...",
        "",
        "Keep squeezing at about twice per second until the cross",
        "disappears. Move only the hand. Keep your arms and",
        "shoulders still, and do not clench your teeth.",
    ])
    scr.countdown(3, "get ready")
    trials = build_trials(config.CLASSES, config.ME_TRIALS_PER_CLASS,
                          config.ME_TRIALS_PER_CLASS * len(config.CLASSES))
    return run_block(scr, rec, trials, "me", "Real movement")


def run_imagery(scr, rec):
    scr.page([
        "BLOCK 2  -  MOTOR IMAGERY",
        "",
        "Same cues, but now you do not move at all.",
        "",
        f"{config.MI_TRIALS_PER_CLASS * len(config.CLASSES)} trials, "
        f"about {config.MI_TRIALS_PER_CLASS * len(config.CLASSES) * 9 / 60:.0f} minutes,",
        f"with a break every {config.MI_TRIALS_PER_RUN} trials.",
    ])
    scr.page([
        "THIS PART MATTERS  -  HOW TO IMAGINE",
        "",
        "Do NOT picture your hand from the outside, like watching a video.",
        "Visual imagery produces almost no signal over motor cortex.",
        "",
        "Instead, feel it from the inside: the tension in your forearm,",
        "the pressure in your palm, the effort of the squeeze.",
        "It should feel like you are about to move but holding back.",
        "",
        "Keep the sensation going the whole time the cross is on screen.",
    ])
    scr.page([
        "AND KEEP STILL",
        "",
        "No actual muscle movement. No jaw clenching, no swallowing,",
        "no eye movement during the cross. Blink in the gaps between",
        "trials, when the screen says so.",
        "",
        "Muscle artifact is the easiest way to fool yourself into",
        "thinking this worked.",
    ])
    scr.countdown(3, "get ready")
    trials = build_trials(config.CLASSES, config.MI_TRIALS_PER_CLASS,
                          config.MI_TRIALS_PER_RUN)
    return run_block(scr, rec, trials, "mi", "Motor imagery")


# ---------------------------------------------------------------------- main
def main():
    print("=" * 62)
    print("  Motor imagery testbed")
    print("=" * 62)

    inlet, meta = lsl_io.find_eeg_stream()
    probe = lsl_io.probe_amplitude(inlet, 2.0)
    if probe is None:
        raise RuntimeError(
            "Connected to the stream but no samples arrived.\n"
            "In BrainVision Recorder, make sure you are in Monitor or "
            "Recording mode, not Impedance Check."
        )
    scale = lsl_io.resolve_units_scale(probe)
    unit_name = "microvolts" if scale == 1e-6 else "volts"
    print(f"Data is flowing. Typical deflection {probe['median_abs']:.3g} "
          f"-> treating as {unit_name}.")
    if probe["flat_channels"]:
        flat = [meta["ch_names"][i] for i in probe["flat_channels"]]
        print(f"[warn] Flat (dead) channels: {', '.join(flat[:12])}")

    rec = lsl_io.Recorder(inlet, meta, scale)
    scr = Screen()
    aborted = False
    rec.start()

    try:
        scr.page([
            "MOTOR IMAGERY SESSION",
            "",
            f"Amplifier: {meta['name']}",
            f"{meta['n_channels']} channels at {meta['sfreq']:.0f} Hz",
            "",
            "Before you start, confirm in BrainVision Recorder that",
            "impedances are below about 10 kOhm, especially at C3, Cz, C4.",
            "",
            "About 22 minutes total. ESC aborts and still saves.",
        ])
        scr.page([
            "GENERAL RULES",
            "",
            "Sit comfortably, both feet on the floor, hands resting",
            "in the same place the whole session.",
            "",
            "Relax your jaw. A clenched jaw swamps the signal.",
            "Blink during the gaps, not during the cross.",
        ])

        run_baseline(scr, rec)
        n_me = run_execution(scr, rec)
        n_mi = run_imagery(scr, rec)

        scr.clear()
        scr.text_block(["Done.", "", f"{n_me} movement trials, "
                                     f"{n_mi} imagery trials.", "",
                        "Saving and analysing..."])
        scr.flip()

    except Aborted:
        aborted = True
        print("\nAborted by user. Saving whatever was recorded.")
    finally:
        rec.stop()
        pygame.quit()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"mi_{stamp}" + ("_partial" if aborted else "")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           config.DATA_DIR)
    fif_path, npz_path = rec.save(out_dir, basename)

    if fif_path:
        print("\nRunning the ERD analysis...\n")
        try:
            import analyze_erd
            analyze_erd.analyze(fif_path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"\n[warn] Analysis failed ({exc}).")
            print(f"Recording is fine. Retry with:\n"
                  f"   python analyze_erd.py {fif_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(1)
