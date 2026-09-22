"""
Closed-loop feedback task.

A cursor moves left or right according to your motor imagery, live. Hit the
highlighted target. This is the training tool: motor imagery improves with
feedback in a way that staring at arrows does not, so this is where your
accuracy actually goes up.

    feedback.bat          /  ./feedback.sh

Keys:  ESC = stop (results are still saved)

The cursor integrates evidence rather than jumping to each decision. A single
1-second window is only ~70% right; accumulated over a couple of seconds that
becomes reliable, at the cost of latency. That trade-off is GAIN and SMOOTH
below.
"""

import datetime
import os
import pickle
import random
import sys
import time

import numpy as np
import pygame

import config
import lsl_io
import preproc


# ------------------------------------------------------------------ tuning
N_TRIALS_PER_CLASS = 20
T_FIXATION = 2.0
T_MAX_FEEDBACK = 6.0     # give up if the target is not reached
T_OUTCOME = 1.0
T_ITI_MIN, T_ITI_MAX = 1.5, 2.5
WARMUP_S = 10.0          # settle the filter and estimate resting bias

GAIN = 1.1               # cursor units per second at full confidence
SMOOTH = 0.72            # evidence accumulation; higher = smoother, slower
BIAS_TAU_S = 20.0        # how fast the auto-centring adapts
HIT_X = 1.0


class Aborted(Exception):
    pass


def load_decoder():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "decoder.pkl")
    if not os.path.exists(path):
        raise RuntimeError("No decoder.pkl found. Run train_decoder first.")
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    print(f"Decoder from {d['source']}, "
          f"offline window accuracy {d['cv_accuracy'] * 100:.1f}%")
    return d


class Screen:
    def __init__(self):
        pygame.init()
        pygame.mouse.set_visible(False)
        if config.FULLSCREEN:
            self.surf = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.surf = pygame.display.set_mode(config.WINDOW_SIZE)
        pygame.display.set_caption("BCI feedback")
        self.w, self.h = self.surf.get_size()
        self.font = pygame.font.SysFont(None, config.FONT_SIZE)
        self.big = pygame.font.SysFont(None, config.FONT_SIZE_BIG)
        self.clock = pygame.time.Clock()

    def pump(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise Aborted()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                raise Aborted()

    def wait_key(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise Aborted()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        raise Aborted()
                    if ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                        return
            self.clock.tick(120)

    def page(self, lines, hint="press SPACE"):
        self.surf.fill(config.BG_COLOR)
        lh = self.font.get_linesize() + 6
        y = (self.h - lh * len(lines)) // 2
        for ln in lines:
            img = self.font.render(ln, True, config.FG_COLOR)
            self.surf.blit(img, ((self.w - img.get_width()) // 2, y))
            y += lh
        img = self.font.render(hint, True, (110, 110, 110))
        self.surf.blit(img, ((self.w - img.get_width()) // 2, self.h - 90))
        pygame.display.flip()
        self.wait_key()

    def draw_task(self, cursor_x, target, outcome=None, confidence=0.0):
        s = self.surf
        s.fill(config.BG_COLOR)
        cy = self.h // 2
        margin = int(self.w * 0.10)
        bw = int(self.w * 0.10)
        half = (self.w // 2) - margin - bw

        for side, sign in (("left", -1), ("right", 1)):
            x = self.w // 2 + sign * (half + bw // 2) - bw // 2
            if outcome == "hit" and target == side:
                col = (60, 200, 90)
            elif outcome == "miss" and target == side:
                col = (90, 90, 90)
            elif target == side:
                col = config.CUE_COLOR
            else:
                col = (55, 55, 55)
            pygame.draw.rect(s, col, (x, cy - 140, bw, 280), border_radius=8)

        # track
        pygame.draw.line(s, (70, 70, 70), (self.w // 2 - half, cy),
                         (self.w // 2 + half, cy), 3)
        # cursor
        px = int(self.w // 2 + np.clip(cursor_x, -1.2, 1.2) * half)
        r = 22 + int(10 * min(1.0, abs(confidence)))
        pygame.draw.circle(s, (240, 240, 240), (px, cy), r)

        pygame.display.flip()

    def message(self, text, seconds, color=None):
        self.surf.fill(config.BG_COLOR)
        img = self.big.render(text, True, color or config.FG_COLOR)
        self.surf.blit(img, ((self.w - img.get_width()) // 2,
                             (self.h - img.get_height()) // 2))
        pygame.display.flip()
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.pump()
            self.clock.tick(120)


class Decoder:
    """Keeps the live buffer, filter state and running evidence."""

    def __init__(self, model, stream_names, sfreq, scale):
        self.m = model
        self.sfreq = sfreq
        self.scale = scale
        idx, missing = preproc.channel_indices(stream_names, model["ch_names"])
        if missing:
            raise RuntimeError(
                f"The decoder needs channels the stream does not have: "
                f"{', '.join(missing)}")
        self.idx = idx
        self.win_n = int(round(model["window_s"] * sfreq))
        self.filt = preproc.CausalBandpass(sfreq, model["band"], len(idx))
        self.buf = np.zeros((len(idx), self.win_n * 3))
        self.filled = 0
        self.evidence = 0.0
        self.bias = 0.0
        self.bias_beta = np.exp(-model["step_s"] / BIAS_TAU_S)
        self.last_p = 0.5

    def push(self, chunk):
        """chunk: (n_samples, n_stream_channels) straight from LSL."""
        x = np.asarray(chunk, dtype=np.float64).T[self.idx] * self.scale
        x = preproc.average_reference(x)
        x = self.filt(x)
        n = x.shape[1]
        if n >= self.buf.shape[1]:
            self.buf = x[:, -self.buf.shape[1]:]
            self.filled = self.buf.shape[1]
        else:
            self.buf = np.roll(self.buf, -n, axis=1)
            self.buf[:, -n:] = x
            self.filled = min(self.buf.shape[1], self.filled + n)

    def ready(self):
        return self.filled >= self.win_n

    def decide(self, update_bias=True):
        w = self.buf[:, -self.win_n:][None, :, :]
        feats = self.m["csp"].transform(w)
        p = float(self.m["lda"].predict_proba(feats)[0, 1])
        self.last_p = p
        d = 2.0 * p - 1.0
        if update_bias:
            self.bias = self.bias_beta * self.bias + (1 - self.bias_beta) * d
        centred = d - self.bias
        self.evidence = SMOOTH * self.evidence + (1 - SMOOTH) * centred
        return self.evidence


def main():
    model = load_decoder()
    inlet, meta = lsl_io.find_eeg_stream()
    probe = lsl_io.probe_amplitude(inlet, 1.5)
    if probe is None:
        raise RuntimeError("Connected but no samples arriving.")
    scale = lsl_io.resolve_units_scale(probe)

    if abs(meta["sfreq"] - model["sfreq"]) > 1.0:
        raise RuntimeError(
            f"Decoder was trained at {model['sfreq']:.0f} Hz but the stream "
            f"is {meta['sfreq']:.0f} Hz. Retrain, or set the amp back.")

    dec = Decoder(model, meta["ch_names"], meta["sfreq"], scale)
    scr = Screen()
    classes = model["classes"]
    step = model["step_s"]

    trials = [i % 2 for i in range(N_TRIALS_PER_CLASS * 2)]
    random.shuffle(trials)

    log = {"target": [], "chosen": [], "hit": [], "time": []}
    trace = []
    aborted = False

    try:
        scr.page([
            "CLOSED-LOOP FEEDBACK",
            "",
            "Move the cursor into the highlighted box using the same",
            "imagery you practised: the FELT sense of squeezing that hand.",
            "",
            "Left box  -> imagine squeezing your LEFT hand",
            "Right box -> imagine squeezing your RIGHT hand",
            "",
            f"{len(trials)} trials, roughly {len(trials) * 11 / 60:.0f} minutes.",
            "Keep still, relax your jaw, blink between trials.",
        ])
        scr.page([
            "THE CURSOR LAGS ON PURPOSE",
            "",
            "It accumulates evidence over a couple of seconds rather than",
            "reacting to every decision. Jerky instant feedback is harder",
            "to learn from than smooth delayed feedback.",
            "",
            "Do not 'push harder' when it stalls. Hold the imagery steady.",
            "Straining recruits muscle, which makes it worse, not better.",
        ])

        # warm-up: settle the filter, estimate the resting bias
        scr.message("settling...", 0.1)
        t_end = time.perf_counter() + WARMUP_S
        last = 0.0
        while time.perf_counter() < t_end:
            scr.pump()
            chunk, _ = inlet.pull_chunk(timeout=0.05, max_samples=4096)
            if chunk:
                dec.push(chunk)
            now = time.perf_counter()
            if dec.ready() and now - last > step:
                dec.decide(update_bias=True)
                last = now
            remain = t_end - time.perf_counter()
            scr.message(f"settling  {remain:0.0f}", 0.02)
        dec.evidence = 0.0

        for ti, cls in enumerate(trials):
            target = classes[cls]

            # fixation
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < T_FIXATION:
                scr.pump()
                chunk, _ = inlet.pull_chunk(timeout=0.01, max_samples=4096)
                if chunk:
                    dec.push(chunk)
                scr.draw_task(0.0, None)
                scr.clock.tick(60)
            dec.evidence = 0.0

            # feedback
            x = 0.0
            last = time.perf_counter()
            t_start = last
            outcome, chosen = None, None
            prev = last
            while True:
                scr.pump()
                chunk, _ = inlet.pull_chunk(timeout=0.005, max_samples=4096)
                if chunk:
                    dec.push(chunk)
                now = time.perf_counter()
                if dec.ready() and now - last >= step:
                    ev = dec.decide(update_bias=True)
                    trace.append((now - t_start, ti, cls, dec.last_p, ev))
                    last = now
                dt = now - prev
                prev = now
                x += dec.evidence * GAIN * dt
                x = float(np.clip(x, -1.3, 1.3))
                scr.draw_task(x, target, None, dec.evidence)

                if abs(x) >= HIT_X:
                    chosen = classes[1] if x > 0 else classes[0]
                    outcome = "hit" if chosen == target else "miss"
                    break
                if now - t_start > T_MAX_FEEDBACK:
                    chosen, outcome = None, "miss"
                    break
                scr.clock.tick(60)

            log["target"].append(target)
            log["chosen"].append(chosen if chosen else "timeout")
            log["hit"].append(outcome == "hit")
            log["time"].append(time.perf_counter() - t_start)

            scr.draw_task(x, target, outcome, dec.evidence)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < T_OUTCOME:
                scr.pump()
                chunk, _ = inlet.pull_chunk(timeout=0.01, max_samples=4096)
                if chunk:
                    dec.push(chunk)
                scr.clock.tick(60)

            # ITI
            iti = random.uniform(T_ITI_MIN, T_ITI_MAX)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < iti:
                scr.pump()
                chunk, _ = inlet.pull_chunk(timeout=0.01, max_samples=4096)
                if chunk:
                    dec.push(chunk)
                scr.surf.fill(config.BG_COLOR)
                img = scr.font.render("relax - blink now", True, (110, 110, 110))
                scr.surf.blit(img, ((scr.w - img.get_width()) // 2,
                                    scr.h - 120))
                pygame.display.flip()
                scr.clock.tick(60)

            if (ti + 1) % 20 == 0 and ti + 1 < len(trials):
                acc = np.mean(log["hit"]) * 100
                scr.page([f"{ti + 1} of {len(trials)} trials.",
                          f"Hit rate so far: {acc:.0f}%",
                          "", "Break as long as you like.",
                          "Stay seated so the cap does not move."])

    except Aborted:
        aborted = True
        print("\nStopped early.")
    finally:
        pygame.quit()

    # -------------------------------------------------------------- results
    n = len(log["hit"])
    if n == 0:
        print("No trials completed.")
        return 0

    hits = np.asarray(log["hit"])
    acc = hits.mean()
    print("\n" + "=" * 56)
    print(f"  Online hit rate: {acc * 100:.1f}%  ({hits.sum()}/{n})")
    timeouts = sum(1 for c in log["chosen"] if c == "timeout")
    print(f"  Timeouts: {timeouts}")
    for c in set(log["target"]):
        m = [h for h, t in zip(log["hit"], log["target"]) if t == c]
        print(f"  {c:6s}: {np.mean(m) * 100:.0f}%  (n={len(m)})")
    reached = [t for t, h in zip(log["time"], log["hit"]) if h]
    if reached:
        print(f"  Median time to hit: {np.median(reached):.1f}s")

    if acc > 0.75:
        print("\n  Strong. You are ready for the robot arm.")
    elif acc > 0.6:
        print("\n  Working. Run this a few more times before the arm -")
        print("  hit rate usually climbs over the first few sessions.")
    else:
        print("\n  Struggling. Check that you are using kinesthetic imagery,")
        print("  and consider retraining the decoder on a fresh session.")
    if timeouts > n * 0.3:
        print("  Many timeouts: try raising GAIN in bci_feedback.py.")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       config.DATA_DIR,
                       f"feedback_{stamp}{'_partial' if aborted else ''}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        target=np.array(log["target"], dtype=object),
        chosen=np.array(log["chosen"], dtype=object),
        hit=hits, time=np.array(log["time"]),
        trace=np.array(trace, dtype=np.float64),
        decoder_source=model["source"],
        offline_accuracy=model["cv_accuracy"],
    )
    print(f"\n  Saved {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(1)
