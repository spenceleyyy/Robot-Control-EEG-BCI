"""
Brain-controlled robot arm.

    robot.bat                                   (menu)
    python sim_arm.py --source replay           your recorded session
    python sim_arm.py --source live             cap on, connector linked
    python sim_arm.py --source keys             arrow keys, no EEG

HOW CONTROL IS SPLIT (shared autonomy)
Your EEG gives about one reliable bit every second or two. That is enough to
choose, not enough to steer seven joints. So:

  you      move the arm left and right above the row of blocks, and choose
           a block by holding the arm over it (a "dwell")
  the arm  does everything else: descends, grasps, lifts, carries the block
           to the drop zone, releases, and comes back

The arm also nudges you toward the nearest block when you are close, which
makes stopping on one much easier. The blocks brighten as the dwell fills.

KEYS (in the MuJoCo window)
  <- / ->   move (keyboard mode)       down   stop (keyboard mode)
  up        grab the nearest block now, in any mode
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("\nMuJoCo is not installed in this environment. Install it with:")
    py = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    print(f"    {py} -m pip install mujoco\n")
    sys.exit(1)

import arm_model as A
import decoder_source as S

# ------------------------------------------------------------------- tuning
GAIN_Y = 0.18          # metres per second at full evidence
DEADZONE = 0.08        # evidence below this does not move the arm
Y_MIN, Y_MAX = -0.25, 0.24

ASSIST_RADIUS = 0.06   # within this distance of a block, the arm helps
ASSIST_K = 1.2         # strength of that help, 1/s
SELECT_TOL = 0.035     # how close counts as "over" a block
DWELL_S = 1.5          # hold this long to grab
ARM_DISTANCE = 0.06    # After the arm comes home it rests over the middle
ARM_AFTER_S = 2.0      # block. Holds don't count until you have either moved
                       # this far, or this much time has passed. That gives you
                       # a head start toward another block, while the middle
                       # one is still yours if you simply wait.

FPS = 60


# --------------------------------------------------------------- controller
class ArmController:
    """Resolved-rate inverse kinematics with damped least squares.

    Each frame: measure where the grasp point is, compute the joint velocity
    that closes the gap to the target (keeping the hand pointing down), and
    integrate it into a joint command that the position servos track.
    """
    KP_POS = 6.0
    KP_ROT = 4.0
    LAMBDA = 0.05
    MAX_QDOT = 2.0
    MAX_LEAD = 0.35   # anti-windup: command never runs this far ahead

    def __init__(self, m, d):
        self.m, self.d = m, d
        oid = mujoco.mj_name2id
        self.sid = oid(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        jids = [oid(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in A.ARM_JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.lo = np.array([m.jnt_range[j][0] for j in jids])
        self.hi = np.array([m.jnt_range[j][1] for j in jids])
        self.act = [oid(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_" + j)
                    for j in A.ARM_JOINTS]
        self.fact = [oid(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                     for a in A.FINGER_ACTUATORS]
        if min([self.sid] + self.act + self.fact) < 0:
            raise RuntimeError("Model is missing a site, joint or actuator "
                               "the controller needs.")
        self.q_cmd = d.qpos[self.qadr].copy()
        self.quat_des = np.zeros(4)
        mujoco.mju_mat2Quat(self.quat_des, A.R_DOWN.flatten())
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def grasp_point(self):
        return self.d.site_xpos[self.sid].copy()

    def update(self, target, grip, dt):
        m, d = self.m, self.d
        mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.sid)
        J = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])

        pos_err = np.asarray(target) - d.site_xpos[self.sid]
        n = np.linalg.norm(pos_err)
        if n > 0.08:
            pos_err *= 0.08 / n

        cur, neg, qerr, rot_err = (np.zeros(4), np.zeros(4), np.zeros(4),
                                   np.zeros(3))
        mujoco.mju_mat2Quat(cur, d.site_xmat[self.sid])
        mujoco.mju_negQuat(neg, cur)
        mujoco.mju_mulQuat(qerr, self.quat_des, neg)
        if qerr[0] < 0:
            qerr *= -1
        mujoco.mju_quat2Vel(rot_err, qerr, 1.0)

        err = np.concatenate([self.KP_POS * pos_err, self.KP_ROT * rot_err])
        qdot = J.T @ np.linalg.solve(J @ J.T + self.LAMBDA ** 2 * np.eye(6), err)
        qdot = np.clip(qdot, -self.MAX_QDOT, self.MAX_QDOT)

        q = d.qpos[self.qadr]
        self.q_cmd = np.clip(self.q_cmd + qdot * dt,
                             q - self.MAX_LEAD, q + self.MAX_LEAD)
        self.q_cmd = np.clip(self.q_cmd, self.lo, self.hi)
        d.ctrl[self.act] = self.q_cmd
        d.ctrl[self.fact] = grip


# --------------------------------------------------------------------- task
def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Task:
    """Choosing (by EEG) and pick-and-place (automatic)."""

    def __init__(self, m, d, ctrl):
        self.m, self.d, self.ctrl = m, d, ctrl
        oid = mujoco.mj_name2id
        self.blocks = [oid(m, mujoco.mjtObj.mjOBJ_BODY, f"block{i}")
                       for i in (1, 2, 3)]
        self.block_geoms = [oid(m, mujoco.mjtObj.mjOBJ_GEOM, f"block{i}_g")
                            for i in (1, 2, 3)]
        self.base_rgba = {g: m.geom_rgba[g].copy() for g in self.block_geoms}
        self.block_qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in self.blocks]
        self.block_dadr = [m.jnt_dofadr[m.body_jntadr[b]] for b in self.blocks]
        self.block_home = [np.array([A.BLOCK_X, y, A.BLOCK_HALF, 1, 0, 0, 0])
                           for y in A.BLOCK_Y]
        self.finger_geoms = [oid(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                             for g in ("finger_l_g", "finger_r_g")]
        fj = [oid(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ("fl", "fr")]
        self.finger_qadr = [m.jnt_qposadr[j] if j >= 0 else -1 for j in fj]
        cursor = oid(m, mujoco.mjtObj.mjOBJ_BODY, "cursor")
        self.mocap = m.body_mocapid[cursor]
        self.cursor_geom = oid(m, mujoco.mjtObj.mjOBJ_GEOM, "cursor_g")

        self.state = "choose"
        self.y = 0.0
        self.target = np.array([A.BLOCK_X, 0.0, A.HOVER_Z])
        self.grip = A.GRIP_OPEN
        self.dwell = 0.0
        self.nearest = None
        self.seq = []
        self.step_t = 0.0
        self.step_from = None
        self.grip_from = A.GRIP_OPEN
        self.carrying = None
        self.slot = 0
        self.idle_t = 0.0
        self.grab_request = False
        self.armed = False
        self.rest_y = 0.0
        self.choose_t = 0.0
        self.stats = {"attempts": 0, "placed": 0, "dropped": 0}
        self.message = ""
        self.on_return = None

    # -- helpers -----------------------------------------------------------
    def log(self, text):
        """A line that survives the rolling status line."""
        print("\n  " + text)

    def grasp_point(self):
        return self.ctrl.grasp_point() if self.ctrl is not None else self.target

    def finger_report(self):
        """How far each finger closed, and whether it is touching the block."""
        try:
            q = [float(self.d.qpos[a]) for a in self.finger_qadr]
            bg = self.block_geoms[self.carrying]
            touch = [0, 0]
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                pair = {c.geom1, c.geom2}
                for k, fg in enumerate(self.finger_geoms):
                    if pair == {fg, bg}:
                        touch[k] += 1
            return q, touch
        except Exception:
            return None, None

    def block_pos(self, i):
        return self.d.xpos[self.blocks[i]].copy()

    def candidates(self):
        out = []
        for i in range(len(self.blocks)):
            p = self.block_pos(i)
            if (p[2] < 0.08 and abs(p[0] - A.BLOCK_X) < 0.12
                    and p[1] < A.DROP_Y - 0.07):
                out.append(i)
        return out

    def reset_blocks(self):
        for qa, da, home in zip(self.block_qadr, self.block_dadr,
                                self.block_home):
            self.d.qpos[qa:qa + 7] = home
            self.d.qvel[da:da + 6] = 0
        self.slot = 0
        mujoco.mj_forward(self.m, self.d)
        self.message = "blocks reset"

    def _highlight(self):
        for k, g in enumerate(self.block_geoms):
            base = self.base_rgba[g]
            if self.state == "choose" and k == self.nearest:
                f = 0.25 + 0.75 * min(1.0, self.dwell / DWELL_S)
                self.m.geom_rgba[g] = base + (np.array([1, 1, 1, 1]) - base) * f * 0.8
            else:
                self.m.geom_rgba[g] = base

    def _cursor(self, cue):
        self.d.mocap_pos[self.mocap] = [A.BLOCK_X, self.y, A.HOVER_Z + 0.2]
        if cue == "left":
            rgba = [1.0, 0.55, 0.2, 0.9]
        elif cue == "right":
            rgba = [0.3, 0.6, 1.0, 0.9]
        else:
            rgba = [1, 1, 1, 0.6]
        self.m.geom_rgba[self.cursor_geom] = rgba

    # -- the sequence the arm runs by itself -------------------------------
    def _start_grab(self, i):
        b = self.block_pos(i)
        sx, sy = A.DROP_SLOTS[self.slot % len(A.DROP_SLOTS)]
        home = [A.BLOCK_X, 0.0, A.HOVER_Z]
        self.carrying = i
        self.stats["attempts"] += 1
        self.seq = [
            ("move", [b[0], b[1], A.HOVER_Z], 0.7),
            ("move", [b[0], b[1], A.GRASP_Z], 1.2),
            ("settle", 0.006, 2.0),          # don't close until actually there
            ("grip", A.GRIP_CLOSED, 0.7),
            ("wait", None, 0.35),            # let the squeeze build before lifting
            ("diag", None, 0.0),
            ("move", [b[0], b[1], A.HOVER_Z], 1.4),
            ("check", None, 0.0),
            ("move", [sx, sy, A.HOVER_Z], 1.8),
            ("move", [sx, sy, A.PLACE_Z], 1.0),
            ("grip", A.GRIP_OPEN, 0.5),
            ("placed", None, 0.0),
            ("move", [sx, sy, A.HOVER_Z], 0.7),
            ("move", home, 1.4),
        ]
        self.state = "auto"
        self.step_t = 0.0
        self.step_from = self.target.copy()
        self.grip_from = self.grip
        self.message = f"grabbing block {i + 1}"
        self.log(f"GRAB block {i + 1}: block at "
                 f"({b[0]:.3f}, {b[1]:+.3f}, {b[2]:.3f})")

    def _run_auto(self, dt):
        if not self.seq:
            self.state = "choose"
            self.y = 0.0
            self.dwell = 0.0
            self.carrying = None
            self.armed = False
            self.rest_y = 0.0
            self.choose_t = 0.0
            if self.on_return:
                self.on_return()
            return
        kind, arg, dur = self.seq[0]
        if kind == "settle":
            self.step_t += dt
            err = np.linalg.norm(self.grasp_point() - self.target)
            if err < arg or self.step_t >= dur:
                gp = self.grasp_point()
                b = self.block_pos(self.carrying)
                ok = "reached" if err < arg else "DID NOT REACH"
                self.log(f"  descend {ok}: gripper at z={gp[2]:.3f} (target "
                         f"{self.target[2]:.3f}, {err * 1000:.0f} mm off), "
                         f"sideways offset from block "
                         f"{np.hypot(*(gp[:2] - b[:2])) * 1000:.0f} mm")
                self.seq.pop(0)
                self._next()
            return
        if kind == "diag":
            q, touch = self.finger_report()
            if q is not None:
                verdict = ("gripping the block" if touch[0] and touch[1] else
                           "only ONE finger touching" if touch[0] or touch[1] else
                           "closed on NOTHING")
                self.log(f"  grip: fingers at {q[0] * 1000:.0f} / "
                         f"{q[1] * 1000:.0f} mm (contact expected ~12, fully "
                         f"closed {A.GRIP_CLOSED * 1000:.0f}); block contacts "
                         f"L={touch[0]} R={touch[1]} -> {verdict}")
            self.seq.pop(0)
            self._next()
            return
        if kind == "check":
            p = self.block_pos(self.carrying)
            self.log(f"  after lift: block at z={p[2]:.3f} "
                     f"({'LIFTED' if p[2] >= 0.10 else 'still on the table'})")
            if p[2] < 0.10:
                self.stats["dropped"] += 1
                self.message = "missed the grasp - returning"
                self.seq = [("grip", A.GRIP_OPEN, 0.4),
                            ("move", [A.BLOCK_X, 0.0, A.HOVER_Z], 1.4)]
            else:
                self.seq.pop(0)
            self._next()
            return
        if kind == "placed":
            self.stats["placed"] += 1
            self.slot += 1
            self.message = f"placed ({self.stats['placed']} so far)"
            self.seq.pop(0)
            self._next()
            return

        self.step_t += dt
        u = smoothstep(self.step_t / dur) if dur > 0 else 1.0
        if kind == "move":
            self.target = self.step_from + (np.asarray(arg) - self.step_from) * u
        elif kind == "wait":
            pass
        elif kind == "grip":
            self.grip = self.grip_from + (arg - self.grip_from) * u
        if self.step_t >= dur:
            self.seq.pop(0)
            self._next()

    def _next(self):
        self.step_t = 0.0
        self.step_from = self.target.copy()
        self.grip_from = self.grip

    # -- per frame ---------------------------------------------------------
    def update(self, evidence, cue, dt):
        if self.state == "auto":
            self._run_auto(dt)
        else:
            cands = self.candidates()
            if not cands:
                self.idle_t += dt
                self.message = "all blocks placed - resetting shortly"
                if self.idle_t > 2.0:
                    self.reset_blocks()
                    self.idle_t = 0.0
            else:
                self.idle_t = 0.0

            v = GAIN_Y * (evidence if abs(evidence) > DEADZONE else 0.0)
            self.nearest = None
            if cands:
                ys = {i: self.block_pos(i)[1] for i in cands}
                self.nearest = min(cands, key=lambda i: abs(ys[i] - self.y))
                off = ys[self.nearest] - self.y
                if self.armed and abs(off) < ASSIST_RADIUS:
                    v += ASSIST_K * off
                if abs(off) < SELECT_TOL:
                    self.dwell += dt
                else:
                    self.dwell = max(0.0, self.dwell - 2 * dt)
            self.y = float(np.clip(self.y + v * dt, Y_MIN, Y_MAX))
            self.target = np.array([A.BLOCK_X, self.y, A.HOVER_Z])

            self.choose_t += dt
            if not self.armed:
                self.dwell = 0.0
                if (abs(self.y - self.rest_y) > ARM_DISTANCE
                        or self.choose_t >= ARM_AFTER_S):
                    self.armed = True
                    self.message = "choose a block"
                elif cands:
                    self.message = (f"ready in {ARM_AFTER_S - self.choose_t:.1f}s"
                                    f" - or start moving")

            if self.nearest is not None and (self.dwell >= DWELL_S
                                             or self.grab_request):
                self._start_grab(self.nearest)
            self.grab_request = False

        self._highlight()
        self._cursor(cue)


# ------------------------------------------------------------------- main
def build_source(args):
    if args.source == "keys":
        return S.KeyboardSource()
    holdout = args.source == "replay"
    path = args.decoder or S.decoder_path(holdout=holdout)
    if holdout and not os.path.exists(path):
        print("No holdout decoder yet - falling back to decoder.pkl.")
        print("For an honest replay run:  train_decoder --holdout 20\n")
        path = S.decoder_path(holdout=False)
    model = S.load_decoder(path)
    print(f"Decoder: {os.path.basename(path)}  (from {model['source']})")
    if args.source == "replay":
        if not model.get("holdout_n"):
            print("[warn] This decoder was trained on every trial, so the "
                  "replay shows it trials it has already seen.\n"
                  "       Expect it to look better than it will live.")
        else:
            print(f"Replaying the {model['holdout_n']} held-out trials the "
                  f"decoder never saw.")
        return S.ReplaySource(model, fif_path=args.fif, speed=args.speed)
    _warn_if_stale(model)
    return S.LiveSource(model)


def _warn_if_stale(model):
    """Live mode on a decoder from an older session works, but worse. Say so."""
    import glob
    import config
    here = os.path.dirname(os.path.abspath(__file__))
    recs = sorted(glob.glob(os.path.join(here, config.DATA_DIR, "*_raw.fif")))
    recs = [r for r in recs if "_partial" not in r] or recs
    if recs and os.path.basename(recs[-1]) != model.get("source"):
        print("\n" + "!" * 64)
        print(f"  This decoder was trained on {model.get('source')}")
        print(f"  but your newest recording is  {os.path.basename(recs[-1])}")
        print("  Decoders transfer poorly between sessions. For today's")
        print("  calibration, close this and run train_decoder.bat first.")
        print("!" * 64 + "\n")


def bar(ev, width=21):
    ev = float(np.clip(ev, -1, 1))
    mid = width // 2
    pos = int(round(mid + ev * mid))
    cells = ["-"] * width
    cells[mid] = "|"
    cells[pos] = "#"
    return "".join(cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", choices=["replay", "live", "keys"],
                    default="replay")
    ap.add_argument("--decoder", help="path to a decoder .pkl")
    ap.add_argument("--fif", help="recording to replay")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed multiplier")
    args = ap.parse_args()

    source = build_source(args)
    m = A.load_model()
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)

    ctrl = ArmController(m, d)
    task = Task(m, d, ctrl)
    task.on_return = source.reset_evidence

    def on_key(keycode):
        if isinstance(source, S.KeyboardSource):
            source.key(keycode)
        if keycode == 265:        # up arrow
            task.grab_request = True

    n_sub = max(1, int(round((1.0 / FPS) / m.opt.timestep)))
    frame_dt = n_sub * m.opt.timestep

    print(f"\nSource: {source.name}.  Close the window to finish.")
    print("Arrow up grabs the nearest block in any mode.\n")

    text_ok = True
    last_print = 0.0
    try:
        viewer_ctx = mujoco.viewer.launch_passive(m, d, key_callback=on_key)
    except RuntimeError as exc:
        if sys.platform == "darwin":
            print("\nOn macOS the MuJoCo viewer must be started with mjpython:")
            print("    .venv/bin/mjpython sim_arm.py --source replay\n")
            return 1
        raise

    with viewer_ctx as viewer:
        viewer.cam.lookat[:] = [0.30, 0.03, 0.10]
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -28

        while viewer.is_running():
            t_frame = time.perf_counter()

            evidence, cue = source.poll()
            task.update(evidence, cue, frame_dt)
            ctrl.update(task.target, task.grip, frame_dt)
            for _ in range(n_sub):
                mujoco.mj_step(m, d)

            line_l = (f"{source.name}\nstate\nevidence\ncue\n"
                      f"placed / tries")
            line_r = (f"\n{task.state}\n{evidence:+.2f}\n{cue or '-'}\n"
                      f"{task.stats['placed']} / {task.stats['attempts']}")
            if text_ok:
                try:
                    viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,
                                       mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                       line_l, line_r)])
                except Exception:
                    text_ok = False
            viewer.sync()

            now = time.perf_counter()
            if now - last_print > 0.25:
                extra = ""
                if isinstance(source, S.ReplaySource):
                    acc, n = source.agreement()
                    extra = (f" t={source.stream_time():6.1f}s"
                             + (f" match={acc * 100:3.0f}%" if acc is not None
                                else ""))
                msg = task.message[:34]
                sys.stdout.write(f"\r[{bar(evidence)}] {evidence:+.2f} "
                                 f"cue={cue or '-':5s} {task.state:6s} "
                                 f"dwell={task.dwell:3.1f}s{extra}  {msg:34s}")
                sys.stdout.flush()
                last_print = now
                if getattr(source, "finished", False) and task.state == "choose":
                    task.message = "replay finished - close the window"

            left = frame_dt - (time.perf_counter() - t_frame)
            if left > 0:
                time.sleep(left)

    print("\n\n" + "=" * 56)
    print(f"  Blocks placed: {task.stats['placed']}   "
          f"attempts: {task.stats['attempts']}   "
          f"missed grasps: {task.stats['dropped']}")
    for line in source.summary():
        print("  " + line)
    print("=" * 56)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(1)
