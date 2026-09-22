"""
Arm self-test. No window, no EEG. Runs the reach-and-grab motion three ways
and reports where it goes wrong:

  1. normal physics
  2. with collisions switched off
  3. with gravity switched off

If (1) fails and (2) works, something is colliding. If (1) fails and (3)
works, the joints are too weak for the arm's weight. If all three fail,
the model and the controller disagree about the arm's geometry.

    selftest.bat
"""

import sys

import numpy as np
import mujoco

import arm_model as A
from sim_arm import ArmController

FRAME = 1 / 60


def body_of_geom(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "world"


def arm_contacts(m, d):
    arm_bodies = {"shoulder", "upperarm", "forearm", "wrist1", "wrist2", "hand",
                  "finger_l", "finger_r"}
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = body_of_geom(m, c.geom1), body_of_geom(m, c.geom2)
        if a in arm_bodies or b in arm_bodies:
            out.add(tuple(sorted((a, b))))
    return out


def run(label, contacts=True, gravity=True, verbose=True):
    m = A.load_model()
    if not contacts:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if not gravity:
        m.opt.gravity[:] = 0
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    ctrl = ArmController(m, d)
    n_sub = max(1, int(round(FRAME / m.opt.timestep)))

    if verbose:
        print(f"\n=== {label} ===")
    by = A.BLOCK_Y[0]
    legs = [("hold home", [A.BLOCK_X, 0.0, A.HOVER_Z], 2.0),
            ("hover block 1", [A.BLOCK_X, by, A.HOVER_Z], 3.0),
            ("down to block", [A.BLOCK_X, by, A.GRASP_Z], 3.0),
            ("back up", [A.BLOCK_X, by, A.HOVER_Z], 3.0),
            ("to drop zone", [A.DROP_SLOTS[0][0], A.DROP_SLOTS[0][1], A.HOVER_Z], 3.0)]
    worst = 0.0
    contacts_seen = set()
    for name, target, secs in legs:
        for _ in range(int(secs / FRAME)):
            ctrl.update(target, A.GRIP_OPEN, n_sub * m.opt.timestep)
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            contacts_seen |= arm_contacts(m, d)
        gp = ctrl.grasp_point()
        err = np.linalg.norm(gp - np.asarray(target)) * 1000
        worst = max(worst, err)
        if verbose:
            q = d.qpos[ctrl.qadr]
            lead = np.abs(ctrl.q_cmd - q).max()
            flag = "ok  " if err < 10 else "MISS"
            print(f"  {flag} {name:14s} want {np.round(target, 3)} got {np.round(gp, 3)}"
                  f"  off {err:6.1f} mm")
            print(f"       joints {np.round(q, 2)}  command-vs-actual gap {lead:.2f} rad")
    if verbose and contacts_seen:
        print("  arm contacts seen: " + ", ".join(f"{a}-{b}" for a, b in sorted(contacts_seen)))
    return worst


def main():
    print(f"MuJoCo {mujoco.__version__}")

    m = A.load_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    gp = d.site_xpos[sid]
    R = d.site_xmat[sid].reshape(3, 3)
    geo_err = np.linalg.norm(gp - [A.BLOCK_X, 0, A.HOVER_Z]) * 1000
    ori_err = np.abs(R - A.R_DOWN).max()
    print(f"\nGeometry check at the home pose:")
    print(f"  gripper at {np.round(gp, 4)}  expected {[A.BLOCK_X, 0.0, A.HOVER_Z]}"
          f"  ({geo_err:.1f} mm off)")
    print(f"  hand orientation error {ori_err:.3f}  (0 = pointing straight down)")
    print(f"  joints {np.round(d.qpos[:6], 3)}")
    if geo_err > 5 or ori_err > 0.05:
        print("  !! The model does not match the controller's geometry.")

    w_normal = run("1. normal physics")
    w_nocoll = run("2. collisions OFF")
    w_nograv = run("3. gravity OFF")

    print("\n" + "=" * 60)
    print(f"  worst miss:  normal {w_normal:6.1f} mm   no-collisions {w_nocoll:6.1f} mm"
          f"   no-gravity {w_nograv:6.1f} mm")
    if w_normal < 10:
        print("  PASS - the arm reaches everything. If it misbehaves in robot.bat,")
        print("  the problem is in the viewer or key handling, not the physics.")
    elif w_nocoll < 10:
        print("  CAUSE: COLLISIONS. Something on the arm is hitting something.")
        print("  The contact list under test 1 names the parts.")
    elif w_nograv < 10:
        print("  CAUSE: GRAVITY. The joint motors are too weak for the arm.")
    else:
        print("  CAUSE: neither collisions nor gravity. The joint data above")
        print("  will show which joint goes wrong.")
    print("=" * 60)
    print("\nCopy everything above and send it back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
