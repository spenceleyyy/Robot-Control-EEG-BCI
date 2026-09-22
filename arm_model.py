"""
The simulation scene.

Written as MJCF in a string so everything runs after `pip install mujoco`,
with nothing else to download.

Layout: the floor plane IS the table top. The arm is bolted at the origin,
three blocks sit in a row in front of it, and a drop zone sits to the right.
The camera looks back at the arm from in front, so world +y is screen-right:
imagining your RIGHT hand moves the arm to the RIGHT on screen.
"""

import numpy as np

# ------------------------------------------------------------------- layout
BLOCK_HALF = 0.03                  # 6 cm cubes
BLOCK_X = 0.40
BLOCK_Y = (-0.18, 0.0, 0.18)
DROP_Y = 0.30
DROP_SLOTS = [(BLOCK_X - 0.075, DROP_Y), (BLOCK_X, DROP_Y),
              (BLOCK_X + 0.075, DROP_Y)]

HOVER_Z = 0.28                     # grasp-point height while cruising
GRASP_Z = 0.042                    # grasp point at pickup: fingertips ~1cm off the table
PLACE_Z = 0.046

# ---------------------------------------------------------------- kinematics
# Keep these in sync with the MJCF below - home_arm_q() depends on them.
SHOULDER_Z = 0.13 + 0.09
L_UPPER = 0.32
L_FORE = 0.29
WRIST_TO_HAND = 0.075 + 0.05
HAND_TO_SITE = 0.083               # the grasp point, midway along the fingers

ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
FINGER_ACTUATORS = ["a_fl", "a_fr"]
GRIP_OPEN = 0.0
GRIP_CLOSED = 0.030                # contact happens at ~0.012; the rest is squeeze

# Hand pointing straight down. Columns are the hand's x, y, z axes in world
# coordinates: palm-out (+x) aims at the floor, fingers close along world y.
R_DOWN = np.array([[0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0]])


def home_arm_q(x=BLOCK_X, z=HOVER_Z):
    """Analytic elbow-up solution putting the grasp point at (x, 0, z) with
    the hand pointing down. Pitch joints rotate about +y, so a positive angle
    tilts a link downward."""
    wrist_z = z + HAND_TO_SITE + WRIST_TO_HAND
    dx, dz = x, wrist_z - SHOULDER_Z
    d2 = dx * dx + dz * dz
    c = np.clip((d2 - L_UPPER ** 2 - L_FORE ** 2) / (2 * L_UPPER * L_FORE),
                -1.0, 1.0)
    phi = float(np.arccos(c))
    alpha = float(np.arctan2(dz, dx))
    beta = float(np.arctan2(L_FORE * np.sin(phi), L_UPPER + L_FORE * np.cos(phi)))
    th1 = -(alpha + beta)
    th2 = th1 + phi
    j4 = np.pi / 2 - th2
    return [0.0, th1, phi, j4, 0.0, 0.0]


def _block(name, y, rgba):
    h = BLOCK_HALF
    return f"""
    <body name="{name}" pos="{BLOCK_X} {y} {h}">
      <freejoint/>
      <geom name="{name}_g" type="box" size="{h} {h} {h}" rgba="{rgba}"
            mass="0.07" friction="1.6 0.05 0.002" condim="4" solref="0.006 1"/>
    </body>"""


def scene_xml():
    q = home_arm_q()
    arm_q = " ".join(f"{v:.5f}" for v in q)
    block_q = " ".join(f"{BLOCK_X} {y} {BLOCK_HALF} 1 0 0 0" for y in BLOCK_Y)
    blocks = (_block("block1", BLOCK_Y[0], ".88 .35 .33 1")
              + _block("block2", BLOCK_Y[1], ".35 .72 .45 1")
              + _block("block3", BLOCK_Y[2], ".38 .55 .92 1"))

    return f"""
<mujoco model="bci_arm">
  <!-- MJCF reads joint ranges in DEGREES by default. Everything here is in
       radians - without this line every joint is limited to about 2 degrees. -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"
          impratio="10"/>
  <visual>
    <quality shadowsize="4096"/>
    <headlight diffuse=".55 .55 .55" ambient=".35 .35 .35" specular="0 0 0"/>
  </visual>

  <default>
    <joint damping="12" armature="0.08"/>
    <position kp="900" forcerange="-200 200"/>
    <default class="link">
      <geom rgba=".62 .64 .70 1"/>
    </default>
    <default class="finger">
      <geom rgba=".25 .27 .32 1" friction="2.5 0.1 0.003" condim="4"
            solref="0.006 1"/>
      <position kp="400" forcerange="-60 60"/>
    </default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1=".22 .24 .28" rgb2=".26 .28 .33"/>
    <material name="tabletop" texture="grid" texrepeat="8 8" reflectance="0.08"/>
  </asset>

  <worldbody>
    <light pos="0.5 0 2.2" dir="0 0 -1" diffuse=".6 .6 .6" castshadow="true"/>
    <light pos="1.4 1.0 1.6" dir="-.6 -.5 -1" diffuse=".25 .25 .25"
           castshadow="false"/>
    <geom name="table" type="plane" size="2 2 .05" material="tabletop"/>

    <geom name="dropzone" type="box" size="0.13 0.06 0.001"
          pos="{BLOCK_X} {DROP_Y} 0.001" rgba=".95 .80 .30 .35"
          contype="0" conaffinity="0"/>

    <body name="cursor" mocap="true" pos="{BLOCK_X} 0 {HOVER_Z + 0.2}">
      <geom name="cursor_g" type="sphere" size="0.022" rgba="1 1 1 .6"
            contype="0" conaffinity="0"/>
    </body>

    <body name="base" pos="0 0 0" childclass="link">
      <geom type="cylinder" size="0.075 0.05" pos="0 0 0.05" rgba=".28 .30 .36 1"/>
      <body name="shoulder" pos="0 0 0.13">
        <joint name="j1" axis="0 0 1" range="-3.0 3.0"/>
        <geom type="capsule" fromto="0 0 -0.02 0 0 0.09" size="0.055"/>
        <body name="upperarm" pos="0 0 0.09">
          <joint name="j2" axis="0 1 0" range="-2.2 2.2"/>
          <geom type="capsule" fromto="0 0 0 {L_UPPER} 0 0" size="0.046"/>
          <body name="forearm" pos="{L_UPPER} 0 0">
            <joint name="j3" axis="0 1 0" range="-2.7 2.7"/>
            <geom type="capsule" fromto="0 0 0 {L_FORE} 0 0" size="0.039"/>
            <body name="wrist1" pos="{L_FORE} 0 0">
              <joint name="j4" axis="0 1 0" range="-2.7 2.7"/>
              <geom type="capsule" fromto="0 0 0 0.075 0 0" size="0.033"/>
              <body name="wrist2" pos="0.075 0 0">
                <joint name="j5" axis="1 0 0" range="-3.0 3.0"/>
                <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.030"/>
                <body name="hand" pos="0.05 0 0">
                  <joint name="j6" axis="0 1 0" range="-2.2 2.2"/>
                  <geom name="palm" type="box" size="0.028 0.058 0.022"
                        pos="0.028 0 0" rgba=".40 .43 .50 1"/>
                  <site name="ee_site" pos="{HAND_TO_SITE} 0 0" size="0.006"
                        rgba="1 .4 .2 0"/>
                  <body name="finger_l" pos="0.055 0.05 0" childclass="finger">
                    <joint name="fl" type="slide" axis="0 -1 0"
                           range="0 0.034" damping="4"/>
                    <geom name="finger_l_g" type="box" size="0.028 0.008 0.016"
                          pos="0.028 0 0"/>
                  </body>
                  <body name="finger_r" pos="0.055 -0.05 0" childclass="finger">
                    <joint name="fr" type="slide" axis="0 1 0"
                           range="0 0.034" damping="4"/>
                    <geom name="finger_r_g" type="box" size="0.028 0.008 0.016"
                          pos="0.028 0 0"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
{blocks}
  </worldbody>

  <actuator>
    <position name="a_j1" joint="j1" ctrlrange="-3.0 3.0"/>
    <position name="a_j2" joint="j2" ctrlrange="-2.2 2.2"/>
    <position name="a_j3" joint="j3" ctrlrange="-2.7 2.7"/>
    <position name="a_j4" joint="j4" ctrlrange="-2.7 2.7"/>
    <position name="a_j5" joint="j5" ctrlrange="-3.0 3.0"/>
    <position name="a_j6" joint="j6" ctrlrange="-2.2 2.2"/>
    <position name="a_fl" class="finger" joint="fl" ctrlrange="0 0.034"/>
    <position name="a_fr" class="finger" joint="fr" ctrlrange="0 0.034"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="{arm_q} 0 0 {block_q}"
         ctrl="{arm_q} 0 0"/>
  </keyframe>
</mujoco>
"""


def load_model():
    import mujoco
    m = mujoco.MjModel.from_xml_string(scene_xml())
    # Guard against unit mix-ups: joint j2 is declared as +/-2.2 rad.
    j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j2")
    if abs(m.jnt_range[j2][1] - 2.2) > 1e-6:
        raise RuntimeError(
            f"Joint limits loaded wrongly (j2 upper limit {m.jnt_range[j2][1]:.4f}, "
            f"expected 2.2 rad). The model's angle units are off.")
    return m
