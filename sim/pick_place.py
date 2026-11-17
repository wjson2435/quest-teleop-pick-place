"""Pick-and-place simulation setup: model building, scene reset and the success check.

Used by teleop (teleop/quest_teleop.py) and policy evaluation (eval.py), so both run the same scene.
  build_sim_model(torque)   scene.xml + TCP site, wrist camera, gravity compensation, mocap target;
                            torque=True turns the arm actuators into motors (see controllers/)
  build_spec(with_cube)     the robot + scene spec (with_cube=False: base of the IK model, teleop/ik.py)
  reset_scene               robot to the "start" key, cube to a random position
  cube_in_bin               success check
"""

from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).parent
_XML = _HERE / "scene.xml"

# ---------------- Parameters ----------------
CONTROL_DT = 0.01  # control period [s] (100 Hz)
# Must be an integer multiple of the sim timestep 0.002 -> 5 mj_steps per control step

TCP_OFFSET = 0.1034  # distance from the hand body origin to the center between the fingertips [m]

ROBOT_BODIES = [f"link{i}" for i in range(8)] + ["hand", "left_finger", "right_finger"]

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

CUBE_HALF = 0.02  # cube half side length [m]

# Place bin (see place_zone in scene.xml)
BIN_FLOOR_TOP = 0.005  # bin floor top surface height above the table [m]
BIN_INNER_HALF = 0.06  # half of the inner width [m]

# Random cube start area for teleop / policy evaluation 
CUBE_X_RANGE = (0.4, 0.6)
CUBE_Y_RANGE = (0.05, 0.3)

# Wrist camera, mounted on the hand body (hand frame: z = toward fingertips, y = finger motion axis).
# With joint7 = +0.785 in the "start" key, hand x = world +x (forward) and hand y = world -y,
# so the fingers close left/right. The camera sits in front of the hand (+x) at flange height,
# looking down and slightly back at the fingers: fingers at the bottom of the image,
# image up = robot forward, image right = robot right.
WRIST_CAM_POS = [0.09, 0.0, 0.0]  # [m], in hand frame
WRIST_CAM_TILT = 15.0  # [deg], view axis tilted from straight down (hand z) back toward the fingers (-x)
WRIST_CAM_FOVY = 70.0  # [deg]


def build_spec(with_cube: bool) -> mujoco.MjSpec:
    """Load scene.xml and add a TCP site to the hand (returns the spec before compiling).

    The site is added in code (MjSpec) so the menagerie panda.xml stays unmodified.
    The IK model has the cube removed so its freejoint is not part of the configuration.
    """
    spec = mujoco.MjSpec.from_file(_XML.as_posix())
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0, 0],
                               rgba=[1, 0.8, 0, 1], group=1)

    # Wrist camera. MuJoCo cameras look along their -z axis with +y as image-up (hand frame):
    #   view direction = (-sin t, 0, cos t),  image up = (cos t, 0, sin t),  image right = hand y
    t = np.deg2rad(WRIST_CAM_TILT)
    cam = spec.body("hand").add_camera(name="wrist", pos=WRIST_CAM_POS, fovy=WRIST_CAM_FOVY)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [0, 1, 0, np.cos(t), 0, np.sin(t)]

    # Gravity compensation (the robot compensates its own weight, like the real Franka).
    #   body.gravcomp = 1          : applies a force that cancels gravity on that body
    #   joint.actgravcomp = True   : routes that force through the actuators -> counted in forcerange (torque limits)
    # Only robot bodies, not the cube.
    for name in ROBOT_BODIES:
        spec.body(name).gravcomp = 1.0
    for i in range(1, 8):
        spec.joint(f"joint{i}").actgravcomp = True  # actuatorgravcomp="true" in XML
    if not with_cube:
        spec.delete(spec.body("cube"))
        spec.delete(spec.key("start"))  # this key contains the cube qpos, so remove it too
    return spec


def build_sim_model(torque: bool = False) -> mujoco.MjModel:
    """Simulation model + mocap 'target' (teleop IK goal marker, in geom group 1 so cameras can hide it).

    torque=False: menagerie position servos, data.ctrl[:7] = joint targets [rad].
    torque=True : arm actuators become motors, data.ctrl[:7] = joint torques [Nm] (see controllers/joint_impedance.py),
                  limited to the same forcerange (87 / 12 Nm). Gravity is still compensated by the sim
                  (gravcomp, see build_spec), like the real Franka where the commanded torque excludes gravity.
    The gripper (actuator8) stays a position actuator in both cases.
    """
    spec = build_spec(with_cube=True)
    if torque:
        for i in range(1, 8):
            act = spec.actuator(f"actuator{i}")
            act.gainprm[:3] = [1, 0, 0]
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            act.biasprm[:3] = [0, 0, 0]
            act.ctrlrange = act.forcerange  # ctrl = torque [Nm]
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    # TeleopMocap grabs mocap index 0 every time it is toggled -> make target the only mocap
    # place_zone becomes a regular fixed body (visual only, so behavior is unchanged)
    spec.body("place_zone").mocap = False
    target = spec.worldbody.add_body(name="target", mocap=True)
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.012, 0, 0],
                    rgba=[1, 0.8, 0, 0.5], contype=0, conaffinity=0,
                    group=1)  # unused by the panda (visual = 2, collision = 3)
    return spec.compile()


def reset_scene(model, data, rng):
    """Robot to the 'start' key, cube to a random position in CUBE_X/Y_RANGE (build_sim_model model)."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
    data.joint("cube_joint").qpos[:2] = [rng.uniform(*CUBE_X_RANGE), rng.uniform(*CUBE_Y_RANGE)]
    data.ctrl[:] = model.key("start").ctrl  # torque model: the arm ctrl is overwritten by the controller
    mujoco.mj_forward(model, data)


def cube_in_bin(data):
    """Success check: cube resting on the bin floor."""
    d = data.body("cube").xpos - data.body("place_zone").xpos
    in_xy = np.all(np.abs(d[:2]) < BIN_INNER_HALF - CUBE_HALF + 0.002)
    on_floor = abs(data.body("cube").xpos[2] - (BIN_FLOOR_TOP + CUBE_HALF)) < 0.005
    return bool(in_xy and on_floor)
