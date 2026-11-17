"""Teleop IK: TCP target pose -> 7 joint targets, with mink (QP-based differential IK).

Same setup as mink's examples/arm_panda.py:
  tasks   eef FrameTask on the "tcp" site (position + orientation) + PostureTask (cost 1e-2) that keeps
          the redundant elbow near the pose at the last reset
  solve   solve_ik (daqp) + integrate, repeated until the eef error is below the thresholds
          (at most MAX_ITERS per call), so each call behaves like an iterative IK

The IK has its own model with the 7 arm joints only: no cube (its freejoint) and no finger joints
(they do not move the TCP and the gripper is commanded separately). Its configuration is not synced
to the sim between calls: it tracks the command, the arm
controller then makes the sim follow it.
"""

import sys
from pathlib import Path

import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import pick_place as pp  # noqa: E402

SOLVER = "daqp"
POS_THRESHOLD = 1e-4  # [m]
ORI_THRESHOLD = 1e-4  # [rad]
MAX_ITERS = 20
DAMPING = 1e-3


def build_ik_model():
    """Sim robot (same TCP site and kinematics) reduced to the 7 arm joints."""
    spec = pp.build_spec(with_cube=False)
    spec.delete(spec.actuator("actuator8"))  # gripper actuator, drives the "split" tendon
    spec.delete(spec.tendon("split"))
    for eq in list(spec.equalities):         # finger_joint2 = finger_joint1
        spec.delete(eq)
    for key in list(spec.keys):              # keyframes have 9 qpos
        spec.delete(key)
    for name in ("finger_joint1", "finger_joint2"):
        spec.delete(spec.joint(name))
    model = spec.compile()
    assert model.nq == 7, model.nq
    return model


class TeleopIK:
    def __init__(self, dt=pp.CONTROL_DT):
        self.dt = dt
        self.model = build_ik_model()
        self.configuration = mink.Configuration(self.model)  # IK state: 7 joint angles + kinematics (Jacobian) # IK
        self.eef_task = mink.FrameTask(frame_name="tcp", frame_type="site",
                                       position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        self.posture_task = mink.PostureTask(model=self.model, cost=1e-2)  # secondary: redundant DoF (elbow) -> reset pose
        self.tasks = [self.eef_task, self.posture_task]

    def reset(self, qpos):
        """Start from the sim joint positions; returns the TCP pose there (position, 3x3 rotation)."""
        self.configuration.update(qpos[:7])
        self.posture_task.set_target_from_configuration(self.configuration)
        T = self.configuration.get_transform_frame_to_world("tcp", "site")
        return T.translation().copy(), T.rotation().as_matrix().copy()

    def solve(self, pos, rot):
        """TCP target (position, mink.SO3) -> 7 joint targets [rad]."""
        self.eef_task.set_target(mink.SE3.from_rotation_and_translation(rot, pos))
        for _ in range(MAX_ITERS):
            vel = mink.solve_ik(self.configuration, self.tasks, self.dt, SOLVER, damping=DAMPING)  # QP: joint velocities (7,) for eef + posture, within joint limits
            self.configuration.integrate_inplace(vel, self.dt)  # q += vel * dt, update kinematics
            err = self.eef_task.compute_error(self.configuration)  # TCP error (6,): position [:3], rotation [3:]
            if np.linalg.norm(err[:3]) <= POS_THRESHOLD and np.linalg.norm(err[3:]) <= ORI_THRESHOLD:
                break
        return self.configuration.q.copy()
