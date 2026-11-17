"""Joint impedance controller: torque control toward joint targets (needs the torque model,
pick_place.build_sim_model(torque=True)).

    tau = Kp * (q_d - q) - Kd * qd          (7 arm joints, [Nm])

Gravity is not in the formula: the sim compensates it (gravcomp), like the real Franka, where the
torque sent through FCI excludes gravity. Gains are the ones of libfranka's joint_impedance_control
example.

The controller runs at the physics rate (500 Hz); q_d comes from the IK (teleop) or the policy and
is held between updates.
"""

import mujoco
import numpy as np

KP = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])  # [Nm/rad]
KD = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])         # [Nm s/rad]


class JointImpedance:
    name = "joint_impedance"
    torque_model = True  # arm actuators must be motors (ctrl = torque)
    target = "joint"     # step() takes 7 joint targets [rad]

    def __init__(self, model, kp=KP, kd=KD):
        self.model = model
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.tau_max = model.actuator_ctrlrange[:7, 1].copy()

    def torque(self, data, q_d):
        """Joint torques [Nm] sent for this state (without gravity)."""
        tau = self.kp * (q_d - data.qpos[:7]) - self.kd * data.qvel[:7]
        return np.clip(tau, -self.tau_max, self.tau_max)

    def step(self, data, q_d, n):
        """n physics steps toward the joint target q_d (torque recomputed every step)."""
        for _ in range(n):
            data.ctrl[:7] = self.torque(data, q_d)
            mujoco.mj_step(self.model, data)
