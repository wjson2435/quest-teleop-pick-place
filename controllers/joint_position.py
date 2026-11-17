"""Joint position control with the menagerie position servos (pick_place.build_sim_model(torque=False)).

data.ctrl[:7] = q_d; the torque is computed inside MuJoCo by the position actuators,
tau = kp * (q_d - q) - kv * qd (menagerie gains, clipped to forcerange). This is how the first
datasets were recorded. No torque() method: this controller does not compute torques itself,
so its datasets have no torque.command.
"""

import mujoco


class JointPosition:
    name = "joint_position"
    torque_model = False  # menagerie position actuators (ctrl = joint target)
    target = "joint"      # step() takes 7 joint targets [rad]

    def __init__(self, model):
        self.model = model

    def step(self, data, q_d, n):
        """n physics steps toward the joint target q_d."""
        data.ctrl[:7] = q_d
        for _ in range(n):
            mujoco.mj_step(self.model, data)
