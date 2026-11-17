"""Arm controllers. Teleop and eval pick one by name (--controller), and the demos must be
evaluated with the controller they were recorded with (the name is stored in the dataset).

Every controller has
  name          registry key, also the dataset suffix (panda_pick_place_<name>)
  torque_model  True: build the sim with pick_place.build_sim_model(torque=True) (arm ctrl = torque)
  target        what step() takes: "joint" = 7 joint targets [rad]
  step(data, target, n)  n physics steps toward the target
and, only if it computes the arm torques itself (torque control):
  torque(data, target)   joint torques [Nm] for this state (recorded as torque.command)
"""

from .joint_impedance import JointImpedance
from .joint_position import JointPosition

CONTROLLERS = {c.name: c for c in (JointImpedance, JointPosition)}
DEFAULT = JointImpedance.name


def get(name):
    try:
        return CONTROLLERS[name]
    except KeyError:
        raise SystemExit(f"Unknown controller {name!r}, choose one of {sorted(CONTROLLERS)}") from None
