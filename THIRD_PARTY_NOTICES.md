# Third-party notices

This project includes or builds on the following third-party work.

## Included as a git submodule

### oculus_reader (`teleop/oculus_reader/`)
- Source: https://github.com/rail-berkeley/oculus_reader
- Copyright (c) the oculus_reader authors (RAIL, UC Berkeley)
- License: Apache License 2.0, full text in `teleop/oculus_reader/LICENSE`
- Included unmodified as a git submodule pointing to the upstream repository
  (Python package, prebuilt APK and app source).
- The Quest app source (`teleop/oculus_reader/app_source/`) contains files based on the
  Oculus Mobile SDK samples, Copyright (c) Facebook Technologies, LLC and its affiliates,
  distributed under their own terms (see the headers of those files).

## Used but not included

### MuJoCo Menagerie: Franka Emika Panda model
- Source: https://github.com/google-deepmind/mujoco_menagerie (`franka_emika_panda/`)
- License: Apache License 2.0
- `sim/scene.xml` includes `panda.xml` and its meshes from a separate menagerie checkout
  (`../../mujoco_menagerie`); the model is modified at load time in `sim/pick_place.py`
  (TCP site, wrist camera, gravity compensation, torque actuators), the files are not.

### mink
- Source: https://github.com/kevinzakka/mink
- License: Apache License 2.0
- `teleop/ik.py` follows the setup of mink's `examples/arm_panda.py`
  (FrameTask + PostureTask, iterative `solve_ik` until convergence).

### libfranka
- Source: https://github.com/frankaemika/libfranka
- License: Apache License 2.0
- The joint impedance gains in `controllers/joint_impedance.py` are taken from its
  `joint_impedance_control` example.

### Python dependencies
Installed separately, under their own licenses: MuJoCo (Apache 2.0), LeRobot (Apache 2.0),
PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), imageio (BSD-2-Clause).
