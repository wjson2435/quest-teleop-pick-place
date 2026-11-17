# Quest Teleop Pick-and-Place

Teleoperate a simulated Franka Emika Panda with a Meta Quest 2, record demonstrations as a
[LeRobot](https://github.com/huggingface/lerobot) dataset, train a Diffusion Policy on them and
evaluate it in the same [MuJoCo](https://mujoco.org) scene.

Task: pick up a red cube placed at a random position on the table and drop it into a green bin.

```
Quest 2 controller pose ─► clutch + filter ─► TCP target ─► IK (mink) ─► joint targets ─► arm controller ─► MuJoCo
                                                                              │
                                    wrist + front camera images, robot state ─┴─► LeRobot dataset (20 Hz)
                                                                                        │
                                                             lerobot-train (Diffusion Policy)
                                                                                        │
                       eval.py: policy ─► joint targets ─► same arm controller ─► MuJoCo ─► success rate
```

## Repository layout

```
controllers/            arm controllers, selected by name with --controller
  joint_position.py       MuJoCo position servos (ctrl = joint targets)
  joint_impedance.py      torque control: tau = Kp (q_d - q) - Kd qd (libfranka gains)
sim/
  scene.xml               table, cube, bin, cameras; includes the Menagerie Panda
  pick_place.py           model building, scene reset, success check
  cameras.py              camera rendering shared by recording and evaluation
teleop/
  quest_teleop.py         Quest teleoperation + recording
  run_quest_teleop.sh     launcher (adb checks, default options)
  ik.py                   differential IK with mink (TCP target -> 7 joint targets)
  lerobot_recorder.py     LeRobot dataset writer
  oculus_reader/          Quest pose/button reader (git submodule, see THIRD_PARTY_NOTICES.md)
train.sh                Diffusion Policy training with lerobot-train
eval.py                 policy evaluation in simulation
```

Recorded datasets go to `datasets/` and training runs to `outputs/` (both git-ignored).

## Workflow

The whole pipeline in three commands (details in [Usage](#usage)):

```bash
./teleop/run_quest_teleop.sh --controller joint_impedance --dataset panda_pick_place  # 1. record demos
DATASET=panda_pick_place_joint_impedance ./train.sh dp_v1                               # 2. train
python eval.py outputs/dp_v1                                                            # 3. evaluate
```

The arm controller is chosen once, when recording, and the later steps follow it:

| Step | Controller | What `--controller` / the dataset decides |
|---|---|---|
| 1. Record | `--controller <name>` (default `joint_impedance`) | the arm controller ([controllers/](controllers/)) and the matching sim model (position servos or torque motors); the dataset folder `datasets/<dataset>_<name>`; the name stored in `meta/controller.json`; whether `torque.command` is recorded (torque controllers only) |
| 2. Train | none, pick the dataset with `DATASET=` | the policy only learns from images, state and action, no simulation runs |
| 3. Evaluate | read automatically | `eval.py` follows the checkpoint to its training dataset and runs the policy with the controller stored there (`--controller` overrides it, with a warning) |

Running step 1 again with the same `--dataset` and `--controller` appends to the same dataset.
Recording into an existing dataset with a different controller is refused, so one dataset always
holds demos of a single controller.

## Installation

Tested on Ubuntu with Python 3.11, MuJoCo 3.17, mink 1.4, LeRobot 0.4.4 and an NVIDIA GPU.

**1. Clone this repository next to MuJoCo Menagerie.** `sim/scene.xml` loads the Panda model from
`../../mujoco_menagerie`, so both must be in the same parent folder. `teleop/oculus_reader` is a
git submodule whose Quest app (APK) is stored with Git LFS, so install git-lfs first
(`sudo apt install git-lfs && git lfs install`) and clone with `--recursive`:

```bash
mkdir robot_ws && cd robot_ws
git clone https://github.com/google-deepmind/mujoco_menagerie.git
git clone --recursive https://github.com/wjson2435/quest-teleop-pick-place.git pick_place
```

```
robot_ws/
├── mujoco_menagerie/
└── pick_place/
```

**2. Create the Python environment.**

```bash
conda create -n mujo python=3.11
conda activate mujo
conda install -c conda-forge ffmpeg=7      # video encoding for LeRobot datasets
pip install mujoco mink daqp lerobot imageio scipy
pip install -e pick_place/teleop/oculus_reader --no-deps
pip install pure-python-adb pyyaml
```

The scripts call `~/anaconda3/envs/mujo/bin/python` and `~/anaconda3/envs/mujo/bin/lerobot-train`;
change `PYTHON` in `teleop/run_quest_teleop.sh` and `LEROBOT_TRAIN` in `train.sh` if your
environment is elsewhere.

**3. Set up the Quest 2.**

- Install adb: `sudo apt install android-tools-adb`
- Enable developer mode on the headset (see the
  [oculus_reader README](teleop/oculus_reader/README.md) for the steps).
- In the headset, turn **off** *Developer settings → Link auto connection*. Otherwise the USB
  connection starts Quest Link and the USB debugging prompt never appears.
- Connect the headset by USB and accept *Allow USB debugging*. `adb devices` should list it as
  `device`.

The reader app (APK) is installed on the headset automatically the first time teleop runs.

## Usage

### 1. Teleoperation and recording

```bash
cd teleop
./run_quest_teleop.sh                               # joint impedance control, recording on
./run_quest_teleop.sh --controller joint_position   # another arm controller
./run_quest_teleop.sh --dataset test                # another dataset name
./run_quest_teleop.sh --no-record                   # teleop only
```

Demos are saved to `datasets/<dataset>_<controller>` (default
`datasets/panda_pick_place_joint_impedance`). Running again appends to the same dataset.

| Button | Action |
|---|---|
| Right grip (hold) | move the robot (clutch: release to reposition your hand) |
| Right trigger (hold) | close the gripper |
| A | save the episode and reset the scene |
| B | discard the episode and reset the scene |
| Y, twice within 2 s | delete the last saved episode |
| X (hold) + push the left controller forward | calibrate "forward" (saved to `teleop/quest_calib.json`) |

Recording starts at the first grip press after a reset. Motion is relative to the viewer: hand
forward moves the robot into the screen, whatever the camera angle (the angle is read when the grip
is pressed). The wrist camera is shown in the bottom-right corner of the viewer and a close-up
front view in the bottom-left corner (the front view is for the operator only and is not recorded).

### 2. Training

```bash
./train.sh dp_v1                                         # 50k steps on the default dataset
DATASET=panda_pick_place_joint_position ./train.sh dp_v2 # another dataset in datasets/
STEPS=500 SAVE_FREQ=500 ./train.sh dp_smoke              # short pipeline check
./train.sh dp_v3 --policy.horizon=32                     # extra options go to lerobot-train
```

Runs are saved to `outputs/<run_name>`. Defaults: Diffusion Policy (LeRobot defaults: 2 observation
frames, horizon 16, 8 executed actions), random crop 216×288, batch size 64, a checkpoint every
10k steps. Other settings can be changed with the `DATASET`, `STEPS`, `SAVE_FREQ`, `BATCH_SIZE`
and `NUM_WORKERS` environment variables.

### 3. Evaluation

```bash
python eval.py outputs/dp_v1                          # last checkpoint, 50 episodes
python eval.py outputs/dp_v1/checkpoints/020000       # a specific checkpoint
python eval.py outputs/dp_v1 --episodes 10 --viewer   # watch it in real time
python eval.py outputs/dp_v1 --video 5                # save the first 5 episodes as mp4
```

Each episode starts with the cube at a random position (the same positions for the same `--seed`)
and succeeds when the cube rests on the bin floor within `--max-steps` policy steps (30 s by
default). Results are written to `<run>/eval/<checkpoint>_<controller>_seed<seed>_n<episodes>/`.

## Controllers

A policy outputs joint targets, so the controller that turns them into motion is part of the
system: **a policy must be evaluated with the controller its demos were recorded with**. The
controller name is appended to the dataset name and stored in `meta/controller.json`; `eval.py`
reads it from the training dataset and uses the same controller (`--controller` overrides it, with
a warning).

| Name | Sim actuators | Command |
|---|---|---|
| `joint_position` | Menagerie position servos | joint targets; MuJoCo computes the torques |
| `joint_impedance` | motors (torque) | `tau = Kp (q_d - q) - Kd qd`, Kp = 600, 600, 600, 600, 250, 150, 50 |

Gravity is compensated by the simulator in both cases, like the real Franka where the commanded
torque excludes gravity.

## Dataset format

LeRobot v3.0 dataset at 20 Hz (physics 500 Hz, control loop 100 Hz).

| Key | Shape | Content |
|---|---|---|
| `observation.images.wrist` | 240×320×3 | wrist camera (video) |
| `observation.images.front` | 240×320×3 | fixed third-person camera (video) |
| `observation.state` | 8 | joint1–7 [rad], gripper width [m] |
| `action` | 8 | joint1–7 targets from the IK [rad], gripper close (0/1) |
| `eef.state` | 10 | TCP position [m], rotation 6D, gripper width |
| `eef.action` | 10 | commanded TCP position, rotation 6D, gripper close |
| `torque.command` | 7 | joint torques of the controller [Nm] (torque controllers only) |
| `sim.qpos`, `sim.qvel` | – | full MuJoCo state, to replay or re-render episodes |

Policies only use `observation.*` and `action`; the other keys are kept for analysis and for
training with other action spaces later.

## Acknowledgements

- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for the Franka Emika Panda model
- [mink](https://github.com/kevinzakka/mink) for differential IK
- [oculus_reader](https://github.com/rail-berkeley/oculus_reader) for reading the Quest controllers
- [LeRobot](https://github.com/huggingface/lerobot) for the dataset format and Diffusion Policy
- [libfranka](https://github.com/frankaemika/libfranka) for the joint impedance gains

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licenses.
