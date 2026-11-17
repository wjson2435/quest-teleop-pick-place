"""Records teleop demos as a LeRobot dataset (v3.0 format, lerobot 0.4.x) for Diffusion Policy.

One frame every RECORD_EVERY control steps (100 Hz / 5 = 20 Hz). Each frame is
(observation at time t, action commanded at time t), taken right before mj_step.

Features
  observation.images.wrist  (240, 320, 3) video  wrist camera
  observation.images.front  (240, 320, 3) video  fixed third-person camera ("front" in scene.xml)
  observation.state         (8,)  joint1..7 [rad], gripper width [m] (sum of both fingers)
  action                    (8,)  joint1..7 targets from the IK [rad], gripper command (1 = close)
                                  (targets of the arm controller, see controllers/)

  Extra keys, ignored by lerobot policies (they only use observation.* and action*):
  eef.state                 (10,) TCP position [m], rotation 6D (first two columns), gripper width
  eef.action                (10,) commanded TCP position, rotation 6D, gripper command
  torque.command            (7,)  joint torques of the controller at this frame [Nm] (without gravity),
                                  only for controllers that compute torques (with a torque() method)
  sim.qpos / sim.qvel       full MuJoCo state, to replay or re-render an episode later

The arm controller the demos were recorded with is stored in meta/controller.json
({"controller": "joint_impedance"}): eval.py reads it to run the policy with the same controller,
and appending with another controller is refused.

Camera images come from sim/cameras.py (also used when evaluating a policy, so the images
look the same as in training).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from cameras import CAMERAS, IMG_H, IMG_W, make_camera_renderer, render_camera, render_cameras  # noqa: F401

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

FPS = 20
CONTROL_HZ = 100
RECORD_EVERY = CONTROL_HZ // FPS  # control steps per recorded frame

DEFAULT_TASK = "pick up the red cube and place it in the green bin"
DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"  # pick_place/datasets

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
ROT6D_NAMES = [f"r{i}" for i in range(6)]


def features(nq, nv, with_torque):
    img = {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]}
    return {
        **{f"observation.images.{c}": dict(img) for c in CAMERAS},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES + ["gripper_width"]},
        "action": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES + ["gripper_close"]},
        "eef.state": {"dtype": "float32", "shape": (10,),
                      "names": ["x", "y", "z"] + ROT6D_NAMES + ["gripper_width"]},
        "eef.action": {"dtype": "float32", "shape": (10,),
                       "names": ["x", "y", "z"] + ROT6D_NAMES + ["gripper_close"]},
        **({"torque.command": {"dtype": "float32", "shape": (7,), "names": JOINT_NAMES}} if with_torque else {}),
        "sim.qpos": {"dtype": "float32", "shape": (nq,), "names": None},
        "sim.qvel": {"dtype": "float32", "shape": (nv,), "names": None},
    }


def dataset_name(base, controller):
    """Dataset folder name: the controller name is appended (panda_pick_place_joint_impedance)."""
    return f"{base}_{controller}"


def read_controller(root):
    """Controller name stored in a dataset (None if the file is missing)."""
    f = Path(root) / "meta" / "controller.json"
    return json.loads(f.read_text())["controller"] if f.exists() else None


def write_controller(root, controller):
    (Path(root) / "meta" / "controller.json").write_text(json.dumps({"controller": controller}) + "\n")


def rot6d(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def make_frame(data, images, q_cmd, tau, cmd_pos, cmd_rot, gripper_close, task):
    """One dataset frame from the current sim state and the command about to be applied.

    q_cmd: joint targets (IK output), tau: controller torque for this state (None: not recorded),
    cmd_pos/cmd_rot: TCP target.
    """
    q = data.qpos
    grip_width = q[7] + q[8]
    tcp = data.site("tcp")
    close = 1.0 if gripper_close else 0.0
    frame = {f"observation.images.{c}": images[c] for c in CAMERAS}
    frame.update({
        "observation.state": np.r_[q[:7], grip_width].astype(np.float32),
        "action": np.r_[q_cmd, close].astype(np.float32),
        "eef.state": np.r_[tcp.xpos, rot6d(tcp.xmat.reshape(3, 3)), grip_width].astype(np.float32),
        "eef.action": np.r_[cmd_pos, rot6d(cmd_rot), close].astype(np.float32),
        "sim.qpos": data.qpos.astype(np.float32),
        "sim.qvel": data.qvel.astype(np.float32),
        "task": task,
    })
    if tau is not None:
        frame["torque.command"] = np.asarray(tau, dtype=np.float32)
    return frame


class Recorder:
    """Thin wrapper around LeRobotDataset: open or create, add frames, save or discard episodes.

    Every saved episode is finalized on disk right away (parquet footers + episode metadata),
    so the dataset stays valid even if the process dies afterwards (the MuJoCo viewer can
    crash on exit). LeRobotDataset cannot keep recording after finalize(), so it is reopened.
    """

    def __init__(self, name, model, controller, with_torque, task=DEFAULT_TASK):
        self.task = task
        self.controller = controller  # controller name, stored in meta/controller.json
        self.with_torque = with_torque  # record torque.command
        self.root = DATASETS_DIR / name
        self.repo_id = f"local/{name}"
        self.model = model
        self.closed = False
        self.n_frames = 0
        self._open(verbose=True)

    def _open(self, verbose=False):
        # A dataset that was created but never got an episode has info.json but no tasks/episodes
        # files, and LeRobotDataset cannot open it. Nothing is lost by starting it over.
        if (self.root / "meta" / "info.json").exists() and not (self.root / "meta" / "tasks.parquet").exists():
            print(f"{self.root} has no saved episodes, creating it again")
            shutil.rmtree(self.root)
        if (self.root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset(self.repo_id, root=self.root)
            expected = set(features(self.model.nq, self.model.nv, self.with_torque))
            stored = set(self.ds.meta.features) - set(DEFAULT_FEATURES)  # minus timestamp, index, ...
            if expected != stored:
                raise SystemExit(f"{self.root} was recorded with other features (missing "
                                 f"{sorted(expected - stored)}, extra {sorted(stored - expected)}). "
                                 "Use a new dataset name (--dataset).")
            self.ds.start_image_writer(num_threads=4)
            if verbose:
                print(f"Appending to {self.root} ({self.num_episodes} episodes so far)")
        else:
            self.ds = LeRobotDataset.create(self.repo_id, fps=FPS,
                                            features=features(self.model.nq, self.model.nv, self.with_torque),
                                            root=self.root, robot_type="panda", use_videos=True,
                                            image_writer_threads=4)
            print(f"Created dataset {self.root}")
        stored = read_controller(self.root)
        if stored is not None and stored != self.controller:
            self._finalize()
            raise SystemExit(f"{self.root} was recorded with controller {stored!r}, not {self.controller!r}. "
                             "Use the same controller or a new dataset name (--dataset).")
        write_controller(self.root, self.controller)  # new dataset, or a copy made by delete_last
        # Leftover frames of an unsaved episode (e.g. after a crash) would end up in the next video
        shutil.rmtree(self.root / "images", ignore_errors=True)

    def _finalize(self):
        self.ds.stop_image_writer()
        self.ds.finalize()

    @property
    def num_episodes(self):
        return self.ds.meta.total_episodes

    def add(self, frame):
        self.ds.add_frame(frame)
        self.n_frames += 1

    def save(self):
        self.ds.save_episode()  # blocks while the episode videos are encoded
        self._finalize()
        self._open()
        self.n_frames = 0

    def discard(self):
        if self.n_frames:
            # clear_episode_buffer only deletes the frames of "image" features, not "video" ones.
            # The PNG frames would stay and be encoded into the next episode with the same index.
            self.ds._wait_image_writer()
            ep = self.ds.episode_buffer["episode_index"]
            for key in self.ds.meta.camera_keys:
                shutil.rmtree(self.ds._get_image_file_dir(ep, key), ignore_errors=True)
            self.ds.clear_episode_buffer(delete_images=False)
        self.n_frames = 0

    def delete_last(self):
        """Delete the last saved episode (and any unsaved frames). Returns its index, or None.

        lerobot can only delete episodes by writing a new copy of the dataset, so the copy is
        written next to it and then swapped in; the original is kept until the copy is complete.
        """
        n = self.num_episodes
        if n == 0:
            return None
        self.discard()
        self._finalize()
        if n == 1:
            shutil.rmtree(self.root)
        else:
            tmp = self.root.with_name(self.root.name + ".tmp_new")
            old = self.root.with_name(self.root.name + ".tmp_old")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(old, ignore_errors=True)
            src = LeRobotDataset(self.repo_id, root=self.root)
            delete_episodes(src, [n - 1], output_dir=tmp, repo_id=self.repo_id)
            self.root.rename(old)
            tmp.rename(self.root)
            shutil.rmtree(old)
        self._open()
        return n - 1

    def close(self):
        """Discard an unfinished episode and close the dataset."""
        if self.closed:
            return
        self.closed = True
        self.discard()
        self._finalize()
        print(f"Dataset closed: {self.num_episodes} episodes in {self.root}")
