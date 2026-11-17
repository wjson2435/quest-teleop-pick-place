"""Quest 2 teleop: the right controller drives the TCP target, and the IK makes the TCP follow it.

Clutch-style relative control:
  - When the grip (RG) is pressed, store the controller pose and the current target pose.
  - While it is held, target = stored target + (controller motion since the press), mapped to the robot frame.
  - When it is released, the target stays where it is (the hand can be repositioned freely).

View-relative control:
  Hand forward = into the screen, hand left = screen left, whatever the viewer azimuth is.
  The azimuth is read when the grip is pressed, so the view can be rotated with the mouse
  between strokes (not during one).

The wrist camera image is shown in the bottom-right corner of the viewer, a front view
(camera "operator", not recorded) in the bottom-left corner.

Recording (--record): demos are saved as a LeRobot dataset in pick_place/datasets/<name>
(see lerobot_recorder.py). Recording starts automatically at the first grip press after a
reset; A saves the episode, B throws it away. Both then reset the scene.

  Every control cycle:
    1. Read the Quest pose/buttons -> update the target pose (clutch)
    2. Low-pass filter + workspace clamp -> mocap 'target' (visual) + FrameTask target
    3. IK (ik.py) -> joint targets q_cmd,  trigger -> gripper
    4. Arm controller (--controller, see controllers/) toward q_cmd for the physics steps of this cycle

Controls:
  Right grip (RG)  hold to move the robot
  Right trigger    hold to close the gripper
  B                reset the scene (robot to start, cube to a random position);
                   when recording: discard the current episode
  A                when recording: save the current episode and reset the scene
  Left Y, twice    when recording: delete the last saved episode (second press within 2 s)
  Left X + push    calibrate "forward": hold X, push the LEFT controller ~20 cm toward the
                   monitor, release X. Saved to quest_calib.json and loaded on the next start.

Usage:
  python quest_teleop.py
  python quest_teleop.py --pos-only     # ignore controller rotation (keep TCP pointing down)
  python quest_teleop.py --scale 1.5    # move the robot 1.5x the hand motion
  python quest_teleop.py --record       # record demos to pick_place/datasets/panda_pick_place_joint_impedance
  python quest_teleop.py --record --dataset test   # -> pick_place/datasets/test_joint_impedance (appends if it exists)
  python quest_teleop.py --record --controller joint_position   # -> panda_pick_place_joint_position
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# OpenGL backend. On this machine (Wayland session) GLFW's default Wayland path falls back to the
# llvmpipe CPU renderer: the viewer drew at ~15 FPS and each wrist image took ~70 ms.
# Forcing GLFW onto X11 (XWayland) with NVIDIA's GLX gives ~60 FPS. The viewer and the wrist
# renderer must use the same GPU (mixing vendors in one process was much slower).
# Must be set before mujoco / glfw are imported.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
os.environ.setdefault("MUJOCO_GL", "glfw")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # controllers
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))  # pick_place, cameras

import mujoco
import mujoco.viewer
import numpy as np

import mink
from oculus_reader.reader import OculusReader

import lerobot_recorder as lr
import pick_place as pp  # sim model, reset, success check
from ik import TeleopIK
import controllers

# ---------------- Parameters ----------------
# Default quest frame -> robot frame (measured once by hand), used when there is no
# calibration file:  robot x (forward) = quest +z,  robot y (left) = quest +x,  robot z (up) = quest +y
R_ROBOT_FROM_QUEST = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
QUEST_UP = np.array([0.0, 1.0, 0.0])  # the Quest frame is gravity-aligned: +y is always up

# Forward calibration (left controller, so it cannot be triggered by accident while teleoperating)
CALIB_HAND = "l"
CALIB_KEY = "X"
CALIB_MIN_DIST = 0.08  # minimum horizontal push [m]
CALIB_FILE = Path(__file__).with_name("quest_calib.json")

HAND = "r"
GRIP_KEY = "RG"  # clutch button (boolean)
TRIGGER_KEY = "rightTrig"  # analog trigger (0..1)
RESET_KEY = "B"
SAVE_KEY = "A"
UNDO_KEY = "Y"  # left controller: delete the last saved episode (press twice)
UNDO_CONFIRM_S = 2.0  # the second Y press must come within this time [s]
MIN_EPISODE_FRAMES = 10  # A is ignored for shorter episodes (0.5 s)
TRIGGER_CLOSE = 0.6  # hysteresis: close above this,
TRIGGER_OPEN = 0.4  # open below this

SMOOTH = 0.2  # low-pass factor per control step (1 = no filtering, smaller = smoother/slower)

# Tracking glitches: when the headset loses sight of a controller (or re-centers its frame), the
# pose can jump tens of cm between two samples. A jump larger than this is not followed: the
# clutch is re-anchored at the new pose, so the robot target stays where it was.
# (Poses arrive at ~70 Hz with gaps up to ~55 ms; 10 cm in one gap is ~2 m/s, faster than teleop.)
JUMP_POS = 0.10  # [m] between two consecutive samples
JUMP_ROT = 0.6  # [rad] between two consecutive samples (~35 deg)

# TCP workspace limits in the robot frame [m] (table top = z 0)
WS_MIN = np.array([0.20, -0.40, 0.005])
WS_MAX = np.array([0.80, 0.40, 0.60])

# Initial viewer camera: diagonal from behind-left so the arm does not hide the workspace.
# (Control is view-relative, so any azimuth works.)
VIEW_LOOKAT = [0.5, 0.0, 0.15]
VIEW_DISTANCE = 1.8
VIEW_AZIMUTH = 45.0  # [deg], 0 = looking along robot +x
VIEW_ELEVATION = -35.0  # [deg], looking down

# Wrist camera inset (bottom-right of the viewer), same size as the recorded images
INSET_W, INSET_H = lr.IMG_W, lr.IMG_H  # [px]
INSET_MARGIN = 10  # [px]
INSET_EVERY = 3  # render every N control steps (~33 Hz)
OPERATOR_CAM = "operator"  # front view inset (bottom-left), only shown to the operator, not recorded

def mapping_from_forward(push):
    """Quest->robot rotation from a 'forward' push vector (quest frame). None if too short."""
    f = push - QUEST_UP * (push @ QUEST_UP)  # keep only the horizontal part
    dist = np.linalg.norm(f)
    if dist < CALIB_MIN_DIST:
        print(f"Calibration ignored: horizontal push {dist * 100:.1f} cm < {CALIB_MIN_DIST * 100:.0f} cm")
        return None
    f /= dist
    left = np.cross(QUEST_UP, f)
    return np.stack([f, left, QUEST_UP])  # rows: robot x, y, z expressed in the quest frame


def load_calibration():
    if CALIB_FILE.exists():
        try:
            R = np.array(json.loads(CALIB_FILE.read_text())["R_robot_from_quest"], dtype=float)
            if R.shape == (3, 3) and np.allclose(R @ R.T, np.eye(3), atol=1e-6):
                print(f"Loaded forward calibration from {CALIB_FILE.name}")
                return R
        except (ValueError, KeyError) as e:
            print(f"Ignoring {CALIB_FILE.name}: {e}")
    print("No calibration file, using the default mapping")
    return R_ROBOT_FROM_QUEST.copy()


def save_calibration(R):
    CALIB_FILE.write_text(json.dumps({"R_robot_from_quest": R.round(6).tolist()}, indent=2))


class ForwardCalibrator:
    """Hold X and push the left controller forward; on release returns the new mapping."""

    def __init__(self):
        self.p0 = None
        self.p_last = None

    def update(self, T_left, pressed):
        if pressed:
            if T_left is not None:
                if self.p0 is None:
                    self.p0 = T_left[:3, 3].copy()
                    print("Calibrating: push the left controller forward, then release X")
                self.p_last = T_left[:3, 3].copy()
            return None
        if self.p0 is None:
            return None
        push = self.p_last - self.p0
        self.p0 = self.p_last = None
        return mapping_from_forward(push)


class InsetUploader:
    """Hands the inset images to the viewer from a background thread.

    viewer.set_images blocks until the viewer draws a frame (~35-45 ms here), which would
    stall the 100 Hz control loop. The thread always uploads only the newest image.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.item = None
        self.running = True
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, items):
        """items: list of (MjrRect, image)."""
        with self.cv:
            self.item = items
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while self.item is None and self.running:
                    self.cv.wait()
                if not self.running:
                    return
                item, self.item = self.item, None
            if self.viewer.is_running():
                self.viewer.set_images(item)

    def stop(self):
        with self.cv:
            self.running = False
            self.cv.notify()
        self.thread.join(timeout=1.0)


class QuestClutch:
    """Turns Quest controller poses into a robot TCP target with a grip clutch."""

    def __init__(self, target_pos, target_rot, scale, pos_only, R_base):
        self.scale = scale
        self.pos_only = pos_only
        self.R_base = R_base  # quest frame -> robot frame for an operator looking along robot +x
        self.reset(target_pos, target_rot)

    def reset(self, target_pos, target_rot):
        self.target_pos = target_pos.copy()
        self.target_rot = target_rot.copy()  # 3x3
        self.engaged = False
        self.ctrl0 = None  # controller pose (4x4, quest frame) at clutch press
        self.pos0 = None
        self.rot0 = None
        self.R = None  # quest frame -> robot frame for the current stroke
        self.last = None  # previous controller pose (for jump detection)

    def update(self, T_ctrl, grip, view_azimuth=0.0):
        """T_ctrl: 4x4 controller pose in the quest frame (None if not tracked).

        view_azimuth: viewer azimuth [deg]. R_base assumes the operator looks along robot +x
        (azimuth 0); for another azimuth the mapping is yawed by the same angle so hand-forward
        stays screen-forward.
        """
        last, self.last = self.last, (T_ctrl.copy() if T_ctrl is not None else None)
        if T_ctrl is None or not grip:
            if self.engaged:
                print("Clutch released")
            self.engaged = False
            return
        if self.engaged and last is not None:
            dpos = np.linalg.norm(T_ctrl[:3, 3] - last[:3, 3])
            cos = (np.trace(T_ctrl[:3, :3] @ last[:3, :3].T) - 1.0) / 2.0
            drot = np.arccos(np.clip(cos, -1.0, 1.0))
            if dpos > JUMP_POS or drot > JUMP_ROT:
                # Re-anchor: continue the stroke from the current target with the new pose
                self.ctrl0 = T_ctrl.copy()
                self.pos0 = self.target_pos.copy()
                self.rot0 = self.target_rot.copy()
                print(f"Tracking jump ignored ({dpos * 100:.0f} cm, {np.rad2deg(drot):.0f} deg). "
                      "If directions now feel wrong, recalibrate with X.")
                return
        if not self.engaged:
            self.engaged = True
            self.ctrl0 = T_ctrl.copy()
            self.pos0 = self.target_pos.copy()
            self.rot0 = self.target_rot.copy()
            a = np.deg2rad(view_azimuth)
            Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                           [np.sin(a), np.cos(a), 0.0],
                           [0.0, 0.0, 1.0]])
            self.R = Rz @ self.R_base  # fixed for this stroke
            print("Clutch engaged")
            return

        R = self.R
        dp_quest = T_ctrl[:3, 3] - self.ctrl0[:3, 3]
        self.target_pos = self.pos0 + self.scale * (R @ dp_quest)

        if not self.pos_only:
            # Rotation change since the press, expressed in the quest world frame,
            # then re-expressed in the robot frame and applied on the world side.
            dR_quest = T_ctrl[:3, :3] @ self.ctrl0[:3, :3].T
            dR_robot = R @ dR_quest @ R.T
            self.target_rot = dR_robot @ self.rot0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos-only", action="store_true", help="ignore controller rotation")
    parser.add_argument("--scale", type=float, default=1.0, help="hand motion -> robot motion scale")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=False,
                        help="record demos as a LeRobot dataset (--no-record to turn off)")
    parser.add_argument("--controller", default=controllers.DEFAULT, choices=sorted(controllers.CONTROLLERS),
                        help="arm controller (also appended to the dataset name)")
    parser.add_argument("--dataset", default="panda_pick_place",
                        help="dataset base name: recorded to pick_place/datasets/<dataset>_<controller>")
    parser.add_argument("--task", default=lr.DEFAULT_TASK, help="task instruction stored with each frame")
    args = parser.parse_args()

    # Arm controller + matching sim model (target sphere in geom group 1: shown in the viewer, hidden in the cameras)
    controller_cls = controllers.get(args.controller)
    model = pp.build_sim_model(torque=controller_cls.torque_model)
    controller = controller_cls(model)
    print(f"Controller: {controller.name}")
    data = mujoco.MjData(model)
    ik = TeleopIK()
    n_substeps = round(pp.CONTROL_DT / model.opt.timestep)
    rng = np.random.default_rng(args.seed)

    def reset():
        pp.reset_scene(model, data, rng)
        return ik.reset(data.qpos)  # TCP pose at the start key

    tcp_pos, tcp_rot = reset()
    clutch = QuestClutch(tcp_pos, tcp_rot, args.scale, args.pos_only, load_calibration())
    calibrator = ForwardCalibrator()
    cmd_pos, cmd_rot = tcp_pos.copy(), mink.SO3.from_matrix(tcp_rot)  # filtered command

    # Camera renderer (inset + recording): no sites (TCP marker), no group-1 geoms (target sphere)
    cam_renderer, cam_opt = lr.make_camera_renderer(model)
    operator_opt = mujoco.MjvOption()  # operator inset: no sites, but the target sphere stays visible
    operator_opt.sitegroup[:] = 0
    has_torque = hasattr(controller, "torque")  # torque control: also record the commanded torques
    recorder = (lr.Recorder(lr.dataset_name(args.dataset, controller.name), model, controller.name,
                            has_torque, args.task)
                if args.record else None)

    reader = OculusReader()
    print(__doc__.split("Controls")[1].split("Usage:")[0])

    gripper_closed = False
    reset_prev = False
    save_prev = False
    undo_prev = False
    undo_armed_t = None  # time of the first Y press
    recording = False  # an episode is being recorded
    rec_step = 0  # control steps since the episode started
    step = 0
    try:
        # Keep the UI panels: on this machine set_images draws nothing when they are hidden.
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = VIEW_LOOKAT
            viewer.cam.distance = VIEW_DISTANCE
            viewer.cam.azimuth = VIEW_AZIMUTH
            viewer.cam.elevation = VIEW_ELEVATION
            uploader = InsetUploader(viewer)
            while viewer.is_running():
                t0 = time.time()

                # 1. Quest -> target (clutch)
                transforms, buttons = reader.get_transformations_and_buttons()
                T_ctrl = transforms.get(HAND) if transforms else None
                buttons = buttons or {}
                clutch.update(T_ctrl, bool(buttons.get(GRIP_KEY, False)), viewer.cam.azimuth)

                T_left = transforms.get(CALIB_HAND) if transforms else None
                R_new = calibrator.update(T_left, bool(buttons.get(CALIB_KEY, False)))
                if R_new is not None:
                    clutch.R_base = R_new  # takes effect from the next grip press
                    save_calibration(R_new)
                    print(f"Forward calibrated (quest frame forward = {np.round(R_new[0], 3)}), "
                          f"saved to {CALIB_FILE.name}")

                trig = buttons.get(TRIGGER_KEY, (0.0,))[0]
                if trig > TRIGGER_CLOSE:
                    gripper_closed = True
                elif trig < TRIGGER_OPEN:
                    gripper_closed = False

                reset_now = bool(buttons.get(RESET_KEY, False))
                save_now = bool(buttons.get(SAVE_KEY, False))
                do_reset = reset_now and not reset_prev
                if recording and do_reset:
                    print(f"Episode discarded ({recorder.n_frames} frames)")
                    recorder.discard()
                    recording = False
                if recording and save_now and not save_prev:
                    if recorder.n_frames < MIN_EPISODE_FRAMES:
                        print(f"Episode too short ({recorder.n_frames} frames), not saved")
                    else:
                        if not pp.cube_in_bin(data):
                            print("Warning: the cube is not in the bin, saving anyway (B discards)")
                        t_save = time.time()
                        n = recorder.n_frames
                        recorder.save()
                        print(f"Episode {recorder.num_episodes - 1} saved: {n} frames "
                              f"({n / lr.FPS:.1f} s), encoding took {time.time() - t_save:.1f} s")
                        recording = False
                        do_reset = True
                reset_prev, save_prev = reset_now, save_now

                undo_now = bool(buttons.get(UNDO_KEY, False))
                if recorder is not None and undo_now and not undo_prev:
                    if recorder.num_episodes == 0:
                        print("No saved episode to delete")
                    elif undo_armed_t is None or time.time() - undo_armed_t > UNDO_CONFIRM_S:
                        undo_armed_t = time.time()
                        print(f"Press Y again within {UNDO_CONFIRM_S:.0f} s to delete "
                              f"episode {recorder.num_episodes - 1}")
                    else:
                        undo_armed_t = None
                        t_del = time.time()
                        idx = recorder.delete_last()  # also drops the frames of an unsaved episode
                        recording = False
                        do_reset = True
                        print(f"Episode {idx} deleted ({time.time() - t_del:.1f} s), "
                              f"{recorder.num_episodes} episodes left")
                undo_prev = undo_now
                if do_reset:
                    tcp_pos, tcp_rot = reset()
                    clutch.reset(tcp_pos, tcp_rot)
                    cmd_pos, cmd_rot = tcp_pos.copy(), mink.SO3.from_matrix(tcp_rot)
                    print("Scene reset")
                if recorder is not None and not recording and clutch.engaged:
                    recording, rec_step = True, 0
                    print(f"Recording episode {recorder.num_episodes} (A = save, B = discard)")

                # 2. Clamp + low-pass filter
                clutch.target_pos = np.clip(clutch.target_pos, WS_MIN, WS_MAX)
                cmd_pos = cmd_pos + SMOOTH * (clutch.target_pos - cmd_pos)
                cmd_rot = cmd_rot.interpolate(mink.SO3.from_matrix(clutch.target_rot), SMOOTH)

                data.mocap_pos[0] = cmd_pos
                data.mocap_quat[0] = cmd_rot.wxyz

                # 3. IK -> joint targets for the arm controller
                q_cmd = ik.solve(cmd_pos, cmd_rot)
                data.ctrl[7] = pp.GRIPPER_CLOSED if gripper_closed else pp.GRIPPER_OPEN

                # 4. Record (observation before the step + the command applied in this step)
                images = None
                if recording:
                    if rec_step % lr.RECORD_EVERY == 0:
                        images = lr.render_cameras(cam_renderer, cam_opt, data)
                        recorder.add(lr.make_frame(data, images, q_cmd, controller.torque(data, q_cmd) if has_torque else None,
                                                   cmd_pos, cmd_rot.as_matrix(), gripper_closed, args.task))
                    rec_step += 1

                # 5. Simulate with the arm controller
                controller.step(data, q_cmd, n_substeps)

                # 6. Insets: wrist camera bottom-right, front (operator) view bottom-left of the 3D view
                #    (window pixels, origin bottom-left; vp.left is the left UI panel width)
                vp = viewer.viewport
                if step % INSET_EVERY == 0 and vp is not None and vp.width > 2 * (INSET_W + INSET_MARGIN):
                    wrist = images["wrist"] if images else lr.render_camera(cam_renderer, cam_opt, data, "wrist")
                    operator = lr.render_camera(cam_renderer, operator_opt, data, OPERATOR_CAM)
                    y = vp.bottom + INSET_MARGIN
                    uploader.submit([
                        (mujoco.MjrRect(vp.left + vp.width - INSET_W - INSET_MARGIN, y, INSET_W, INSET_H), wrist),
                        (mujoco.MjrRect(vp.left + INSET_MARGIN, y, INSET_W, INSET_H), operator),
                    ])
                step += 1

                viewer.sync()
                time.sleep(max(0.0, pp.CONTROL_DT - (time.time() - t0)))
            uploader.stop()
            # Close the dataset before leaving the viewer (the viewer can crash on exit)
            if recorder is not None:
                recorder.close()
    finally:
        if recorder is not None:
            recorder.close()  # no-op if already closed (e.g. after Ctrl+C)
        reader.stop()  # the logcat thread is not a daemon; stop it or the process will not exit
        cam_renderer.close()


if __name__ == "__main__":
    main()
