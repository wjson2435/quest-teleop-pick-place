"""Evaluate a trained policy in the pick-and-place sim: success rate over random cube positions.

Runs the policy the same way the demos were recorded:
  - every 1/20 s (lerobot_recorder.FPS): render wrist + front (sim/cameras.py), state = 7 joints +
    gripper width -> policy -> action (7 joint targets + gripper close)
  - the action (joint targets) is held for 5 control steps (100 Hz) = 25 physics steps (500 Hz)
  - the arm controller (controllers/) must be the one the demos were recorded with: by default it is
    read from the training dataset (meta/controller.json); --controller overrides it (with a warning)
  - the cube starts at a random position (pick_place.reset_scene, same range as teleop)
  - success = cube resting on the bin floor (pick_place.cube_in_bin), checked after every action

The same --seed gives the same cube positions, so checkpoints can be compared fairly.

Usage:
  python eval.py outputs/dp_pick_place_v1                 # last checkpoint, 50 episodes
  python eval.py outputs/dp_pick_place_v1/checkpoints/020000
  python eval.py <run> --episodes 10 --viewer                        # watch it (real time)
  python eval.py <run> --video 5                                     # save the first 5 episodes as mp4
  python eval.py <run> --controller joint_position                   # force a controller

Results (summary + per episode) go to <run>/eval/<checkpoint>_<controller>_seed<seed>_n<episodes>/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Same OpenGL setup as quest_teleop.py (X11 GLFW + NVIDIA GLX); must be set before mujoco is imported
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
os.environ.setdefault("MUJOCO_GL", "glfw")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))             # controllers
sys.path.insert(0, str(HERE / "sim"))     # pick_place, cameras
sys.path.insert(0, str(HERE / "teleop"))  # lerobot_recorder (FPS, task string)

import imageio
import mujoco
import mujoco.viewer
import numpy as np
import torch

import cameras
import controllers
import lerobot_recorder as lr
import pick_place as pp
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

SETTLE_S = 0.5  # let the cube settle after a reset (in teleop the scene runs before the first grip)
LIFT_Z = 0.05   # cube raised this much above its start = "lifted" (to tell grasp from place failures)


def resolve_checkpoint(path):
    """Accepts a run dir, a checkpoint dir or a pretrained_model dir."""
    p = Path(path).expanduser().resolve()
    for c in (p, p / "pretrained_model", p / "checkpoints" / "last" / "pretrained_model"):
        if (c / "config.json").exists():
            return c.resolve()  # also resolves the "last" symlink to the real step
    raise SystemExit(f"No checkpoint found in {p}")


def dataset_controller(ckpt):
    """Controller the training dataset was recorded with (None if unknown)."""
    cfg = json.loads((ckpt / "train_config.json").read_text())
    root = cfg["dataset"]["root"]
    return lr.read_controller(root) if root else None


def load_policy(ckpt, device):
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(ckpt),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def observation(data, images):
    """Same keys and values as lerobot_recorder.make_frame."""
    q = data.qpos
    obs = {f"observation.images.{c}": images[c] for c in cameras.CAMERAS}
    obs["observation.state"] = np.r_[q[:7], q[7] + q[8]].astype(np.float32)
    return obs


def run_episode(model, data, controller, policy, pre, post, renderer, opt, rng, args, viewer=None, frames=None):
    pp.reset_scene(model, data, rng)
    policy.reset()  # clears the observation history and the queued actions
    cube0 = data.body("cube").xpos.copy()
    controller.step(data, data.qpos[:7].copy(), round(SETTLE_S / model.opt.timestep))

    n_physics = round(pp.CONTROL_DT / model.opt.timestep) * lr.RECORD_EVERY  # per action
    device = torch.device(args.device)
    max_z = cube0[2]
    for step in range(args.max_steps):
        t0 = time.time()
        images = cameras.render_cameras(renderer, opt, data)
        if frames is not None:
            frames.append(np.hstack([images["front"], images["wrist"]]))
        action = predict_action(observation(data, images), policy, device, pre, post,
                                use_amp=False, task=args.task, robot_type="panda")
        action = action.cpu().numpy().reshape(-1)  # (1, 8) -> (8,)
        data.ctrl[7] = pp.GRIPPER_CLOSED if action[7] > 0.5 else pp.GRIPPER_OPEN
        controller.step(data, action[:7], n_physics)
        max_z = max(max_z, data.body("cube").xpos[2])

        if viewer is not None:
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            time.sleep(max(0.0, 1.0 / lr.FPS - (time.time() - t0)))
        if pp.cube_in_bin(data):
            return dict(success=True, steps=step + 1, lifted=True, cube_start=cube0[:2].tolist())
    return dict(success=False, steps=args.max_steps, lifted=bool(max_z > cube0[2] + LIFT_Z),
                cube_start=cube0[:2].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="run dir (uses the last checkpoint) or checkpoint dir")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="cube positions (same seed = same positions)")
    parser.add_argument("--max-steps", type=int, default=600, help="policy steps per episode (600 = 30 s)")
    parser.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer (runs in real time)")
    parser.add_argument("--video", type=int, default=0, metavar="N", help="save the first N episodes as mp4")
    parser.add_argument("--controller", choices=sorted(controllers.CONTROLLERS), default=None,
                        help="arm controller (default: the one stored in the training dataset)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default=lr.DEFAULT_TASK)
    args = parser.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    recorded = dataset_controller(ckpt)
    if args.controller is None:
        if recorded is None:
            raise SystemExit("The training dataset has no meta/controller.json (or was moved): pass --controller")
        args.controller = recorded
    elif recorded is not None and args.controller != recorded:
        print(f"WARNING: the demos were recorded with {recorded!r}, evaluating with {args.controller!r}")
    step_name = ckpt.parent.name  # e.g. 020000
    run_dir = ckpt.parents[2] if ckpt.parent.parent.name == "checkpoints" else ckpt.parent
    out_dir = run_dir / "eval" / f"{step_name}_{args.controller}_seed{args.seed}_n{args.episodes}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint {ckpt}\nController {args.controller}\nResults    {out_dir}")

    policy, pre, post = load_policy(ckpt, args.device)
    controller_cls = controllers.get(args.controller)
    model = pp.build_sim_model(torque=controller_cls.torque_model)  # same model as in teleop
    controller = controller_cls(model)
    data = mujoco.MjData(model)
    renderer, opt = cameras.make_camera_renderer(model)
    rng = np.random.default_rng(args.seed)

    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    results = []
    try:
        for i in range(args.episodes):
            frames = [] if i < args.video else None
            t0 = time.time()
            r = run_episode(model, data, controller, policy, pre, post, renderer, opt, rng, args, viewer, frames)
            results.append(r)
            outcome = "SUCCESS" if r["success"] else ("FAIL (lifted, not placed)" if r["lifted"] else "FAIL (not lifted)")
            n_ok = sum(x["success"] for x in results)
            print(f"[{i + 1:3d}/{args.episodes}] {outcome:<26} {r['steps'] / lr.FPS:5.1f} s sim, "
                  f"{time.time() - t0:5.1f} s wall   cube ({r['cube_start'][0]:.2f}, {r['cube_start'][1]:.2f})"
                  f"   success so far {n_ok}/{i + 1}")
            if frames:
                name = out_dir / f"ep{i:03d}_{'success' if r['success'] else 'fail'}.mp4"
                imageio.mimsave(name, frames, fps=lr.FPS)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        if viewer is not None:
            viewer.close()
        renderer.close()

    if not results:
        return
    n = len(results)
    ok = [r for r in results if r["success"]]
    summary = dict(
        checkpoint=str(ckpt), controller=args.controller, seed=args.seed, episodes=n, max_steps=args.max_steps,
        success_rate=len(ok) / n,
        lifted_not_placed=sum(not r["success"] and r["lifted"] for r in results) / n,
        not_lifted=sum(not r["lifted"] for r in results) / n,
        mean_success_time_s=float(np.mean([r["steps"] for r in ok]) / lr.FPS) if ok else None,
    )
    (out_dir / "results.json").write_text(json.dumps(dict(summary=summary, episodes=results), indent=2))
    print(f"\nSuccess {len(ok)}/{n} = {summary['success_rate'] * 100:.0f}%   "
          f"(lifted but not placed {summary['lifted_not_placed'] * 100:.0f}%, "
          f"not lifted {summary['not_lifted'] * 100:.0f}%)")
    if ok:
        print(f"Mean time of successful episodes: {summary['mean_success_time_s']:.1f} s")


if __name__ == "__main__":
    main()
