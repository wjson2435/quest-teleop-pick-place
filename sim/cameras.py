"""Camera images for the policy (wrist + front), shared by data recording and policy evaluation.

Rendered without sites (TCP marker) and without geom group 1 (teleop target sphere), and
without shadows/reflections. Training and evaluation must use the same settings.
"""

import mujoco

IMG_H, IMG_W = 240, 320
CAMERAS = ["wrist", "front"]  # MuJoCo camera names -> observation.images.<name>


def make_camera_renderer(model):
    """Renderer + scene options. Create one per process and reuse it."""
    renderer = mujoco.Renderer(model, IMG_H, IMG_W)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0  # no TCP marker
    opt.geomgroup[1] = 0  # no teleop target sphere
    return renderer, opt


def render_camera(renderer, opt, data, camera):
    renderer.update_scene(data, camera, scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    return renderer.render()  # new (H, W, 3) uint8 array


def render_cameras(renderer, opt, data):
    return {c: render_camera(renderer, opt, data, c) for c in CAMERAS}
