"""Render one push-kick trial (same mechanism as eval_push_kick.py) to GIF,
for visual side-by-side comparison. EGL software rendering (no GPU needed),
same pattern as playground/dcubot/render_checkpoint.py.

Usage:
    MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
      .venv/bin/python tools/render_push_kick.py --which dcubot_post_fix --magnitude 0.8 --theta 0 --out dcubot_kick.gif
"""
import argparse
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, "tools")
import eval_push_kick as epk


def render_kick(infer, foot_names, magnitude, theta, out, fps=20, size=(360, 270), settle_s=1.0, total_s=4.0):
    epk.reset_idle(infer)
    m, d = infer.model, infer.data
    torso_id = 1
    m.vis.global_.offwidth, m.vis.global_.offheight = size
    r = mujoco.Renderer(m, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso_id
    cam.distance, cam.azimuth, cam.elevation = 1.0, 90, -20

    n_settle = int(settle_s / (infer.sim_dt * infer.decimation))
    for _ in range(n_settle):
        epk.step_policy(infer)

    d.qvel[0] += magnitude * np.cos(theta)
    d.qvel[1] += magnitude * np.sin(theta)

    n_total = int(total_s / (infer.sim_dt * infer.decimation))
    frames = []
    every = max(1, int(1.0 / (fps * infer.sim_dt * infer.decimation)))
    for i in range(n_total):
        epk.step_policy(infer)
        if i % every == 0:
            r.update_scene(d, camera=cam)
            frames.append(Image.fromarray(r.render()))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"saved {out} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=list(epk.VARIANTS.keys()))
    ap.add_argument("--magnitude", type=float, default=0.8)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = "/home/ahn/dcubot/DCUBOT/rl_training"
    infer, foot_names, _ = epk.VARIANTS[args.which](base)
    render_kick(infer, foot_names, args.magnitude, args.theta, args.out)


if __name__ == "__main__":
    main()
