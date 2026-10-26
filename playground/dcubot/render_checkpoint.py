"""Headless render of a trained dcubot policy to a GIF (no ffmpeg / no display).

Runs the same deploy loop as mujoco_infer.py (incl. the gait-phase obs) with a
fixed command and records frames offscreen. Software rendering keeps it off the
GPU so it can run while training is using it:

  MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
    python -m playground.dcubot.render_checkpoint -o <policy.onnx> --vx 0.15
"""
import argparse

import mujoco
import numpy as np
from PIL import Image

from playground.dcubot.mujoco_infer import MjInfer
from playground.dcubot import debug_vis


def render(onnx, vx, vy, wz, duration, out, model_path, fps=20, size=(360, 270)):
    inf = MjInfer(model_path, onnx, False)
    inf.commands = [vx, vy, wz, 0.0, 0.0, 0.0]
    m, d = inf.model, inf.data
    torso = m.body("base_link").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link") >= 0 else 1
    m.vis.global_.offwidth, m.vis.global_.offheight = size
    r = mujoco.Renderer(m, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = torso
    cam.distance, cam.azimuth, cam.elevation = 1.3, 125, -22
    frames, cnt = [], 0
    every = max(1, int(1.0 / (fps * inf.sim_dt)))
    for _ in range(int(duration / inf.sim_dt)):
        mujoco.mj_step(m, d)
        cnt += 1
        if cnt % inf.decimation == 0:
            obs = inf.get_obs(d, inf.commands)
            a = inf.policy.infer(obs)
            inf.last_last_last_action = inf.last_last_action.copy()
            inf.last_last_action = inf.last_action.copy()
            inf.last_action = a.copy()
            inf.motor_targets = inf.default_actuator + a * inf.action_scale
            d.ctrl = inf.motor_targets.copy()
        if cnt % every == 0:
            r.update_scene(d, camera=cam)
            debug_vis.draw(r.scene, d, inf._imu_id, inf.commands, inf.get_gyro(d)[2])
            frames.append(Image.fromarray(r.render()))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    x = d.qpos[0]
    print(f"saved {out} ({len(frames)} frames), base x={x:.2f}m")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--onnx_model_path", required=True)
    p.add_argument("--vx", type=float, default=0.15)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--output", default="replay.gif")
    p.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    a = p.parse_args()
    render(a.onnx_model_path, a.vx, a.vy, a.wz, a.duration, a.output, a.model_path)
