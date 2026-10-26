"""Headless replay of a trained DCUBOT policy -> a few PNG stills + a contact
sheet (and an mp4 if ffmpeg is available). Uses the dcubot MjInfer (not the
duck one) so the obs/gait-phase computation matches what the policy was
trained with.

Usage:
    .venv/bin/python tools/record_dcubot_checkpoint.py -o <onnx> [--vx 0.08] [--duration 6] [--out_dir DIR]
"""
import argparse
import os

import mujoco
import numpy as np
from PIL import Image

from playground.dcubot.mujoco_infer import MjInfer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", required=True)
    ap.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    ap.add_argument("--vx", type=float, default=0.08, help="forward command, m/s (trained range 0.05-0.10)")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", default="/home/ahn/dcubot/발표자료_DCUBOT_레퍼런스/07_RL학습_결과")
    ap.add_argument("--n_stills", type=int, default=6)
    ap.add_argument("--still_window", type=float, nargs=2, default=None,
                     help="t_start t_end (s): take stills evenly from this slice instead of the whole clip")
    ap.add_argument("--distance", type=float, default=0.9)
    ap.add_argument("--sheet_name", default="checkpoint_gait_sheet")
    args = ap.parse_args()

    infer = MjInfer(args.model_path, args.onnx_model_path, standing=False)
    infer.debug_vis = False
    infer.commands = [args.vx, 0.0, 0.0, 0.0, 0.0, 0.0]

    renderer = mujoco.Renderer(infer.model, height=480, width=640)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.17]
    cam.distance = args.distance
    cam.azimuth = 100
    cam.elevation = -10

    n_steps = int(args.duration / infer.sim_dt)
    render_every = max(1, int(1.0 / (args.fps * infer.sim_dt)))
    frames = []
    root_z = []
    counter = 0
    for _ in range(n_steps):
        mujoco.mj_step(infer.model, infer.data)
        counter += 1
        if counter % infer.decimation == 0:
            obs = infer.get_obs(infer.data, infer.commands)
            action = infer.policy.infer(obs)
            infer.last_last_last_action = infer.last_last_action.copy()
            infer.last_last_action = infer.last_action.copy()
            infer.last_action = action.copy()
            infer.motor_targets = infer.default_actuator + action * infer.action_scale
            infer.motor_targets = np.clip(
                infer.motor_targets,
                infer.prev_motor_targets - infer.max_motor_velocity * (infer.sim_dt * infer.decimation),
                infer.prev_motor_targets + infer.max_motor_velocity * (infer.sim_dt * infer.decimation),
            )
            infer.prev_motor_targets = infer.motor_targets.copy()
            infer.data.ctrl = infer.motor_targets.copy()
        root_z.append(infer.data.qpos[2])
        if counter % render_every == 0:
            cam.lookat[0] = infer.data.qpos[0]  # follow the robot forward (+X)
            cam.lookat[1] = infer.data.qpos[1]
            renderer.update_scene(infer.data, camera=cam, scene_option=scene_option)
            frames.append(renderer.render())

    root_z = np.array(root_z)
    print(f"frames: {len(frames)}  root z min/mean/max: {root_z.min():.3f}/{root_z.mean():.3f}/{root_z.max():.3f} m")
    fell = root_z.min() < 0.10  # rough collapse threshold, home z ~0.19-0.2
    print("FELL (root z < 0.10 m at some point)" if fell else "stayed upright for the whole clip")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.still_window:
        lo = int(args.still_window[0] * args.fps)
        hi = int(args.still_window[1] * args.fps)
        idxs = np.linspace(lo, min(hi, len(frames) - 1), args.n_stills).astype(int)
    else:
        idxs = np.linspace(0, len(frames) - 1, args.n_stills).astype(int)
    stills = [frames[i] for i in idxs]
    for n, i in enumerate(idxs):
        t = i / args.fps
        Image.fromarray(stills[n]).save(f"{args.out_dir}/{args.sheet_name}_still_{n}_t{t:.2f}s.png")

    # contact sheet: stills side by side
    h, w, _ = stills[0].shape
    sheet = Image.new("RGB", (w * len(stills), h))
    for n, img in enumerate(stills):
        sheet.paste(Image.fromarray(img), (n * w, 0))
    sheet.save(f"{args.out_dir}/{args.sheet_name}.png")
    print(f"saved {len(stills)} stills + contact sheet to {args.out_dir}")

    try:
        import imageio_ffmpeg
        import mediapy as media
        # This venv has no system ffmpeg; point mediapy at the static binary
        # imageio-ffmpeg downloads instead (no sudo needed).
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        media.write_video(f"{args.out_dir}/{args.sheet_name}_replay.mp4", frames, fps=args.fps)
        print(f"saved {args.sheet_name}_replay.mp4")
    except Exception as e:
        print(f"(no mp4: {e})")


if __name__ == "__main__":
    main()
