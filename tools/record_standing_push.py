"""Record a push-and-recover clip for a DCUBOT standing checkpoint: settle,
single push (velocity kick, same mechanism as training), then watch it
recover (or fall). Saves an mp4 plus a root-height/tilt plot over time.

Usage:
    .venv/bin/python tools/record_standing_push.py -o <ckpt.onnx> --push_mag 0.8 --push_dir 0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mujoco
import numpy as np

from playground.dcubot.mujoco_infer_standing import StandingInfer

for _p in fm.findSystemFonts(fontpaths=["/usr/share/fonts/opentype/noto"]):
    if "NotoSansCJK-Regular" in _p:
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", required=True)
    ap.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    ap.add_argument("--push_mag", type=float, default=0.8, help="push velocity kick, m/s")
    ap.add_argument("--push_dir_deg", type=float, default=0.0, help="push direction, deg (0=+X)")
    ap.add_argument("--settle_s", type=float, default=1.5)
    ap.add_argument("--after_s", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", default="/home/ahn/dcubot/발표자료_DCUBOT_레퍼런스/07_RL학습_결과")
    ap.add_argument("--tag", default="standing_push_demo")
    args = ap.parse_args()

    infer = StandingInfer(args.model_path, args.onnx_model_path)
    mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)

    renderer = mujoco.Renderer(infer.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.17]
    cam.distance = 0.9
    cam.azimuth = 100
    cam.elevation = -10

    frames, root_z, tilt_deg, times = [], [], [], []
    render_every = max(1, int(1.0 / (args.fps * infer.sim_dt)))
    counter = 0
    pushed = False
    total_s = args.settle_s + args.after_s
    n_steps = int(total_s / infer.sim_dt)
    push_step = int(args.settle_s / infer.sim_dt)
    theta = np.radians(args.push_dir_deg)

    for i in range(n_steps):
        if i == push_step:
            infer.data.qvel[0] += args.push_mag * np.cos(theta)
            infer.data.qvel[1] += args.push_mag * np.sin(theta)
            pushed = True
        mujoco.mj_step(infer.model, infer.data)
        counter += 1
        if counter % infer.decimation == 0:
            obs = infer.get_obs(infer.data, infer.commands)
            action = infer.policy.infer(obs)
            infer.last_last_last_action = infer.last_last_action.copy()
            infer.last_last_action = infer.last_action.copy()
            infer.last_action = action.copy()
            motor_targets = infer.default_actuator + action * infer.action_scale
            infer.data.ctrl = motor_targets.copy()
        g = infer.get_gravity(infer.data)
        tilt = np.degrees(np.arccos(np.clip(g[2], -1, 1)))  # upvector sensor: [0,0,1] upright -> 0 deg
        root_z.append(infer.data.qpos[2])
        tilt_deg.append(tilt)
        times.append(i * infer.sim_dt)
        if counter % render_every == 0:
            cam.lookat[0] = infer.data.qpos[0]
            cam.lookat[1] = infer.data.qpos[1]
            renderer.update_scene(infer.data, camera=cam)
            frames.append(renderer.render())

    fell = bool(infer.get_gravity(infer.data)[-1] < 0.0)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"push {args.push_mag} m/s @ {args.push_dir_deg} deg -> {'FELL' if fell else 'survived'}, "
          f"final root z={root_z[-1]:.3f} m, final tilt={tilt_deg[-1]:.1f} deg")

    # mp4
    try:
        import imageio_ffmpeg
        import mediapy as media
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        media.write_video(f"{args.out_dir}/{args.tag}.mp4", frames, fps=args.fps)
        print(f"saved {args.tag}.mp4")
    except Exception as e:
        print(f"(no mp4: {e})")

    # root height + tilt over time, with push marked
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), dpi=150, sharex=True)
    axes[0].plot(times, root_z, color="#1f6feb")
    axes[0].axvline(args.settle_s, color="#d1242f", ls="--", label=f"push {args.push_mag} m/s")
    axes[0].set_ylabel("몸통 높이 (m)"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(times, tilt_deg, color="#2da44e")
    axes[1].axvline(args.settle_s, color="#d1242f", ls="--")
    axes[1].set_ylabel("기울기 (deg, 0=직립)"); axes[1].set_xlabel("시간 (s)"); axes[1].grid(alpha=0.3)
    fig.suptitle(f"standing 푸시 테스트 — {args.push_mag} m/s, 결과: {'쓰러짐' if fell else '버팀'}")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/{args.tag}_trace.png")
    print(f"saved {args.tag}_trace.png")


if __name__ == "__main__":
    main()
