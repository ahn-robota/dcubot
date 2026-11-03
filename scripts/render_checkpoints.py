"""Offline, CPU-renderable video of N parallel policy rollouts composited
into one scene, labeled with the checkpoint's iteration/step number.

Standalone -- meant to be run from a separate terminal while training runs
elsewhere (e.g. on the lab PC's GPU). Does not hook into training at all
(no brax progress_fn / checkpoint-save callback); it is opt-in by nature --
you run it yourself, whenever you want, on whichever .onnx file you point it
at. This keeps training throughput completely unaffected.

Each of the N environments gets a different, randomly sampled command and a
different grid position (offset in X/Y) so they don't overlap visually. The
first environment's data drives the scene via mjv_updateScene; the rest are
added into the same scene via mjv_addGeoms (their world positions already
differ thanks to the grid offset, so no extra geom-shifting is needed).

For real CPU rendering (so it doesn't compete with an active GPU training
run), set MUJOCO_GL=osmesa before running this script. That needs
`libosmesa6-dev` installed system-wide (sudo apt install libosmesa6-dev) --
without it, MuJoCo falls back to whatever GPU-backed renderer is available.
Writing to .mp4 needs ffmpeg on PATH; without it, this script falls back to
saving a handful of .png frames (same fallback as mujoco_infer_headless.py).

Usage:
    MUJOCO_GL=osmesa uv run python scripts/render_checkpoints.py \
        checkpoints/<run>/<step>.onnx --num_envs 4 --duration 5 \
        --out checkpoints/<run>/<step>_rollout.mp4
    (add --lin_vel_x/--lin_vel_y/--ang_vel_yaw MIN MAX to sample commands from
    the range the policy was trained on)
"""

import argparse
import os
import re

import mediapy as media
import mujoco
import numpy as np

try:
    import imageio_ffmpeg

    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
except ImportError:
    pass  # falls back to a system ffmpeg on PATH, or the PNG-frame fallback below

from playground.dcubot.mujoco_infer import MjInfer, SymmetricPolicy
from playground.dcubot.joystick import default_config

GRID_SPACING = 1.0  # meters between adjacent envs in the composited scene


def _label_for(onnx_path: str) -> str:
    """"<iteration N / step N>" from a checkpoint filename like
    *_<step>.onnx (brax/this project's own naming, e.g. 2026_10_20_..._11468800.onnx)."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    m = re.search(r"(\d+)$", stem)
    step = m.group(1) if m else "?"
    return f"step {step}"


def _sample_command(rng: np.random.Generator, ranges: dict) -> list:
    vx = rng.uniform(*ranges["lin_vel_x"])
    vy = rng.uniform(*ranges["lin_vel_y"])
    vyaw = rng.uniform(*ranges["ang_vel_yaw"])
    # [lin_x, lin_y, ang, neck_pitch, head_pitch, head_yaw] -- head held neutral.
    return [vx, vy, vyaw, 0.0, 0.0, 0.0]


def make_envs(model_path: str, onnx_path: str, num_envs: int, seed: int, ranges: dict,
              symmetric_trained: bool = False):
    rng = np.random.default_rng(seed)
    envs = []
    grid_cols = int(np.ceil(np.sqrt(num_envs)))
    for i in range(num_envs):
        infer = MjInfer(model_path, onnx_path, standing=False)
        if symmetric_trained:
            infer.policy = SymmetricPolicy(infer.policy, pre_tanh=True)
        mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)
        row, col = divmod(i, grid_cols)
        infer.data.qpos[0] += col * GRID_SPACING
        infer.data.qpos[1] += row * GRID_SPACING
        mujoco.mj_forward(infer.model, infer.data)
        infer.commands = _sample_command(rng, ranges)
        envs.append(infer)
    return envs


def step_envs(envs):
    for infer in envs:
        mujoco.mj_step(infer.model, infer.data)


def maybe_act(envs, counter):
    if counter % envs[0].decimation != 0:
        return
    for infer in envs:
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


_PERT = mujoco.MjvPerturb()


def render_composited(envs, renderer, scene_option, camera, label: str):
    scn = renderer.scene
    mujoco.mjv_updateScene(
        envs[0].model, envs[0].data, scene_option, _PERT, camera, mujoco.mjtCatBit.mjCAT_ALL, scn
    )
    for infer in envs[1:]:
        mujoco.mjv_addGeoms(
            infer.model, infer.data, scene_option, _PERT, mujoco.mjtCatBit.mjCAT_ALL, scn
        )
    frame = renderer.render()
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame)
        ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 0))
        frame = np.array(img)
    except ImportError:
        pass  # label skipped if Pillow isn't installed; video content is unaffected
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model_path", type=str)
    parser.add_argument("--model_path", type=str, default="playground/dcubot/xmls/scene_flat_terrain.xml")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="checkpoint_rollout.mp4")
    parser.add_argument(
        "--symmetric_trained", action="store_true",
        help="policy trained with runner.py --symmetric_policy",
    )
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"):
        parser.add_argument(
            f"--{name}", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
            help=f"command range to sample {name} from (default: joystick.py's); "
            "match the range the policy was trained on",
        )
    args = parser.parse_args()

    config = default_config()
    ranges = {
        name: getattr(args, name) or list(getattr(config, name))
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
    }
    envs = make_envs(args.model_path, args.onnx_model_path, args.num_envs, args.seed, ranges,
                     args.symmetric_trained)
    label = _label_for(args.onnx_model_path)

    renderer = mujoco.Renderer(envs[0].model, height=args.height, width=args.width)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    # Free camera framing the whole env grid, not just env 0.
    grid_cols = int(np.ceil(np.sqrt(args.num_envs)))
    grid_rows = int(np.ceil(args.num_envs / grid_cols))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [
        (grid_cols - 1) * GRID_SPACING / 2,
        (grid_rows - 1) * GRID_SPACING / 2,
        0.2,
    ]
    camera.distance = max(grid_cols, grid_rows) * GRID_SPACING * 1.8 + 1.0
    camera.azimuth = 135
    camera.elevation = -25

    n_steps = int(args.duration / envs[0].sim_dt)
    render_every = max(1, int(1.0 / (args.fps * envs[0].sim_dt)))
    frames = []
    for counter in range(1, n_steps + 1):
        step_envs(envs)
        maybe_act(envs, counter)
        if counter % render_every == 0:
            frames.append(render_composited(envs, renderer, scene_option, camera, label))

    try:
        media.write_video(args.out, frames, fps=args.fps)
        print(f"Saved {len(frames)} frames to {args.out}")
    except RuntimeError:
        frame_dir = args.out.rsplit(".", 1)[0] + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
        step = max(1, len(frames) // 8)
        for i in range(0, len(frames), step):
            media.write_image(f"{frame_dir}/frame_{i:04d}.png", frames[i])
        print(f"ffmpeg not found; saved PNG frames to {frame_dir}")


if __name__ == "__main__":
    main()
