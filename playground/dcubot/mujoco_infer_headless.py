"""Headless (EGL offscreen) replay of a trained policy, saved as mp4.

Same physics/policy loop as mujoco_infer.py, but renders offscreen instead of
opening an interactive mujoco.viewer window, so it can run without a display.
"""

import argparse

import mediapy as media
import mujoco
import numpy as np

from playground.open_duck_mini_v2.mujoco_infer import MjInfer


def run_headless(
    infer: MjInfer,
    duration_s: float,
    command: list,
    output_path: str,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
):
    infer.commands = list(command)

    renderer = mujoco.Renderer(infer.model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    infer.model.vis.scale.jointlength = 0.05
    infer.model.vis.scale.jointwidth = 0.015
    frames = []
    counter = 0
    render_every = max(1, int(1.0 / (fps * infer.sim_dt)))

    n_steps = int(duration_s / infer.sim_dt)
    for _ in range(n_steps):
        mujoco.mj_step(infer.model, infer.data)
        counter += 1

        if counter % infer.decimation == 0:
            if not infer.standing:
                infer.imitation_i += 1.0 * infer.phase_frequency_factor
                infer.imitation_i = infer.imitation_i % infer.PRM.nb_steps_in_period
                infer.imitation_phase = np.array(
                    [
                        np.cos(infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi),
                        np.sin(infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi),
                    ]
                )
            obs = infer.get_obs(infer.data, infer.commands)
            action = infer.policy.infer(obs)

            infer.last_last_last_action = infer.last_last_action.copy()
            infer.last_last_action = infer.last_action.copy()
            infer.last_action = action.copy()

            infer.motor_targets = infer.default_actuator + action * infer.action_scale
            infer.motor_targets = np.clip(
                infer.motor_targets,
                infer.prev_motor_targets
                - infer.max_motor_velocity * (infer.sim_dt * infer.decimation),
                infer.prev_motor_targets
                + infer.max_motor_velocity * (infer.sim_dt * infer.decimation),
            )
            infer.prev_motor_targets = infer.motor_targets.copy()
            infer.data.ctrl = infer.motor_targets.copy()

        if counter % render_every == 0:
            renderer.update_scene(
                infer.data,
                camera="track" if _has_camera(infer.model, "track") else -1,
                scene_option=scene_option,
            )
            frames.append(renderer.render())

    try:
        media.write_video(output_path, frames, fps=fps)
        print(f"Saved {len(frames)} frames to {output_path}")
    except RuntimeError:
        import os

        frame_dir = output_path.rsplit(".", 1)[0] + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
        step = max(1, len(frames) // 8)
        for i in range(0, len(frames), step):
            media.write_image(f"{frame_dir}/frame_{i:04d}.png", frames[i])
        print(f"ffmpeg not found; saved {len(range(0, len(frames), step))} PNG frames to {frame_dir}")


def _has_camera(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--vx", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="replay.mp4")
    args = parser.parse_args()

    infer = MjInfer(args.model_path, args.reference_data, args.onnx_model_path, args.standing)
    run_headless(
        infer,
        duration_s=args.duration,
        command=[args.vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        output_path=args.output,
    )
