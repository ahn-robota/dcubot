"""Replay a trained policy in the MuJoCo viewer, cycling through
forward -> backward -> strafe left -> strafe right -> turn left -> turn right,
SEGMENT_SECONDS each, back to back.

NOTE: the current joystick.py trains FORWARD-ONLY (lin_vel_y and
ang_vel_yaw command ranges are both [0.0, 0.0] -- see
docs/CHANGES_FROM_ORIGINAL.md #2, reference motion is forward-only). The
policy has never seen a nonzero backward/strafe/turn command during
training, so those segments are expected to look broken/undefined -- that's
not a bug in this script, it's exactly the current training scope. Forward
uses the trained range; the other five are illustrative magnitudes only.

Usage:
    .venv/bin/python scripts/demo_directions.py -o <path/to/policy.onnx>
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from playground.dcubot.mujoco_infer import MjInfer

SEGMENT_SECONDS = 3.0
# (label, [lin_x, lin_y, ang_vel])
SEQUENCE = [
    ("forward", [0.07, 0.0, 0.0]),
    ("backward", [-0.07, 0.0, 0.0]),
    ("strafe_left", [0.0, 0.07, 0.0]),
    ("strafe_right", [0.0, -0.07, 0.0]),
    ("turn_left", [0.0, 0.0, 0.3]),
    ("turn_right", [0.0, 0.0, -0.3]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/dcubot/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument(
        "--loop", action="store_true", help="repeat the sequence forever"
    )
    args = parser.parse_args()

    mjinfer = MjInfer(args.model_path, args.onnx_model_path, standing=False)

    with mujoco.viewer.launch_passive(
        mjinfer.model, mjinfer.data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        mjinfer.model.vis.scale.jointlength = 0.05
        mjinfer.model.vis.scale.jointwidth = 0.015

        counter = 0
        seq_start = time.time()
        seg_idx = -1
        total_segments = len(SEQUENCE)

        while True:
            elapsed = time.time() - seq_start
            idx = int(elapsed // SEGMENT_SECONDS)
            if idx >= total_segments:
                if args.loop:
                    seq_start = time.time()
                    idx = 0
                else:
                    break
            if idx != seg_idx:
                seg_idx = idx
                label = SEQUENCE[seg_idx][0]
                print(f"[{label}] {seg_idx * SEGMENT_SECONDS:.0f}-{(seg_idx + 1) * SEGMENT_SECONDS:.0f}s")

            _, cmd = SEQUENCE[seg_idx]
            mjinfer.commands[0] = cmd[0]
            mjinfer.commands[1] = cmd[1]
            mjinfer.commands[2] = cmd[2]

            step_start = time.time()
            mujoco.mj_step(mjinfer.model, mjinfer.data)
            counter += 1

            if counter % mjinfer.decimation == 0:
                obs = mjinfer.get_obs(mjinfer.data, mjinfer.commands)
                action = mjinfer.policy.infer(obs)

                mjinfer.last_last_last_action = mjinfer.last_last_action.copy()
                mjinfer.last_last_action = mjinfer.last_action.copy()
                mjinfer.last_action = action.copy()

                mjinfer.motor_targets = (
                    mjinfer.default_actuator + action * mjinfer.action_scale
                )
                mjinfer.motor_targets = np.clip(
                    mjinfer.motor_targets,
                    mjinfer.prev_motor_targets
                    - mjinfer.max_motor_velocity * (mjinfer.sim_dt * mjinfer.decimation),
                    mjinfer.prev_motor_targets
                    + mjinfer.max_motor_velocity * (mjinfer.sim_dt * mjinfer.decimation),
                )
                mjinfer.prev_motor_targets = mjinfer.motor_targets.copy()
                mjinfer.data.ctrl = mjinfer.motor_targets.copy()

            viewer.sync()
            time_until_next = mjinfer.model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    print("done")


if __name__ == "__main__":
    main()
