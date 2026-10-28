"""Does a policy actually follow velocity commands?

For each fixed command, run the policy for 8 s from the home pose (plain
MuJoCo through playground/dcubot/mujoco_infer.py's MjInfer: same observation
as training, no pushes, no noise) and report the mean body-frame velocity
after the first 2 s, plus whether the robot fell. Prints one summary line per
policy with a tracking score: mean over the moving commands of
achieved/commanded along the commanded axis (1.0 = perfect, ~0 = stands
still, negative = wrong way).

Usage (from rl_training/):
    .venv/bin/python scripts/eval_command_tracking.py <policy.onnx> [...]
    .venv/bin/python scripts/eval_command_tracking.py checkpoints/<run>/*.onnx
"""

import argparse
import sys

import mujoco
import numpy as np

from playground.dcubot.mujoco_infer import MjInfer, USE_MOTOR_SPEED_LIMITS

XML = "playground/dcubot/xmls/scene_flat_terrain.xml"
SECONDS = 8.0
SETTLE_SECONDS = 2.0
# (name, (vx, vy, wz), axis index the command is on, or None for stand)
COMMANDS = [
    ("stand", (0.0, 0.0, 0.0), None),
    ("fwd", (0.2, 0.0, 0.0), 0),
    ("back", (-0.15, 0.0, 0.0), 0),
    ("left", (0.0, 0.15, 0.0), 1),
    ("right", (0.0, -0.15, 0.0), 1),
    ("turn_l", (0.0, 0.0, 0.1), 2),
    ("turn_r", (0.0, 0.0, -0.1), 2),
]


def run(onnx_path, command):
    m = MjInfer(XML, onnx_path, False)
    m.commands[:3] = list(command)
    model, data = m.model, m.data
    imu = model.site("imu").id
    vel, fell_at = [], None
    for counter in range(1, int(SECONDS / m.sim_dt) + 1):
        mujoco.mj_step(model, data)
        if counter % m.decimation:
            continue
        action = m.policy.infer(m.get_obs(data, m.commands))
        m.last_last_last_action = m.last_last_action.copy()
        m.last_last_action = m.last_action.copy()
        m.last_action = action.copy()
        targets = m.default_actuator + action * m.action_scale
        if USE_MOTOR_SPEED_LIMITS:
            limit = m.max_motor_velocity * m.sim_dt * m.decimation
            targets = np.clip(
                targets, m.prev_motor_targets - limit, m.prev_motor_targets + limit
            )
            m.prev_motor_targets = targets.copy()
        m.motor_targets = targets
        data.ctrl = targets.copy()

        rot = data.site_xmat[imu].reshape(3, 3)
        if rot[2, 2] < 0.5:
            fell_at = counter * m.sim_dt
            break
        if counter * m.sim_dt > SETTLE_SECONDS:
            local_lin = rot.T @ data.qvel[:3]
            vel.append([local_lin[0], local_lin[1], data.qvel[5]])
    mean = np.mean(vel, axis=0) if vel else np.zeros(3)
    return mean, fell_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", nargs="+")
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="multiply every command, e.g. 0.4 for a policy trained on 0.05-0.1 m/s",
    )
    args = parser.parse_args()

    for onnx_path in args.onnx:
        ratios, falls, rows = [], 0, []
        for name, command, axis in COMMANDS:
            command = tuple(c * args.scale for c in command)
            mean, fell_at = run(onnx_path, command)
            falls += fell_at is not None
            if axis is not None:
                ratios.append(mean[axis] / command[axis] if fell_at is None else 0.0)
            rows.append(
                f"    {name:6s} cmd=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}) "
                f"vx={mean[0]:+.3f} vy={mean[1]:+.3f} wz={mean[2]:+.3f} "
                + (f"FELL@{fell_at:.1f}s" if fell_at else "ok")
            )
        print(
            f"{onnx_path}  tracking={np.mean(ratios):+.2f} "
            f"(fwd/back/left/right/turn_l/turn_r: {' '.join(f'{r:+.2f}' for r in ratios)})  "
            f"falls={falls}/{len(COMMANDS)}"
        )
        print("\n".join(rows))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
