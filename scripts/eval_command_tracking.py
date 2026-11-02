"""Does a policy actually follow velocity commands?

For each fixed command, run the policy for 8 s from the home pose (plain
MuJoCo through playground/dcubot/mujoco_infer.py's MjInfer: same observation
as training, no pushes, no noise) and report the mean body-frame velocity
after the first 2 s, plus whether the robot fell. Prints one summary line per
policy with a tracking score: mean over the moving commands of
achieved/commanded along the commanded axis (1.0 = perfect, ~0 = stands
still, negative = wrong way).

With --env N it instead steps N robots in parallel through the training env
itself (playground/dcubot/joystick.py, task flat_terrain_train: observation
noise, action/IMU delays and pushes on, nominal model), each from a different
random seed, and averages over them -- far less noisy than one deterministic
run, at ~1 min per checkpoint. A robot counts as fallen once the env
terminates it.

Usage (from rl_training/):
    .venv/bin/python scripts/eval_command_tracking.py <policy.onnx> [...]
    .venv/bin/python scripts/eval_command_tracking.py checkpoints/<run>/*.onnx
    .venv/bin/python scripts/eval_command_tracking.py --env 256 <policy.onnx>
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


def run_env(onnx_path, command, n_envs, env_fns={}):
    """Mean velocity over the robots that stayed up, and the fallen fraction."""
    import jax
    import jax.numpy as jp
    from playground.common.onnx_infer import OnnxInfer

    if not env_fns:
        from playground.dcubot.joystick import Joystick

        env = Joystick(task="flat_terrain_train")
        env_fns.update(
            env=env,
            reset=jax.jit(jax.vmap(env.reset)),
            step=jax.jit(jax.vmap(env.step)),
            vel=jax.jit(jax.vmap(lambda d: jp.concatenate(
                [env.get_local_linvel(d)[:2], env.get_gyro(d)[2:]]))),
        )
    policy = OnnxInfer(onnx_path, awd=True)
    cmd = jp.tile(jp.array([*command, 0.0, 0.0, 0.0]), (n_envs, 1))
    state = env_fns["reset"](jax.random.split(jax.random.PRNGKey(0), n_envs))
    state.info["command"] = cmd
    fallen = np.zeros(n_envs, bool)
    vel_sum, vel_n = np.zeros((n_envs, 3)), np.zeros(n_envs)
    dt = env_fns["env"].dt
    for i in range(int(SECONDS / dt)):
        obs = np.asarray(state.obs["state"])
        action = np.stack([policy.infer(o) for o in obs])
        state = env_fns["step"](state, jp.asarray(action))
        state.info["command"] = cmd
        fallen |= np.asarray(state.done) > 0
        if (i + 1) * dt > SETTLE_SECONDS:
            up = ~fallen
            vel_sum[up] += np.asarray(env_fns["vel"](state.data))[up]
            vel_n[up] += 1
    ok = vel_n > 0
    mean = (vel_sum[ok] / vel_n[ok, None]).mean(axis=0) if ok.any() else np.zeros(3)
    return mean, fallen.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", nargs="+")
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="multiply every command, e.g. 0.4 for a policy trained on 0.05-0.1 m/s",
    )
    parser.add_argument(
        "--turn", type=float, default=None,
        help="turn command magnitude in rad/s, applied after --scale (a scaled-down "
        "turn command can be smaller than the robot's own yaw drift)",
    )
    parser.add_argument(
        "--env", type=int, default=0, metavar="N",
        help="average N robots in the training env instead of one MjInfer run",
    )
    args = parser.parse_args()

    for onnx_path in args.onnx:
        ratios, falls, rows = [], 0, []
        for name, command, axis in COMMANDS:
            command = tuple(c * args.scale for c in command)
            if axis == 2 and args.turn is not None:
                command = (0.0, 0.0, np.sign(command[2]) * args.turn)
            if args.env:
                mean, fell_frac = run_env(onnx_path, command, args.env)
                falls += fell_frac
                status = f"fell {100 * fell_frac:.0f}%"
                fell = fell_frac >= 0.5
            else:
                mean, fell_at = run(onnx_path, command)
                falls += fell_at is not None
                status = f"FELL@{fell_at:.1f}s" if fell_at else "ok"
                fell = fell_at is not None
            if axis is not None:
                ratios.append(mean[axis] / command[axis] if not fell else 0.0)
            rows.append(
                f"    {name:6s} cmd=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}) "
                f"vx={mean[0]:+.3f} vy={mean[1]:+.3f} wz={mean[2]:+.3f} {status}"
            )
        print(
            f"{onnx_path}  tracking={np.mean(ratios):+.2f} "
            f"(fwd/back/left/right/turn_l/turn_r: {' '.join(f'{r:+.2f}' for r in ratios)})  "
            f"falls={falls:.2f}/{len(COMMANDS)}"
        )
        print("\n".join(rows))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
