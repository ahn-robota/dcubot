"""Left/right mirror symmetry for DCUBOT's policy.

Observation (94, see MjInfer.get_obs / Joystick._get_obs "state") and action
(13) mirrors are a fixed permutation plus sign flip, so they work the same on
numpy and jax arrays and on batches: `x[..., PERM] * SIGN`.

Joint order (actuators): left leg 5, right leg 5, neck_pitch, head_pitch,
head_yaw. The MJCF's right-leg joint axes are the mirror of the left ones, so
a mirrored joint angle keeps its sign; only head_yaw flips.

`make_symmetric_ppo_networks` builds brax PPO networks whose policy is
pi(o) = (f(o) + M f(M o)) / 2 on the pre-tanh mean (and the swapped std), so
the trained policy is exactly symmetric. The exported ONNX holds f only;
`SymmetricPolicy(..., pre_tanh=True)` in mujoco_infer.py rebuilds pi from it.
"""

import dataclasses

import numpy as np

JOINT_PERM = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10, 11, 12])
JOINT_SIGN = np.array([1.0] * 12 + [-1.0])


def _obs_mirror():
    perm, sign = [], []

    def add(idx, sgn):
        perm.extend(idx)
        sign.extend(sgn)

    add([0, 1, 2], [-1, 1, -1])  # gyro (angular velocity: pseudovector)
    add([3, 4, 5], [1, -1, 1])  # accelerometer
    add([6, 7, 8, 9, 10, 11], [1, -1, -1, 1, 1, -1])  # vx, vy, yaw, neck, head pitch, head yaw
    for start in range(12, 90, 13):  # angles, velocities, 3 past actions, motor targets
        add(list(start + JOINT_PERM), list(JOINT_SIGN))
    add([91, 90], [1, 1])  # foot contacts swap
    add([92, 93], [-1, -1])  # gait phase shifted by half a cycle (other foot leads)
    return np.array(perm), np.array(sign, dtype=np.float32)


OBS_PERM, OBS_SIGN = _obs_mirror()


def mirror_obs(obs):
    return obs[..., OBS_PERM] * OBS_SIGN


def mirror_joints(x):
    return x[..., JOINT_PERM] * JOINT_SIGN


def make_symmetric_ppo_networks(*args, **kwargs):
    """brax make_ppo_networks, with the policy symmetrized (see module doc)."""
    import jax.numpy as jp
    from brax.training import networks
    from brax.training.agents.ppo import networks as ppo_networks

    nets = ppo_networks.make_ppo_networks(*args, **kwargs)
    base = nets.policy_network
    obs_key = kwargs.get("policy_obs_key", "state")

    def apply(processor_params, policy_params, obs):
        if isinstance(obs, dict):
            mirrored = dict(obs)
            mirrored[obs_key] = mirror_obs(obs[obs_key])
        else:
            mirrored = mirror_obs(obs)
        out = base.apply(processor_params, policy_params, obs)
        out_m = base.apply(processor_params, policy_params, mirrored)
        loc, scale = jp.split(out, 2, axis=-1)
        loc_m, scale_m = jp.split(out_m, 2, axis=-1)
        loc = 0.5 * (loc + mirror_joints(loc_m))
        scale = 0.5 * (scale + scale_m[..., JOINT_PERM])
        return jp.concatenate([loc, scale], axis=-1)

    policy = networks.FeedForwardNetwork(init=base.init, apply=apply)
    return dataclasses.replace(nets, policy_network=policy)
