"""DCUBOT-specific reward helpers.

The imitation reward follows the duck's reward_imitation, but the reference
layout and the joint slicing are DCUBOT's own:
  - the reference frame is the 28-dim layout written by fit_poly.py:
    joint_pos[0:13], joint_vel[13:26], foot_contacts[26:28]
  - the Placo recording orders joints as left leg(5), head(3), right leg(5),
    while the MuJoCo actuator order is left leg(5), right leg(5), head(3),
    so the leg slices are taken explicitly.
"""

import jax.numpy as jp


def reward_imitation_dcubot(
    joints_qpos, joints_qvel, contacts, reference_frame, cmd,
    base_local_linvel=None, base_local_angvel=None,
):
    """Bounded imitation reward for the 10 leg joints and the foot contacts.

    Each term is exp(-error), so the reward stays in [0, 2] and never goes
    hugely negative. (The earlier squared-error form reached -6e4 per step
    while the robot was falling, and the env clips total reward at 0, which
    removed the learning signal.)
    Returns 0 when the command is (almost) zero, since the reference is a walk.

    With base_local_linvel/base_local_angvel given, upstream's base-velocity
    terms are added (same forms and weights as upstream reward_imitation:
    lin xy exp(-8 err^2) x1, lin z x1, ang xy exp(-2 err^2) x0.5, ang z x0.5)
    against the reference's base_linear_vel[28:31]/base_angular_vel[31:34].
    These are what pay for actually moving at the reference's speed.
    """
    # Placo recording order: left leg [0:5], head [5:8], right leg [8:13].
    ref_joint_pos = jp.concatenate([reference_frame[0:5], reference_frame[8:13]])
    ref_joint_vel = jp.concatenate([reference_frame[13:18], reference_frame[21:26]])
    ref_contacts = jp.where(
        reference_frame[26:28] > 0.5,
        jp.ones_like(reference_frame[26:28]),
        jp.zeros_like(reference_frame[26:28]),
    )

    # MuJoCo actuator order: left leg [0:5], right leg [5:10], head [10:13].
    joint_pos = joints_qpos[:10]
    joint_vel = joints_qvel[:10]

    pos_err = jp.sum(jp.square(joint_pos - ref_joint_pos))
    vel_err = jp.sum(jp.square(joint_vel - ref_joint_vel))
    # Kernel 1.0 (was 0.1): with 0.1, joints just 0.2 rad off the reference
    # scored exp(-4) ~ 0.02 -- no gradient toward the gait (auto exp04).
    joint_pos_rew = jp.exp(-pos_err / 1.0)
    joint_vel_rew = jp.exp(-0.1 * vel_err)
    contact_rew = jp.mean((contacts == ref_contacts).astype(jp.float32))

    move_cmd_norm = jp.linalg.norm(cmd[:3])
    reward = 0.5 * joint_pos_rew + 0.25 * joint_vel_rew + 0.25 * contact_rew

    if base_local_linvel is not None:
        ref_lin, ref_ang = reference_frame[28:31], reference_frame[31:34]
        reward += (
            jp.exp(-8.0 * jp.sum(jp.square(base_local_linvel[:2] - ref_lin[:2])))
            + jp.exp(-8.0 * jp.square(base_local_linvel[2] - ref_lin[2]))
            + 0.5 * jp.exp(-2.0 * jp.sum(jp.square(base_local_angvel[:2] - ref_ang[:2])))
            + 0.5 * jp.exp(-2.0 * jp.square(base_local_angvel[2] - ref_ang[2]))
        )
    return jp.nan_to_num(reward * (move_cmd_norm > 0.01))


def reward_feet_gait(
    gait_step, dt, period, contact, command,
    offsets=(0.75, 0.25), stance_threshold=0.56, command_threshold=0.02,
):
    """Reward foot contact that matches an alternating stance/swing schedule
    (left/right half a cycle apart), as in Unitree's mjlab velocity task.

    Default offsets put the schedule in phase with the omnidirectional
    reference's own foot contacts (95% of gait steps agree; offsets (0, 0.5)
    agree on only 55%), so it does not fight the imitation contact term.
    """
    global_phase = gait_step * dt / period
    leg_phase = (global_phase + jp.array(offsets)) % 1.0
    is_stance = leg_phase < stance_threshold
    match = (is_stance == contact).astype(jp.float32)
    active = jp.linalg.norm(command[:3]) > command_threshold
    return jp.mean(match) * active


def cost_feet_clearance_dcubot(
    foot_z, foot_vel, target_z, command, command_threshold=0.02
):
    """Penalize swing-foot height deviating from target_z, weighted by the
    foot's horizontal speed (planted feet are not penalized). Only while a
    move command is given."""
    vel_norm = jp.linalg.norm(foot_vel[..., :2], axis=-1)
    cost = jp.sum(jp.abs(foot_z - target_z) * vel_norm)
    active = jp.linalg.norm(command[:3]) > command_threshold
    return jp.nan_to_num(cost * active)
