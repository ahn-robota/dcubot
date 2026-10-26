"""DCUBOT-specific reward helpers.

The imitation reward follows the duck's reward_imitation, but the reference
layout and the joint slicing are DCUBOT's own:
  - the reference frame is the 34-dim layout written by fit_poly.py:
    joint_pos[0:13], joint_vel[13:26], foot_contacts[26:28]
    (base velocities are left out: Placo's fbase velocity is not reliable)
  - the Placo recording orders joints as left leg(5), head(3), right leg(5),
    while the MuJoCo actuator order is left leg(5), right leg(5), head(3),
    so the leg slices are taken explicitly.
"""

import jax.numpy as jp


def cost_stand_still_dcubot(
    local_linvel,
    local_angvel,
    qpos,
    qvel,
    default_pose,
    lin_vel_threshold: float = 0.3,
    ang_vel_threshold: float = 1.0,
    reduction_factor: float = 0.2,
    ignore_head: bool = False,
):
    """Penalize drifting from the default pose/zero velocity -- but only while
    actually at rest. Gated on *measured* base velocity (unitree_rl_lab's
    joint_position_penalty/stand_still_scale pattern), not on command size:
    standing's move command is always [0,0,0] (it's a stand-still task), so
    command-gating was permanently on and penalized every recovery attempt as
    hard as idle drift. Confirmed by tools/diagnose_reward_terms.py: during a
    push recovery (tilt > 35deg), this cost outweighed reward_recovery_dcubot
    on 96% of steps (net -0.646/step average) -- it was fighting the exact
    behavior we want. Gating on real velocity turns the penalty down whenever
    something (a push, a recovery attempt) is actually moving the robot,
    regardless of what was commanded.
    """
    if not ignore_head:
        pose_cost = jp.sum(jp.abs(qpos - default_pose))
        vel_cost = jp.sum(jp.abs(qvel))
    else:
        left_leg_pos = qpos[:5]
        right_leg_pos = qpos[8:]
        left_leg_vel = qvel[:5]
        right_leg_vel = qvel[8:]
        left_leg_default = default_pose[:5]
        right_leg_default = default_pose[8:]
        pose_cost = jp.sum(jp.abs(left_leg_pos - left_leg_default)) + jp.sum(
            jp.abs(right_leg_pos - right_leg_default)
        )
        vel_cost = jp.sum(jp.abs(left_leg_vel)) + jp.sum(jp.abs(right_leg_vel))

    moving = (jp.linalg.norm(local_linvel) > lin_vel_threshold) | (
        jp.linalg.norm(local_angvel) > ang_vel_threshold
    )
    gate = jp.where(moving, reduction_factor, 1.0)
    return jp.nan_to_num(pose_cost + vel_cost) * gate


def cost_head_pos_dcubot(joints_qpos, joints_qvel, cmd, gate_on_move: bool = True):
    """Penalize head-joint deviation from the commanded head pose (cmd[3:]).

    gate_on_move=True (default, joystick.py's walking-mode use): only active
    while there's a nonzero body-movement command (cmd[:3]) -- the original
    behavior. gate_on_move=False (joystick.py's standing-mode dispatch after
    the standing/joystick merge): skip that gate. Standing-mode commands have
    cmd[:3]==0 by construction, so the move-based gate would always zero this
    out -- this is exactly what made the term dead weight in standing.py
    before the merge (reward_config.scales.head_pos=-2.0 there, but cmd[:3]
    is hardcoded to 0 in standing.py's own sample_command, so this never
    fired despite the nonzero scale).
    """
    del joints_qvel
    move_cmd_norm = jp.linalg.norm(cmd[:3])
    head_cmd = cmd[3:]
    head_pos = joints_qpos[5:8]
    head_pos_error = jp.sum(jp.square(head_pos - head_cmd))
    gate = (move_cmd_norm > 0.01) if gate_on_move else 1.0
    return jp.nan_to_num(head_pos_error) * gate


def reward_recovery_settle_dcubot(gravity, prev_gravity, tilt_threshold_deg: float = 35.0):
    """Bonus for actively reducing tilt once significantly tilted (e.g. after
    a push). Gated on tilt ONLY, never on height/pose -- gating on height lets
    a policy farm the bonus by crouching into a low "safe" pose instead of
    actually recovering balance (seen in pollen-robotics/microduck_rl).
    `gravity`/`prev_gravity` are the "upvector" sensor reading ([0,0,1] upright,
    tilts toward [0,0,-1] as the robot falls; same signal cost_orientation uses).

    This is the "settle" half of reward_recovery_blend_dcubot below (posture
    convergence once the fall has slowed down); kept standalone too since
    tools/diagnose_reward_terms.py and standing.py's USE_RECOVERY_BLEND=False
    path both call it directly under its original name.
    """
    tilt = jp.arccos(jp.clip(gravity[-1], -1.0, 1.0))
    prev_tilt = jp.arccos(jp.clip(prev_gravity[-1], -1.0, 1.0))
    improvement = jp.clip(prev_tilt - tilt, 0.0, None)  # only reward decreasing tilt
    gated = improvement * (tilt > jp.radians(tilt_threshold_deg))
    return jp.nan_to_num(gated)


# Kept as the pre-existing name: tools/diagnose_reward_terms.py imports this
# directly, and standing.py's USE_RECOVERY_BLEND=False path uses it unchanged.
reward_recovery_dcubot = reward_recovery_settle_dcubot


def reward_recovery_impact_dcubot(
    gyro, prev_gyro, linvel_z, prev_linvel_z, tilt, tilt_threshold_deg: float = 35.0
):
    """Early-fall term, Disney 'Robot Crash Course' style (arXiv:2511.10635):
    while still tilted past the threshold, reward damping rotational speed
    and vertical fall speed -- i.e. a soft landing -- rather than posture.
    This is deliberately indifferent to orientation: a policy that slows its
    tumble is rewarded even if it ends up on its side, which is the point
    (impact minimization first, right the posture afterwards). Gated on the
    same tilt condition as reward_recovery_settle_dcubot so the two terms
    are only ever both active during an actual fall/recovery episode.
    """
    speed = jp.linalg.norm(gyro) + jp.abs(linvel_z)
    prev_speed = jp.linalg.norm(prev_gyro) + jp.abs(prev_linvel_z)
    improvement = jp.clip(prev_speed - speed, 0.0, None)
    gated = improvement * (tilt > jp.radians(tilt_threshold_deg))
    return jp.nan_to_num(gated)


def _cubic_smoothstep(t):
    """3t^2 - 2t^3 on [0, 1]: the standard cubic (Hermite) smoothstep spline,
    zero slope at both ends so the blend doesn't kink reward gradients at the
    moment the fall starts or at the end of the blend window."""
    t = jp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def reward_recovery_blend_dcubot(
    gravity,
    prev_gravity,
    gyro,
    prev_gyro,
    linvel_z,
    prev_linvel_z,
    fall_elapsed,
    tilt_threshold_deg: float = 35.0,
    blend_duration: float = 0.5,
):
    """Blend reward_recovery_impact_dcubot (early: minimize impact) and
    reward_recovery_settle_dcubot (late: converge to a safe/upright posture)
    with a cubic-spline weight u(t) = _cubic_smoothstep(fall_elapsed /
    blend_duration): u=0 right as the tilt threshold is crossed (prioritize
    reducing fall speed), u=1 once `blend_duration` seconds into the fall
    (prioritize reducing tilt back toward upright).

    Idea from Disney Research's RSS 2024 "Design and Control of a Bipedal
    Robotic Character" and 2025 "Robot Crash Course: Learning Soft and
    Stylized Falling" (arXiv:2511.10635) -- their hardware-specific pieces
    (actuator system identification, motion-capture fall data) are not used
    here, only the two-phase reward-shaping idea.
    """
    tilt = jp.arccos(jp.clip(gravity[-1], -1.0, 1.0))
    u = _cubic_smoothstep(fall_elapsed / blend_duration)
    impact_term = reward_recovery_impact_dcubot(
        gyro, prev_gyro, linvel_z, prev_linvel_z, tilt, tilt_threshold_deg
    )
    settle_term = reward_recovery_settle_dcubot(
        gravity, prev_gravity, tilt_threshold_deg
    )
    return jp.nan_to_num((1.0 - u) * impact_term + u * settle_term)


# Per-body-part weights for cost_contact_impact_dcubot, adapted from Disney's
# head > shoulder > elbow > pelvis/legs ordering in "Robot Crash Course" --
# DCUBOT has no arms (no shoulder/elbow bodies exist in dcubot.xml), so the
# ordering is re-mapped onto the bodies DCUBOT actually has: head/neck stays
# the most sensitive tier, base/base_link (the only "upper body" mass) stands
# in for the missing shoulder tier, then hip/thigh, then shin. Feet are
# intentionally left out (weight 0): normal gait foot-ground contact is not
# a fall "impact" and must not be penalized by this term.
CONTACT_IMPACT_BODY_WEIGHTS = {
    "head_1": 5.0,
    "head_mount_1": 5.0,
    "neck_1": 5.0,
    "base": 2.0,
    "base_link": 2.0,
    "hip_yaw_L_1": 0.5,
    "hip_yaw_R_1": 0.5,
    "hip_roll_L_1": 0.5,
    "hip_roll_R_1": 0.5,
    "thigh_L_1": 0.5,
    "thigh_R_1": 0.5,
    "shin_L_1": 0.2,
    "shin_R_1": 0.2,
}


def cost_contact_impact_dcubot(cfrc_ext, body_ids, body_weights):
    """Weighted sum of external-contact force magnitude per body, using
    CONTACT_IMPACT_BODY_WEIGHTS above. `cfrc_ext` is data._impl.cfrc_ext
    (shape [nbody, 6], force in [:, :3]); `body_ids`/`body_weights` are the
    precomputed arrays built once in Standing._post_init from that dict.
    """
    forces = jp.linalg.norm(cfrc_ext[body_ids, :3], axis=-1)
    return jp.nan_to_num(jp.sum(forces * body_weights))


def gait_phase_obs(gait_step, dt, period, command, command_threshold=0.02):
    """[sin, cos] of the gait clock; zeroed for (near-)zero commands so the
    policy can tell 'stand' from 'walk'."""
    phase = (gait_step * dt / period) % 1.0
    obs = jp.array([jp.sin(2 * jp.pi * phase), jp.cos(2 * jp.pi * phase)])
    active = jp.linalg.norm(command[:3]) > command_threshold
    return jp.where(active, obs, jp.zeros(2))


def reward_feet_gait(
    gait_step, dt, period, contact, command,
    offsets=(0.0, 0.5), stance_threshold=0.56, command_threshold=0.02,
):
    """Reward foot contact that matches an alternating stance/swing schedule
    (left/right half a cycle apart), as in Unitree's mjlab velocity task."""
    global_phase = (gait_step * dt / period)
    leg_phase = (global_phase + jp.array(offsets)) % 1.0
    is_stance = leg_phase < stance_threshold
    match = (is_stance == contact).astype(jp.float32)
    active = jp.linalg.norm(command[:3]) > command_threshold
    return jp.mean(match) * active


def cost_feet_clearance(foot_z, foot_vel, target_z, command, command_threshold=0.02):
    """Penalize swing-foot height deviating from target, weighted by foot
    horizontal speed (so planted feet are not penalized)."""
    vel_norm = jp.linalg.norm(foot_vel[..., :2], axis=-1)
    cost = jp.sum(jp.abs(foot_z - target_z) * vel_norm)
    active = jp.linalg.norm(command[:3]) > command_threshold
    return jp.nan_to_num(cost * active)


def reward_imitation_dcubot(
    joints_qpos, joints_qvel, contacts, reference_frame, cmd
):
    """Bounded imitation reward for the 10 leg joints and the foot contacts.

    Each term is exp(-error), so the reward stays in [0, 2] and never goes
    hugely negative. (The earlier squared-error form reached -6e4 per step
    while the robot was falling, and the env clips total reward at 0, which
    removed the learning signal.)
    Returns 0 when the command is (almost) zero, since the reference is a walk.
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
    joint_pos_rew = jp.exp(-pos_err / 0.1)  # ~0.37 at 0.1 rad RMS-ish error over 10 joints
    joint_vel_rew = jp.exp(-0.1 * vel_err)
    contact_rew = jp.mean((contacts == ref_contacts).astype(jp.float32))

    move_cmd_norm = jp.linalg.norm(cmd[:3])
    reward = 0.5 * joint_pos_rew + 0.25 * joint_vel_rew + 0.25 * contact_rew
    return jp.nan_to_num(reward * (move_cmd_norm > 0.01))
