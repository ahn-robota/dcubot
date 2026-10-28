# Automated experiments, lab PC, 2026-10-28 ~ 10-30

Goal: get a policy that actually follows velocity commands (the 50M main
run stands still -- `docs/RESULTS_omni_1024_50m.md`). User-approved: the
lab PC runs, evaluates and modifies code on its own for ~36 h. All config
changes live on branch `auto/2026-10-28`; master is untouched apart from a
pointer in `CLAUDE.md`.

Common to every run unless noted: lab PC RTX 3080, `--task
flat_terrain_train --num_envs 1024`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`.
Every checkpoint is scored by `scripts/eval_command_tracking.py` (8 s fixed
commands fwd 0.2 / back -0.15 / left 0.15 / right -0.15 / turn +-0.1, no
pushes): **tracking** = mean achieved/commanded over those 6 commands
(1.0 perfect, 0 = stands still). The 50M main run scores 0.00 (16.2M:
-0.02).

## Findings before the runs

- `reward_tracking_lin_vel` ignores lateral error up to `y_tol=0.1`, and the
  lateral command range is +-0.158: standing still under a full-range strafe
  command costs only `(0.058)^2` -- strafing is almost free to ignore.
- Per step, `alive` pays 20 while perfect linear tracking pays at most 2.5
  (angular 6.0). Standing still already collects part of the tracking
  reward (`exp(-c^2/sigma)`), so walking gains little and risks falling
  (losing `alive`).
- Open-loop playback of the reference gait through the position actuators
  (no policy) moves the robot the commanded way (fwd 0.196 -> +0.08 m/s,
  back -> -0.09 m/s) before tipping over after 1-2 s: the gait is physically
  meaningful, it needs a balancing policy.
- The reference asks for faster joint motion than the motor-speed limit
  (`max_motor_velocity=5.24` rad/s, STS3215) allows: fwd 0.196 needs up to
  15.7 (L hip_pitch), 25.0 (L knee), 10.2 (L ankle), 11.0 (R knee) rad/s,
  over the limit on 10-25% of the gait-cycle steps for hip_pitch/knee/ankle
  (the L/R asymmetry, 25 vs 11, looks like polynomial-fit spikes). With the
  limit applied, open-loop playback still moves the robot (fwd +0.07,
  back -0.27 m/s before tipping), so this makes the reference harder to
  follow but not impossible.
- How to read `tracking_lin_vel` during training: under the training
  command distribution a robot that stands still earns 0.407/step (sigma
  0.01, baseline) or 0.257/step (sigma 0.003). The 50M baseline ends at
  0.400 -- exactly standing.

## Runs

| run | change vs. master `ba2fc7a` | steps | best tracking | verdict |
|---|---|---|---|---|
| exp01 | none (sigma 0.003, ang_vel_yaw +-0.3 as committed by the laptop) | 3.6M (stopped) | 0.00 (0 / 1.8M / 3.6M all 0.00, no falls) | stopped early: same as baseline at this stage, and alive dominance untouched -- would need >16M to judge |
| exp02 | `alive` 20 -> 2 | 1.8M (stopped) | 0.00 (turn -0.05/+0.07) | stopped: costs exceed rewards, see below |
| exp03 | `alive` 20 -> 5 | 10.8M (stopped) | -0.02 (all checkpoints 0.00 +-0.02, no falls) | balance learned faster than baseline (ep_len 500 at 9M vs 438), still no command following |
| exp04 | exp03 + imitation shaping: joint-pos kernel 0.1 -> 1.0, upstream's base-velocity terms | 0 (stopped) | -- | replaced by the direction curriculum below (same reward) |
| cur1 | exp04 reward, **forward only**: `--lin_vel_x 0.05 0.3 --lin_vel_y 0 0 --ang_vel_yaw 0 0` | 7.2M (stopped) | 0.00 (fwd vx 0.000 at every checkpoint; checked in the training env too) | balance OK (ep_len 272 at 5.4M), imitation creeping up 1.45 -> 1.58/step, swing 1.9 -> 3.3, but no forward motion |
| cur1b | cur1 restored from its 7.2M checkpoint, forward range of the old walking runs: `--lin_vel_x 0.05 0.1` | 1.8M (stopped) | fwd 0.08 -> vx +0.001 (eval `--scale 0.4`) | ep_len 409, feet in contact 99% of the time, foot bodies move ~5 mm: it never lifts a foot |
| cur1c | cur1b + foot rewards of the pre-restart walking runs: `feet_air_time` 2.0, `feet_gait` 6.0 (schedule aligned to the reference contacts), `feet_clearance` -5.0 (target 3 cm swing), restored from cur1b 1.8M | 12.6M planned | | running |

### cur1c: rewards for lifting the feet

The removed `compare_original_vs_dcubot.md` (deleted in `314d246`) records
that the pre-restart dcubot walking runs had three rewards upstream doesn't:
`feet_air_time` 2.0, `feet_gait` 6.0, `feet_clearance` -5.0, all gated on a
move command. They were dropped in the restart for structural fidelity
(CHANGES #5), and nothing has since paid the policy to lift a foot -- every
policy here keeps both feet down 99% of the time. Restored:

- `feet_air_time`: `common/rewards.py` `reward_feet_air_time` (unchanged).
- `feet_gait`: old dcubot version (contact matching an alternating
  stance/swing schedule). Its default offsets (0, 0.5) agree with the
  omnidirectional reference's foot contacts on only 55% of gait steps;
  offsets (0.75, 0.25) agree on 95%, so they no longer fight imitation's
  contact term. Standing still scores half (3.0/step), stepping on schedule
  6.0/step.
- `feet_clearance`: old dcubot version (swing-foot height error weighted by
  horizontal foot speed), target foot-site height 0.07 m (0.040 at the
  home pose, i.e. a 3 cm swing).

### cur1: still no forward motion

Ruled out along the way: the reference's joint-speed demand is not what
separates the old walking runs from these -- the old forward-only
`polynomial_coefficients_slow.pkl` asks for the same 13-15 rad/s peaks,
over the 5.24 rad/s limit on 45-50% of gait steps. Physical capacity is
fine too (1.36 kg robot, 3.23 Nm per servo; open-loop reference playback
saturates hip_pitch/knee/ankle 4-18% of the time with ~0.1 rad tracking
error). The clearest remaining difference from the user's earlier
forward-walking runs is the commanded speed: 0.05-0.10 m/s there vs
0.05-0.30 here.

### Direction curriculum (user's idea, 2026-10-28 18:40)

Upstream trains every direction at once: each axis sampled uniformly and
independently, resampled every 500 steps, 10% zero commands -- no
curriculum. The user's earlier forward-only dcubot runs learned to walk
forward quickly, while every omnidirectional run so far stands still. With
random directions, any early stepping motion moves the robot in some
direction that matches the command only by chance, so on average "move"
earns nothing over "stand".

Plan: stage-by-stage, **cumulative** command ranges (earlier directions stay
in the mix so the policy doesn't forget them), each stage restoring the
previous stage's last checkpoint (`--restore_checkpoint_path`):

1. cur1 forward: x [0.05, 0.3]
2. cur2 + backward: x [-0.2, 0.3]
3. cur3 + strafe: y [-0.158, 0.158]
4. cur4 + turn: yaw [-0.3, 0.3]

Move to the next stage once the current direction's eval ratio is >= 0.5
(or the stage's steps run out). Budget (user extended the run to
2026-10-29 ~18:45, ~70M steps at ~814 steps/s): cur1 up to ~15M, cur2-4 up
to ~12M each, then all directions mixed for the rest. A stage that is not
there yet at its launch length is continued from its last checkpoint. `runner.py` gained `--lin_vel_x/--lin_vel_y/
--ang_vel_yaw MIN MAX` for this (passed to `Joystick(config_overrides=...)`).

### exp03: the policy never finds the gait

Per step in the training env (tracking terms /weight /ep_len):

| step | lin tracking (stand 0.257) | ang tracking (stand 0.245) | imitation | ep_len |
|---|---|---|---|---|
| 3.6M | 0.131 | 0.027 | 0.210 | 175 |
| 7.2M | 0.157 | 0.032 | 0.210 | 396 |

With sigma 0.003 walking at a 0.2 m/s command would pay ~2.5/step more than
standing, so the incentive is there -- the policy just never discovers the
gait. Imitation stays flat at ~0.21, which is almost all the contact term:
dcubot's joint term `0.5 * exp(-sum(err^2) / 0.1)` is ~0.02 with every leg
joint only 0.2 rad off, so it gives no gradient toward the reference.
Upstream's is `-15 * sum(err^2)` (pulls toward the reference from anywhere)
plus base-velocity terms that pay for moving at the reference's speed.

### exp02: per-step reward clip

The env clips each step's total reward at 0 (`jp.clip(sum * dt, 0, 1e4)`,
same as upstream). At step 0 with `alive=2` the episode costs (action_rate
257 + stand_still 15 + torques 2) exceed all rewards (alive 151 + tracking
26 + imitation 16), so most steps are clipped to exactly 0 and carry no
learning signal; ep_len at 1.8M was 85.8 vs 95.6 for the baseline. With
`alive=20` the sum is always positive, which is why upstream never hits the
clip. `alive=5` keeps the per-step sum positive (~5.5 reward vs ~3.6 cost at
the start, costs fall as training proceeds) while still giving tracking 4x
more weight relative to alive than upstream.
