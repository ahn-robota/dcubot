# Changes from upstream open_duck_mini_v2

Baseline policy (2026-10-27): follow upstream `open_duck_mini_v2/joystick.py`'s
reward/obs/config structure as-is. Only the robot model (MJCF), the
reference motion, and values tied to the robot's own shape/speed are
changed. Everything below is a deliberate, reasoned deviation; anything not
listed here was kept identical to upstream.

## 1. Imitation reward function (custom_rewards.py)

Upstream's `reward_imitation` indexes its reference_frame with hardcoded
slice offsets tied to the duck's own PolyReferenceMotion pkl layout (16
joints: left leg(5) + neck/head/antennas(6) + right leg(5) = 40-dim frame).
dcubot's own layout is 13 joints (left leg(5) + head(3), no antennas +
right leg(5) = 34-dim frame, Placo recording order left-leg/head/right-leg),
confirmed by `scripts/inspect_reference.py` reading the raw recording
JSON's own `"Joints"` list. Using upstream's function on dcubot's pkl would
silently read the wrong dims. Used dcubot's own `reward_imitation_dcubot`
instead (already proven over a real 21M-step run, 2026-10-07). It already
self-gates on `cmd_norm > 0.01`, same semantics as upstream's
`reward *= cmd_norm > 0.01`.

## 2. Reference motion is forward-only; commands narrowed to match

`scripts/inspect_reference.py` run against
`polynomial_coefficients_slow.pkl` (the pkl `joystick.py` actually uses)
confirmed: 5 keys, dx in [0.049, 0.098] m/s, dy=0 and dtheta=0 on every key
(no backward/strafe/turn data at all). Per instruction, command ranges are
narrowed to match exactly, not widened:

```
lin_vel_x = [0.049, 0.098]
lin_vel_y = [0.0, 0.0]
ang_vel_yaw = [0.0, 0.0]
```

A true omnidirectional reference motion remains future work. It was
attempted on 2026-10-20 (`reference_motion_generator/
omnidirectional_phase1_report.md`): a 210-combo dx*dy*dtheta sweep was
built, but per-combo achieved velocity turned out to be dominated by noise
once all three axes were pushed to their extremes simultaneously (e.g. one
dtheta value's 42 combos averaged to near-zero net turning despite a large
intended dtheta), and `PolyReferenceMotion.process()`'s dense-grid
requirement means recordings can't simply be relabeled by real achieved
velocity without breaking the grid. Not resolved; not used here.

## 3. Head command is 3-dim (not 4-dim), ranges from dcubot's own joint limits

dcubot's head has `neck_pitch`/`head_pitch`/`head_yaw` only (no
`head_roll`, no antennas) -- confirmed from the actuator list in
`xmls/dcubot.xml` and the Placo recording's `"Joints"` field. `sample_command`
therefore returns a 6-dim vector (3 move + 3 head), not upstream's 7-dim.
Ranges are read directly from `xmls/dcubot.xml`'s joint `range=` attributes
(previously these were copied verbatim from upstream's duck-specific head
joint ranges, which don't correspond to dcubot's actual physical limits):

| joint | dcubot XML range (rad) | previous (copied from duck) |
|---|---|---|
| neck_pitch | [-0.785398, 0.785398] | [-0.34, 1.1] |
| head_pitch | [-1.570796, 1.570796] | [-0.78, 0.78] |
| head_yaw | [-1.570796, 1.570796] | [-1.5, 1.5] |

No reward currently penalizes head tracking error (`cost_head_pos` is not
wired into `joystick.py`'s `_get_reward`), so this had no effect on past
training outcomes -- it only matters once/if a head-tracking cost is added.

## 4. tracking_sigma lowered 0.01 -> 0.004

Upstream's sigma=0.01 was tuned for a command range of roughly +-0.15 m/s
(span 0.3). dcubot's forward-only range (0.049-0.098, span 0.049, max
magnitude 0.098) is 3-6x narrower. Keeping sigma=0.01 unchanged would make
`reward_tracking_lin_vel` proportionally far more forgiving for dcubot: a
worst-case tracking error equal to dcubot's *entire* command range would
still score `exp(-0.0096/0.01) ~= 0.38`, versus upstream's own worst-case
relative error scoring `exp(-0.0225/0.01) ~= 0.11`. Lowering sigma to 0.004
restores comparable relative sensitivity
(`exp(-0.0096/0.004) ~= 0.095`, close to upstream's ~0.11).

`y_tol=0.1` in `reward_tracking_lin_vel` is left unchanged per instruction,
pending a reference-replay measurement of how much the reference itself
wobbles laterally (see Step 4 below) -- not yet done.

`action_scale=0.25` (rad, per-step delta added to default pose) was checked
against dcubot's tightest joint range (hip_pitch, +-0.523599 rad total span)
and found not problematic: 0.25 is well under the half-range even for the
tightest joint, and the motor-speed-limit clip plus the actuator's
`inheritrange` clamp bound any remaining excess. Left unchanged.

## 5. feet_air_time/feet_gait/feet_clearance dropped

These were dcubot-specific additions from an earlier session, not present
in upstream `open_duck_mini_v2/joystick.py` at all. Dropped for structural
fidelity to "use upstream structure as-is" -- not because they were found
harmful. Can be re-added later if gait quality needs it.

## 6. Home keyframe pose (xmls/scene_flat_terrain*.xml)

Previously a placeholder: legs perfectly straight, all joint angles zero,
base height 0.2475, no trunk pitch (comment literally said "Not yet a tuned
standing pose"). Replaced with the reference motion's own standing posture,
taken from `motions_dcubot/standing_0.0_0.0_0.0.json` (steady-state average
over the last 100 frames of that recording, left/right-symmetrized since
the raw recording has nonzero sway/noise):

- base height: 0.2495 (was 0.2475)
- base orientation: quat `[0.99885, 0, 0.04796, 0]`, i.e. ~5.5 deg forward
  trunk pitch (was identity/upright) -- matches the recording's
  `walk_trunk_pitch=5.5` Placo config, confirmed present in every recording
  checked (both the dedicated standing recording and forward-walk
  recordings), so it's a fixed physical target, not situational.
- leg joint angles: hip_pitch=0.0769, knee=0.0365, ankle=-0.0173 (both
  legs; was all-zero) -- a real crouch, consistent with Placo's
  `walk_com_height=0.19` target requiring the legs to bend to lower the CoM
  for dcubot's own leg geometry.
- Quaternion note: the recording JSON stores `root_quat` in scipy's
  `[x, y, z, w]` convention (`gait_generator.py` uses
  `scipy.spatial.transform.Rotation.as_quat()`), reordered here to
  MuJoCo's `[w, x, y, z]` for the keyframe.

Verified in isolation (no RL env, raw MuJoCo step loop): the new pose holds
under gravity + position actuators for 2s with the base settling only
~1.7mm and the upvector staying at ~0.994 (no fall, no penetration).

## Unchanged (confirmed identical to upstream)

Reward weights (tracking_lin_vel=2.5, tracking_ang_vel=6.0, torques=-1e-3,
action_rate=-0.5, stand_still=-0.2, alive=20.0, imitation=1.0),
imitation/stand_still gating (`cmd_norm > 0.01` / `< 0.01`), tracking
always-on, `push_config` (0.1-1.0 m/s magnitude, 5-10s interval),
`episode_length=1000`, `ctrl_dt=0.02`, noise/delay config,
`max_motor_velocity=5.24` rad/s with `USE_MOTOR_SPEED_LIMITS=True`,
domain randomization (`randomize.py`; only `FLOOR_GEOM_ID`/`TORSO_BODY_ID`
differ from `common/randomize.py`, because dcubot's compiled model indexes
those differently -- a necessary robot-specific value, not a behavior
change), fall termination, PPO hyperparameters (BerkeleyHumanoid preset
except `num_envs`, scaled down to 1024 for this machine's 6GB GPU), network
size.

## Also fixed (bugs, not upstream-vs-dcubot choices)

`common/rewards.py`'s `cost_stand_still(ignore_head=True)` branch and
`cost_head_pos` previously hardcoded numeric slices (`qpos[:5]`/`qpos[9:]`/
`qpos[5:9]`) matching the duck's 16-joint layout. For dcubot's 13-joint
layout these slices would silently pick up the wrong joints (part of the
head block, not the leg block). Neither path is currently exercised
(`joystick.py` always calls `cost_stand_still(ignore_head=False)` and never
calls `cost_head_pos`), so this had no effect on past training, but both
functions now take explicit `leg_indices`/`head_indices` parameters (to be
derived from joint names by the caller) instead of hardcoded slices, so
they're correct if ever turned on.
