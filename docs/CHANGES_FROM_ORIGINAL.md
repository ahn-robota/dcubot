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

## 2. Omnidirectional reference motion; command ranges from REAL ACHIEVED velocity, not pkl labels

`joystick.py` uses `polynomial_coefficients_omnidirectional.pkl` (210
keys, dx/dy/dtheta all nonzero -- forward, backward, strafe, and turn are
all covered). This supersedes an earlier pass of this file (same day) that
used a forward-only pkl (`polynomial_coefficients_slow.pkl`, 5 keys,
dx-only) -- that was a reversion made when the omnidirectional pkl was
believed to be data-quality broken; it was later restored after finding
that conclusion was entangled with an unrelated replay-script bug (joint
remap issue in a visualization script, not the reference data itself; see
2026-10-20's `reference_motion_generator/omnidirectional_phase1_report.md`
for the original sweep, and project memory for the restoration). The user
explicitly wants the same training structure as upstream
open_duck_mini_v2, which trains omnidirectionally (confirmed: upstream's
own pkl has 240 keys, dx -0.148..0.222, dy -0.111..0.111, dtheta
-1.111..1.222).

Command ranges are NOT the pkl's raw intention labels -- they're each
axis's REAL ACHIEVED velocity, checked directly against the pure
single-axis recordings' own `Vel_x`/`Vel_y`/`Yaw` fields in
`reference_motion_generator/recordings_omnidirectional/`:

| axis | intention label range | real achieved range | ratio |
|------|------------------------|----------------------|-------|
| dx (fwd/back) | 0.098-0.294 | 0.100-0.300 | ~1.0x (clean) |
| dy (strafe)   | 0.098-0.294 | 0.057-0.158 | ~0.55x |
| dtheta (turn) | 0.98-1.961  | 0.024-0.103 rad/s | ~0.03-0.05x |

dx and dy are close enough to 1:1 (or a consistent linear scale) to use
directly. dtheta is NOT: commanding near the label's scale (up to 1.961
rad/s) would ask for a turn rate the gait never gets close to (real
achieved is only ~0.03-0.1 rad/s), putting `tracking_ang_vel` (rewarding
fast turning) and `imitation` (rewarding the much-slower reference gait)
into direct conflict. So the config is:

```
lin_vel_x = [-0.20, 0.30]
lin_vel_y = [-0.158, 0.158]
ang_vel_yaw = [-0.103, 0.103]
```

One consequence of the `ang_vel_yaw` choice: at this achieved scale,
`PolyReferenceMotion.vel_to_index`'s nearest-neighbor snap mostly lands on
the dtheta=0 (straight-gait) reference frame, since 0.103 is much closer
to the grid's 0 point than to its next label (0.98). Turning is therefore
driven mostly by `tracking_ang_vel`, with `imitation` contributing little
guidance for that axis specifically -- a data limitation of the current
Placo sweep (its dtheta parameter isn't a direct rad/s dial), not a code
bug. Revisiting the sweep generation for faster real turning is future
work.

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

## 4. tracking_sigma kept at upstream's 0.01

An earlier pass of this file (same day) lowered sigma to 0.004 to
compensate for the forward-only command range being much narrower than
upstream's. With the omnidirectional ranges above, `lin_vel_x`'s max
magnitude (0.30) and `lin_vel_y`'s (0.158) are close enough to upstream's
own (0.222/0.111) that upstream's sigma=0.01 is a reasonable fit again.
`ang_vel_yaw`'s range (+-0.103) is much narrower than upstream's
(+-1.111..1.222) -- sigma=0.01 will be proportionally more forgiving for
turning than for forward/strafe tracking, but since the real achievable
turn rate is tiny in absolute terms anyway, absolute errors stay small
regardless, so this isn't expected to be a major issue in practice. Watch
turning behavior quality as a flag for revisiting this.

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
