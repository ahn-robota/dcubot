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

## 4. tracking_sigma 0.01 -> 0.003, ang_vel_yaw range +-0.103 -> +-0.3 (2026-10-28 experiment)

Upstream's sigma=0.01 was kept at first. The 50M-step main run
(`docs/RESULTS_omni_1024_50m.md`) and the 3.6M laptop verification run both
learned to stand still regardless of command. `reward_tracking_*` is
`exp(-err^2 / sigma)` with an absolute error scale, so standing still under a
command `c` scores `exp(-c^2 / sigma)`. Upstream's command ranges are well
beyond `sqrt(sigma)=0.1` (ang_vel_yaw +-1.0, ratio 10), but dcubot's
real-achieved-velocity ang_vel_yaw range (+-0.103, #2) is ~1 sigma-width:
standing scores ~0.37 even at the command extreme. Experiment (user-approved,
not yet validated): sigma 0.003 sharpens tracking on every axis, and
ang_vel_yaw +-0.3 asks for more turning than the gait reaches (~0.1) so that
"turn as much as possible" clearly beats "stand". Full reasoning in
`joystick.py`'s module docstring, deviation #4.

`y_tol=0.1` in `reward_tracking_lin_vel` is left unchanged per instruction,
pending a reference-replay measurement of how much the reference itself
wobbles laterally (see Step 4 below) -- not yet done.

`action_scale=0.25` (rad, per-step delta added to default pose) was checked
against dcubot's tightest joint range (hip_pitch, +-0.523599 rad total span)
and found not problematic: 0.25 is well under the half-range even for the
tightest joint, and the motor-speed-limit clip plus the actuator's
`inheritrange` clamp bound any remaining excess. Left unchanged.

## 5. feet_air_time/feet_gait/feet_clearance restored (branch auto/2026-10-28)

dcubot-specific additions from the pre-restart walking runs, not in
upstream. The restart dropped them for structural fidelity; afterwards no
policy ever lifted a foot (both feet down 99% of the time) and none walked.
Restored with their old weights, all gated on a move command:
`feet_air_time` 2.0 (`common/rewards.py`), `feet_gait` 6.0 (contact vs an
alternating stance/swing schedule; offsets (0.75, 0.25) so it agrees with
the reference's own foot contacts on 95% of gait steps instead of 55%),
`feet_clearance` -5.0 (swing-foot height error x horizontal foot speed,
target foot-site height `feet_clearance_target_z`, a 3 cm swing: 0.07 on the
old model, 0.03 on the final-URDF MJCF whose foot sites sit at the sole).
This is what made the forward stage walk (`docs/AUTO_EXPERIMENTS_2026-10-28.md`, cur1c).

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

## 7. alive 20 -> 5 (branch auto/2026-10-28)

alive paid 20/step against 2.5 for perfect linear tracking, so falling risk
outweighed walking. 2 was too low: costs exceeded rewards early on and the
env's per-step clip at 0 removed the learning signal; 5 keeps the sum
positive.

## 8. Imitation reward: wider joint kernel, upstream's base-velocity terms (branch auto/2026-10-28)

`reward_imitation_dcubot`'s joint term was `exp(-sum(err^2)/0.1)`, ~0.02
once the legs are 0.2 rad off the reference -- no gradient toward the gait.
Kernel 0.1 -> 1.0. Added upstream's base-velocity terms (lin xy/z
`exp(-8 err^2)` x1, ang xy/z `exp(-2 err^2)` x0.5) against the reference's
base velocity, in the local frame.

## 9. ref_yaw_scale = 10 (branch auto/2026-10-28; replaced by #15)

Yaw commands of +-0.05..0.1 rad/s all snapped to the reference grid's
straight dtheta=0 key (grid 0, +-0.98, +-1.96), so imitation asked for
straight stepping while tracking asked for turning. The yaw command is
multiplied by 10 for the reference lookup only, selecting the turning keys.

## 10. Single-axis commands, 30% (2026-11-04, lab PC)

Upstream samples x, y and yaw independently, so almost every command mixes
all three; a pure turn in place or a pure strafe is practically never
trained, yet that is how the robot is driven and evaluated. After 20M more
steps (long1) pure-command following had drifted (forward overshoot up to
1.6x, left turns never learned) while the training reward kept rising.
`sample_command` now keeps only one axis, chosen uniformly, for
`single_axis_command_prob = 0.3` of the commands (after the 10% zero
commands: ~10% stand / ~27% single-axis / ~63% mixed).

## 11. Symmetric policy (`runner.py --symmetric_policy`, 2026-11-04, lab PC)

Every run so far broke left/right symmetry on its own: turning works one
way only (old model left, new model right), although the robot (whole-body
CoM 0.4 mm off center) and the reference (symmetric turning keys) are
symmetric. Averaging a trained policy with its mirror at inference
(`mujoco_infer.py --symmetric`) fixed drift but can't create a turn the
policy never learned. With `--symmetric_policy` the policy PPO trains is
pi(o) = (f(o) + M f(M o)) / 2 on the pre-tanh mean (std swapped L/R),
`playground/dcubot/symmetry.py`, so it is exactly equivariant during
training. The ONNX export holds f; run it with
`mujoco_infer.py --symmetric_trained`, which rebuilds pi exactly.

## 12. tracking_ang_vel on a low-passed yaw rate (2026-11-04, lab PC)

Upstream scores the instantaneous gyro yaw rate. Stepping makes it swing by
0.16 (stand) / 0.25 (turn in place) / 0.75 (forward) rad/s std, far beyond
sqrt(tracking_sigma) = 0.055: with a 0.15 rad/s turn command a robot that
doesn't turn at all scored 0.18/step, about what a correct turn with the
same swing scores -- no gradient toward turning, which no run ever learned
properly. (Upstream's +-1.0 rad/s turn commands dwarf the swing.)
The reward now uses an exponential moving average of the gyro yaw rate,
`ang_vel_filter_tau = 0.4` s (one gait cycle; cuts the swing ~6x). The
observation is unchanged.

## 13. Imitation base-velocity targets: the command, not the reference (2026-11-04, lab PC)

Upstream's imitation base-velocity terms (#8) pull the base toward the
reference frame's base velocity. The Placo reference does not move at the
command's speed: at the eval commands it goes 1.37x forward (0.15 -> 0.206
m/s), 0.94x back, 0.65x sideways (0.09 -> 0.059) and 0.73x turning
(0.15 x ref_yaw_scale -> 0.11 rad/s, with a 0.09 rad/s swing). The
policies' 64-robot eval ratios sat on exactly those values (long4: forward
1.14-1.22, strafe 0.47-0.71, turn 0.39-0.55) whatever tracking_lin_vel and
tracking_ang_vel asked for. The planar targets (lin xy, ang z) are now the
command; ang z uses the low-passed yaw rate of #12. Lin z and ang xy still
follow the reference. Weights and kernels unchanged.

## 14. Reference lookup interpolated between grid points (2026-11-04, lab PC)

Upstream's `PolyReferenceMotion.vel_to_index` snaps the command to the
nearest (dx, dy, dtheta) grid point. DCUBOT's grid is 0.098 m/s apart in x
and y, so every forward command from 0.147 to 0.2 imitated the 0.196 gait
(0.206 m/s), and long5 (with #13) still walked forward at 1.4x the command:
the joint terms set the stride. `playground/dcubot/reference_motion.py`
blends the coefficients of the 8 surrounding grid points trilinearly (same
period for every gait; a polynomial is linear in its coefficients). The
reference's own speed at the eval commands is now forward 0.158 for 0.15,
back -0.116 for -0.11. Strafe (0.6x) and turning (0.07 rad/s for 0.15) stay
below the command: that is the Placo sweep itself (deviation #2), which only
regenerating the reference can fix.

## 15. Reference looked up by achieved speed (2026-11-04, lab PC; replaces #9)

The pkl's keys are Placo's intention labels, not the speeds the gaits reach:
x 1:1, y ~0.57x, turn 0.025 rad/s at key 0.98 and 0.11 at 1.961. With #14,
`ref_yaw_scale` 10 gave a 0.15 rad/s command a 0.067 rad/s reference turn.
`InterpolatedPolyReferenceMotion` now measures each pure single-axis gait's
mean base velocity at load time and maps every command axis to the key that
reaches it (`jp.interp`, clamped at the fastest gait: x -0.207..0.31, y
+-0.162, turn +-0.11). Single-axis commands now get a reference moving at the
command's speed (turn capped at 0.11 rad/s); `ref_yaw_scale` is gone. Mixed
commands still blend the grid's mixed gaits, which move erratically (e.g.
(0.1, 0.05, 0.05) -> (0.01, 0.11, 0.02)); the base-velocity terms target the
command anyway (#13).

## 16. Torso orientation cost, weight -20 (2026-11-05, lab PC)

The user wants the torso upright when standing: the trained policies stood
with the torso pitched ~10 deg forward, and the real robot, posed like that
unpowered, tips forward. The cause is the reference: its leg angles pitch
base_link ~11 deg forward when the feet are flat (5.5 deg relative to the
foot frames, plus the 5.5 deg the all-zero pose already has), at every
command. Since the reference is not regenerated, upstream's
`cost_orientation` (sum of squared x/y of the IMU z axis; IMU site aligned
with base_link) is added with weight -5: 0.16/step at 10 deg, about 4x what
the imitation joint term loses by bending hips/ankles 11 deg off the
reference. That did not move the policy (long9: 11.1 / 10.9 / 11.3 deg
standing at 0 / 1.8M / 3.6M; the cost was ~1% of the per-step reward), so
the weight is -20 (0.7/step at 11 deg). With it, slow backward commands
stopped working (long12: -0.1 m/s -> 0.00, vs -0.13 without the cost; the
gait walks back only leaning forward), so the cost is off while the x
command is below -0.02 m/s.

## 17. Sim-to-real: smoother actions, wider randomization, slower top speed (2026-11-05, lab PC)

The user found the gait good but the stride long for the real robot and
asked for all three of:
- `action_rate` -0.5 -> -1.0 (smoother joint targets).
- Wider domain randomization in `randomize.py`: upstream Open Duck's own
  "was ..." originals (floor friction 0.4-1.0, torso mass +-0.2 kg, qpos0
  +-0.05 rad, kp 0.8-1.2) plus frictionloss 0.8-1.2, armature 1.0-1.1,
  link masses 0.85-1.15. Torso CoM jitter stays +-1 cm.
- Top forward speed back to 0.2 m/s (`--lin_vel_x -0.15 0.2 --lin_vel_y
  -0.12 0.12`): the reference's gait period is fixed (0.4 s), so stride
  scales with speed; 0.2 instead of 0.3 m/s is 2/3 the stride.

## 18. Head stays down while standing/walking: stand_still -0.2 -> -1.0, cost_head_pos wired in (2026-11-11, first real-hardware standing test)

The first real standing test (torque on, zero velocity command) showed the
neck drooping forward almost to horizontal, and the torso leaning forward
too -- but `scripts/replay_directions_mujoco.py --source pkl` shows the
reference motion itself is not tilted this way, so the droop is the
policy's own learned behavior, not something inherited from the reference.

Two gaps, found by reading `common/rewards.py`:
- `cost_stand_still` (zero-command case, already called with
  `ignore_head=False`) sums `|qpos - default_pose|` over all 13 joints at
  weight -0.2 -- small next to `alive` (5.0) and `feet_gait` (6.0), so a
  large head deviation barely matters to the total reward. Raised to -1.0.
- `cost_head_pos` (the moving case: `move_cmd_norm > 0.01`) was never
  called at all for dcubot -- nothing kept the head near its command (fixed
  at 0 for now, see `docs/SIM2REAL_PLAN.md`) while walking. Wired in at
  weight -1.0, with `head_indices` derived from `self.actuator_names` by
  name (never hardcoded -- see that function's own docstring), in `cmd[3:]`
  order (neck_pitch, head_pitch, head_yaw).

Not yet re-trained/evaluated as of this commit -- see
`docs/AUTO_EXPERIMENTS_2026-10-28.md` for the result once the next run
completes.

## 19. Floor friction randomization lower bound 0.4 -> 0.15 (2026-11-11, same test)

Standing up on the real floor, the robot's PLA feet slipped considerably --
visibly sliding forward while trying to balance out the neck/torso lean
from #18, looking almost like marching in place. The training floor
friction range (0.4-1.0, CHANGES #17) never covered anything this slick, so
the policy was never exposed to it. Widened to 0.15-1.0
(`playground/dcubot/randomize.py`). Not empirically measured against the
real floor yet -- a simple incline test (tilt a sample of the floor
material with the foot part on it; `mu = tan(angle at first slip)`) would
give a real number to check this range against.

## 20. Strafe range halved: lin_vel_y -0.12/0.12 -> -0.06/0.06 (2026-11-11, same test)

Same reasoning as #17's forward top speed: the reference's gait period is
fixed (0.4 s), so stride scales with commanded speed, and the user found
the strafe stride too long for the real robot (~2x what looks controllable
on hardware). Halved `lin_vel_y` in `default_config()` and the recipe
command below; `--lin_vel_x`/`--ang_vel_yaw` are unchanged.

## 21. Upright reference motion and home pose: walk_trunk_pitch 5.5 -> 0 deg; #20 reverted (2026-11-11, lab PC)

The real robot and every trained policy lean forward. The cause is the
reference, not the reward: with the stance foot flat, the old reference
legs (and the old home keyframe, which was built from the reference's own
standing pose) put the torso at **5.5 deg forward** in every one of the
210 combos, because the Placo generator used `walk_trunk_pitch: 5.5`.
The orientation cost (#16, -20) could only pull the policy 2-3 deg away
from what imitation asks for (long15 standing ~7 deg).

- Generator (`dcubot-reference-motion-generator`):
  `robots/dcubot/placo_presets/medium.json` and `placo_defaults.json`
  `walk_trunk_pitch` 5.5 -> 0. Everything else unchanged (same
  `auto_gait.json` grid, com height 0.22). Regenerated the 210-combo sweep
  (`auto_waddle.py --duck dcubot --sweep -j 12 --output_dir
  recordings_omnidirectional_trunk0`) and fit at degree 9
  (`FIT_DEGREE=9 fit_poly.py`), giving
  `playground/dcubot/data/polynomial_coefficients_omnidirectional_trunk0_deg9.pkl`;
  `joystick.py` now points here (head-v2 pkl stays in place).
- Checks on the new pkl/recordings: implied torso pitch with a flat stance
  foot -0.5..+0.5 deg over all 210 recordings (old: 5.5 deg in all); 0
  exploded (combo, phase) pairs out of 4200; left hip_pitch range -22..21
  deg (limit +-30); base speeds of the pure single-axis keys identical to
  the old pkl (fwd/back 0.206/-0.207, strafe +-0.11, turn +-0.025).
- **Home keyframe** (`scene_flat_terrain.xml` and `_train.xml`, qpos and
  ctrl): base quat 1 0 0 0 (was 5.5 deg pitch), legs
  hip_pitch/knee/ankle = -0.104527 / 0.050879 / 0.053608 rad (was
  0.024435 / 0.059952 / 0.011606), the new standing recording's
  steady-state mean (frames 100-450) averaged over left/right, z 0.264145
  so the soles touch the floor. Whole-body CoM is over the feet in both
  poses (0.6 mm vs 0.4 mm).
  **Hardware:** `hardware/pi/policy_runner.py` reads its default pose from
  this keyframe, so a policy trained after this change must be deployed
  together with the new xml. long14 and earlier need the old xml.
- **#20 reverted:** `lin_vel_y` back to -0.12/0.12. long15 (trained at
  +-0.06) barely strafed in the viewer and its 64-robot strafe ratios fell
  to 0.78/0.70; stride length on hardware can be limited from the
  command side instead.

## 22. Neck held at its command while standing too: head_pos -1 -> -20, always on (2026-11-12, lab PC)

With the upright reference (#21) the torso stands at 0-1.5 deg, but the
neck still sits ~11 deg forward at a standstill (every long16 checkpoint
-10.9..-12.4 deg; long14 was the same). Nothing holds it there: imitation
only covers the legs, upstream's `cost_head_pos` is multiplied by
`|cmd[:3]| > 0.01` (off while standing), and `stand_still` sums |q - home|
over all 13 joints, so 0.19 rad on one joint is a small share, and the
zero command is only drawn 10% of the time.

- `playground/common/rewards.py` `cost_head_pos`: new `always` argument
  (default False = upstream behaviour); True drops the moving-only gate.
- `joystick.py`: `cost_head_pos(..., always=True)`, scale `head_pos`
  -1.0 -> -20.0 (same size as the orientation cost that straightened the
  torso, #16).
- `head_range_factor` 1.0 -> 0.0: head commands (`cmd[3:]`) are now always
  0, as on the real robot and in the viewer. They used to be drawn from the
  full joint ranges (neck +-45, head +-90 deg), far beyond the +-14.3 deg
  the policy can reach from home, so at -20 the cost was mostly an
  unreachable random target (episode cost ~15000 vs ~200 for the others,
  train reward 148 -> 24) and the standing neck did not move in the first
  4.3M steps (-11.7 deg). Restarted long17 with this; reward at step 0 170.

## 23. Neck fixed at home: the policy's neck output is ignored (2026-11-13, lab PC)

#22 moved the standing neck only -11.7 -> -10.2 deg in 21M steps (long17),
so the neck is now taken away from the policy: `fix_neck=True` in
`default_config()` sets the neck_pitch motor target to its home value (0)
every step, after `default + action * action_scale` and before the motor
speed clip. The policy still outputs 13 actions (ONNX shape unchanged) and
its neck action still appears in the observation's last actions; only the
motor target ignores it. head_pitch and head_yaw are unchanged (still
policy-driven, held near 0 by head_pos).

- Viewer: `mujoco_infer.py --fix_neck`; eval: `eval_command_tracking.py
  --fix_neck` (also sets the env config in `--env` mode, so policies from
  before #23 are evaluated without it unless asked).
- **Real robot:** `hardware/pi/policy_runner.py` must also set the
  neck_pitch target to home after computing `default + action * 0.25`,
  for every policy trained with this (long18 onward). Without it the
  robot moves the neck the way the policy happens to output, which it
  never did in training.
- long17 21.3M with the neck fixed, before any retraining (64-env): falls
  unchanged (~33%), torso 0.5 deg standing, but strafe 0.57/0.49 and
  `|r-1|` 0.22 (0.11 with its own neck).

## 24. Servo tracking noise, weaker servos, contact dropout, torque cost x50 (2026-11-13, lab PC)

On the real robot the legs move fast and jerky while the same policy is
smooth in MuJoCo (checklist on `hardware/bringup`:
`docs/SIM2REAL_DEBUG_2026-11-13.md`). The user asked for the policy to be
trained against servos that do not track well, as in actuator-noise domain
randomization papers. Until now only kp (0.8-1.2) and a 0-3 step action
delay touched the servo side.

- `randomize.py`: kp *U(0.7, 1.2) (was 0.8-1.2); new per-actuator max
  torque `actuator_forcerange` *U(0.7, 1.05) (3.35 N*m is at nominal
  voltage, the real battery has sagged to 6.2 V).
- `joystick.py` `step()`: the target sent to the simulated servo is the
  policy's target + a per-episode offset U(-0.03, 0.03) rad per joint
  (`noise_config.scales.target_bias`, zero/calibration error) + per-step
  N(0, 0.02 rad) (`target_noise`). `info["motor_targets"]`, which the
  observation shows, stays the clean target, as on the robot. Applied
  after the neck fix (#23), so the neck target is noisy too.
- `_get_obs()`: each foot-contact observation reads False with probability
  0.05 per step (`contact_dropout`); rewards still use the true contact.
- Reward scale `torques` -1e-3 -> -0.05: at -1e-3 it was ~4 per episode
  against 174-357 for orientation/action_rate/head_pos (long17), i.e. no
  effect. Torque is roughly proportional to servo current.
- All three noise terms scale with `noise_config.level` like the existing
  ones.

## Unchanged (confirmed identical to upstream)

Reward weights (tracking_lin_vel=2.5, tracking_ang_vel=6.0, torques=-1e-3,
action_rate=-0.5, stand_still=-0.2, imitation=1.0; alive differs, #7),
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

`randomize.py` `TORSO_BODY_ID` 1 -> 2 (2026-11-03). Body 1 is "base", the
massless free-joint frame; the torso mass is on body 2, "base_link". The
torso mass (+-0.1 kg) / CoM (+-5 cm) randomization therefore never touched
the torso: it added a +-0.1 kg point mass (negative half the time) at the
base frame instead. True of every run before this date, the walking cur5
policy included.
With the torso really randomized, upstream's +-5 cm CoM jitter turned out
too large for dcubot: torso 0.833 of 2.07 kg and a narrow stance, and a
standing policy tips over within 1-2.6 s once the torso CoM is shifted
>=2 cm sideways (4/6 shift directions at 2 cm, 5/6 at 5 cm); training
episodes stalled at ~80 steps. Jitter reduced to **+-1 cm** (2/6 at 1 cm
for a policy never trained against it).

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

## Final URDF remodel (2026-11-02, branch `model/final-urdf`)

New Fusion 360 CAD remodel with real measured masses (shell + head + battery
pack), replacing the earlier placeholder-mass model. Total mass 1.358kg ->
**2.072kg** (+53%). Kinematic joint-to-joint distances unchanged (within
0.2mm of the old model, confirmed via direct URDF forward-kinematics
comparison for all 13 joints -- axis directions also confirmed identical,
cos similarity 1.000 on every joint).

### URDF post-processing (`robot_description/dcubot.urdf`)

- 3 right-leg joint axis signs (`right_hip_yaw`/`right_hip_roll`/
  `right_hip_pitch`) flipped from the raw Fusion export to match the old
  model's sign convention (the raw export mirrors axes naively, which
  doesn't match what the Placo generator / MJCF / joystick.py expect).
  Joint limits were already symmetric so left unchanged.
- 3 links had imprecise inertia (Fusion/fusion2urdf truncates to 6 decimal
  *places*, not significant figures, so very small values lose precision):
  `hip_yaw_L_1`, `hip_yaw_R_1`, `head_mount_1`. Recomputed from mesh volume
  via `trimesh` (mass-matched to the URDF's declared mass, CoM kept as
  given by the URDF; mesh-computed CoM matched the URDF's declared CoM
  within 1.8-5.5 micrometers for all three, cross-validating the method).
- All 13 joints: `effort=1.91` N*m, `velocity=5.24` rad/s (STS3215-C001
  servo spec: 7.4V, 19.5kg*cm stall torque = 1.91 N*m), replacing the
  exporter's generic placeholder values. `effort` was initially set to 3.23
  N*m (~32.9kg*cm, matching the 12V/30kg*cm STS3215 variant, not this
  robot's actual 7.4V/C001 servos) -- corrected after confirming the real
  part number with the user. `velocity` (5.24 rad/s = 300deg/s) wasn't
  re-verified against a 7.4V-specific datasheet figure (only a 6V number,
  ~252deg/s, was found), but scales plausibly with voltage so was left
  unchanged.
- MJCF actuator `kp` / `forcerange` (2026-11-03, lab PC, user-approved):
  13.37 / 1.91 -> **17.8 / 3.35**, i.e. upstream Open Duck Mini v2's values
  (`tools/gen_dcubot_mjcf.py`, both MJCFs regenerated). Same servo: upstream
  takes them from Rhoban BAM's identification of `feetech_sts3215_7_4V`
  (`docs/sim2real.md` there; BAM's `forcerange = vin * kt / R` with the
  identified kt 1.18 N*m/A, R 2.48 ohm gives ~3.5 N*m at 7.4 V), and its
  policies transfer to the real robot with them. The datasheet stall torque
  (1.91) is below what the identified motor model produces; with it, the
  reference gait saturated hip_pitch/knee/ankle 21-33% of the time, with
  BAM's values 10-16%. BAM identifies at the output shaft, so the 1:345
  gear ratio is already in these numbers. The URDF's `effort=1.91` is left
  as the datasheet figure.
- Joint *limits* (ranges) came from the CAD export as-is and are tighter
  than the old model on knee (-90~90 deg -> **-45~30 deg**) and ankle
  (-45~45 deg -> -45~40 deg); neck_pitch widened (-45~45 deg -> -90~45 deg).
  Not a deliberate choice -- this is just what the new CAD's joint stops are.
- Added `trunk`/`left_foot`/`right_foot` dummy fixed-joint frames (required
  by Placo's `HumanoidRobot`, not part of the normal 14-link/13-joint
  model). `trunk` offset is now `(0,0,0)` relative to `base_link` (the new
  model's origin is already at the base_link mesh's own reference point,
  unlike the old model which needed a 26.9mm Y correction). Foot-frame
  (ankle-joint-origin -> sole-center) recomputed from the new foot mesh via
  `trimesh`: `(14.58, 18.27, -38.82)` mm vs old `(12.2, 18.5, -38.8)` mm --
  only x moved meaningfully (+2.4mm).

### Reference motion regeneration (`playground/dcubot/data/polynomial_coefficients_omnidirectional_newurdf_deg9.pkl`)

- **`walk_com_height` 0.19 -> 0.22** (`placo_defaults.json` and
  `placo_presets/medium.json` in the generator). The new model's heavier
  base_link+head raises its natural (straight-leg) whole-body CoM height
  from 186mm to ~213-234mm above the foot sole (measured two ways: Placo's
  own `com_world()`, and a mass-weighted sum over all 14 links' MuJoCo-
  compiled CoMs). At the old target (0.19, barely below the old model's own
  186mm), the new model's standing IK solve falls into a degenerate,
  asymmetric branch (one leg pinned exactly at its joint limit, left/right
  knee signs opposite). Swept 0.19->0.25 in 0.005 steps: solver is broken
  through 0.21, clean (small, symmetric, old-model-like angles) from 0.215
  on. Picked 0.22 for margin above that boundary. Real achieved
  velocities for the same raw sweep dx/dy/dtheta are unchanged from the old
  model (checked against the 3 extreme single-axis combos: forward 0.3003
  -> 0.3002 m/s, strafe -0.1569 -> -0.1568 m/s, turn 0.1032 -> 0.1040
  rad/s) -- com_height does not affect the step-length/cadence -> velocity
  mapping, only crouch depth.
- Regenerated the 210-combo omnidirectional sweep using the UNCHANGED
  `robots/dcubot/auto_gait.json` (byte-identical to the one that produced
  the currently-used pkl -- it was initially suspected stale because its
  raw sweep bounds, e.g. `max_sweep_x: 0.06`, don't match the pkl's own key
  labels, e.g. `dx` up to `0.294`; this is NOT a mismatch, the key labels
  are `raw_value * 2 / period` -- `gait_generator.py`'s "computed xvel"
  formula -- confirmed by reproducing the exact label set from the raw
  sweep bounds). Grid confirmed identical to the existing pkl via
  `scripts/inspect_reference.py`: 210 keys, same dx/dy/dtheta value sets,
  period 0.408s, fps 50.
- **Found and fixed a real bug affecting the EXISTING pkl too**: fitting
  used `degree=15` (float64, per the existing convention), but the actual
  consumer (`playground/common/poly_reference_motion.py`, used inside the
  JAX-jitted training loop) evaluates the polynomial via
  `jax.numpy.polyval`, which silently runs in **float32** (`jax_enable_x64`
  is off by default). A degree-15 fit over only 20 samples is already
  ill-conditioned (coefficients up to ~1e8, confirmed by direct
  inspection); float32 evaluation of that ill-conditioned polynomial
  explodes for roughly the last third of each gait cycle (phase-steps
  ~13-19 of 20). Quantified on the existing (currently-trained-on) pkl:
  **27.4% of (combo, phase-step) pairs show joint angles whose magnitude
  exceeds 3x the physical joint limit** (some by hundreds of degrees;
  foot_contact channel also affected the same way). The new pkl at
  degree=15 showed the same rate (29.4%) -- confirms this is not new-model-
  specific, it's inherited from the shared fitting/evaluation code. Swept
  fit degree 7-15 and measured the float32-vs-float64 evaluation gap
  directly: negligible through degree 9 (0.04 deg), degree 10 already
  0.63 deg, degree 12+ catastrophic (tens to thousands of degrees) --
  matches the `fit_poly.py` comment "float32 breaks deg>=10 fits", which
  was apparently only ever applied to the *fitting* step, not evaluation.
  **Fixed by refitting the new pkl at `degree=9`** (`FIT_DEGREE=9` env var)
  -- re-verified 0% bad pairs across all 210 combos x 20 phase-steps, and
  foot_contact / left-right hip_pitch phase symmetry both look like a
  normal gait. Did not touch global `jax_enable_x64` (would affect the
  entire JAX/Brax/MJX training pipeline's performance/memory, a much
  bigger decision) or the existing pkl (already used for real training
  runs; whether to regenerate it at degree=9 too is left for the user to
  decide separately).
- Reproduction commands:
  ```bash
  cd reference_motion_generator
  uv run python scripts/auto_waddle.py --duck dcubot --sweep -j 6 \
    --output_dir recordings_omnidirectional_newurdf
  FIT_DEGREE=9 uv run python scripts/fit_poly.py \
    --ref_motion recordings_omnidirectional_newurdf
  ```
  (`-j 6`, not higher -- `-j 16` caused an OOM kill on this laptop's 15GB
  RAM, silently losing ~85% of the sweep with no error, only a truncated
  per-combo log.)

### Head v2 (2026-11-04, branch `model/head-v2`)

After real assembly, the head shape was revised: `head_1` 221.0 -> 224.0g
(mass 2.072 -> 2.075kg), height 91mm -> 56mm (-38.5%), volume -14.3%
despite the mass increase. Only `head_1`'s mesh/mass/inertia/CoM changed
(confirmed via diff at every stage: raw export, URDF, MJCF); nothing else
in the robot moved. Whole-body CoM (base_link-relative) shifted X -3.34mm,
Y ~0mm, Z -1.13mm (3.52mm total, under the 5mm threshold originally set
for redoing the reference motion).

Initially kept the existing `..._newurdf_deg9.pkl` (head-v1) as-is per
that threshold. Regenerated anyway after all, at the user's request, to
visually re-check turning directly against the actual head-v2 model in
Placo (`gait_playground`) rather than the old one -- same pipeline as the
final-URDF remodel (same `auto_gait.json`/grid, `degree=9` fit), confirmed
grid-identical and 0% bad pairs.
`playground/dcubot/data/polynomial_coefficients_omnidirectional_headv2_deg9.pkl`,
`joystick.py` now points here; `..._newurdf_deg9.pkl` (head-v1) stays in
place, untouched.

Also fixed and then reverted: `scripts/replay_directions_mujoco.py`'s
`--source pkl` mode integrates Placo's own base_linear_vel/base_angular_vel
(not used by training -- known unreliable, see the "Final URDF remodel"
imitation-reward note) to draw the base's visual trajectory, which made
turn segments visibly swerve instead of rotating in place. Tried
averaging over one gait cycle and re-rotating by accumulated yaw (fixed
the swerve in a numeric check -- turn_left's drift dropped from
oscillating up to 16cm/step to 3.2cm total over 3s) but the user reported
it looked worse in the actual viewer, so this was reverted; the script's
visualization of turning is still not fully correct. Not a problem with
the reference data itself (`replay_reference.py`'s joint pos/vel components
are 1.000, and the actual training reward never reads these fields).

Actuator `kp=17.8`/`forcerange=3.35` (BAM-identified, see "Final URDF
remodel") and all site positions unchanged.

### Also found: two more dead CLI arguments (bugs, not used -- discovered while testing)

`open_duck_reference_motion_generator/gait_generator.py`'s `--walk_com_height`
and `--dx`/`--dy`/`--dtheta` flags are parsed but never applied to the
actual Placo parameters -- only the `--preset` JSON file's own values are
used. `auto_waddle.py`'s sweep mode already works around this correctly (it
edits the preset dict directly in Python before writing the temp preset
file), so the 210-combo sweep itself was never affected -- this only
matters for anyone invoking `gait_generator.py` directly with these flags
expecting them to override the preset (they silently don't).
