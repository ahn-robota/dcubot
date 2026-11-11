# Automated experiments, lab PC, 2026-10-28 ~ 10-30

Goal: get a policy that actually follows velocity commands (the 50M main
run stands still -- `docs/RESULTS_omni_1024_50m.md`). User-approved: the
lab PC runs, evaluates and modifies code on its own for ~36 h. Done on
branch `auto/2026-10-28`, merged into master on 2026-11-03. The distilled,
repeatable procedure is [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md); this file
is the full record, including every attempt that didn't work.

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
| cur1c | cur1b + foot rewards of the pre-restart walking runs: `feet_air_time` 2.0, `feet_gait` 6.0 (schedule aligned to the reference contacts), `feet_clearance` -5.0 (target 3 cm swing), restored from cur1b 1.8M | 3.6M (stage done) | **fwd 0.91** (0.08 -> 0.070 m/s) | **walks forward**: 0.05/0.08/0.10 -> 0.046/0.070/0.076 m/s, stands on 0, feet lift 5-9 mm; veers left up to 0.38 m / 8 s |
| cur2 | cur1c 3.6M + backward: `--lin_vel_x -0.1 0.1` | 9.0M (stage done) | fwd 1.16 / **back 0.47** (back 0.00 -> 0.01 -> 0.12 -> 0.32 -> 0.45 -> 0.47) | ep_len 541 at 7.2M; forward overshoots slightly (0.08 -> 0.093 m/s) |
| cur3 | cur2 9.0M + strafe: `--lin_vel_y -0.1 0.1` | 3.6M (stage done) | fwd 1.27 / back 0.34 / **left 0.83 / right 0.88** | strafe learned within 1.8M (0.75/0.62); backward dipped 0.47 -> 0.33 and stayed |
| cur4 | cur3 3.6M + turning: `--ang_vel_yaw -0.1 0.1` | 12.6M (done) | most balanced 1.8M: fwd 0.89 / back 0.29 / left 0.87 / right 0.89 / turn 0.01, 0.21, no falls | no falls from 1.8M on; translation peaked at 5.4M (back 0.70) then declined to 12.6M (right -0.14); turning never learned (scores follow a drifting left/right bias) |
| cur5 | cur4 1.8M + `ref_yaw_scale=10` (yaw command x10 for the reference lookup only), all directions at the slow ranges | 14.4M (stopped) | **5.4M: 0.67** (env eval 0.55) -- fwd 1.15 / back 0.44 / left 1.06 / right 0.66 / turn L 0.40 / turn R 0.32, no falls | first checkpoint with every direction (both turns) the right way; then 7.2M 0.68, 9.0M 0.64, 10.8M 0.53, 12.6M 0.33, 14.4M 0.32 (env eval 0.33) while training reward kept rising |
| cur6 | cur5 5.4M + tracking weights x2: `tracking_lin_vel` 2.5 -> 5, `tracking_ang_vel` 6 -> 12 | 9.0M (done) | single-run 0.72 at 5.4M and 9.0M | stronger fwd/left/turn-left, but strafe right broken (-0.00..-0.29 in the env eval); weights reverted after the run |

## Final result (2026-10-29 17:10)

**Chosen policy: cur5 7.2M** -- `checkpoints/cur5/2026_10_29_110034_7208960.onnx`
on the lab PC (not in git). Its training config is this branch's HEAD
(`joystick.py` as committed: tracking 2.5/6.0, alive 5, imitation kernel
1.0 + base-velocity terms, foot rewards, `ref_yaw_scale` 10) with
`--lin_vel_x -0.1 0.1 --lin_vel_y -0.1 0.1 --ang_vel_yaw -0.1 0.1`.
Lineage: cur1 -> cur1b -> cur1c -> cur2 -> cur3 -> cur4 (1.8M) -> cur5, each
restoring the previous stage's checkpoint.

Final comparison, `eval_command_tracking.py --env 64 --scale 0.4 --turn 0.1`
(64 robots in the training env with noise, delays and pushes; commands fwd
0.08, back -0.06, strafe +-0.06 m/s, turn +-0.1 rad/s):

| checkpoint | mean | fwd | back | left | right | turn L | turn R | fell (8 s) |
|---|---|---|---|---|---|---|---|---|
| cur4 1.8M | 0.39 | 0.56 | 0.33 | 0.85 | 0.46 | -0.13 | 0.30 | ~31% |
| cur5 5.4M | 0.55 | 0.65 | 0.54 | 0.74 | 0.33 | 1.32 | -0.27 | ~30% |
| **cur5 7.2M** | 0.53 | 0.63 | 0.61 | 0.59 | 0.49 | 0.64 | 0.19 | ~32% |
| cur5 9.0M | 0.50 | 0.63 | 0.57 | 0.84 | 0.35 | 0.38 | 0.21 | ~31% |
| cur6 5.4M | 0.61 | 0.82 | 0.44 | 0.93 | -0.00 | 1.58 | -0.10 | ~32% |
| cur6 7.2M | 0.59 | 0.72 | 0.42 | 1.07 | -0.23 | 1.39 | 0.19 | ~32% |
| cur6 9.0M | 0.54 | 0.68 | 0.47 | 1.13 | -0.29 | 0.86 | 0.43 | ~32% |

cur5 7.2M is the only one with every direction the right way; it also
stands still on a zero command (env eval: vx -0.002, vy -0.004, wz -0.02).

Open issues, for the user to decide:

- **Speeds are small.** Trained on +-0.1 m/s and +-0.1 rad/s only; the
  original goal was up to 0.3 m/s forward. Next step would be a stage that
  widens the ranges from this checkpoint.
- **~30% fall within 8 s under the training env's pushes** (0.1-1.0 m/s
  kicks every 5-10 s), even on a stand command. Without pushes (MjInfer)
  none fall.
- **Turning is weak and left/right asymmetric**: the reference's turning
  gait itself only turns ~0.025 rad/s. Regenerating the reference (stronger
  turning keys) is the user's call.
- **Command following peaks at ~5-7M steps after each change and then
  decays** while the training reward keeps rising (cur4, cur5); doubling the
  tracking weights (cur6) helped some directions and broke strafe right.
  Mirror (left/right) augmentation is a candidate for the asymmetry.
- The robot model will gain a shell and a head battery; since observation
  and action sizes stay the same, this checkpoint can be fine-tuned on the
  new model rather than retrained from scratch.

### cur5: command following peaks early, then decays

`eval_command_tracking.py` gained `--env N`: N robots in the training env
itself (noise, delays, pushes; nominal model), averaged. It agrees with the
single-run eval -- cur5 5.4M 0.55 vs 14.4M 0.33 -- so the decline is real,
not eval noise. In the training env every per-step term keeps improving
(lin tracking 0.267 -> 0.356, gait 0.58 -> 0.76, action_rate 1.47 -> 0.69
from 0 to 10.8M), but those are averaged over random mixed-direction
commands with gait (6), alive (5) and imitation (~2/step) paying more than
tracking (2.5 / 6): after ~5-7M the policy trades pure-direction command
following for them, in cur4 and cur5 alike. cur6 doubles both tracking
weights from the best cur5 checkpoint. Also visible in the env eval: 30-40%
of robots fall within 8 s with pushes on.

cur4, every checkpoint with `eval_command_tracking.py --scale 0.4 --turn 0.1`:

| ckpt | fwd | back | left | right | turn L | turn R | falls |
|---|---|---|---|---|---|---|---|
| 0 | 1.27 | 0.34 | 0.83 | 0.88 | 0.00 | 0.00 | 2/7 |
| 1.8M | 0.89 | 0.29 | 0.87 | 0.89 | 0.01 | 0.21 | 0/7 |
| 3.6M | 1.08 | 0.27 | 1.03 | 0.46 | 0.37 | -0.11 | 0/7 |
| 5.4M | 1.13 | 0.70 | 1.00 | 0.37 | 1.76 | -1.19 | 0/7 |
| 7.2M | 1.14 | 0.38 | 0.98 | 0.12 | 1.35 | -0.64 | 0/7 |
| 9.0M | 0.99 | 0.19 | 1.15 | 0.34 | 0.00 | -0.45 | 0/7 |
| 10.8M | 1.06 | 0.28 | 1.14 | 0.13 | 0.06 | -0.78 | 0/7 |
| 12.6M | 0.81 | 0.17 | 1.44 | -0.14 | 0.28 | -1.03 | 0/7 |

Likely cause of the decline: with yaw commands snapping to the straight
reference key, tracking asks for turning while imitation asks for straight
stepping. cur5 scales the yaw command x10 before the reference lookup, so
+-0.05..0.1 rad/s selects the +-0.98 turning key (which does use hip_yaw,
-0.12..+0.10 rad); `tracking_ang_vel` still uses the real command.

### cur4: turning

The 0.4x-scaled eval's turn commands (+-0.04 rad/s) are within the robot's
own yaw drift, so those two scores are noise. At the full trained rate
(+-0.1 rad/s) the 1.8M and 3.6M policies turn +-0.01..0.06 rad/s,
about as much as they drift with no command; adding a yaw command to a
forward command also stops the forward motion. Likely reason: a yaw
command of 0.1 snaps to the reference grid's dtheta=0 key (grid 0,
+-0.98, +-1.96), so imitation asks for straight stepping; and even the
reference's own turning keys only turn ~0.025 rad/s (label 0.98). Only
`tracking_ang_vel` asks for turning, with no gait to copy -- a
reference-motion limitation (regenerating it is the user's call).

5.4M (full-rate check, MjInfer 8 s): backward up to 0.70 (eval scale 0.4);
the policy now turns, but always left -- stand 0 -> wz +0.06 (and creeps
fwd 0.04 m/s), turn +0.1 -> +0.18, turn -0.1 -> +0.12, strafe right ->
wz +0.14. Not from the reference (symmetric: mean hip_yaw 0, hip_roll
-0.004/+0.004) and hardly from the model: right-leg segment CoMs sit 2-4 mm
closer to the midline than the left (world y L +0.037 vs R -0.033 for
thigh/shin), whole-body CoM 0.8 mm off center.

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

**First command-following policy** (cur1c 1.8M, MjInfer, 8 s, no pushes):

| fwd command | achieved | foot contact L / R | max foot lift |
|---|---|---|---|
| 0.00 | 0.001 m/s | 0.94 / 0.85 | ~0 mm |
| 0.05 | 0.011 m/s | 0.85 / 0.71 | 2 mm |
| 0.08 | 0.030 m/s | 0.76 / 0.72 | 2 mm |
| 0.10 | 0.035 m/s | 0.66 / 0.75 | 1 mm |

Speed now scales with the command and it stands on a zero command; feet
leave the ground 25-35% of the time (was 1%) but barely lift -- a shuffle.

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

## Final-URDF model (2.072 kg), 2026-11-03

User merged the final-URDF remodel (laptop) and this branch into master and
asked for the direction curriculum **from scratch** on the new model, running
02:10 -> 12:00. Same reward config as the chosen cur5 policy, except
`feet_clearance_target_z` 0.07 -> 0.03 (the new MJCF's foot sites sit at the
sole: 0.000 at the home pose). The `feet_gait` offsets (0.75, 0.25) still
agree with the new reference's contacts on 95% of gait steps. New model:
servo force range 1.91 N·m (was 3.23); open-loop reference playback hits it
21-33% of the time on hip_pitch/knee/ankle (old model 4-18%).

| run | stage | steps | result | notes |
|---|---|---|---|---|
| nm1 | forward only `--lin_vel_x 0.05 0.1 --lin_vel_y 0 0 --ang_vel_yaw 0 0`, from scratch (restarted 02:20 with actuator kp/forcerange 17.8/3.35, see below) | 7.2M (stopped) | fwd 0.23 / 0.19 / 0.41 / 0.46 at 1.8 / 3.6 / 5.4 / 7.2M | moves forward from 1.8M but drifts sideways (vy -0.10 at a 0.08 fwd command); balance slow: ep_len 74 / 89 / 108 / 144 (old model ~95 / 160 / 250 / 335) |
| nm1b | nm1 7.2M + `randomize.py` torso fix (`TORSO_BODY_ID` 1 -> 2) | 1.8M (stage done) | **fwd 0.82**, back 0.46 (untrained) | ep_len 144 -> 76 at the restore (torso mass/CoM randomization now really applied) -> 89; still drifts sideways (vy -0.09) |
| nm2 | nm1b 1.8M + backward `--lin_vel_x -0.1 0.1` | 1.8M (stopped) | fwd 0.90 / **back 0.61** | ep_len stuck at ~82: the +-5 cm torso CoM jitter (live since the fix) is near-unsurvivable sideways; reduced to +-1 cm |
| nm2b | nm2 1.8M + torso CoM jitter +-1 cm | 1.8M (stage done) | **fwd 1.17 / back 0.63** | ep_len 82 -> 183 at the restore (same weights, gentler jitter) -> 217; sideways drift on fwd vy -0.106 -> -0.037; turns while backing up (wz -0.23) |
| nm3 | nm2b 1.8M + strafe `--lin_vel_y -0.1 0.1` | 1.8M (stage done) | fwd 1.34 / back 0.42 / **left 0.97 / right 1.07** | strafe learned within one checkpoint and symmetric; ep_len 55 at the restore -> 201 |
| nm4 | nm3 1.8M + turning `--ang_vel_yaw -0.1 0.1` | 12.6M (done) | best balanced **9.0M: 0.79** -- fwd 0.88 / back 0.98 / left 0.75 / right 1.15 / turn L 0.71 / turn R 0.25, no falls | no falls from 1.8M on; ep_len 53 -> 165 / 227 / 318 / 373 / 427 / 493; 7.2M 0.75 (all positive), 10.8M 0.89 / 12.6M 0.90 but turn L or strafe L drops; turning bias flips between checkpoints (right at 3.6-5.4M) |
| nm5 | nm4 9.0M + wider speeds `--lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15` | 12.6M (done 10:56) | see final comparison | ep_len 324 -> 489; no falls in any single-run eval; raw turning stays one-sided |

### Final result, final-URDF model (2026-11-03 11:20)

**Chosen: nm5 3.6M with symmetric inference** --
`checkpoints/nm5/2026_11_03_090557_3604480.onnx` (lab PC, not in git),
run with `mujoco_infer.py --symmetric`. Config: master as committed
(BAM actuators, torso CoM jitter +-1 cm, reward/curriculum as cur5) with
`--lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15`.
Lineage: nm1 -> nm1b -> nm2 -> nm2b -> nm3 -> nm4 (9.0M) -> nm5.

`eval_command_tracking.py --env 64 --scale 0.75 --turn 0.15` (64 robots in
the training env, noise/delays/pushes, nominal model; commands fwd 0.15,
back -0.11, strafe +-0.11 m/s, turn +-0.15 rad/s). `|r-1|` = mean
distance of the six ratios from 1 (overshoot counts):

| checkpoint | mode | `|r-1|` | fwd | back | left | right | turn L | turn R | fell (8 s) |
|---|---|---|---|---|---|---|---|---|---|
| nm4 7.2M | sym | 0.39 | 0.56 | 0.46 | 0.58 | 0.46 | 0.73 | 0.85 | 31% |
| nm4 9.0M | sym | 0.32 | 0.62 | 0.55 | 0.77 | 0.62 | 0.72 | 0.81 | 29% |
| **nm5 3.6M** | **sym** | **0.28** | 0.74 | 0.63 | 0.84 | 0.74 | 0.59 | 0.77 | 31% |
| nm5 5.4M | sym | 0.28 | 0.82 | 0.66 | 0.94 | 0.86 | 0.48 | 0.56 | 32% |
| nm5 7.2M | sym | 0.40 | 0.95 | 0.72 | 0.97 | 0.94 | -0.05 | 0.07 | 31% |
| nm5 12.6M | sym | 0.32 | 1.18 | 0.82 | 0.91 | 0.85 | 0.26 | 0.39 | 32% |
| nm4 7.2M | raw | 0.60 | 0.47 | 0.50 | 0.61 | 0.44 | -0.35 | 0.75 | 33% |
| nm4 9.0M | raw | 0.66 | 0.50 | 0.56 | 0.62 | 0.61 | -0.17 | -0.11 | 32% |
| nm5 3.6M | raw | 0.49 | 0.65 | 0.62 | 0.79 | 0.72 | 0.40 | -0.12 | 32% |
| nm5 5.4M | raw | 0.33 | 0.76 | 0.65 | 0.76 | 0.80 | 0.49 | 0.56 | 34% |
| nm5 7.2M | raw | 0.43 | 0.82 | 0.75 | 0.80 | 0.73 | 0.50 | -0.19 | 35% |
| nm5 12.6M | raw | 0.26 | 1.00 | 0.81 | 0.87 | 0.64 | 0.19 | 1.07 | 33% |

nm5 3.6M symmetric is the most even (every direction >= 0.59, both turns);
nm5 5.4M symmetric ties on `|r-1|` but turns weaker; raw nm5 12.6M scores
0.26 but turns left only 0.19. Without pushes (single-run eval) none fall.

Open issues: ~30% of robots fall within 8 s under the training env's pushes
(same as the old model); speeds are still small (0.15 m/s fwd) next to the
original 0.3 m/s goal; slight forward creep (~0.03 m/s) on a stand command.

Symmetric inference (`mujoco_infer.py --symmetric`, a = (pi(o) + M pi(M o))/2,
inference only): on nm4 5.4M the yaw rate on forward went -0.51 -> +0.00,
standing -0.28 -> +0.01, backward -0.20 -> -0.01, and turn left/right both
went the right way (+0.04/-0.03, raw -0.02/-0.14); strafe right weakened.
The robot model is symmetric (whole-body CoM y +0.4 mm), so the per-run
turning bias is the policy's.

Actuators: the first nm1 start (02:10) used the datasheet torque 1.91 N*m /
kp 13.37 and was stopped after ~30 min (no checkpoints kept). Upstream Open
Duck Mini v2 uses the same 7.4 V STS3215 with BAM-identified kp 17.8 /
forcerange 3.35; switched to those (`docs/CHANGES_FROM_ORIGINAL.md`, final
URDF remodel section). Reference-gait saturation on hip_pitch/knee/ankle:
21-33% -> 10-16% of the time.

## Long run, 200M steps (head-v2 model), started 2026-11-04

User asked for ~200M steps (vs ~27M for the current policy; upstream trains
150-300M), ~40 h on the lab PC at ~1400 steps/s, while the robot hardware is
being tested. Run in ~25M-step segments from the current policy (nm5 3.6M,
which already follows every direction on the head-v2 model: symmetric env
eval `|r-1|` 0.29), each restoring the best checkpoint of the previous one,
widening the command ranges toward the 0.3 m/s goal. Every checkpoint keeps
its eval line and a 30 s / 20-robot video (nothing deleted this time).

The reference motion is not regenerated (user's decision, 2026-11-05): strafe
(~0.57x of the key) and turning (at most 0.11 rad/s) stay limited by the
current Placo sweep; CHANGES #13-#15 get what they can out of it.

| run | ranges | steps | result | notes |
|---|---|---|---|---|
| long1 | `--lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15` (same as nm5), from nm5 3.6M | 19.8M (stopped) | env eval (64 robots, scale 0.75 / turn 0.15): best 14.4M sym `|r-1|` 0.37 / raw 0.33, vs nm5 3.6M sym **0.28** | translation stronger (fwd/back ~1.0-1.25) but turning lost in symmetric mode (0.13/0.15; raw turns right only); falls unchanged ~32%; single-run `|r-1|` peaked at 14.4M (0.37) then 0.40 / 0.48 / 0.55 |
| long2 | same ranges, from **nm5 3.6M**, + 30% single-axis commands (CHANGES #10) | 7.2M (stopped) | single-run raw `|r-1|` 0.60 -> 0.42 / 0.46 / 0.31 / **0.29** (7.2M: fwd 1.21 / back 0.94 / left 1.05 / right 1.09 / turn L -0.01 / turn R 1.35) | translation close to 1 without long1's overshoot; turning still right-only, and symmetric inference turns weakly (5.4M: 0.38/0.21) |
| long3 | long2 7.2M + **symmetric policy** (`--symmetric_policy`, CHANGES #11), same ranges | 3.6M (stopped) | sym-trained single-run `|r-1|` 0.42 / 0.46 / 0.56; turning symmetric but ~0 (0.28/0.15 -> -0.02/-0.05) | found why turning never trains: CHANGES #12 |
| long4 | long3 3.6M + yaw tracking on a 0.4 s low-passed yaw rate (CHANGES #12), symmetric policy, same ranges | 25.2M | 64-robot env eval (sym-trained) best 12.6M: `|r-1|` 0.30, fwd/back/left/right/turn_l/turn_r 1.14/1.08/0.71/0.61/0.55/0.54, ~33% fall in 8 s; later checkpoints 0.36-0.40 (strafe decays to ~0.5) | turning now learns (0 -> ~0.5) but ratios sit on the reference's own speeds (fwd 1.37x, strafe 0.65x, turn 0.73x of the command): CHANGES #13 |
| long5 | long4 12.6M + imitation base-velocity targets = command (CHANGES #13), same ranges | 7.2M (stopped) | sym-trained single-run `|r-1|` 0.43 / 0.53 / 0.46 / 0.51; forward stays 1.4x, strafe 0.7, turning 0.05-0.33 | the reference lookup snaps to a 0.098 m/s grid, so the joint terms still ask for the 0.196 gait: CHANGES #14 |
| long6 | long4 12.6M + #13 + interpolated reference lookup (CHANGES #14), same ranges | 7.2M (stopped) | 64-robot env eval: 5.4M `|r-1|` 0.38 (1.15/1.05/0.68/0.59/0.37/0.29), 7.2M 0.41; long5 5.4M 0.33 (1.10/1.10/0.69/0.61/0.46/0.47); ~33% fall | no gain over long4 12.6M (0.30); with ref_yaw_scale 10 the interpolated turn reference got slower (0.067 rad/s at 0.15): CHANGES #15 |
| long7 | long4 12.6M + #13 + reference looked up by achieved speed (CHANGES #15), same ranges | 25.2M | 64-robot env eval (sym-trained): 25.2M `|r-1|` 0.30, fwd/back/left/right/turn_l/turn_r 1.02/1.06/0.55/0.47/0.63/0.66; 18.0M-23.4M 0.31-0.33; ~34% fall in 8 s | forward/back now ~1.0 and turning best so far (0.6), strafe down to ~0.5. Single-run evals swing turning 0-0.5 between checkpoints while the training env's tracking_ang_vel climbs steadily: judge on the 64-robot eval only |
| long8 | long7 25.2M, wider ranges `--lin_vel_x -0.2 0.3 --lin_vel_y -0.15 0.15 --ang_vel_yaw -0.15 0.15` | 7.2M (stopped) | reward 167 -> 173, ep_len 525 -> 540 | stopped for the upright-torso change |
| long9 | long8 7.2M + torso orientation cost -5 (CHANGES #16), same ranges as long8 | 3.6M (stopped) | torso pitch standing 11.1 -> 10.9 -> 11.3 deg: no effect | weight too small (~1% of the step reward) |
| long10 | long9 3.6M + orientation cost -20, same ranges | 25.2M | torso pitch standing 11.3 -> 10.0 (7.2M) -> 9.1 (14.4M) -> 7.9 deg (25.2M); single-run `|r-1|` 0.35 at 7.2M, but backward collapses from 18M (1.34 -> 0.00) and strafe drops to ~0.2 | forcing the hips back in sim stands upright but stops forward walking: the reference gait walks leaning forward. User picked continuing to straighten while walking (from 14.4M) |
| long11 | long10 14.4M, same config | stopped at start | | replaced by long12 at the user's sim-to-real request |
| long12 | long10 14.4M + CHANGES #17 (action_rate -1.0, wider randomization, `--lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15`), orientation -20 | 25.2M | 64-robot env eval (with the wider randomization): 12.6M `|r-1|` **0.29** (1.02/0.76/0.65/0.60/0.60/0.65), 7.2M 0.37, 16.2M 0.38, 25.2M 0.40; ~33% fall in 8 s. Torso pitch standing 9.1 -> 8.2 (12.6M) -> 6.3 deg (25.2M) | slow backward (-0.1) stops from 7.2M on (single-run back ratio swings 0-1.4 around the eval command's threshold); strafe decays after 18M |
| long13 | long12 12.6M + orientation cost off while walking backward (CHANGES #16), same config | 25.2M | 64-robot env eval: **16.2M `|r-1|` 0.19** (0.95/1.01/0.80/0.72/0.65/0.73), 10.8M 0.20, 18.0M 0.22, 23.4M 0.26, 25.2M 0.25; ~33% fall in 8 s. Torso pitch standing 8.1-8.9 deg | best policy so far; slow backward (-0.1) still stands in the noise-free single run |
| long14 | long13 16.2M, same config, last ~22M of the 200M | 23.6M | 64-robot env eval: **7.9M `|r-1|` 0.11** (0.98/0.98/0.97/0.85/0.76/0.84), 5.9M 0.16, 9.8M 0.20, 11.8M 0.16, 15.7M 0.20, 19.7M 0.21, 23.6M 0.25; ~33% fall in 8 s. Torso pitch 8.5 / 8.1 / 5.5 deg (stand / forward / turn) at 7.9M | **final policy** of the long run. Noise-free single run still stands still on very slow commands (+-0.05 m/s, strafe 0.06), -0.1 back now walks |
| long15 | long14 7.9M + CHANGES #18-#20 (stand_still -1.0, head_pos -1.0, floor friction 0.15-1.0), `--lin_vel_x -0.15 0.2 --lin_vel_y -0.06 0.06 --ang_vel_yaw -0.15 0.15` | 25.2M | 64-env (same new env, scale 0.75 turn 0.15): long15 3.6M 0.14, 7.2M 0.16, 10.8M 0.21, 14.4M 0.19, 18.0M 0.19, 21.6M 0.19, 25.2M 0.20 vs long14 7.9M re-evaluated 0.12; falls ~33% for all | Not adopted: long14 7.9M stays deployed. Train reward 138 -> 110, ep_len 544 -> 427; strafe and turn ratios dropped (25.2M left/right 0.78/0.70, turn 0.67/0.73). Torso stand pitch 8.5 -> 6.8 deg, walking 7-10 deg; neck stand -11.7 -> -12.8 deg, so head_pos -1.0 did not lift the head. Note for the real-robot neck droop: in sim long14 stands with neck_pitch -11.4 deg (forward; actual follows the target within 0.3 deg at 0.085 N*m), and the policy cannot command more than +-14.3 deg from home (tanh action x 0.25 rad), so a near-horizontal (~-90 deg) droop on hardware is not something the policy asked for: check the neck servo (holding torque, zero, sign, battery voltage) first |
| long16 | long14 7.9M + CHANGES #21 (upright reference and home pose, strafe back to +-0.12), `--lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15 --symmetric_policy` | 100M planned | | running (started 2026-11-11 20:47). Step 0 (long14 on the new pose): reward 47, ep_len 235, torso stand -2.5 deg, neck -10.0 deg. Run with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.78` (0.85 hit CUDA OOM / cuSolver errors with two MuJoCo viewers open) |

The reference turning keys are symmetric (base yaw rate +-0.025 at +-0.98,
+-0.110 at +-1.961, same hip_yaw swing), so the one-sided turning is the
policy's own symmetry breaking. Training commands mix all three axes, while
the eval (and the robot) use pure ones -- long2 adds single-axis commands.

### Long run summary (2026-11-04 03:00 -> 2026-11-06 03:20, ~202M steps)

Final policy: `checkpoints/long14/2026_11_06_000746_7864320.onnx` (lab PC
only), trained with `--symmetric_policy`, run with `--symmetric_trained`.
64-robot env eval (wider randomization, pushes), command ratios
fwd/back/left/right/turn_l/turn_r 0.98/0.98/0.97/0.85/0.76/0.84, `|r-1|`
0.11 (nm5, the policy before the long run: 0.29 with forward/back at
0.74/0.63). Torso pitch ~8 deg standing (nm5 ~11). Falls under pushes
unchanged at ~33% within 8 s.

What moved it, in order: turning reward on a low-passed yaw rate (#12),
reference looked up by achieved speed (#14, #15) with command-targeted base
velocity (#13), and the sim-to-real config (#16 orientation cost off while
walking back, #17 action_rate -1.0, wider randomization, 0.2 m/s top speed).
Judge checkpoints on the 64-robot eval only: the per-checkpoint single-run
eval swings turning and backward between 0 and 1.8.
