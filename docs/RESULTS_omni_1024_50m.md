# Results: omni_1024_50m (lab PC, 2026-10-27/28)

**Outcome: the policy does not follow velocity commands. It stays upright
and steps in place.** Training itself ran cleanly; the problem is what it
learned.

## Run

- Lab PC (RTX 3080 10GB), 2026-10-27 15:21 -> 10-28 09:39 (~18 h), no errors.
- `--task flat_terrain_train --num_envs 1024 --num_timesteps 50000000`
  (`--num_evals` default 30), `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`. Everything
  else as in the laptop verification run (`docs/VERIFICATION_2026-10-27.md`).
- brax rounds to whole 163,840-step chunks: 29 x 1,802,240 = 52,264,960 steps.
- 814 steps/s steady state.
- Checkpoints (30, each with `.onnx`) stay on the lab PC in
  `checkpoints/omni_1024_50m/` (not in git).

## Learning curve

| step | reward | ep_len |
|---|---|---|
| 0 | 25.6 | 74 |
| 3.6M | 57.0 | 162 |
| 9.0M | 161.5 | 438 |
| 16.2M | 242.1 | 620 |
| 16M-52M | 230-273 | 575-653 |
| 52.3M (final) | 270.9 | 621 |

Best eval: 273.0 / 639.9 at 41,451,520. After 16M the total is flat.

## Reward components

TensorBoard's `eval/episode_reward/*` and `eval/episode_cost/*` are
per-episode sums **already multiplied by their weight**; `episode_reward` =
0.02 (ctrl_dt) x (sum of rewards - sum of costs) -- checked exactly at 23.4M.
Below each term is divided by its weight and by ep_len, i.e. a per-step
average (tracking terms max 1.0; costs: lower is better).

| step | lin_vel | ang_vel | imitation | action_rate | torques | stand_still | swing_peak (raw) |
|---|---|---|---|---|---|---|---|
| 0 | 0.171 | 0.041 | 0.208 | 6.82 | 20.3 | 0.98 | 1.97 |
| 9.0M | 0.261 | 0.045 | 0.196 | 4.89 | 17.2 | 1.15 | 7.45 |
| 16.2M | 0.335 | 0.063 | 0.188 | 3.15 | 12.6 | 1.50 | 8.60 |
| 27.0M | 0.360 | 0.092 | 0.190 | 1.88 | 8.4 | 0.60 | 7.04 |
| 41.5M | 0.405 | 0.149 | 0.178 | 1.24 | 6.0 | 0.61 | 5.68 |
| 52.3M | 0.400 | 0.205 | 0.179 | 1.02 | 5.0 | 0.44 | 5.02 |

`alive` is ~98% of the total reward (e.g. 243 of 271 at 52.3M), so the total
tracks ep_len.

## Command-tracking check

Policy stepped in the training env itself (`Joystick`, task
`flat_terrain_train`, pushes on), command held fixed, 8 s, mean world-frame
base velocity after the first 2 s:

| command | final 52.3M | 41.5M | 16.2M |
|---|---|---|---|
| fwd 0.3 m/s | vx -0.002 | -0.027 | -0.014 |
| fwd 0.15 m/s | -0.053 (pushed, fell) | -0.029 | -0.053 |
| back -0.2 m/s | -0.002 | -0.000 | -0.001 |
| left 0.158 m/s | vy -0.029 (pushed, fell) | +0.000 | -0.024 |

The ~0.05 m/s values are pushes, not commanded motion. `mujoco_infer.py`
(after the fixes below) gives the same picture: upright for 8 s on all
seven commands, net velocity ~0.

So the rising tracking terms above do not mean the robot moves: with these
small command ranges and `tracking_sigma=0.01`, standing still already
scores `exp(-v_cmd^2 / 0.01)`, e.g. 0.37 for a 0.1 m/s command.

## Likely cause (not yet confirmed)

Standing still is a cheap local optimum: `alive` (weight 20) dominates and
zero velocity already earns a large part of the tracking reward over this
command range. To check next:

- Did the laptop verification run (3.6M) policy walk? If not, this was there
  from the start.

  **Checked (2026-10-28, laptop).** Same test, laptop's final checkpoint
  (`checkpoints_verify_omnidirectional/2026_10_27_053737_3604480.onnx`),
  `Joystick`/`flat_terrain_train`, pushes on, 8s, mean world linvel after 2s:

  | command | mean world linvel | done |
  |---|---|---|
  | fwd 0.3 | (-0.049, -0.032, -0.063) | **1.0 (fell)** |
  | fwd 0.15 | (-0.047, -0.031, -0.040) | **1.0 (fell)** |
  | back -0.2 | (-0.0008, 0.0002, 0.0000) | 0.0 |
  | left 0.158 | (-0.0001, 0.003, 0.00003) | 0.0 |

  So: not walking even at 3.6M -- backward/left already show the same
  near-zero velocity as the 52.3M policy; forward commands fall rather than
  stand (less trained, more exposed to `push_config` kicks). This confirms
  the problem was present from early in training, not something that
  emerged only after 16M -- consistent with a structural cause (command
  range / `tracking_sigma` / `alive` dominance) rather than "just needs
  more steps."
- Compare with upstream: command ranges vs `tracking_sigma`, and upstream's
  300M steps / 8192 envs.

  **Checked (2026-10-28, laptop).** Upstream (`open_duck_mini_v2/joystick.py`):
  `tracking_sigma=0.01` (same value dcubot currently uses), `lin_vel_x=[-0.15,0.15]`,
  `lin_vel_y=[-0.2,0.2]`, `ang_vel_yaw=[-1.0,1.0]`, `alive=20.0` (same as dcubot).

  `reward_tracking_*` is `exp(-error^2/sigma)` -- an ABSOLUTE error scale,
  independent of the command range. So standing still (v=0) when commanded
  `cmd` scores `exp(-cmd^2/sigma)`. The ratio `max_command / sqrt(sigma)`
  (sqrt(0.01)=0.1 for both) says how much that reward actually drops at
  the edge of the command range -- a low ratio means even the extreme
  command barely distinguishes "stand" from "comply":

  | axis | upstream range | upstream ratio | dcubot range | dcubot ratio |
  |---|---|---|---|---|
  | lin_vel_x | +-0.15 | 1.5 | [-0.20, 0.30] | 2.0-3.0 |
  | lin_vel_y | +-0.2 | 2.0 | +-0.158 | 1.58 |
  | ang_vel_yaw | +-1.0 | **10.0** | +-0.103 | **1.03** |

  `lin_vel_x`/`lin_vel_y` are roughly in the same ballpark as upstream (not
  the likely culprit by themselves). `ang_vel_yaw` is the outlier: at
  upstream's ratio of 10, standing still at the command extreme scores
  `exp(-100)` (~0) -- essentially forced to turn to get any
  `tracking_ang_vel` reward. At dcubot's ratio of ~1, standing still at the
  command extreme already scores `exp(-1)` ~= 0.37, and for the ~40% of
  draws with `|cmd| < 0.1` (within the command's own 0.2-0.29 span) it
  scores well above that. Combined with `alive` being ~98% of total reward,
  there's very little pressure left to actually move for this axis -- this
  matches deviation #2's own module-docstring caveat in `joystick.py`
  ("ang_vel_yaw... driven mostly by tracking_ang_vel... not expected to be
  a major issue in practice"), which this result suggests was too
  optimistic.

  This does NOT fully explain why forward/backward/strafe also show ~0
  velocity, though (their ratios aren't obviously bad vs upstream). Two
  candidates, not distinguished yet: (a) `alive`'s outsized share of reward
  makes standing a safe/cheap optimum regardless of axis, and the policy
  generalizes "don't move" across all axes once it learns that for
  `ang_vel_yaw`; (b) training scale -- 52.3M steps / 1024 envs here vs
  upstream's up to 300M steps / 8192 envs (far more parallel exploration),
  so dcubot's policy may simply not have had enough exploration pressure to
  find/commit to the riskier "actually walk" strategy regardless of the
  sigma issue above.
- Any reward/command change needs approval and an entry in
  `docs/CHANGES_FROM_ORIGINAL.md`.

## Side finding: accelerometer offset is a no-op in training

`joystick.py` `_get_obs` calls `accelerometer.at[0].set(accelerometer[0] + 1.3)`
without assigning the result (JAX arrays are immutable), so the policy is
trained on the raw accelerometer. `mujoco_infer.py` used to add +1.3 and
now does not.

## mujoco_infer.py fixes

It was still the pre-restart version (`ba85c94`) and did not match the
current `joystick.py`:

- imported the deleted `debug_vis` module (would not start) -> removed
- gait phase was `[sin, cos]` and zeroed for small commands -> `[cos, sin]`,
  always running, period from `default_config().gait_period`
- accelerometer +1.3 -> removed (see above)
- arrow-key command ranges were old (yaw +-1.0) -> read from
  `default_config()` (`lin_vel_x/lin_vel_y/ang_vel_yaw`)
