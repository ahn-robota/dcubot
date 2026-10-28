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
- Compare with upstream: command ranges vs `tracking_sigma`, and upstream's
  300M steps / 8192 envs.
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
