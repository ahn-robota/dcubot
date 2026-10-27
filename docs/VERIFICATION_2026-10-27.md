# Verification results (2026-10-27, laptop RTX 4050 6GB)

Baseline for comparison before the lab computer's main training run
(30-50M steps). See `docs/CHANGES_FROM_ORIGINAL.md` for what the config
actually is; this doc is just the verification numbers.

## Dimensions (smoke test)

```
.venv/bin/python -c "
import jax
from playground.dcubot.joystick import Joystick
env = Joystick(task='flat_terrain')
state = env.reset(jax.random.PRNGKey(0))
print(env.observation_size, env.action_size, state.obs['state'].shape)
"
```

- `observation_size`: `{'privileged_state': (159,), 'state': (94,)}`
- `action_size`: 13
- No crash, no NaN on reset + step.

Sampled command ranges (200 draws of `env.sample_command`), confirmed
within the config's intended bounds:
```
lin_vel_x:   -0.198 .. 0.300   (config: [-0.20, 0.30])
lin_vel_y:   -0.156 .. 0.157   (config: [-0.158, 0.158])
ang_vel_yaw: -0.102 .. 0.103   (config: [-0.103, 0.103])
```

## Reference replay test (reward_imitation_dcubot slicing check)

```
.venv/bin/python scripts/replay_reference.py --keys 0.098_0.0_0.0
.venv/bin/python scripts/replay_reference.py --keys=-0.098_0.0_0.0
.venv/bin/python scripts/replay_reference.py   # all 210 keys
```

(`--keys` is `nargs="*"`, so repeating the flag keeps only the last value --
one key per run, or no `--keys` for all keys.)

Teleports qpos/qvel directly from the fitted
`polynomial_coefficients_omnidirectional.pkl` and checks
`reward_imitation_dcubot`'s components:

- `joint_pos_rew`: 1.000 (max) for every key/frame checked -- confirms the
  Placo-order -> actuator-order remap (`[left(5), right(5), head(3)]`) is
  correct.
- `joint_vel_rew`: 1.000 (max), same confirmation for velocities.
- `contact_rew`: ~0.40-0.60 (not max) -- this is a known limitation of this
  *test method*, not the reward code: the script holds the robot's base
  pose fixed at the home keyframe instead of also replaying the
  recording's own base trajectory, so the feet don't land at the same
  height/position the recording assumes. Confirmed separately: the actual
  visual check used a different script
  (`scripts/replay_directions_mujoco.py`, which DOES replay the full base
  trajectory) and looked correct once a joint-order remap bug in *that*
  script was fixed -- see git history / project memory for that bug.

## Short verification training (completed, not resumed into main training)

Command used:
```
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
.venv/bin/python -u -m playground.dcubot.runner \
  --task flat_terrain_train \
  --output_dir checkpoints_verify_omnidirectional \
  --num_timesteps 2000000 \
  --num_evals 12
```
(`--num_envs` defaulted to 1024 here; this flag didn't exist yet at
training time, added afterward -- see `playground/dcubot/runner.py`.)

Ran from ~02:51 to ~05:37 KST. Overshot the nominal 2,000,000-step target
to 3,604,480 due to brax's chunk-size rounding with `--num_evals 12` (not
an error -- same documented mechanic as other runs this project has done).
No crashes, no NaNs, no errors. Final checkpoint:
`checkpoints_verify_omnidirectional/2026_10_27_053737_3604480` (+ `.onnx`,
not tracked in git -- checkpoints are gitignored by design).

| step | reward | reward_std | ep_len | sps |
|---|---|---|---|---|
| 0 | 25.4 | 11.2 | 73.6 | — |
| 327,680 | 25.4 | 11.5 | 73.7 | 451 |
| 655,360 | 26.1 | 11.8 | 75.9 | 465 |
| 983,040 | 29.4 | 14.5 | 84.8 | 494 |
| 1,310,720 | 33.4 | 16.7 | 96.2 | 494 |
| 1,638,400 | 34.4 | 21.6 | 98.4 | 495 |
| 1,966,080 | 39.0 | 24.0 | 110.8 | 495 |
| 2,293,760 | 40.6 | 25.5 | 116.0 | 495 |
| 2,621,440 | 42.6 | 26.0 | 120.8 | 495 |
| 2,949,120 | 48.1 | 27.0 | 137.2 | 495 |
| 3,276,800 | 57.7 | 40.6 | 162.0 | 495 |
| 3,604,480 (final) | 59.8 | 37.3 | 166.4 | 495 |

Monotonic improvement throughout, no regressions, ~495 steps/s steady
state on this GPU (RTX 4050 laptop, 6GB, `num_envs=1024`).

## Visual confirmation (not numeric, recorded for completeness)

```
.venv/bin/python scripts/replay_directions_mujoco.py
```
Replays the RAW Placo recordings (not this pkl) for forward/backward/
strafe-left/strafe-right/turn-left/turn-right, 3s each, in the live MuJoCo
viewer. User visually confirmed this looks correct after a joint-order
remap bug in this script was fixed (left/right/head block ordering).

## Known caveats carried into the lab run

- `ang_vel_yaw` command range (+-0.103 rad/s) is real-achieved-velocity
  based, much smaller than the pkl's raw intention labels (up to +-1.961)
  -- see `docs/CHANGES_FROM_ORIGINAL.md` #2. Turning is driven mostly by
  `tracking_ang_vel`, not `imitation`, for this reason.
- `cost_feet_clearance`/`y_tol` lateral-wobble measurement (mentioned as
  pending in `docs/CHANGES_FROM_ORIGINAL.md` #4) was not done.

## Lab PC reproduction (2026-10-27, RTX 3080 10GB)

Same commands as above, after `uv sync --locked` (Python 3.13.13, jax/jaxlib
0.9.0, playground 0.0.5, mujoco/mujoco-mjx 3.12.0, brax 0.14.2).

- `jax.devices()`: `[CudaDevice(id=0)]`
- Dimensions: identical (`privileged_state` 159, `state` 94, action 13).
  50 zero-action steps after reset: no NaN, no termination.
- Command sampling (200 draws, different seed): lin_vel_x -0.200..0.295,
  lin_vel_y -0.155..0.157, ang_vel_yaw -0.102..0.102 -- within config bounds.
- Reference replay, all 210 keys: `joint_pos_rew` 1.000 and
  `joint_vel_rew` 1.000 (min 1.000 on every key); `contact_rew` 0.400 on
  every key (sim contacts are `[0, 0]` throughout -- the fixed home base
  pose keeps the feet off the floor, the test-method limitation described
  above).

Matches the laptop results.
