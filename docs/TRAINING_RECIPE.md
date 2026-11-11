# Training recipe: command-following walking policy

How the current best policy was trained (final-URDF model, 2.072 kg, lab PC
2026-11-03), and how to repeat it. The reasoning and every failed attempt
are in [`AUTO_EXPERIMENTS_2026-10-28.md`](AUTO_EXPERIMENTS_2026-10-28.md);
every deviation from upstream Open Duck is in
[`CHANGES_FROM_ORIGINAL.md`](CHANGES_FROM_ORIGINAL.md).

## Result

`long14` 7.9M, the end of the 200M-step long run (2026-11-06), trained with
`--symmetric_policy` and run with `--symmetric_trained`.
`eval_command_tracking.py --env 64 --scale 0.75 --turn 0.15 --symmetric_trained`
(64 robots in the training env with noise, delays, pushes and the wider
randomization of CHANGES #17; fixed commands):

| fwd 0.15 | back -0.11 | left 0.11 | right -0.11 | turn L 0.15 | turn R -0.15 | fell within 8 s |
|---|---|---|---|---|---|---|
| 0.98 | 0.98 | 0.97 | 0.85 | 0.76 | 0.84 | ~33% |

(achieved / commanded along the commanded axis; `|r-1|` 0.11.) Torso pitch
~8 deg forward standing. Lab PC only:
`checkpoints/long14/2026_11_06_000746_7864320.onnx`.

The policy before it, `nm5` 3.6M (`|r-1|` 0.29), came from the curriculum
below; the long run continued from it with the changes listed in
`AUTO_EXPERIMENTS_2026-10-28.md` "Long run summary" (CHANGES #12-#17). To
repeat the result, run the curriculum, then continue with the committed
config, `--symmetric_policy` and
`--lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15`,
picking checkpoints by the 64-robot eval.

## Next (queued, 2026-11-11): head droop + floor slip + strafe stride

First real standing/walking test found the neck/torso drooping forward and
the real floor far slicker than anything trained (CHANGES #18/#19/#20:
`stand_still` -0.2 -> -1.0, `cost_head_pos` wired in, floor friction
0.15-1.0, `lin_vel_y` halved). Continue from `long14`'s own native
checkpoint (lab PC only, under `checkpoints/long14/` -- the step matching
`2026_11_06_000746_7864320.onnx`, NOT that `.onnx` file itself, which is an
inference-only export):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=<비율> uv run python -u -m playground.dcubot.runner \
  --task flat_terrain_train --num_envs <N> --symmetric_policy \
  --output_dir checkpoints/<실험이름> --num_timesteps <스텝> --num_evals <N> \
  --lin_vel_x -0.15 0.2 --lin_vel_y -0.06 0.06 --ang_vel_yaw -0.15 0.15 \
  --restore_checkpoint_path $PWD/checkpoints/long14/<체크포인트>
```

Judge with the usual 64-robot eval (`--symmetric_trained`); also watch
standing torso/neck pitch specifically (not just command tracking), since
that's what #18 targets and the existing eval script doesn't report it.

## What makes it walk

All in `joystick.py` / `custom_rewards.py` / `randomize.py` as committed:

1. Foot rewards `feet_air_time` 2.0, `feet_gait` 6.0, `feet_clearance` -5.0
   (CHANGES #5). Without them no policy ever lifted a foot.
2. `alive` 5 instead of 20 (#7), imitation joint kernel 1.0 plus base-velocity
   terms (#8), reference looked up by achieved speed (#14, #15), imitation base-velocity targets = command (#13), `tracking_sigma` 0.003 (#4).
3. Actuators: BAM-identified kp 17.8 / forcerange 3.35 (final URDF remodel
   section). Torso randomization on the right body, CoM jitter +-1 cm
   (Also fixed).
4. A direction curriculum (below). Training every direction at once from
   scratch converged to standing still.

## Curriculum

Each stage restores the previous stage's checkpoint and keeps the earlier
directions in the command mix. Move on once the new direction follows its
command at >= 0.5 (`eval_command_tracking.py --scale 0.4`, add `--turn 0.1`
from the turning stage on). On the lab PC (RTX 3080, ~1400 steps/s) one
1.8M-step checkpoint takes ~21 min; the whole run took ~9 h.

| stage | ranges | steps used |
|---|---|---|
| 1 forward | `--lin_vel_x 0.05 0.1 --lin_vel_y 0 0 --ang_vel_yaw 0 0` | 9.0M |
| 2 + backward | `--lin_vel_x -0.1 0.1 --lin_vel_y 0 0 --ang_vel_yaw 0 0` | 3.6M |
| 3 + strafe | `--lin_vel_x -0.1 0.1 --lin_vel_y -0.1 0.1 --ang_vel_yaw 0 0` | 1.8M |
| 4 + turning | `--lin_vel_x -0.1 0.1 --lin_vel_y -0.1 0.1 --ang_vel_yaw -0.1 0.1` | 9.0M |
| 5 wider speeds | `--lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15` | 3.6M |

(Stages 1-2 took longer in the recorded run because the actuator values and
the randomization were fixed mid-way; with the committed config they should
need less.)

Per stage, from the repo root (lab PC values; see `CLAUDE.md` for the
laptop's):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run python -u -m playground.dcubot.runner \
  --task flat_terrain_train --num_envs 1024 \
  --output_dir checkpoints/<stage> --num_timesteps 12000000 --num_evals 8 \
  <ranges> \
  --restore_checkpoint_path $PWD/checkpoints/<previous stage>/<checkpoint dir>
```

`--num_evals 8` puts a checkpoint every ~1.8M steps; stop the run once the
stage's criterion is met and restore from that checkpoint. The restore path
must be absolute (orbax).

## Picking the final checkpoint

Command following tends to peak a few million steps into a stage and then
drift, while the training reward keeps rising, so the last checkpoint is
not necessarily the best. Score the candidates with the multi-robot eval,
both raw and `--symmetric`, and take the one whose six ratios are closest to
1 (`|ratio-1|`, overshoot counts) with no direction weak:

```bash
uv run python scripts/eval_command_tracking.py --env 64 --scale 0.75 --turn 0.15 \
  [--symmetric] checkpoints/<stage>/*.onnx
```

(`--env` needs the GPU; run it when no training is using it, or with
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`.)

## Running and viewing

```bash
# keyboard-driven viewer: arrows = fwd/back/strafe, q/e = turn, space = stop
uv run python -m playground.dcubot.mujoco_infer -o <policy.onnx> \
  --lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15 \
  --symmetric_trained --heading_hold 1.0   # --symmetric for nm5-style policies

# 30 s video of 20 robots with random commands in the trained range
MUJOCO_GL=egl uv run python scripts/render_checkpoints.py <policy.onnx> \
  --num_envs 20 --duration 30 --width 1280 --height 720 \
  --lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15 \
  --out <video.mp4>
```

- `--symmetric`: a = (pi(o) + M pi(M o)) / 2 with the left/right mirror of
  observation and action. Removes the policy's own turning bias (the robot
  model is symmetric; each run picks up a different bias).
- `--heading_hold K`: with no turn or strafe command, adds a yaw command
  K x heading error from the IMU. Off while strafing.

Both are inference-side only, so the real robot's runtime needs the same
two pieces to behave like the simulation.

## Open issues

- ~30% fall within 8 s under the training env's pushes (0.1-1.0 m/s kicks
  every 5-10 s).
- Top speed is 0.2 m/s: 0.3 m/s made the stride long for the real robot
  (the reference's gait period is fixed), CHANGES #17.
- Very slow commands (+-0.05 m/s) stand still in the noise-free simulator.
- Slight forward creep (~0.03 m/s) on a stand command.
