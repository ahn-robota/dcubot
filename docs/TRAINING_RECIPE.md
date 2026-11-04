# Training recipe: command-following walking policy

How the current best policy was trained (final-URDF model, 2.072 kg, lab PC
2026-11-03), and how to repeat it. The reasoning and every failed attempt
are in [`AUTO_EXPERIMENTS_2026-10-28.md`](AUTO_EXPERIMENTS_2026-10-28.md);
every deviation from upstream Open Duck is in
[`CHANGES_FROM_ORIGINAL.md`](CHANGES_FROM_ORIGINAL.md).

## Result

`nm5` 3.6M, run with symmetric inference (`--symmetric`).
`eval_command_tracking.py --env 64 --scale 0.75 --turn 0.15 --symmetric`
(64 robots in the training env with noise, delays and pushes; fixed commands):

| fwd 0.15 | back -0.11 | left 0.11 | right -0.11 | turn L 0.15 | turn R -0.15 | fell within 8 s |
|---|---|---|---|---|---|---|
| 0.74 | 0.63 | 0.84 | 0.74 | 0.59 | 0.77 | ~31% |

(achieved / commanded along the commanded axis). No falls without pushes.
The checkpoint lives on the lab PC only:
`checkpoints/nm5/2026_11_03_090557_3604480.onnx`.

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
  --symmetric --heading_hold 1.0

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
- Speeds are still small (0.2 m/s forward max) next to the 0.3 m/s goal; a
  further widening stage from this checkpoint is the next step.
- Slight forward creep (~0.03 m/s) on a stand command.
