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

## Runs

| run | change vs. master `ba2fc7a` | steps | best tracking | verdict |
|---|---|---|---|---|
| exp01 | none (sigma 0.003, ang_vel_yaw +-0.3 as committed by the laptop) | 3.6M (stopped) | 0.00 (0 / 1.8M / 3.6M all 0.00, no falls) | stopped early: same as baseline at this stage, and alive dominance untouched -- would need >16M to judge |
| exp02 | `alive` 20 -> 2 | 1.8M (stopped) | 0.00 (turn -0.05/+0.07) | stopped: costs exceed rewards, see below |
| exp03 | `alive` 20 -> 5 | 9.0M planned | | running |

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
