"""Metrics 2 and 3 for the presentation:
  (2) push-recovery success rate at every saved checkpoint of a run (the
      "did it actually learn to recover" curve -- as opposed to reward).
  (3) termination-reason distribution (timeout vs fell_over) for a single
      checkpoint, under the training-realistic push schedule (random pushes
      every 5-10s, magnitude 0.1-1.0 m/s, same as push_config in standing.py),
      run to full episode length (1000 steps / 20s) or until it falls.

Both with_policy (the ONNX checkpoint) and without_policy (ctrl held fixed
at the default pose -- the passive baseline) are measured. The passive
condition doesn't depend on the checkpoint's weights, so for metric 2 it's
measured once, not per-checkpoint.

Usage:
    .venv/bin/python tools/presentation_push_metrics.py
"""
import csv
import json
import os
import sys

import mujoco
import numpy as np

from playground.dcubot.mujoco_infer_standing import StandingInfer

OUT_DIR = "/home/ahn/dcubot/발표자료_DCUBOT_레퍼런스/07_RL학습_결과/presentation_data"
MODEL_PATH = "playground/dcubot/xmls/scene_flat_terrain.xml"
PUSH_MAG = 0.5
TRIALS_PER_CHECKPOINT = 10
TERMINATION_TRIALS = 30
RNG_SEED = 0


def reset(infer):
    mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)
    infer.last_action[:] = 0
    infer.last_last_action[:] = 0
    infer.last_last_action[:] = 0
    infer.last_last_last_action[:] = 0


def step_once(infer, use_policy):
    for _ in range(infer.decimation):
        mujoco.mj_step(infer.model, infer.data)
    if use_policy:
        obs = infer.get_obs(infer.data, infer.commands)
        action = infer.policy.infer(obs)
        infer.last_last_last_action = infer.last_last_action.copy()
        infer.last_last_action = infer.last_action.copy()
        infer.last_action = action.copy()
        infer.data.ctrl = infer.default_actuator + action * infer.action_scale
    else:
        infer.data.ctrl = infer.default_actuator.copy()


def fallen(infer):
    return infer.get_gravity(infer.data)[-1] < 0.0


def single_push_trial(infer, use_policy, rng, settle_s=1.0, after_s=4.0):
    """One settle -> single push at a random direction -> recover-or-fall window."""
    reset(infer)
    for _ in range(int(settle_s / (infer.sim_dt * infer.decimation))):
        step_once(infer, use_policy)
    theta = rng.uniform(0, 2 * np.pi)
    infer.data.qvel[0] += PUSH_MAG * np.cos(theta)
    infer.data.qvel[1] += PUSH_MAG * np.sin(theta)
    for _ in range(int(after_s / (infer.sim_dt * infer.decimation))):
        step_once(infer, use_policy)
        if fallen(infer):
            return False
    return True


def success_rate(infer, use_policy, trials, seed):
    rng = np.random.default_rng(seed)
    successes = sum(single_push_trial(infer, use_policy, rng) for _ in range(trials))
    return successes / trials


def full_episode_termination_test(infer, use_policy, trials, seed, max_steps=1000,
                                   push_interval_range=(5.0, 10.0),
                                   push_magnitude_range=(0.1, 1.0)):
    """Realistic full episode: random pushes on the SAME schedule standing.py
    trains with, run until it falls or hits the episode length cap."""
    rng = np.random.default_rng(seed)
    results = []  # "timeout" or "fell_over"
    for _ in range(trials):
        reset(infer)
        next_push_step = rng.uniform(*push_interval_range) / (infer.sim_dt * infer.decimation)
        step = 0
        outcome = "timeout"
        while step < max_steps:
            if step >= next_push_step:
                theta = rng.uniform(0, 2 * np.pi)
                mag = rng.uniform(*push_magnitude_range)
                infer.data.qvel[0] += mag * np.cos(theta)
                infer.data.qvel[1] += mag * np.sin(theta)
                next_push_step += rng.uniform(*push_interval_range) / (infer.sim_dt * infer.decimation)
            step_once(infer, use_policy)
            step += 1
            if fallen(infer):
                outcome = "fell_over"
                break
        results.append(outcome)
    return results


def run_era(era_name, checkpoints, final_checkpoint_path):
    """checkpoints: list of (absolute_step, onnx_path), ascending."""
    out_dir = os.path.join(OUT_DIR, era_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Metric 2: success-rate learning curve (with_policy per checkpoint) ---
    curve_path = os.path.join(out_dir, "push_recovery_success_curve.csv")
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "success_rate_with_policy"])
        for step, path in checkpoints:
            infer = StandingInfer(MODEL_PATH, path)
            rate = success_rate(infer, True, TRIALS_PER_CHECKPOINT, seed=RNG_SEED)
            w.writerow([step, rate])
            print(f"  [{era_name}] step {step}: success_rate_with_policy={rate:.2f}", flush=True)
    # passive baseline: single value, doesn't depend on the checkpoint's weights
    infer = StandingInfer(MODEL_PATH, final_checkpoint_path)
    baseline = success_rate(infer, False, TRIALS_PER_CHECKPOINT, seed=RNG_SEED)
    with open(os.path.join(out_dir, "push_recovery_baseline_without_policy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["success_rate_without_policy_baseline"])
        w.writerow([baseline])
    print(f"  [{era_name}] success_rate_without_policy_baseline={baseline:.2f}", flush=True)

    # --- Metric 3: termination-reason distribution, final checkpoint only ---
    term_path = os.path.join(out_dir, "termination_reason_distribution.csv")
    infer = StandingInfer(MODEL_PATH, final_checkpoint_path)
    with open(term_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "condition", "reason", "percentage"])
        for cond_name, use_policy in [("with_policy", True), ("without_policy", False)]:
            outcomes = full_episode_termination_test(infer, use_policy, TERMINATION_TRIALS, seed=RNG_SEED)
            n = len(outcomes)
            for reason in ("timeout", "fell_over"):
                pct = 100.0 * outcomes.count(reason) / n
                w.writerow([os.path.basename(final_checkpoint_path), cond_name, reason, pct])
            print(f"  [{era_name}] {cond_name}: "
                  f"timeout={100*outcomes.count('timeout')/n:.0f}% "
                  f"fell_over={100*outcomes.count('fell_over')/n:.0f}%", flush=True)
    print(f"saved {curve_path}")
    print(f"saved {term_path}")


BEFORE_DIR = "checkpoints_dcubot_standing"
BEFORE_CHECKPOINTS = [
    (0, "2026_10_11_044952_0.onnx"),
    (819200, "2026_10_11_051813_819200.onnx"),
    (1638400, "2026_10_11_055248_1638400.onnx"),
    (2457600, "2026_10_11_062457_2457600.onnx"),
    (3276800, "2026_10_11_065711_3276800.onnx"),
    (4096000, "2026_10_11_072923_4096000.onnx"),
    (4915200, "2026_10_11_080136_4915200.onnx"),
    (5734400, "2026_10_11_083343_5734400.onnx"),
    (6553600, "2026_10_11_090549_6553600.onnx"),
    (7372800, "2026_10_11_093754_7372800.onnx"),
    (8192000, "2026_10_11_101004_8192000.onnx"),
    (9011200, "2026_10_11_104213_9011200.onnx"),
    (9830400, "2026_10_11_111423_9830400.onnx"),
    (10649600, "2026_10_11_114632_10649600.onnx"),
    (11468800, "2026_10_11_121841_11468800.onnx"),
]

AFTER_DIR = "checkpoints_dcubot_standing_v3"
AFTER_CHECKPOINTS = [
    (0, "2026_10_13_033534_0.onnx"),
    (163840, "2026_10_13_034111_163840.onnx"),
    (327680, "2026_10_13_035017_327680.onnx"),
    (491520, "2026_10_13_035910_491520.onnx"),
    (655360, "2026_10_13_040802_655360.onnx"),
    (819200, "2026_10_13_041653_819200.onnx"),
    (983040, "2026_10_13_042544_983040.onnx"),
    (1146880, "2026_10_13_044829_163840.onnx"),
    (1310720, "2026_10_13_045851_327680.onnx"),
    (1474560, "2026_10_13_050743_491520.onnx"),
    (1638400, "2026_10_13_051633_655360.onnx"),
    (1802240, "2026_10_13_052524_819200.onnx"),
    (1966080, "2026_10_13_053414_983040.onnx"),
    (2129920, "2026_10_13_060204_163840.onnx"),
    (2293760, "2026_10_13_061228_327680.onnx"),
    (2457600, "2026_10_13_062119_491520.onnx"),
    (2621440, "2026_10_13_063009_655360.onnx"),
    (2785280, "2026_10_13_063900_819200.onnx"),
    (2949120, "2026_10_13_064749_983040.onnx"),
]


if __name__ == "__main__":
    era = sys.argv[1] if len(sys.argv) > 1 else "both"
    if era in ("before", "both"):
        print("=== BEFORE ===", flush=True)
        ckpts = [(s, os.path.join(BEFORE_DIR, p)) for s, p in BEFORE_CHECKPOINTS]
        run_era("before", ckpts, ckpts[-1][1])
    if era in ("after", "both"):
        print("=== AFTER ===", flush=True)
        ckpts = [(s, os.path.join(AFTER_DIR, p)) for s, p in AFTER_CHECKPOINTS]
        run_era("after", ckpts, ckpts[-1][1])
