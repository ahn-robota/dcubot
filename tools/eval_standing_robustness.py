"""Evaluate a DCUBOT standing checkpoint: push-recovery success rate (the
classic "push magnitude vs survival" curve from balance-RL papers), posture
stability (roll/pitch/height variation while undisturbed), and torque/energy
use. Deterministic replay (no domain randomization), CPU only.

Usage:
    .venv/bin/python tools/eval_standing_robustness.py -o checkpoints_dcubot_standing/<ckpt>.onnx \
        --out_dir <dir> [--push_magnitudes 0.1 0.3 0.5 0.8 1.0 1.3 1.5 2.0] [--trials 5]
"""
import argparse
import csv
import json
import os

import mujoco
import numpy as np

from playground.dcubot.mujoco_infer_standing import StandingInfer


def reset(infer):
    mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)
    infer.last_action[:] = 0
    infer.last_last_action[:] = 0
    infer.last_last_last_action[:] = 0
    infer.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def step_policy(infer, torque_log=None):
    for _ in range(infer.decimation):
        mujoco.mj_step(infer.model, infer.data)
    obs = infer.get_obs(infer.data, infer.commands)
    action = infer.policy.infer(obs)
    infer.last_last_last_action = infer.last_last_action.copy()
    infer.last_last_action = infer.last_action.copy()
    infer.last_action = action.copy()
    motor_targets = infer.default_actuator + action * infer.action_scale
    infer.data.ctrl = motor_targets.copy()
    if torque_log is not None:
        torque_log.append(np.abs(infer.data.actuator_force).copy())


def fallen(infer):
    return infer.get_gravity(infer.data)[-1] < 0.0


def posture_stability(infer, seconds=8.0):
    """Undisturbed: roll/pitch (from gravity vector) and base-height variation."""
    reset(infer)
    n_ctrl_steps = int(seconds / (infer.sim_dt * infer.decimation))
    rolls, pitches, heights, torques = [], [], [], []
    for _ in range(n_ctrl_steps):
        step_policy(infer, torque_log=torques)
        g = infer.get_gravity(infer.data)  # body-frame gravity direction, [0,0,-1] when upright
        # upvector sensor: g ~ [0,0,1] upright, so g[2] (not -g[2]) is the upright reference.
        rolls.append(np.degrees(np.arctan2(g[1], g[2])))
        pitches.append(np.degrees(np.arctan2(-g[0], g[2])))
        heights.append(infer.data.qpos[2])
    torques = np.array(torques)
    return {
        "roll_std_deg": float(np.std(rolls)),
        "pitch_std_deg": float(np.std(pitches)),
        "height_std_m": float(np.std(heights)),
        "height_mean_m": float(np.mean(heights)),
        "fell": bool(fallen(infer)),
        "mean_abs_torque_Nm": float(torques.mean()),
        "peak_abs_torque_Nm": float(torques.max()),
    }


def push_sweep(infer, magnitudes, trials, settle_s=1.0, recover_s=4.0, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for mag in magnitudes:
        successes = 0
        for t in range(trials):
            reset(infer)
            for _ in range(int(settle_s / (infer.sim_dt * infer.decimation))):
                step_policy(infer)
            theta = rng.uniform(0, 2 * np.pi)
            infer.data.qvel[0] += mag * np.cos(theta)
            infer.data.qvel[1] += mag * np.sin(theta)
            fell_during = False
            for _ in range(int(recover_s / (infer.sim_dt * infer.decimation))):
                step_policy(infer)
                if fallen(infer):
                    fell_during = True
                    break
            if not fell_during:
                successes += 1
        rate = successes / trials
        print(f"  push {mag:.2f} m/s: {successes}/{trials} survived ({rate*100:.0f}%)", flush=True)
        results.append({"magnitude_mps": mag, "trials": trials, "successes": successes, "success_rate": rate})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", required=True)
    ap.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    ap.add_argument("--push_magnitudes", type=float, nargs="+",
                     default=[0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out_dir", default="/home/ahn/dcubot/발표자료_DCUBOT_레퍼런스/07_RL학습_결과")
    ap.add_argument("--tag", default="standing", help="filename prefix for this checkpoint's results")
    args = ap.parse_args()

    infer = StandingInfer(args.model_path, args.onnx_model_path)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== 자세 안정성 (방해 없이, 8초) ===", flush=True)
    posture = posture_stability(infer)
    for k, v in posture.items():
        print(f"  {k}: {v}")

    print("=== 푸시 강인성 (크기별, 1초 대기 후 밀고 4초 회복) ===", flush=True)
    sweep = push_sweep(infer, args.push_magnitudes, args.trials)

    out = {
        "onnx": os.path.basename(args.onnx_model_path),
        "posture_stability": posture,
        "push_sweep": sweep,
    }
    json_path = f"{args.out_dir}/{args.tag}_robustness_eval.json"
    json.dump(out, open(json_path, "w"), indent=2, ensure_ascii=False)

    csv_path = f"{args.out_dir}/{args.tag}_push_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["magnitude_mps", "trials", "successes", "success_rate"])
        w.writeheader()
        for row in sweep:
            w.writerow(row)

    print(f"\nsaved {json_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
