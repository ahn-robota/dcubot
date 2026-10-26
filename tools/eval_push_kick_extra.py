"""Validity-check extensions to eval_push_kick.py (2026-10-20):
  --mode idle: 10s undisturbed standing (no kick), tilt max/std + full-window
    joint-motion-sum, as a baseline "natural fidgeting" level for interpreting
    the post-kick joint-motion numbers in push_kick_{variant}.csv.
  --mode finesweep: magnitudes 0.1-0.5 m/s in 0.05 steps x 4 cardinal
    directions, same kick_trial() mechanism as eval_push_kick.py, to find
    each variant's max-survivable-kick-magnitude threshold.

Reuses eval_push_kick.py's variant constructors / kick_trial exactly (same
mechanism, same fall/tilt definitions) -- run as its own process per variant,
same reason as eval_push_kick.py (shared top-level `playground` package name
collision between the two projects' imports).

Usage:
    .venv/bin/python tools/eval_push_kick_extra.py --mode idle --variant dcubot_post_fix
    .venv/bin/python tools/eval_push_kick_extra.py --mode finesweep --variant original_pretrained
"""
import argparse
import csv
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.eval_push_kick import (
    VARIANTS, reset_idle, step_policy, tilt_deg, fallen, foot_geom_ids,
    feet_airborne_mask, kick_trial,
)


def idle_run(infer, seconds=10.0):
    reset_idle(infer)
    n = int(seconds / (infer.sim_dt * infer.decimation))
    tilts = []
    joint_series = []
    for _ in range(n):
        step_policy(infer)
        tilts.append(tilt_deg(infer))
        joints = (infer.get_actuator_joints_qpos(infer.data.qpos)
                  if hasattr(infer, "get_actuator_joints_qpos")
                  else infer.data.qpos[-infer.num_dofs:])
        joint_series.append(np.array(joints))
    joint_series = np.array(joint_series)
    joint_motion = float(np.sum(np.abs(np.diff(joint_series, axis=0))))
    return {
        "max_tilt_deg": float(np.max(tilts)),
        "std_tilt_deg": float(np.std(tilts)),
        "final_tilt_deg": float(tilts[-1]),
        "joint_motion_sum_10s": joint_motion,
        "joint_motion_per_s": joint_motion / seconds,
    }


def finesweep_run(infer, foot_ids):
    magnitudes = [round(0.1 + 0.05 * i, 2) for i in range(9)]  # 0.10..0.50
    directions = ["+X", "-X", "+Y", "-Y"]
    dir_thetas = {"+X": 0.0, "-X": np.pi, "+Y": np.pi / 2, "-Y": -np.pi / 2}
    rows = []
    max_survived = None
    for mag in magnitudes:
        any_fell = False
        for dname in directions:
            r = kick_trial(infer, foot_ids, mag, dir_thetas[dname])
            r.update({"magnitude": mag, "direction": dname})
            rows.append(r)
            if r["fell"]:
                any_fell = True
            print(f"  mag={mag:.2f} dir={dname}: fell={r['fell']} max_tilt={r['max_tilt_deg']:.1f}", flush=True)
        if not any_fell:
            max_survived = mag
    return rows, max_survived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["idle", "finesweep"], required=True)
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), required=True)
    ap.add_argument("--out_dir", default="eval_results")
    args = ap.parse_args()
    base = "/home/ahn/dcubot/DCUBOT/rl_training"
    os.makedirs(os.path.join(base, args.out_dir), exist_ok=True)

    infer, foot_names, _ = VARIANTS[args.variant](base)
    foot_ids = foot_geom_ids(infer, foot_names)

    if args.mode == "idle":
        res = idle_run(infer)
        res["variant"] = args.variant
        print(json.dumps(res, indent=2))
        json.dump(res, open(os.path.join(base, args.out_dir, f"idle_{args.variant}.json"), "w"), indent=2)
    else:
        rows, max_survived = finesweep_run(infer, foot_ids)
        os.chdir(base)
        csv_path = os.path.join(args.out_dir, f"push_kick_fine_{args.variant}.csv")
        fieldnames = ["magnitude", "direction", "fell", "max_tilt_deg", "final_tilt_deg",
                      "time_to_recover_s", "joint_motion_sum_1s", "foot_airborne_events_1s"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fieldnames})
        print(f"\nmax survived magnitude (no fall in any direction): {max_survived}")
        print(f"saved {csv_path}")
        json.dump({"variant": args.variant, "max_survived_magnitude": max_survived, "rows": rows},
                   open(os.path.join(base, args.out_dir, f"push_kick_fine_{args.variant}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
