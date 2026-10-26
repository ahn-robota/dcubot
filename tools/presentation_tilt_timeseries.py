"""Metric 4 for the presentation: one representative push trial's tilt/height
over time, with_policy vs without_policy. Push direction is fixed at +X
(theta=0) to match the original "Barbie doll" demo
(tools/compare_policy_vs_passive.py), which is also +X-only -- this is the
specific direction where the problem was first visually discovered.

Usage:
    .venv/bin/python tools/presentation_tilt_timeseries.py <onnx_path> <out_csv> [--push_mag 0.5]
"""
import argparse
import csv

import mujoco
import numpy as np

from playground.dcubot.mujoco_infer_standing import StandingInfer


def run_one(infer, push_mag, settle_s, after_s, use_policy):
    mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)
    infer.last_action[:] = 0
    infer.last_last_action[:] = 0
    infer.last_last_last_action[:] = 0
    infer.data.ctrl = infer.default_actuator.copy()

    n_steps = int((settle_s + after_s) / infer.sim_dt)
    push_step = int(settle_s / infer.sim_dt)
    counter = 0
    times, root_z, tilt_deg = [], [], []
    for i in range(n_steps):
        if i == push_step:
            infer.data.qvel[0] += push_mag
        mujoco.mj_step(infer.model, infer.data)
        counter += 1
        if counter % infer.decimation == 0:
            if use_policy:
                obs = infer.get_obs(infer.data, infer.commands)
                action = infer.policy.infer(obs)
                infer.last_last_last_action = infer.last_last_action.copy()
                infer.last_last_action = infer.last_action.copy()
                infer.last_action = action.copy()
                infer.data.ctrl = infer.default_actuator + action * infer.action_scale
            else:
                infer.data.ctrl = infer.default_actuator.copy()
        g = infer.get_gravity(infer.data)
        tilt = np.degrees(np.arccos(np.clip(g[2], -1, 1)))
        root_z.append(infer.data.qpos[2])
        tilt_deg.append(tilt)
        times.append(i * infer.sim_dt)
    return times, root_z, tilt_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("out_csv")
    ap.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    ap.add_argument("--push_mag", type=float, default=0.5)
    ap.add_argument("--settle_s", type=float, default=1.0)
    ap.add_argument("--after_s", type=float, default=4.0)
    args = ap.parse_args()

    infer = StandingInfer(args.model_path, args.onnx_path)

    rows = []
    for cond_name, use_policy in [("with_policy", True), ("without_policy", False)]:
        t, z, tilt = run_one(infer, args.push_mag, args.settle_s, args.after_s, use_policy)
        for ti, zi, tilti in zip(t, z, tilt):
            rows.append({"time_sec": ti, "tilt_deg": tilti, "height_m": zi, "condition": cond_name})

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_sec", "tilt_deg", "height_m", "condition"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
