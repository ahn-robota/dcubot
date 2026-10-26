"""Step 1 diagnostic (no reward-code changes): replay a push on the current
standing checkpoint and log each reward term's *actual scaled contribution*
per control step, using the exact same functions/scales as training
(playground/dcubot/standing.py's default_config + custom_rewards.py).

Goal: confirm or refute, with real numbers, whether cost_stand_still_dcubot's
scaled magnitude exceeds reward_recovery_dcubot's scaled magnitude during the
window where the robot is significantly tilted (tilt > recovery_tilt_threshold_deg)
after a push -- i.e. whether stand_still is actually outcompeting/suppressing
the recovery incentive, rather than just assuming it theoretically would.

Usage:
    .venv/bin/python tools/diagnose_reward_terms.py -o <ckpt.onnx> --push_mag 0.5
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mujoco
import numpy as np

from playground.dcubot.mujoco_infer_standing import StandingInfer
from playground.dcubot.standing import default_config
from playground.common.rewards import (
    cost_orientation, cost_torques, cost_action_rate, reward_alive,
    reward_tracking_lin_vel, reward_tracking_ang_vel,
)
from playground.dcubot.custom_rewards import cost_stand_still_dcubot, reward_recovery_dcubot

for _p in fm.findSystemFonts(fontpaths=["/usr/share/fonts/opentype/noto"]):
    if "NotoSansCJK-Regular" in _p:
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", required=True)
    ap.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    ap.add_argument("--push_mag", type=float, default=0.5)
    ap.add_argument("--settle_s", type=float, default=1.0)
    ap.add_argument("--after_s", type=float, default=4.0)
    ap.add_argument("--out_dir", default="/home/ahn/dcubot/발표자료_DCUBOT_레퍼런스/07_RL학습_결과")
    ap.add_argument("--tag", default="reward_term_diagnosis")
    args = ap.parse_args()

    cfg = default_config().reward_config
    scales = cfg.scales
    tilt_gate = cfg.recovery_tilt_threshold_deg

    infer = StandingInfer(args.model_path, args.onnx_model_path)
    mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)
    infer.last_action[:] = 0
    infer.last_last_action[:] = 0
    infer.last_last_last_action[:] = 0

    prev_gravity = infer.get_gravity(infer.data).copy()

    n_steps = int((args.settle_s + args.after_s) / infer.sim_dt)
    push_step = int(args.settle_s / infer.sim_dt)
    counter = 0
    rows = []

    for i in range(n_steps):
        if i == push_step:
            infer.data.qvel[0] += args.push_mag
        mujoco.mj_step(infer.model, infer.data)
        counter += 1
        if counter % infer.decimation == 0:
            obs = infer.get_obs(infer.data, infer.commands)
            action = infer.policy.infer(obs)
            prev_action = infer.last_action.copy()
            infer.last_last_last_action = infer.last_last_action.copy()
            infer.last_last_action = infer.last_action.copy()
            infer.last_action = action.copy()
            infer.data.ctrl = infer.default_actuator + action * infer.action_scale

            data = infer.data
            cmd = np.array(infer.commands)
            gravity = infer.get_gravity(data)
            tilt_deg = float(np.degrees(np.arccos(np.clip(gravity[-1], -1.0, 1.0))))

            raw = {
                "alive": float(reward_alive()),
                "orientation": float(cost_orientation(gravity)),
                "torques": float(cost_torques(data.actuator_force)),
                "action_rate": float(cost_action_rate(action, prev_action)),
                "stand_still": float(cost_stand_still_dcubot(
                    infer.get_linvel(data), infer.get_gyro(data),
                    infer.get_actuator_joints_qpos(data.qpos),
                    infer.get_actuator_joints_qvel(data.qvel), infer.default_actuator,
                    cfg.stand_still_lin_vel_threshold, cfg.stand_still_ang_vel_threshold,
                    cfg.stand_still_reduction_factor, True)),
                "tracking_lin_vel": float(reward_tracking_lin_vel(
                    cmd, infer.get_linvel(data), cfg.tracking_sigma_lin_vel)),
                "tracking_ang_vel": float(reward_tracking_ang_vel(
                    cmd, infer.get_gyro(data), cfg.tracking_sigma_ang_vel)),
                "recovery": float(reward_recovery_dcubot(gravity, prev_gravity, tilt_gate)),
            }
            scaled = {k: v * float(scales[k]) for k, v in raw.items()}
            prev_gravity = gravity.copy()

            rows.append({"step": i, "t": i * infer.sim_dt, "tilt_deg": tilt_deg,
                         **{f"raw_{k}": v for k, v in raw.items()},
                         **{f"scaled_{k}": v for k, v in scaled.items()}})

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = f"{args.out_dir}/{args.tag}_{args.push_mag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path}")

    # The window we actually care about: tilt above the recovery gate.
    crisis = [r for r in rows if r["tilt_deg"] > tilt_gate]
    print(f"\npush {args.push_mag} m/s -- {len(crisis)}/{len(rows)} steps with tilt > {tilt_gate} deg")
    if crisis:
        ss = np.array([r["scaled_stand_still"] for r in crisis])  # already negative (cost)
        rec = np.array([r["scaled_recovery"] for r in crisis])
        print(f"  during that window:")
        print(f"    scaled stand_still cost : mean {ss.mean():+.3f}  min {ss.min():+.3f}  (more negative = bigger penalty)")
        print(f"    scaled recovery bonus   : mean {rec.mean():+.3f}  max {rec.max():+.3f}")
        outweighed = np.mean(np.abs(ss) > rec)
        print(f"    |stand_still| > recovery on {outweighed*100:.0f}% of those steps")
        net = ss + rec
        print(f"    net (stand_still + recovery): mean {net.mean():+.3f}  "
              f"({'penalty wins' if net.mean() < 0 else 'recovery wins'} on average)")
    else:
        print("  (never crossed the recovery gate during this run -- lower the push or widen after_s)")

    # Plot: tilt, and the two competing terms, over time.
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=150, sharex=True)
    t = [r["t"] for r in rows]
    axes[0].plot(t, [r["tilt_deg"] for r in rows], color="#444")
    axes[0].axhline(tilt_gate, color="#d1242f", ls="--", label=f"recovery gate ({tilt_gate}deg)")
    axes[0].axvline(args.settle_s, color="#888", ls=":")
    axes[0].set_ylabel("기울기 (deg)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].plot(t, [r["scaled_stand_still"] for r in rows], color="#d1242f", label="stand_still (비용, 음수)")
    axes[1].plot(t, [r["scaled_recovery"] for r in rows], color="#1f6feb", label="recovery (보너스)")
    axes[1].axhline(0, color="#ccc", lw=0.8)
    axes[1].axvline(args.settle_s, color="#888", ls=":")
    axes[1].set_ylabel("스텝당 보상 기여"); axes[1].set_xlabel("시간 (s)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle(f"stand_still vs recovery 실측 기여도 — push {args.push_mag} m/s")
    fig.tight_layout()
    png = f"{args.out_dir}/{args.tag}_{args.push_mag}.png"
    fig.savefig(png)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
