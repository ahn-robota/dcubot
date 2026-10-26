"""Does the standing checkpoint actually balance, or is the push-recovery just
passive mechanical stability (like a cup settling back onto its base)?
Same push, two conditions:
  (a) policy active (normal)
  (b) policy disabled -- ctrl held at the fixed default (home) pose the whole
      time, no feedback at all (pure PD-servo-to-a-constant-target + physics)
If (b) also recovers, the "balancing" is just passive geometry. If (b) falls
but (a) recovers, the policy is doing real corrective work.

Usage:
    .venv/bin/python tools/compare_policy_vs_passive.py -o <ckpt.onnx> --push_mag 0.3
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mujoco
import numpy as np

from playground.dcubot.mujoco_infer_standing import StandingInfer

for _p in fm.findSystemFonts(fontpaths=["/usr/share/fonts/opentype/noto"]):
    if "NotoSansCJK-Regular" in _p:
        fm.fontManager.addfont(_p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def run_one(infer, push_mag, settle_s, after_s, use_policy):
    mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)
    infer.last_action[:] = 0
    infer.last_last_action[:] = 0
    infer.last_last_last_action[:] = 0
    infer.data.ctrl = infer.default_actuator.copy()

    n_steps = int((settle_s + after_s) / infer.sim_dt)
    push_step = int(settle_s / infer.sim_dt)
    counter = 0
    root_z, tilt_deg, times = [], [], []
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
                infer.data.ctrl = infer.default_actuator.copy()  # fixed target, no feedback
        g = infer.get_gravity(infer.data)
        tilt_deg.append(np.degrees(np.arccos(np.clip(g[2], -1, 1))))
        root_z.append(infer.data.qpos[2])
        times.append(i * infer.sim_dt)
    return np.array(times), np.array(root_z), np.array(tilt_deg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", required=True)
    ap.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    ap.add_argument("--push_mag", type=float, default=0.3)
    ap.add_argument("--settle_s", type=float, default=1.0)
    ap.add_argument("--after_s", type=float, default=4.0)
    ap.add_argument("--out_dir", default="/home/ahn/dcubot/발표자료_DCUBOT_레퍼런스/07_RL학습_결과")
    args = ap.parse_args()

    infer = StandingInfer(args.model_path, args.onnx_model_path)

    t1, z1, tilt1 = run_one(infer, args.push_mag, args.settle_s, args.after_s, use_policy=True)
    t2, z2, tilt2 = run_one(infer, args.push_mag, args.settle_s, args.after_s, use_policy=False)

    print(f"[정책 있음]  최종 기울기 {tilt1[-1]:.1f}deg  최종 높이 {z1[-1]:.3f}m  최대 기울기 {tilt1.max():.1f}deg")
    print(f"[정책 없음]  최종 기울기 {tilt2[-1]:.1f}deg  최종 높이 {z2[-1]:.3f}m  최대 기울기 {tilt2.max():.1f}deg")

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), dpi=150, sharex=True)
    axes[0].plot(t1, z1, color="#1f6feb", label="정책 있음")
    axes[0].plot(t2, z2, color="#d1242f", label="정책 없음 (기본 자세 고정)")
    axes[0].axvline(args.settle_s, color="#888", ls="--")
    axes[0].set_ylabel("몸통 높이 (m)"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(t1, tilt1, color="#1f6feb")
    axes[1].plot(t2, tilt2, color="#d1242f")
    axes[1].axvline(args.settle_s, color="#888", ls="--")
    axes[1].set_ylabel("기울기 (deg)"); axes[1].set_xlabel("시간 (s)"); axes[1].grid(alpha=0.3)
    fig.suptitle(f"정책 vs 수동 자세 고정 — 같은 푸시 {args.push_mag} m/s")
    fig.tight_layout()
    out = f"{args.out_dir}/policy_vs_passive_{args.push_mag}.png"
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
