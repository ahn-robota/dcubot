"""Inspect a DCUBOT polynomial reference-motion pkl: key ranges, frame length,
and what each index of the reference vector means.

Usage (run from rl_training/):
    .venv/bin/python scripts/inspect_reference.py \
        --pkl playground/dcubot/data/polynomial_coefficients_slow.pkl

Run it once per candidate pkl if you want to compare
polynomial_coefficients.pkl vs polynomial_coefficients_slow.pkl (both exist on
disk right now; joystick.py currently points at the "_slow" one via
DCUBOT_REFERENCE_PKL).

The per-index layout below is reconstructed by reading, not guessed:
  - reference_motion_generator/scripts/fit_poly.py concatenates
    [joints_pos, joints_vel, foot_contacts, base_linear_vel, base_angular_vel]
    into the vector these coefficients reproduce.
  - reference_motion_generator/open_duck_reference_motion_generator/
    gait_generator.py writes "Joints": list(pwe.get_angles().keys()) into each
    raw recording .json -- that's the authoritative joint order for the
    joints_pos/joints_vel blocks. If a raw recording .json is found next to
    the pkl (same sweep), this script reads that list directly instead of
    assuming it; otherwise it falls back to the order confirmed by hand on
    2026-10-27 against motions_dcubot/forward_slow_0.01_0.0_0.0.json.
"""

import argparse
import glob
import json
import os
import pickle

FALLBACK_JOINT_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def find_joint_order(pkl_path: str):
    search_dirs = [
        os.path.join(os.path.dirname(pkl_path), "..", "..", "..", "reference_motion_generator"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "reference_motion_generator"),
        "../reference_motion_generator",
        "../../reference_motion_generator",
    ]
    for base in search_dirs:
        for pattern in ("motions_dcubot/*.json", "recordings_*/*.json", "*/recordings/*.json"):
            for f in glob.glob(os.path.join(base, pattern)):
                try:
                    d = json.load(open(f))
                    if "Joints" in d:
                        return d["Joints"], f
                except Exception:
                    continue
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, required=True)
    args = parser.parse_args()

    data = pickle.load(open(args.pkl, "rb"))
    keys = list(data.keys())

    dxs, dys, dthetas = [], [], []
    for k in keys:
        dx, dy, dtheta = (float(x) for x in k.split("_"))
        dxs.append(dx)
        dys.append(dy)
        dthetas.append(dtheta)

    sample = data[keys[0]]
    period = sample["period"]
    fps = sample["fps"]
    frame_offsets = sample.get("frame_offsets")
    n_dims = len(sample["coefficients"])

    joint_order, joint_src = find_joint_order(args.pkl)
    if joint_order is None:
        joint_order = FALLBACK_JOINT_ORDER
        joint_src = "fallback (hardcoded, confirmed 2026-10-27 against motions_dcubot/forward_slow_0.01_0.0_0.0.json)"
    n_joints = len(joint_order)

    print("=" * 70)
    print(f"pkl: {args.pkl}")
    print("=" * 70)
    print(f"number of keys (dx_dy_dtheta combos): {len(keys)}")
    print(f"dx  range: [{min(dxs):.4f}, {max(dxs):.4f}]  (unique values: {sorted(set(dxs))})")
    print(f"dy  range: [{min(dys):.4f}, {max(dys):.4f}]  (unique values: {sorted(set(dys))})")
    print(f"dtheta range: [{min(dthetas):.4f}, {max(dthetas):.4f}]  (unique values: {sorted(set(dthetas))})")
    print()
    has_backward = any(dx < 0 for dx in dxs)
    has_strafe = any(dy != 0 for dy in dys)
    has_turn = any(dtheta != 0 for dtheta in dthetas)
    print(f"backward (dx<0) present: {has_backward}")
    print(f"strafe   (dy!=0) present: {has_strafe}")
    print(f"turn     (dtheta!=0) present: {has_turn}")
    if not (has_backward and has_strafe and has_turn):
        print(">>> This pkl is NOT omnidirectional (forward-only or partial). If used as-is,")
        print(">>> command ranges (lin_vel_x/y, ang_vel_yaw) should be narrowed to match --")
        print(">>> do not widen commands beyond what this pkl actually covers.")
    print()
    print(f"period: {period} s   fps: {fps}   steps/period: {int(period * fps)}")
    print(f"raw frame_offsets (from the recording, for reference): {frame_offsets}")
    print()
    print(f"reference vector length actually in this pkl: {n_dims}")
    print(f"joint order source: {joint_src}")
    print(f"number of joints per pos/vel block: {n_joints}")
    print()
    print("Index layout (fit_poly.py concatenation order: joints_pos, joints_vel,")
    print("foot_contacts, base_linear_vel, base_angular_vel):")
    print("-" * 70)
    idx = 0
    for j in joint_order:
        print(f"  [{idx:2d}] pos  {j}")
        idx += 1
    for j in joint_order:
        print(f"  [{idx:2d}] vel  {j}")
        idx += 1
    for side in ("left", "right"):
        print(f"  [{idx:2d}] foot_contact {side}")
        idx += 1
    for axis in ("x", "y", "z"):
        print(f"  [{idx:2d}] base_linear_vel {axis}")
        idx += 1
    for axis in ("x", "y", "z"):
        print(f"  [{idx:2d}] base_angular_vel {axis}")
        idx += 1
    print("-" * 70)
    print(f"total dims listed above: {idx}  (should equal n_dims={n_dims} above)")
    if idx != n_dims:
        print(">>> MISMATCH -- the joint order/count assumption above does not match this")
        print(">>> pkl's actual coefficient count. Do not trust the index table until resolved.")
    print()
    print("NOTE: playground/dcubot/custom_rewards.py's reward_imitation_dcubot only reads")
    print("indices [0:28] of this vector (joints_pos+joints_vel+foot_contacts) -- the")
    print("trailing base_linear_vel/base_angular_vel (indices 28:34) are written to the pkl")
    print("but intentionally not used for imitation (Placo's base velocity isn't reliable).")


if __name__ == "__main__":
    main()
