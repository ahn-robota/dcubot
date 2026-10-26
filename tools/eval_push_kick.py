"""Objective "did it flail/step back" test: apply a single instantaneous
qvel kick (same mechanism as training's push_config -- NOT a sustained
force/mouse-drag) to a standing policy and measure fall/tilt/recovery plus
two numeric stand-ins for "바둥거림" (active leg flailing): summed |joint
angle change| and foot-airborne-event count over the 1s immediately after
the kick. Supports dcubot checkpoints, a "no policy" (hold home pose)
baseline, and the original open_duck_mini_v2 project's own pretrained
policy (different obs/action dims, different MjInfer signature -- handled
via a thin adapter). Native CPU mujoco throughout, no mjx/GPU needed.

Usage:
    .venv/bin/python tools/eval_push_kick.py
"""
import argparse
import csv
import json
import os
import sys

import mujoco
import numpy as np


# ---- dcubot variant ----------------------------------------------------
def make_dcubot_infer(onnx_path):
    sys.path.insert(0, "/home/ahn/dcubot/DCUBOT/rl_training")
    from playground.dcubot.mujoco_infer import MjInfer

    inf = MjInfer("playground/dcubot/xmls/scene_flat_terrain.xml", onnx_path, False)
    inf.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    foot_names = ["left_foot_collision", "right_foot_collision"]
    return inf, foot_names, None


def make_dcubot_no_policy():
    """Pure passive baseline: ctrl held at default_actuator (home pose) every step, no onnx at all."""
    sys.path.insert(0, "/home/ahn/dcubot/DCUBOT/rl_training")
    from playground.dcubot.mujoco_infer_base import MJInferBase

    class NoPolicyInfer(MJInferBase):
        def __init__(self, model_path):
            super().__init__(model_path)
            self.action_scale = 0.25
            self.decimation = 10
            self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.last_action = np.zeros(self.num_dofs)
            self.last_last_action = np.zeros(self.num_dofs)
            self.last_last_last_action = np.zeros(self.num_dofs)
            self.motor_targets = self.default_actuator.copy()

        def get_obs(self, data, commands):
            return np.zeros(1)  # unused by _ZeroPolicy, just needs to exist

        class _ZeroPolicy:
            def infer(self, obs):
                return np.zeros(13)

        policy = _ZeroPolicy()

    inf = NoPolicyInfer("playground/dcubot/xmls/scene_flat_terrain.xml")
    foot_names = ["left_foot_collision", "right_foot_collision"]
    return inf, foot_names, None


# ---- original open_duck_mini_v2 variant --------------------------------
def make_original_infer(onnx_path):
    os.chdir("/home/ahn/dcubot/OpenDuckMini/rl_training")
    sys.path.insert(0, "/home/ahn/dcubot/OpenDuckMini/rl_training")
    from playground.open_duck_mini_v2.mujoco_infer import MjInfer as OrigMjInfer

    inf = OrigMjInfer(
        "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
        "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
        onnx_path,
        False,
    )
    inf.commands = [0.0] * len(inf.commands)
    foot_names = ["left_foot_bottom_tpu", "right_foot_bottom_tpu"]
    return inf, foot_names, None


# ---- shared trial logic --------------------------------------------------
def tilt_deg(infer):
    g = infer.get_gravity(infer.data)
    return float(np.degrees(np.arccos(np.clip(g[2], -1.0, 1.0))))


def fallen(infer):
    return infer.get_gravity(infer.data)[-1] < 0.0


def reset_idle(infer):
    mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)
    infer.last_action[:] = 0
    infer.last_last_action[:] = 0
    infer.last_last_last_action[:] = 0
    infer.motor_targets = infer.default_actuator.copy()
    if hasattr(infer, "prev_motor_targets"):
        infer.prev_motor_targets = infer.default_actuator.copy()
    if hasattr(infer, "gait_step"):
        infer.gait_step = 0


def step_policy(infer):
    for _ in range(infer.decimation):
        mujoco.mj_step(infer.model, infer.data)
    obs = infer.get_obs(infer.data, infer.commands)
    action = infer.policy.infer(obs)
    infer.last_last_last_action = infer.last_last_action.copy()
    infer.last_last_action = infer.last_action.copy()
    infer.last_action = action.copy()
    motor_targets = infer.default_actuator + action * infer.action_scale
    infer.data.ctrl = motor_targets.copy()


def foot_geom_ids(infer, foot_names):
    return [mujoco.mj_name2id(infer.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in foot_names]


def feet_airborne_mask(infer, foot_ids):
    """True per foot if that foot geom is in zero active contacts right now."""
    contacting = set()
    for i in range(infer.data.ncon):
        c = infer.data.contact[i]
        if c.geom1 in foot_ids:
            contacting.add(c.geom1)
        if c.geom2 in foot_ids:
            contacting.add(c.geom2)
    return [fid not in contacting for fid in foot_ids]


def kick_trial(infer, foot_ids, magnitude, theta, settle_s=1.0, window_s=1.0, post_s=3.0):
    reset_idle(infer)
    n_settle = int(settle_s / (infer.sim_dt * infer.decimation))
    for _ in range(n_settle):
        step_policy(infer)

    infer.data.qvel[0] += magnitude * np.cos(theta)
    infer.data.qvel[1] += magnitude * np.sin(theta)

    n_window = int(window_s / (infer.sim_dt * infer.decimation))
    n_post = int(post_s / (infer.sim_dt * infer.decimation))

    joint_pos_series = []
    tilts = []
    fell_during = False
    air_prev = [False, False]
    air_events = [0, 0]

    for i in range(n_post):
        step_policy(infer)
        t = tilt_deg(infer)
        tilts.append(t)
        if fallen(infer):
            fell_during = True
        if i < n_window:
            joints = infer.get_actuator_joints_qpos(infer.data.qpos) if hasattr(infer, "get_actuator_joints_qpos") else infer.data.qpos[-infer.num_dofs:]
            joint_pos_series.append(np.array(joints))
            air_now = feet_airborne_mask(infer, foot_ids)
            for fi in range(2):
                if air_now[fi] and not air_prev[fi]:
                    air_events[fi] += 1
            air_prev = air_now

    joint_pos_series = np.array(joint_pos_series)
    joint_motion = float(np.sum(np.abs(np.diff(joint_pos_series, axis=0)))) if len(joint_pos_series) > 1 else 0.0
    recovered_idx = next((i for i, t in enumerate(tilts) if t < 10.0), None)
    time_to_recover = recovered_idx * infer.sim_dt * infer.decimation if recovered_idx is not None else None

    return {
        "fell": fell_during,
        "max_tilt_deg": float(np.max(tilts)),
        "final_tilt_deg": float(tilts[-1]),
        "time_to_recover_s": time_to_recover,
        "joint_motion_sum_1s": joint_motion,
        "foot_airborne_events_1s": sum(air_events),
    }


def run_variant(name, make_fn, magnitudes, directions, trials_random=0, seed=0):
    print(f"\n=== variant: {name} ===", flush=True)
    infer, foot_names, _ = make_fn()
    foot_ids = foot_geom_ids(infer, foot_names)
    rng = np.random.default_rng(seed)
    results = []
    dir_thetas = {"+X": 0.0, "-X": np.pi, "+Y": np.pi / 2, "-Y": -np.pi / 2}
    for mag in magnitudes:
        for dname in directions:
            theta = dir_thetas[dname]
            r = kick_trial(infer, foot_ids, mag, theta)
            r.update({"variant": name, "magnitude": mag, "direction": dname})
            print(f"  mag={mag} dir={dname}: fell={r['fell']} max_tilt={r['max_tilt_deg']:.1f} "
                  f"joint_motion={r['joint_motion_sum_1s']:.3f} airborne_events={r['foot_airborne_events_1s']}", flush=True)
            results.append(r)
        for t in range(trials_random):
            theta = rng.uniform(0, 2 * np.pi)
            r = kick_trial(infer, foot_ids, mag, theta)
            r.update({"variant": name, "magnitude": mag, "direction": f"rand{t}"})
            results.append(r)
    return results, infer


VARIANTS = {
    "dcubot_post_fix": lambda base: make_dcubot_infer(
        f"{base}/checkpoints_tracking_fix_verify/2026_10_19_211742_1966080.onnx"),
    "dcubot_pre_fix": lambda base: make_dcubot_infer(
        f"{base}/checkpoints_standing_joystick_merge/2026_10_18_104047_2129920.onnx"),
    "dcubot_no_policy": lambda base: make_dcubot_no_policy(),
    "original_pretrained": lambda base: make_original_infer(
        f"{base}/.scratch_open_duck_mini_pretrained/BEST_WALK_ONNX.onnx"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--random_trials", type=int, default=0)
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), required=True,
                     help="Run one variant per process -- 'original_pretrained' imports a "
                          "same-named top-level 'playground' package as the dcubot variants, "
                          "which Python's import cache can't disambiguate within one process. "
                          "Run each variant as its own process invocation.")
    args = ap.parse_args()
    base = "/home/ahn/dcubot/DCUBOT/rl_training"
    os.makedirs(os.path.join(base, args.out_dir), exist_ok=True)

    magnitudes = [0.3, 0.5, 0.8, 1.0]
    directions = ["+X", "-X", "+Y", "-Y"]

    results, _ = run_variant(args.variant, lambda: VARIANTS[args.variant](base),
                              magnitudes, directions, args.random_trials)

    os.chdir(base)
    csv_path = os.path.join(args.out_dir, f"push_kick_{args.variant}.csv")
    fieldnames = ["variant", "magnitude", "direction", "fell", "max_tilt_deg",
                  "final_tilt_deg", "time_to_recover_s", "joint_motion_sum_1s",
                  "foot_airborne_events_1s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k) for k in fieldnames})
    json.dump(results, open(os.path.join(base, args.out_dir, f"push_kick_{args.variant}.json"), "w"), indent=2)
    print(f"\nsaved {csv_path}")


if __name__ == "__main__":
    main()
