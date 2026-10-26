"""Evaluate a DCUBOT checkpoint's fallen-init recovery (USE_FALLEN_INIT,
joystick.py Step 2 curriculum): reset tilted/moving the same way
fallen_init_config does during training (roll/pitch up to
roll_pitch_range_deg, base linvel +-lin_vel_range, base angvel/joint vel
+-ang_vel_range), hold a zero (standing) command, and check whether the
policy actively drives tilt back down (recovery) or just collapses further
while barely actuating ("barbie doll" -- the exact failure mode this
curriculum is meant to fix). Deterministic replay, CPU only.

Usage:
    .venv/bin/python tools/eval_fallen_recovery.py -o <checkpoint>.onnx \
        [--angles_deg 15 25 30 35] [--trials 5] [--duration 4.0] [--out_dir DIR] [--tag NAME]
"""
import argparse
import json
import os

import mujoco
import numpy as np

from playground.dcubot.mujoco_infer import MjInfer


def axis_angle_quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array(axis, dtype=float), float(angle))
    return q


def quat_mul(q1, q2):
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q1, q2)
    return q


def tilt_deg(infer):
    g = infer.get_gravity(infer.data)  # ~[0,0,1] upright (upvector sensor, see base.py)
    return float(np.degrees(np.arccos(np.clip(g[2], -1.0, 1.0))))


def fallen(infer):
    return infer.get_gravity(infer.data)[-1] < 0.0


def reset_idle(infer):
    """Plain upright reset -- used to measure the idle/passive torque baseline."""
    mujoco.mj_resetDataKeyframe(infer.model, infer.data, 0)
    infer.last_action[:] = 0
    infer.last_last_action[:] = 0
    infer.last_last_last_action[:] = 0
    infer.motor_targets = infer.default_actuator.copy()
    infer.prev_motor_targets = infer.default_actuator.copy()
    infer.gait_step = 0
    infer.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def reset_fallen(infer, roll_deg, pitch_deg, lin_vel, ang_vel, joint_vel):
    """Mirrors joystick.py reset()'s USE_FALLEN_INIT branch: tilt the base by
    roll_deg/pitch_deg (same axis_angle_to_quat + quat_mul order as
    training), overwrite base linvel/angvel and all joint vel."""
    reset_idle(infer)
    addr = infer._floating_base_qpos_addr
    quat0 = infer.data.qpos[addr + 3 : addr + 7].copy()
    roll_q = axis_angle_quat([1.0, 0.0, 0.0], np.radians(roll_deg))
    pitch_q = axis_angle_quat([0.0, 1.0, 0.0], np.radians(pitch_deg))
    new_quat = quat_mul(quat0, quat_mul(roll_q, pitch_q))
    infer.data.qpos[addr + 3 : addr + 7] = new_quat

    vaddr = infer._floating_base_qvel_addr
    infer.data.qvel[vaddr : vaddr + 3] = lin_vel
    infer.data.qvel[vaddr + 3 : vaddr + 6] = ang_vel
    infer.data.qvel[infer.actuator_qvel_addr] = joint_vel
    mujoco.mj_forward(infer.model, infer.data)


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


def idle_torque_baseline(infer, seconds=2.0):
    """Mean |torque| while standing undisturbed -- the 'doing basically
    nothing' reference point for the barbie-doll check below."""
    reset_idle(infer)
    n_steps = int(seconds / (infer.sim_dt * infer.decimation))
    torques = []
    for _ in range(n_steps):
        step_policy(infer, torque_log=torques)
    return float(np.mean(np.abs(torques)))


def recovery_trial(infer, roll_deg, pitch_deg, lin_vel, ang_vel, joint_vel, duration):
    reset_fallen(infer, roll_deg, pitch_deg, lin_vel, ang_vel, joint_vel)
    initial_tilt = tilt_deg(infer)
    n_steps = int(duration / (infer.sim_dt * infer.decimation))
    tilts, torques = [], []
    fell_during = False
    for _ in range(n_steps):
        step_policy(infer, torque_log=torques)
        tilts.append(tilt_deg(infer))
        if fallen(infer):
            fell_during = True
    final_tilt = tilts[-1]
    return {
        "initial_tilt_deg": initial_tilt,
        "final_tilt_deg": final_tilt,
        "min_tilt_deg": float(np.min(tilts)),
        "max_tilt_deg": float(np.max(tilts)),
        "fell": fell_during,
        "mean_abs_torque_Nm": float(np.mean(np.abs(torques))),
        "recovered": (not fell_during) and (final_tilt < 15.0),
    }


def sweep(infer, angles_deg, trials, duration, lin_vel_range=2.0, ang_vel_range=0.5, seed=0):
    rng = np.random.default_rng(seed)
    nu = infer.num_dofs
    results = []
    for angle in angles_deg:
        trial_results = []
        for _ in range(trials):
            roll_sign = rng.choice([-1, 1])
            pitch_sign = rng.choice([-1, 1])
            roll_deg = roll_sign * angle
            pitch_deg = pitch_sign * angle
            lin_vel = rng.uniform(-lin_vel_range, lin_vel_range, size=3)
            ang_vel = rng.uniform(-ang_vel_range, ang_vel_range, size=3)
            joint_vel = rng.uniform(-ang_vel_range, ang_vel_range, size=nu)
            trial_results.append(
                recovery_trial(infer, roll_deg, pitch_deg, lin_vel, ang_vel, joint_vel, duration)
            )
        n_recovered = sum(t["recovered"] for t in trial_results)
        mean_torque = float(np.mean([t["mean_abs_torque_Nm"] for t in trial_results]))
        mean_final_tilt = float(np.mean([t["final_tilt_deg"] for t in trial_results]))
        print(
            f"  tilt {angle:.0f}deg: {n_recovered}/{trials} recovered "
            f"(final tilt mean={mean_final_tilt:.1f}deg, torque mean={mean_torque:.2f}Nm)",
            flush=True,
        )
        results.append({
            "angle_deg": angle,
            "trials": trials,
            "recovered": n_recovered,
            "recovery_rate": n_recovered / trials,
            "mean_final_tilt_deg": mean_final_tilt,
            "mean_abs_torque_Nm": mean_torque,
            "trial_detail": trial_results,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", required=True)
    ap.add_argument("--model_path", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    ap.add_argument("--angles_deg", type=float, nargs="+", default=[15, 25, 30, 35])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--tag", default="fallen_recovery")
    args = ap.parse_args()

    infer = MjInfer(args.model_path, args.onnx_model_path, False)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== idle torque baseline (2s, undisturbed) ===", flush=True)
    baseline = idle_torque_baseline(infer)
    print(f"  mean |torque| = {baseline:.3f} Nm")

    print(f"=== fallen-init recovery sweep (angles={args.angles_deg}, trials={args.trials}, {args.duration}s each) ===", flush=True)
    results = sweep(infer, args.angles_deg, args.trials, args.duration)

    # "Barbie doll" check: recovery torque barely above idle baseline despite
    # high tilt and low recovery rate -- i.e. the policy isn't actually
    # trying. 1.5x is a conservative margin (real recovery pushes torque
    # noticeably above idle sway); tune after seeing real numbers.
    overall_torque = float(np.mean([r["mean_abs_torque_Nm"] for r in results]))
    overall_recovery_rate = float(np.mean([r["recovery_rate"] for r in results]))
    barbie_doll = overall_recovery_rate < 0.5 and overall_torque < 1.5 * baseline

    out = {
        "onnx": os.path.basename(args.onnx_model_path),
        "idle_torque_baseline_Nm": baseline,
        "sweep": results,
        "overall_recovery_rate": overall_recovery_rate,
        "overall_mean_torque_Nm": overall_torque,
        "barbie_doll_suspected": barbie_doll,
    }
    json_path = os.path.join(args.out_dir, f"{args.tag}.json")
    json.dump(out, open(json_path, "w"), indent=2, ensure_ascii=False)
    print(f"\noverall recovery rate: {overall_recovery_rate*100:.0f}%  "
          f"overall torque: {overall_torque:.2f}Nm (idle baseline {baseline:.2f}Nm)")
    if barbie_doll:
        print("!!! BARBIE DOLL SUSPECTED: low recovery rate and torque near idle baseline -- "
              "policy does not appear to be actively trying to recover.")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
