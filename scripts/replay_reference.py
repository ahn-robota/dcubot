"""Step 4 validation: for each pkl command, set the robot's qpos/qvel DIRECTLY
from the reference frame (teleport, not actuator-driven) at every phase step
of one gait period, then check reward_imitation_dcubot's components. Since
the robot is forced to literally be the reference pose, each component
should be near its max (pos/vel ~1.0, contact ~1.0) if the slicing/remapping
in custom_rewards.py is correct; a low component means the slice is wrong.

Usage (from rl_training/):
    .venv/bin/python scripts/replay_reference.py
"""

import numpy as np
import mujoco
import pickle
import jax.numpy as jp

PKL = "playground/dcubot/data/polynomial_coefficients_slow.pkl"
XML = "playground/dcubot/xmls/scene_flat_terrain.xml"


def sample_polynomial(t, coeffs):
    # coeffs are already flipped once (highest-degree-first, np.polyval's
    # convention) -- matches poly_reference_motion.py's load-time flip.
    return np.array([np.polyval(c, t) for c in coeffs])


def main():
    data = pickle.load(open(PKL, "rb"))
    m = mujoco.MjModel.from_xml_path(XML)
    floor_id = m.geom("floor").id
    left_geom = m.geom("left_foot_collision").id
    right_geom = m.geom("right_foot_collision").id

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    home_qpos_base = d.qpos[:7].copy()  # keep base pose fixed at the home keyframe

    for key in sorted(data.keys(), key=lambda k: float(k.split("_")[0])):
        entry = data[key]
        period, fps = entry["period"], entry["fps"]
        nb_steps = int(period * fps)
        raw_coeffs = [np.flip(np.array(v)) for v in entry["coefficients"].values()]

        pos_rews, vel_rews, contact_rews = [], [], []
        for i in range(nb_steps):
            t = (i % nb_steps) / nb_steps
            ref = sample_polynomial(t, raw_coeffs)  # 34-dim

            # Placo order: left(5) pos[0:5], head(3) pos[5:8], right(5) pos[8:13]
            # Actuator order: left(5), right(5), head(3) -- same remap as
            # reward_imitation_dcubot/joystick.py use for the actual joints.
            qpos_actuator_order = np.concatenate([ref[0:5], ref[8:13], ref[5:8]])
            qvel_actuator_order = np.concatenate([ref[13:18], ref[21:26], ref[18:21]])

            d.qpos[:7] = home_qpos_base
            d.qpos[7:20] = qpos_actuator_order
            d.qvel[:6] = 0.0
            d.qvel[6:19] = qvel_actuator_order
            mujoco.mj_forward(m, d)

            contact_left = any(
                (c.geom1 == left_geom and c.geom2 == floor_id)
                or (c.geom2 == left_geom and c.geom1 == floor_id)
                for c in d.contact[: d.ncon]
            )
            contact_right = any(
                (c.geom1 == right_geom and c.geom2 == floor_id)
                or (c.geom2 == right_geom and c.geom1 == floor_id)
                for c in d.contact[: d.ncon]
            )
            contacts = jp.array([float(contact_left), float(contact_right)])

            reference_frame = jp.array(ref)
            ref_joint_pos = jp.concatenate([reference_frame[0:5], reference_frame[8:13]])
            ref_joint_vel = jp.concatenate([reference_frame[13:18], reference_frame[21:26]])
            ref_contacts = jp.where(reference_frame[26:28] > 0.5, 1.0, 0.0)

            joint_pos = jp.array(qpos_actuator_order[:10])
            joint_vel = jp.array(qvel_actuator_order[:10])
            pos_err = jp.sum(jp.square(joint_pos - ref_joint_pos))
            vel_err = jp.sum(jp.square(joint_vel - ref_joint_vel))
            pos_rew = float(jp.exp(-pos_err / 0.1))
            vel_rew = float(jp.exp(-0.1 * vel_err))
            contact_rew = float(jp.mean((contacts == ref_contacts).astype(jp.float32)))

            pos_rews.append(pos_rew)
            vel_rews.append(vel_rew)
            contact_rews.append(contact_rew)

            if contact_rew < 1.0:
                print(
                    f"    [contact mismatch] key={key} i={i} t={t:.2f} "
                    f"sim_contacts={contacts} ref_contacts={ref_contacts} "
                    f"ref_foot_contact_raw={np.round(ref[26:28], 3)}"
                )

        print(f"key={key}  nb_steps={nb_steps}")
        print(f"  joint_pos_rew: mean={np.mean(pos_rews):.3f} min={np.min(pos_rews):.3f}")
        print(f"  joint_vel_rew: mean={np.mean(vel_rews):.3f} min={np.min(vel_rews):.3f}")
        print(f"  contact_rew:   mean={np.mean(contact_rews):.3f} min={np.min(contact_rews):.3f}")
        combined = 0.5 * np.mean(pos_rews) + 0.25 * np.mean(vel_rews) + 0.25 * np.mean(contact_rews)
        print(f"  combined (0.5*pos+0.25*vel+0.25*contact): {combined:.3f}  (max=1.0)")
        print()


if __name__ == "__main__":
    main()
