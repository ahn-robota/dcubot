"""Replay the RAW Placo reference-motion recordings (not the trained policy,
not any fitted pkl) directly in the dcubot MuJoCo viewer: forward -> backward
-> strafe left -> strafe right -> turn left -> turn right, SEGMENT_SECONDS
each, back to back.

Each segment teleports qpos (base position/orientation + all 13 joint
angles) straight from the recording's own frames every step -- this is the
actual Placo-simulated motion, kinematically replayed, not run through
physics or any policy. Base resets to that recording's own origin at the
start of each segment so all six are easy to watch without the robot
drifting off-screen.

Two sources, pick with --source:
  omni (default): pure single-axis slices of the 2026-10-20 210-combo
    omnidirectional sweep (recordings_omnidirectional/) -- the "well-made"
    full sweep the user built against the 5 validation criteria
    (foot-contact, hip_pitch limit, L/R symmetry, no foot overlap,
    commanded:achieved ratio). Picking only the pure single-axis points
    (the other two axes exactly 0) deliberately avoids the simultaneous
    3-axis-extreme combos that a later investigation in this session found
    to be degenerate/noisy -- confirmed clean here via each file's own
    Vel_x/Vel_y/Yaw fields (e.g. 116_0.0_0.196_0.0.json: Vel_x=-0.0002,
    Vel_y=0.1075, matching its pure-strafe intent).
  single_axis: the older 2026-10-15 motions_dcubot/ 9-file set (already
    validated via gait_playground separately).
  pkl: the fitted polynomial_coefficients_omnidirectional.pkl that
    playground/dcubot/joystick.py actually trains on (in git, so this works
    on any machine). Joint angles come from the polynomials; the pkl has no
    base pose, so the base stays at the home keyframe's height/trunk pitch
    and x/y/yaw are integrated from the pkl's base linear/angular velocity.

omni/single_axis read raw recordings that live outside the repo (laptop
only). This script is for inspecting the reference *motion data itself*,
independent of training state.

Usage (from rl_training/):
    .venv/bin/python scripts/replay_directions_mujoco.py [--source omni|single_axis|pkl]
"""

import argparse
import json
import time

import mujoco
import mujoco.viewer
import numpy as np

from playground.common.poly_reference_motion_numpy import PolyReferenceMotion

XML = "playground/dcubot/xmls/scene_flat_terrain.xml"
PKL = "playground/dcubot/data/polynomial_coefficients_omnidirectional.pkl"
SEGMENT_SECONDS = 3.0

# Same six directions as the "omni" recordings, as (dx, dy, dtheta) pkl keys.
PKL_SEQUENCE = [
    ("forward", (0.196, 0.0, 0.0)),
    ("backward", (-0.196, 0.0, 0.0)),
    ("strafe_left", (0.0, 0.196, 0.0)),
    ("strafe_right", (0.0, -0.196, 0.0)),
    ("turn_left", (0.0, 0.0, 0.98)),
    ("turn_right", (0.0, 0.0, -0.98)),
]

SOURCES = {
    "omni": (
        "/home/ahn/dcubot/DCUBOT/reference_motion_generator/recordings_omnidirectional",
        [
            ("forward", "106_0.196_0.0_0.0.json"),
            ("backward", "102_-0.196_0.0_0.0.json"),
            ("strafe_left", "116_0.0_0.196_0.0.json"),
            ("strafe_right", "92_0.0_-0.196_0.0.json"),
            ("turn_left", "146_0.0_0.0_0.98.json"),
            ("turn_right", "62_0.0_0.0_-0.98.json"),
        ],
    ),
    "single_axis": (
        "/home/ahn/dcubot/DCUBOT/reference_motion_generator/motions_dcubot",
        [
            ("forward", "forward_medium_0.02_0.0_0.0.json"),
            ("backward", "backward_-0.04_0.0_0.0.json"),
            ("strafe_left", "strafe_left_0.0_0.04_0.0.json"),
            ("strafe_right", "strafe_right_0.0_-0.04_0.0.json"),
            ("turn_left", "turn_left_0.0_0.0_0.5.json"),
            ("turn_right", "turn_right_0.0_0.0_-0.5.json"),
        ],
    ),
}


def load_recording(path):
    d = json.load(open(path))
    fo = d["Frame_offset"][0]
    frames = np.array(d["Frames"])
    fps = d["FPS"]
    root_pos = frames[:, fo["root_pos"] : fo["root_pos"] + 3]
    # scipy's R.from_matrix(...).as_quat() convention: [x, y, z, w].
    # Reorder to MuJoCo's [w, x, y, z].
    root_quat_xyzw = frames[:, fo["root_quat"] : fo["root_quat"] + 4]
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    # Recording/Joints order: left leg(5), head(3), right leg(5).
    # MuJoCo qpos (actuator/body-declaration) order: left leg(5), right
    # leg(5), head(3). Remap here -- NOT doing this previously put head
    # angles into the right-hip qpos slots and right-leg angles into the
    # head/right-knee/ankle slots, which is exactly why the right leg
    # looked broken while the left leg (unaffected, same slot order in
    # both conventions) looked fine.
    joints_pos_recording_order = frames[:, fo["joints_pos"] : fo["left_toe_pos"]]
    joints_pos = np.concatenate(
        [
            joints_pos_recording_order[:, 0:5],  # left leg
            joints_pos_recording_order[:, 8:13],  # right leg
            joints_pos_recording_order[:, 5:8],  # head
        ],
        axis=1,
    )
    return root_pos, root_quat_wxyz, joints_pos, fps


def load_pkl_motion(prm, command, home_qpos_base):
    """One SEGMENT_SECONDS clip from the pkl, in load_recording's format.

    Frame layout (see custom_rewards.py / scripts/inspect_reference.py):
    joints_pos[0:13], joints_vel[13:26], foot_contacts[26:28],
    base_linear_vel[28:31], base_angular_vel[31:34]; joints in Placo order
    left leg(5), head(3), right leg(5).
    """
    fps = prm.fps
    n = int(SEGMENT_SECONDS * fps)
    home_quat = home_qpos_base[3:7]
    xy, yaw = np.zeros(2), 0.0
    root_pos, root_quat, joints_pos = [], [], []
    for i in range(n):
        ref = np.array(prm.get_reference_motion(*command, i))
        joints_pos.append(np.concatenate([ref[0:5], ref[8:13], ref[5:8]]))
        root_pos.append([xy[0], xy[1], home_qpos_base[2]])
        yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, yaw_quat, home_quat)
        root_quat.append(quat)
        xy += ref[28:30] / fps
        yaw += ref[33] / fps
    return np.array(root_pos), np.array(root_quat), np.array(joints_pos), fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=list(SOURCES.keys()) + ["pkl"], default="omni"
    )
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    recordings = {}
    if args.source == "pkl":
        prm = PolyReferenceMotion(PKL)
        sequence = [(label, f"{label} {cmd}") for label, cmd in PKL_SEQUENCE]
        for (label, name), (_, cmd) in zip(sequence, PKL_SEQUENCE):
            recordings[name] = load_pkl_motion(prm, cmd, d.qpos[:7].copy())
    else:
        motions_dir, sequence = SOURCES[args.source]
        for label, fname in sequence:
            recordings[fname] = load_recording(f"{motions_dir}/{fname}")

    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
        m.vis.scale.jointlength = 0.05
        m.vis.scale.jointwidth = 0.015

        while True:
            for label, fname in sequence:
                root_pos, root_quat, joints_pos, fps = recordings[fname]
                n = len(root_pos)
                # Recenter X/Y only (so the robot doesn't drift off-screen
                # across the loop) -- Z must stay as recorded, it's the
                # actual standing/walking height (~0.25m), not an offset to
                # cancel out. Zeroing it too would plant the base at z=0,
                # i.e. on/in the floor.
                origin_xy = root_pos[0, :2].copy()
                print(f"[{label}] {fname}  ({n} frames @ {fps}fps)")

                seg_start = time.time()
                i = 0
                while time.time() - seg_start < SEGMENT_SECONDS:
                    frame_start = time.time()
                    idx = i % n
                    d.qpos[0:2] = root_pos[idx, :2] - origin_xy
                    d.qpos[2] = root_pos[idx, 2]
                    d.qpos[3:7] = root_quat[idx]
                    d.qpos[7:20] = joints_pos[idx]
                    mujoco.mj_forward(m, d)
                    viewer.sync()
                    i += 1
                    dt = 1.0 / fps - (time.time() - frame_start)
                    if dt > 0:
                        time.sleep(dt)


if __name__ == "__main__":
    main()
