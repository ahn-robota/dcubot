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

NOTE: neither source is what playground/dcubot/joystick.py currently trains
on -- the active pkl (polynomial_coefficients_slow.pkl) is forward-only (see
docs/CHANGES_FROM_ORIGINAL.md #2). This script is for inspecting the
reference *motion data itself*, independent of training state.

Usage (from rl_training/):
    .venv/bin/python scripts/replay_directions_mujoco.py [--source omni|single_axis]
"""

import argparse
import json
import time

import mujoco
import mujoco.viewer
import numpy as np

XML = "playground/dcubot/xmls/scene_flat_terrain.xml"
SEGMENT_SECONDS = 3.0

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(SOURCES.keys()), default="omni")
    args = parser.parse_args()
    motions_dir, sequence = SOURCES[args.source]

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    recordings = {}
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
