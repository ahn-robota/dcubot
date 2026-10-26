"""Visual check of DCUBOT left/right reference points in MuJoCo.

Loads the playground scene at the 'home' keyframe, prints the numbers, and draws
markers in the viewer (translucent robot so the markers stay visible):

  red     : world origin (0, 0, 0)            -- URDF base frame (MJCF still pre-recentre)
  green   : body centre (base_link inertial)
  blue    : hip midline (between hip_yaw L/R)
  yellow  : ankle midline (foot_L_1 / foot_R_1 body origins = ankle axes)
  magenta : sole midline (foot collision mesh bbox centres, world)

Expected after the fixes: green, blue and magenta lie on one line (y ~ same),
yellow is ~18 mm away from them on both sides (ankle axis vs sole, same for both feet).

Run (from Open_Duck_Playground):
    .venv/bin/python tools/show_offset.py
    .venv/bin/python tools/show_offset.py --alpha 0.15
    .venv/bin/python tools/show_offset.py --xml playground/dcubot/xmls/dcubot.xml
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

MARKERS = {
    "origin":       [1.0, 0.1, 0.1, 1.0],
    "body_centre":  [0.1, 1.0, 0.1, 1.0],
    "hip_midline":  [0.1, 0.4, 1.0, 1.0],
    "ankle_midline": [1.0, 0.9, 0.1, 1.0],
    "sole_midline": [1.0, 0.2, 1.0, 1.0],
}


def add_marker(scn, pos, rgba, radius=0.006):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
        np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, np.float32),
    )
    scn.ngeom += 1


def world_mesh_centre(m, d, gid):
    """Centre of a collision mesh's bounding box, in world coordinates."""
    mid = m.geom_dataid[gid]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[va:va + vn].astype(float)
    v_world = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
    return (v_world.min(0) + v_world.max(0)) / 2


def compute_points(m, d):
    hip_l = d.xpos[m.body("hip_yaw_L_1").id]
    hip_r = d.xpos[m.body("hip_yaw_R_1").id]
    ankle_l = d.xpos[m.body("foot_L_1").id]
    ankle_r = d.xpos[m.body("foot_R_1").id]
    body_centre = d.xipos[m.body("base_link").id].copy()
    sole_l = world_mesh_centre(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"))
    sole_r = world_mesh_centre(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"))

    pts = {
        "origin": np.zeros(3),
        "body_centre": body_centre,
        "hip_midline": (hip_l + hip_r) / 2,
        "ankle_midline": (ankle_l + ankle_r) / 2,
        "sole_midline": (sole_l + sole_r) / 2,
    }
    info = {
        "hip_spacing_mm": (hip_l[1] - hip_r[1]) * 1e3,
        "body_centre_y_mm": body_centre[1] * 1e3,
        "hip_midline_y_mm": pts["hip_midline"][1] * 1e3,
        "ankle_midline_y_mm": pts["ankle_midline"][1] * 1e3,
        "sole_midline_y_mm": pts["sole_midline"][1] * 1e3,
        "left_sole_rel_body_mm": (sole_l[1] - body_centre[1]) * 1e3,
        "right_sole_rel_body_mm": (sole_r[1] - body_centre[1]) * 1e3,
        "sole_midline_minus_body_mm": (pts["sole_midline"][1] - body_centre[1]) * 1e3,
        "ankle_minus_sole_midline_mm": (pts["ankle_midline"][1] - pts["sole_midline"][1]) * 1e3,
    }
    return pts, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="playground/dcubot/xmls/scene_flat_terrain.xml")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="robot transparency (0=invisible, 1=opaque); markers stay visible")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    m.geom_rgba[m.geom_bodyid != 0, 3] = args.alpha
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)  # 'home' keyframe
    mujoco.mj_forward(m, d)

    pts, info = compute_points(m, d)
    print("== reference points at home keyframe (mm, world y) ==")
    for k, v in info.items():
        print(f"  {k:30s} {v:+9.2f}")
    print("colours: red=origin, green=body centre, blue=hip midline,")
    print("         yellow=ankle midline, magenta=sole midline")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.02, 0.12]
        viewer.cam.distance = 0.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            for name, rgba in MARKERS.items():
                add_marker(viewer.user_scn, pts[name], rgba)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
