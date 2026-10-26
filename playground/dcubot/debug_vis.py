"""Direction / coordinate-frame overlay for the dcubot viewers (like mjlab's
velocity-command arrows): body axes + commanded vs actual velocity arrows.

  body axes   : X red (forward), Y green (left), Z blue (up)
  blue arrow  : commanded linear velocity (body frame -> world)
  cyan arrow  : actual linear velocity
  green / light-green vertical arrow: commanded / actual yaw rate (up = +z)
"""
import mujoco
import numpy as np

LIN_SCALE = 2.0  # arrow length (m) per m/s
ANG_SCALE = 0.2  # arrow length (m) per rad/s


def add_arrow(scn, p0, p1, rgba, width=0.006):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if scn.ngeom >= scn.maxgeom or np.linalg.norm(p1 - p0) < 1e-4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
        np.asarray(rgba, np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, p0, p1)
    scn.ngeom += 1


def draw(scn, data, imu_site_id, command, gyro_z):
    R = data.site_xmat[imu_site_id].reshape(3, 3)
    pos = data.site_xpos[imu_site_id]
    # axes drawn on the floor under the robot so they are not hidden by the body
    ground = np.array([pos[0], pos[1], 0.006])
    for axis, rgba in zip(range(3), ([1, 0, 0, 1], [0, 0.8, 0, 1], [0, 0, 1, 1])):
        add_arrow(scn, ground, ground + R[:, axis] * 0.15, rgba, 0.006)

    top = pos + np.array([0.0, 0.0, 0.22])
    cmd_w = R @ np.array([command[0], command[1], 0.0])
    add_arrow(scn, top, top + cmd_w * LIN_SCALE, [0.2, 0.2, 1, 1])  # cmd lin (blue)
    add_arrow(scn, top, top + np.asarray(data.qvel[:3]) * LIN_SCALE, [0, 1, 1, 1])  # actual (cyan)
    up = top + np.array([0.0, 0.0, 0.05])
    add_arrow(scn, up, up + np.array([0, 0, command[2] * ANG_SCALE]), [0, 0.7, 0, 1])
    add_arrow(scn, up, up + np.array([0, 0, gyro_z * ANG_SCALE]), [0.5, 1, 0.5, 1])


def local_velocity(data, imu_site_id):
    R = data.site_xmat[imu_site_id].reshape(3, 3)
    return R.T @ np.asarray(data.qvel[:3])
