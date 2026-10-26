"""Why does dcubot fall in ~1.4 s at step 0 during training eval while the duck
survives ~10 s? Hold the default pose (action = 0) under the training
randomizations/pushes, one factor at a time, and report time-to-fall.
Plain MuJoCo on CPU (GPU is busy training); mirrors playground/dcubot/randomize.py.
"""
import sys
import numpy as np
import mujoco
from etils import epath

which_pkg = sys.argv[1] if len(sys.argv) > 1 else "dcubot"
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 12
if which_pkg == "dcubot":
    from playground.dcubot import base, joystick
    xml = "playground/dcubot/xmls/scene_flat_terrain_train.xml"
    assets = base.get_assets(lowpoly=True)
else:
    from playground.open_duck_mini_v2 import base, joystick
    xml = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    assets = base.get_assets()
cfg = joystick.default_config()
pc = cfg.push_config
M0 = mujoco.MjModel.from_xml_string(epath.Path(xml).read_text(), assets=assets)
FLOOR = M0.geom("floor").id
kf = M0.keyframe("home")
default_ctrl = kf.ctrl.copy()
ACT_DOFS = np.array([M0.jnt_dofadr[M0.actuator_trnid[i, 0]] for i in range(M0.nu)])


def randomized(rng, which):
    m = mujoco.MjModel.from_xml_string(epath.Path(xml).read_text(), assets=assets)
    if "friction" in which:
        m.geom_friction[FLOOR, 0] = rng.uniform(0.5, 1.0)
    if "joint_fl" in which:
        m.dof_frictionloss[ACT_DOFS] *= rng.uniform(0.9, 1.1, M0.nu)
    if "armature" in which:
        m.dof_armature[ACT_DOFS] *= rng.uniform(1.0, 1.05, M0.nu)
    if "com_jitter" in which:
        m.body_ipos[1] += rng.uniform(-0.05, 0.05, 3)
    if "link_mass" in which:
        m.body_mass[:] *= rng.uniform(0.9, 1.1, m.nbody)
    if "torso_mass" in which:
        m.body_mass[1] += rng.uniform(-0.1, 0.1)
    if "kp" in which:
        f = rng.uniform(0.9, 1.1, m.nu)
        kp = m.actuator_gainprm[:, 0].copy()
        m.actuator_gainprm[:, 0] = kp * f
        m.actuator_biasprm[:, 1] = -kp * f
    return m


def episode(rng, which, push):
    m = randomized(rng, which)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    yaw = rng.uniform(-3.14, 3.14)
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qpos[:2] += rng.uniform(-0.05, 0.05, 2)
    imu = m.site("imu").id
    next_push = rng.uniform(*pc.interval_range)
    for step in range(1000):
        d.ctrl[:] = default_ctrl
        for _ in range(10):
            mujoco.mj_step(m, d)
        if push and d.time >= next_push:
            ang = rng.uniform(0, 2 * np.pi); mag = rng.uniform(*pc.magnitude_range)
            d.qvel[0] += mag * np.cos(ang); d.qvel[1] += mag * np.sin(ang)
            next_push = d.time + rng.uniform(*pc.interval_range)
        if d.site_xmat[imu].reshape(3, 3)[2, 2] < 0.0 or not np.isfinite(d.qpos).all():
            return step + 1
    return 1000


CONFIGS = [
    ("none, no push", [], False),
    ("push only", [], True),
    ("friction", ["friction"], True),
    ("joint frictionloss", ["joint_fl"], True),
    ("armature", ["armature"], True),
    ("COM jitter (wrapper)", ["com_jitter"], True),
    ("link mass x0.9-1.1", ["link_mass"], True),
    ("torso mass +-0.1", ["torso_mass"], True),
    ("kp x0.9-1.1", ["kp"], True),
    ("ALL, no push", ["friction", "joint_fl", "armature", "com_jitter", "link_mass", "torso_mass", "kp"], False),
    ("ALL + push (=training)", ["friction", "joint_fl", "armature", "com_jitter", "link_mass", "torso_mass", "kp"], True),
]
print(f"== {which_pkg}: zero action, {N_EP} episodes/config, max 1000 steps (20 s) ==", flush=True)
for name, which, push in CONFIGS:
    rng = np.random.default_rng(0)
    L = np.array([episode(rng, which, push) for _ in range(N_EP)])
    print(f"{name:26s} mean len {L.mean():6.0f}  median {np.median(L):5.0f}  survived20s {np.mean(L>=1000)*100:3.0f}%  min {L.min():4d}", flush=True)
