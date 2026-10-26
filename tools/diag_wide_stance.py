"""Does a wider foot spacing (hip_roll splay, zero action) help dcubot survive the training pushes?
No ankle-roll axis exists, so splaying tilts the soles by the roll angle; we report that too.
"""
import sys
import numpy as np, mujoco
sys.argv = ["x", "dcubot", "12"]
exec(open("tools/diag_zero_action_stability.py").read().split("CONFIGS = [")[0])

names = [M0.actuator(i).name for i in range(M0.nu)]
def pose_ctrl(a, s):
    c = np.zeros(M0.nu)
    c[names.index("left_hip_roll")] = s * a
    c[names.index("right_hip_roll")] = s * a   # right roll axis is mirrored (-X)
    return c

def foot_min_z_and_sep(m, d):
    zs, ys = [], []
    for g in range(m.ngeom):
        n = m.geom(g).name
        if n in ("left_foot_collision", "right_foot_collision"):
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            zs.append(w[:, 2].min()); ys.append(w[:, 1].mean())
    return min(zs), abs(ys[0] - ys[1])

def setup(m, ctrl):
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    for i in range(m.nu): d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] = ctrl[i]
    mujoco.mj_forward(m, d)
    zmin, sep = foot_min_z_and_sep(m, d)
    d.qpos[2] -= zmin - 0.001
    mujoco.mj_forward(m, d)
    return d, sep

def sign_that_widens():
    best = {}
    for s in (+1, -1):
        d, sep = setup(M0, pose_ctrl(0.15, s)); best[s] = sep
    print("foot sep at a=0.15:", best)
    return max(best, key=best.get)
SIGN = sign_that_widens()

def episode(rng, which, push, a):
    ctrl = pose_ctrl(a, SIGN)
    m = randomized(rng, which); d, _ = setup(m, ctrl)
    yaw = rng.uniform(-3.14, 3.14); d.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    imu = m.site("imu").id; nxt = rng.uniform(*pc.interval_range)
    for step in range(1000):
        d.ctrl[:] = ctrl
        for _ in range(10): mujoco.mj_step(m, d)
        if push and d.time >= nxt:
            ang = rng.uniform(0, 2*np.pi); mg = rng.uniform(*pc.magnitude_range)
            d.qvel[0] += mg*np.cos(ang); d.qvel[1] += mg*np.sin(ang); nxt = d.time + rng.uniform(*pc.interval_range)
        if d.site_xmat[imu].reshape(3, 3)[2, 2] < 0.0 or not np.isfinite(d.qpos).all(): return step + 1
    return 1000

ALL = ["friction", "joint_fl", "armature", "com_jitter", "link_mass", "torso_mass", "kp"]
print("== hip_roll splay a | ALL randomization + push, zero action ==", flush=True)
for a in (0.0, 0.05, 0.10, 0.15, 0.20, 0.26):
    _, sep = setup(M0, pose_ctrl(a, SIGN))
    for label, push in (("no push", False), ("push", True)):
        rng = np.random.default_rng(0)
        L = np.array([episode(rng, ALL, push, a) for _ in range(12)])
        print(f"a={a:.2f} ({np.degrees(a):4.1f} deg) foot sep {sep*100:4.1f} cm | {label:7s} mean {L.mean():5.0f} median {np.median(L):5.0f} survived20s {np.mean(L>=1000)*100:3.0f}% min {L.min()}", flush=True)
