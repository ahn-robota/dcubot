"""Does a bent-knee home pose make dcubot survive the training pushes (zero action)?
Feet stay flat: hip + knee + ankle pitch sum to 0 (all three share one axis).
Sign is chosen so the knee is in front of the hip-ankle line (natural bend).
"""
import sys
import numpy as np, mujoco
from etils import epath
sys.argv = ["x", "dcubot", "12"]
exec(open("tools/diag_zero_action_stability.py").read().split("CONFIGS = [")[0])

names = [M0.actuator(i).name for i in range(M0.nu)]
def pose_ctrl(k, s):
    c = np.zeros(M0.nu)
    for side in ("left", "right"):
        c[names.index(f"{side}_hip_pitch")] = s * k / 2
        c[names.index(f"{side}_knee")] = -s * k
        c[names.index(f"{side}_ankle")] = s * k / 2
    return c

def knee_is_forward(s, k=0.8):
    d = mujoco.MjData(M0); c = pose_ctrl(k, s)
    for i in range(M0.nu): d.qpos[M0.jnt_qposadr[M0.actuator_trnid[i, 0]]] = c[i]
    mujoco.mj_forward(M0, d)
    x = lambda b: d.xpos[M0.body(b).id][0]
    return x("shin_L_1") - (x("hip_roll_L_1") + x("foot_L_1")) / 2   # >0: knee ahead of hip-ankle line

for s in (+1, -1):
    print(f"sign {s:+d}: knee ahead of hip-ankle line by {knee_is_forward(s):+.3f} m")
SIGN = +1 if knee_is_forward(+1) > 0 else -1
print("using sign", SIGN, flush=True)

def episode_pose(rng, which, push, k):
    global default_ctrl
    ctrl = pose_ctrl(k, SIGN); default_ctrl = ctrl
    m = randomized(rng, which); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for i in range(m.nu): d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] = ctrl[i]
    mujoco.mj_forward(m, d)
    dz = d.site_xpos[m.site("left_foot").id][2] - 0.039     # put foot site at standing height
    d.qpos[2] -= dz
    yaw = rng.uniform(-3.14, 3.14); d.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    imu = m.site("imu").id; nxt = rng.uniform(*pc.interval_range)
    for step in range(1000):
        d.ctrl[:] = ctrl
        for _ in range(10): mujoco.mj_step(m, d)
        if push and d.time >= nxt:
            a = rng.uniform(0, 2*np.pi); mg = rng.uniform(*pc.magnitude_range)
            d.qvel[0] += mg*np.cos(a); d.qvel[1] += mg*np.sin(a); nxt = d.time + rng.uniform(*pc.interval_range)
        if d.site_xmat[imu].reshape(3, 3)[2, 2] < 0.0 or not np.isfinite(d.qpos).all(): return step + 1
    return 1000

ALL = ["friction", "joint_fl", "armature", "com_jitter", "link_mass", "torso_mass", "kp"]
print("== crouch depth k (knee flexion, rad) | ALL randomization + push, zero action ==", flush=True)
for k in (0.0, 0.3, 0.6, 0.9, 1.2):
    rng = np.random.default_rng(0)
    L = np.array([episode_pose(rng, ALL, True, k) for _ in range(12)])
    print(f"k={k:.1f} (knee {np.degrees(k):3.0f} deg)  mean len {L.mean():5.0f}  median {np.median(L):5.0f}  survived20s {np.mean(L>=1000)*100:3.0f}%  min {L.min()}", flush=True)
