"""Step-0 policies output random actions. Time-to-fall (no push, no randomization)
for i.i.d. Gaussian actions a~N(0,sigma), ctrl = home + 0.25*tanh(a), dcubot vs duck."""
import sys, numpy as np, mujoco
from etils import epath
pkg = sys.argv[1]
if pkg == "dcubot":
    from playground.dcubot import base, joystick
    xml, assets = "playground/dcubot/xmls/scene_flat_terrain_train.xml", base.get_assets(lowpoly=True)
else:
    from playground.open_duck_mini_v2 import base, joystick
    xml, assets = "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml", base.get_assets()
m = mujoco.MjModel.from_xml_string(epath.Path(xml).read_text(), assets=assets)
home = m.keyframe("home").ctrl.copy()
imu = m.site("imu").id
print(f"== {pkg}: random actions, no push, 12 eps/sigma ==", flush=True)
for sigma in (0.1, 0.3, 0.7):
    rng = np.random.default_rng(1); L = []
    for _ in range(12):
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
        for s in range(1000):
            a = np.tanh(rng.normal(0, sigma, m.nu))
            d.ctrl[:] = home + 0.25 * a
            for _ in range(10): mujoco.mj_step(m, d)
            if d.site_xmat[imu].reshape(3, 3)[2, 2] < 0.0: break
        L.append(s + 1)
    L = np.array(L)
    print(f"sigma={sigma}: mean len {L.mean():5.0f} median {np.median(L):5.0f} survived20s {np.mean(L>=1000)*100:3.0f}%", flush=True)
