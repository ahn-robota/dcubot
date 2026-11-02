"""Generate playground/dcubot/xmls/dcubot.xml from robot_description/dcubot.urdf.

Mirrors the OLD dcubot.xml's structure exactly (same body nesting order,
default classes, sensors, actuators) -- only geometry/mass/inertia/joint
values come from the URDF. See docs/CHANGES_FROM_ORIGINAL.md for the
final-URDF remodel this supports.

Usage: uv run python tools/gen_dcubot_mjcf.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "robot_description" / "dcubot.urdf"
OUT = ROOT / "playground" / "dcubot" / "xmls" / "dcubot.xml"
OUT_TRAIN = ROOT / "playground" / "dcubot" / "xmls" / "dcubot_train.xml"
ASSETS_DIR = ROOT / "playground" / "dcubot" / "xmls" / "assets"

tree = ET.parse(URDF)
urdf_root = tree.getroot()

links = {l.get("name"): l for l in urdf_root.findall("link")}
joints = {j.get("name"): j for j in urdf_root.findall("joint")}


def link_inertial(name):
    inertial = links[name].find("inertial")
    origin = inertial.find("origin")
    mass = float(inertial.find("mass").get("value"))
    pos = origin.get("xyz") if origin is not None else "0 0 0"
    i = inertial.find("inertia")
    ixx, iyy, izz = float(i.get("ixx")), float(i.get("iyy")), float(i.get("izz"))
    ixy, ixz, iyz = float(i.get("ixy")), float(i.get("ixz")), float(i.get("iyz"))
    full = f"{ixx:.6g} {iyy:.6g} {izz:.6g} {ixy:.6g} {iyz:.6g} {ixz:.6g}"
    return pos, mass, full


def link_visual_pos(name):
    visual = links[name].find("visual")
    origin = visual.find("origin") if visual is not None else None
    return origin.get("xyz") if origin is not None else "0 0 0"


def joint_origin(name):
    return joints[name].find("origin").get("xyz")


def joint_axis(name):
    return joints[name].find("axis").get("xyz")


def joint_range(name):
    lim = joints[name].find("limit")
    return f"{lim.get('lower')} {lim.get('upper')}"


# Foot sole-center offset (ankle-joint-origin -> sole), identical for both
# feet in their own local frame (confirmed via trimesh on the new foot
# meshes -- see docs/CHANGES_FROM_ORIGINAL.md). Used for the foot sites,
# which must be symmetric (sign-different only in Y) per the work order,
# unlike the old model's placeholder `pos="0 0 0"` site.
FOOT_SITE_OFFSET = "0.01457818 0.01826794 -0.038825"

# base.z (standing height) and the IMU site's local z are both derived from
# the model's own zero-pose kinematics (foot sole must land at floor z=0;
# IMU site must sit 240mm above the floor) -- computed in a first pass
# below and substituted into a second, final pass.
BASE_Z_PLACEHOLDER = 0.21
IMU_SITE_POS_PLACEHOLDER = "0.07 0 0"

bl_pos, bl_mass, bl_inertia = link_inertial("base_link")

mesh_names = [
    "base_link", "hip_yaw_R_1", "hip_yaw_L_1", "hip_roll_R_1", "hip_roll_L_1",
    "thigh_R_1", "thigh_L_1", "shin_R_1", "shin_L_1", "foot_R_1", "foot_L_1",
    "neck_1", "head_mount_1", "head_1",
]
asset_lines = "\n".join(
    f'    <mesh name="{m}" file="{m}.stl" scale="0.001 0.001 0.001"/>' for m in mesh_names
)


def leg_body(side):
    hy, hr, hp = f"{side}_hip_yaw", f"{side}_hip_roll", f"{side}_hip_pitch"
    kn, an = f"{side}_knee", f"{side}_ankle"
    L = "L" if side == "left" else "R"
    hip_yaw_link, hip_roll_link = f"hip_yaw_{L}_1", f"hip_roll_{L}_1"
    thigh_link, shin_link, foot_link = f"thigh_{L}_1", f"shin_{L}_1", f"foot_{L}_1"

    hy_i_pos, hy_mass, hy_in = link_inertial(hip_yaw_link)
    hr_i_pos, hr_mass, hr_in = link_inertial(hip_roll_link)
    th_i_pos, th_mass, th_in = link_inertial(thigh_link)
    sh_i_pos, sh_mass, sh_in = link_inertial(shin_link)
    ft_i_pos, ft_mass, ft_in = link_inertial(foot_link)

    foot_site_name = f"{side}_foot"
    foot_collision_name = f"{side}_foot_collision"

    return f"""        <body name="{hip_yaw_link}" pos="{joint_origin(hy)}">
          <inertial pos="{hy_i_pos}" mass="{hy_mass:.8g}" fullinertia="{hy_in}"/>
          <joint name="{hy}" type="hinge" axis="{joint_axis(hy)}" range="{joint_range(hy)}" class="sts3215"/>
          <geom class="sts3215" pos="{link_visual_pos(hip_yaw_link)}" type="mesh" rgba="0.7 0.7 0.7 1" mesh="{hip_yaw_link}"/>
          <body name="{hip_roll_link}" pos="{joint_origin(hr)}">
            <inertial pos="{hr_i_pos}" mass="{hr_mass:.8g}" fullinertia="{hr_in}"/>
            <joint name="{hr}" type="hinge" axis="{joint_axis(hr)}" range="{joint_range(hr)}" class="sts3215"/>
            <geom class="sts3215" pos="{link_visual_pos(hip_roll_link)}" type="mesh" rgba="0.7 0.7 0.7 1" mesh="{hip_roll_link}"/>
            <body name="{thigh_link}" pos="{joint_origin(hp)}">
              <inertial pos="{th_i_pos}" mass="{th_mass:.8g}" fullinertia="{th_in}"/>
              <joint name="{hp}" type="hinge" axis="{joint_axis(hp)}" range="{joint_range(hp)}" class="sts3215"/>
              <geom class="sts3215" pos="{link_visual_pos(thigh_link)}" type="mesh" rgba="0.7 0.7 0.7 1" mesh="{thigh_link}"/>
              <body name="{shin_link}" pos="{joint_origin(kn)}">
                <inertial pos="{sh_i_pos}" mass="{sh_mass:.8g}" fullinertia="{sh_in}"/>
                <joint name="{kn}" type="hinge" axis="{joint_axis(kn)}" range="{joint_range(kn)}" class="sts3215"/>
                <geom class="sts3215" pos="{link_visual_pos(shin_link)}" type="mesh" rgba="0.7 0.7 0.7 1" mesh="{shin_link}"/>
                <body name="{foot_link}" pos="{joint_origin(an)}">
                  <inertial pos="{ft_i_pos}" mass="{ft_mass:.8g}" fullinertia="{ft_in}"/>
                  <joint name="{an}" type="hinge" axis="{joint_axis(an)}" range="{joint_range(an)}" class="sts3215"/>
                  <geom name="{foot_collision_name}" pos="{link_visual_pos(foot_link)}" type="mesh" rgba="0.7 0.7 0.7 1" mesh="{foot_link}"/>
                  <site name="{foot_site_name}" pos="{FOOT_SITE_OFFSET}"/>
                </body>
              </body>
            </body>
          </body>
        </body>"""


neck_i_pos, neck_mass, neck_in = link_inertial("neck_1")
hm_i_pos, hm_mass, hm_in = link_inertial("head_mount_1")
head_i_pos, head_mass, head_in = link_inertial("head_1")

head_body = f"""        <body name="neck_1" pos="{joint_origin('neck_pitch')}">
          <inertial pos="{neck_i_pos}" mass="{neck_mass:.8g}" fullinertia="{neck_in}"/>
          <joint name="neck_pitch" type="hinge" axis="{joint_axis('neck_pitch')}" range="{joint_range('neck_pitch')}" class="sts3215"/>
          <geom class="sts3215" pos="{link_visual_pos('neck_1')}" type="mesh" rgba="0.7 0.7 0.7 1" mesh="neck_1"/>
          <body name="head_mount_1" pos="{joint_origin('head_pitch')}">
            <inertial pos="{hm_i_pos}" mass="{hm_mass:.8g}" fullinertia="{hm_in}"/>
            <joint name="head_pitch" type="hinge" axis="{joint_axis('head_pitch')}" range="{joint_range('head_pitch')}" class="sts3215"/>
            <geom class="sts3215" pos="{link_visual_pos('head_mount_1')}" type="mesh" rgba="0.7 0.7 0.7 1" mesh="head_mount_1"/>
            <body name="head_1" pos="{joint_origin('head_yaw')}">
              <inertial pos="{head_i_pos}" mass="{head_mass:.8g}" fullinertia="{head_in}"/>
              <joint name="head_yaw" type="hinge" axis="{joint_axis('head_yaw')}" range="{joint_range('head_yaw')}" class="sts3215"/>
              <geom class="sts3215" pos="{link_visual_pos('head_1')}" type="mesh" rgba="0.7 0.7 0.7 1" mesh="head_1"/>
            </body>
          </body>
        </body>"""

def build_mjcf(base_z, imu_site_pos, meshdir="assets"):
    return f"""<?xml version="1.0"?>
<!-- Generated by tools/gen_dcubot_mjcf.py from robot_description/dcubot.urdf
     (final Fusion 360 remodel, 2.072kg, 2026-11-02). See
     docs/CHANGES_FROM_ORIGINAL.md "Final URDF remodel" for what changed. -->
<mujoco model="dcubot">

  <option iterations="1" ls_iterations="5">
    <flag eulerdamp="disable"/>
  </option>
  <compiler angle="radian" meshdir="{meshdir}"/>

  <default>
    <default class="dcubot">
      <joint frictionloss="0.1" armature="0.005"/>
      <position kp="50" dampratio="1"/>
    </default>
  </default>

  <sensor>
    <gyro site="imu" name="gyro"/>
    <velocimeter site="imu" name="local_linvel"/>
    <accelerometer site="imu" name="accelerometer"/>
    <framezaxis objtype="site" objname="imu" name="upvector"/>
    <framexaxis objtype="site" objname="imu" name="forwardvector"/>
    <framelinvel objtype="site" objname="imu" name="global_linvel"/>
    <frameangvel objtype="site" objname="imu" name="global_angvel"/>
    <framepos objtype="site" objname="imu" name="position"/>
    <framequat objtype="site" objname="imu" name="orientation"/>
    <framelinvel objtype="site" objname="right_foot" name="right_foot_global_linvel"/>
    <framelinvel objtype="site" objname="left_foot" name="left_foot_global_linvel"/>
    <framexaxis objtype="site" objname="left_foot" name="left_foot_upvector"/>
    <framexaxis objtype="site" objname="right_foot" name="right_foot_upvector"/>
    <framepos objtype="site" objname="left_foot" name="left_foot_pos"/>
    <framepos objtype="site" objname="right_foot" name="right_foot_pos"/>
  </sensor>

  <default>
    <default class="sts3215">
      <geom contype="0" conaffinity="0"/>
      <joint damping="0.56" frictionloss="0.068" armature="0.027"/>
      <!-- kp / forcerange: Rhoban BAM identification of the same servo
           (feetech_sts3215_7_4V), as used by upstream Open Duck Mini v2;
           the datasheet stall torque (19.5 kg*cm = 1.91 N*m) is lower than
           what the identified motor model produces. -->
      <position kp="17.8" kv="0.0" forcerange="-3.35 3.35"/>
    </default>
    <default class="backlash">
      <joint damping="0.01" frictionloss="0" armature="0.01" limited="true"
        range="-0.008726646259971648 0.008726646259971648"/>
    </default>
  </default>

  <asset>
{asset_lines}
  </asset>

  <worldbody>
    <body name="base" pos="0 0 {base_z:.6f}">
      <freejoint name="dcubot_freejoint"/>
      <site name="imu" pos="{imu_site_pos}"/>

      <body name="base_link" childclass="dcubot">
        <!-- root inertial kept explicitly (MuJoCo's URDF import drops it for the world-welded root) -->
        <inertial pos="{bl_pos}" mass="{bl_mass:.8g}" fullinertia="{bl_inertia}"/>
        <geom class="sts3215" type="mesh" rgba="0.7 0.7 0.7 1" pos="{link_visual_pos('base_link')}" mesh="base_link"/>

{leg_body('left')}
{leg_body('right')}
{head_body}
      </body>
    </body>
  </worldbody>

  <actuator>
    <position class="sts3215" name="left_hip_yaw" joint="left_hip_yaw" inheritrange="1"/>
    <position class="sts3215" name="left_hip_roll" joint="left_hip_roll" inheritrange="1"/>
    <position class="sts3215" name="left_hip_pitch" joint="left_hip_pitch" inheritrange="1"/>
    <position class="sts3215" name="left_knee" joint="left_knee" inheritrange="1"/>
    <position class="sts3215" name="left_ankle" joint="left_ankle" inheritrange="1"/>
    <position class="sts3215" name="right_hip_yaw" joint="right_hip_yaw" inheritrange="1"/>
    <position class="sts3215" name="right_hip_roll" joint="right_hip_roll" inheritrange="1"/>
    <position class="sts3215" name="right_hip_pitch" joint="right_hip_pitch" inheritrange="1"/>
    <position class="sts3215" name="right_knee" joint="right_knee" inheritrange="1"/>
    <position class="sts3215" name="right_ankle" joint="right_ankle" inheritrange="1"/>
    <position class="sts3215" name="neck_pitch" joint="neck_pitch" inheritrange="1"/>
    <position class="sts3215" name="head_pitch" joint="head_pitch" inheritrange="1"/>
    <position class="sts3215" name="head_yaw" joint="head_yaw" inheritrange="1"/>
  </actuator>
  <equality/>
</mujoco>
"""


# --- Pass 1: write with placeholder base.z / IMU pos, then measure ---
OUT.write_text(build_mjcf(BASE_Z_PLACEHOLDER, IMU_SITE_POS_PLACEHOLDER))

import mujoco  # noqa: E402

m = mujoco.MjModel.from_xml_path(str(OUT))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
left_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
base_z_at_placeholder = d.xpos[base_bid][2]
foot_offset_from_base = d.site_xpos[left_site][2] - base_z_at_placeholder

final_base_z = -foot_offset_from_base  # puts the foot sole at floor z=0
final_imu_z = 0.240 - final_base_z
final_imu_pos = f"0.07 0 {final_imu_z:.6f}"

# --- Pass 2: write with the final, kinematically-derived values ---
OUT.write_text(build_mjcf(final_base_z, final_imu_pos, meshdir="assets"))
OUT_TRAIN.write_text(build_mjcf(final_base_z, final_imu_pos, meshdir="assets_lowpoly"))
print(f"Wrote {OUT}")
print(f"Wrote {OUT_TRAIN}")
print(f"base.z = {final_base_z:.6f}  imu site pos = {final_imu_pos}")
