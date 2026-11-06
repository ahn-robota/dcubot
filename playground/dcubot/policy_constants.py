"""Inference-time constants shared by playground/dcubot/mujoco_infer.py (sim)
and hardware/pi/policy_runner.py (real robot). Pure Python/no third-party
deps -- importable on the Raspberry Pi, which has neither jax nor mujoco.

These values MIRROR playground/dcubot/joystick.py's default_config() --
joystick.py can't be imported here or on the Pi (it pulls in jax and
mujoco_playground at module level), so this is a second, deliberately
narrow source of truth. If default_config() changes one of these, update
this file to match. See docs/SIM_TO_REAL.md.
"""

ACTION_SCALE = 0.25
DOF_VEL_SCALE = 0.05
GAIT_PERIOD = 0.4  # s
MAX_MOTOR_VELOCITY = 5.24  # rad/s
CONTROL_DT = 0.02  # s (joystick.py's ctrl_dt == mujoco_infer.py's sim_dt * decimation)

LIN_VEL_X_RANGE = (-0.20, 0.30)
LIN_VEL_Y_RANGE = (-0.158, 0.158)
ANG_VEL_YAW_RANGE = (-0.3, 0.3)

# playground/dcubot/xmls/dcubot.xml's <actuator> declaration order -- the
# ONNX policy's observation/action vectors are flat arrays in exactly this
# order, not keyed by name, so this list is what turns our name-keyed
# sensor dicts into the array the policy actually expects.
ACTUATOR_ORDER = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
]
