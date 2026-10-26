"""Replay a DCUBOT standing-task (playground/dcubot/standing.py) policy.

Separate from mujoco_infer.py's MjInfer: standing.py's observation (79-dim)
is NOT the same as the joystick/walking one (94-dim, has motor_targets and
a gait-phase clock that standing has no use for) and standing.py does not
add the +1.3 accelerometer[0] bias that joystick.py does. Loading a standing
checkpoint through the joystick MjInfer would either crash (wrong input
shape) or silently feed it an observation it was never trained on.
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse

from playground.common.onnx_infer import OnnxInfer
from playground.dcubot.mujoco_infer_base import MJInferBase


class StandingInfer(MJInferBase):
    def __init__(self, model_path: str, onnx_model_path: str):
        super().__init__(model_path)
        self.action_scale = 0.25
        self.dof_vel_scale = 0.05
        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-2.7, 2.7]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        # Matches standing.py's command: [0, 0, 0, neck_pitch, head_pitch, head_yaw].
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.max_motor_velocity = 5.24  # rad/s

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")

    def get_obs(self, data, command):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)  # no +1.3 bias -- standing.py doesn't add it
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        contacts = self.get_feet_contacts(data)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                contacts,
            ]
        )
        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        neck_pitch = 0.0
        head_pitch = 0.0
        head_yaw = 0.0
        if keycode == 265:  # arrow up
            head_pitch = self.NECK_PITCH_RANGE[1]
        if keycode == 264:  # arrow down
            head_pitch = self.NECK_PITCH_RANGE[0]
        if keycode == 263:  # arrow left
            head_yaw = self.HEAD_YAW_RANGE[1]
        if keycode == 262:  # arrow right
            head_yaw = self.HEAD_YAW_RANGE[0]
        self.commands[3] = neck_pitch
        self.commands[4] = head_pitch
        self.commands[5] = head_yaw

    def run(self):
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=True, show_right_ui=True,
            key_callback=self.key_callback,
        ) as viewer:
            print("[안내] ctrl+오른쪽 드래그로 로봇을 밀어서 버티는지 확인할 수 있습니다 "
                  "(MuJoCo 뷰어 기본 기능, 학습 때 쓴 push와는 별개입니다).")
            print("[안내] 1초마다 상태를 출력합니다 -- action이 0 근처에 멈춰있으면 "
                  "정책이 안 움직이는 것이고, 값이 계속 바뀌면 정상 동작 중입니다.")
            counter = 0
            last_print = 0.0
            qpos0 = self.get_actuator_joints_qpos(self.data.qpos).copy()
            while True:
                step_start = time.time()
                mujoco.mj_step(self.model, self.data)
                counter += 1
                if counter % self.decimation == 0:
                    obs = self.get_obs(self.data, self.commands)
                    action = self.policy.infer(obs)
                    self.last_last_last_action = self.last_last_action.copy()
                    self.last_last_action = self.last_action.copy()
                    self.last_action = action.copy()
                    motor_targets = self.default_actuator + action * self.action_scale
                    self.data.ctrl = motor_targets.copy()
                viewer.sync()
                if step_start - last_print > 1.0:
                    last_print = step_start
                    g = self.get_gravity(self.data)
                    
                    tilt_deg = np.degrees(np.arccos(np.clip(g[2], -1, 1)))
                    joint_dev = np.max(np.abs(self.get_actuator_joints_qpos(self.data.qpos) - qpos0))
                    print(f"[상태] action 평균|값|={np.abs(self.last_action).mean():.3f}  "
                          f"관절 변화(시작 대비 최대)={np.degrees(joint_dev):.1f}deg  "
                          f"기울기={tilt_deg:.1f}deg  몸통높이={self.data.qpos[2]:.3f}m", flush=True)
                dt_left = self.model.opt.timestep - (time.time() - step_start)
                if dt_left > 0:
                    time.sleep(dt_left)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="playground/dcubot/xmls/scene_flat_terrain.xml")
    args = parser.parse_args()
    StandingInfer(args.model_path, args.onnx_model_path).run()
