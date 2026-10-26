import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
import glob
import os
from playground.common.onnx_infer import OnnxInfer

from playground.dcubot.mujoco_infer_base import MJInferBase
from playground.dcubot import debug_vis

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(self, model_path: str, onnx_model_path: str, standing: bool):
        super().__init__(model_path)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.policy = OnnxInfer(onnx_model_path, awd=True)
        self.watch_dir = None
        self.debug_vis = True
        self._imu_id = self.model.site("imu").id
        self._last_print = 0.0
        self._loaded_onnx = onnx_model_path

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        # DCUBOT has no head_roll: [lin_x, lin_y, ang, neck_pitch, head_pitch, head_yaw]
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0
        # Must match joystick.py: gait clock period and its step counter
        # (one increment per policy step, obs at step n uses gait_step=n).
        self.gait_period = 0.4
        self.gait_step = 0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def _gait_phase(self, command):
        dt = self.sim_dt * self.decimation
        phase = (self.gait_step * dt / self.gait_period) % 1.0
        if np.linalg.norm(np.asarray(command[:3])) <= 0.02:
            return np.zeros(2)
        return np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

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
                self.motor_targets,
                contacts,
                self._gait_phase(command),
            ]
        )
        self.gait_step += 1

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        if keycode == 72:  # h
            self.head_control_mode = not self.head_control_mode
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if not self.head_control_mode:
            if keycode == 265:  # arrow up
                lin_vel_x = self.COMMANDS_RANGE_X[1]
            if keycode == 264:  # arrow down
                lin_vel_x = self.COMMANDS_RANGE_X[0]
            if keycode == 263:  # arrow left
                lin_vel_y = self.COMMANDS_RANGE_Y[1]
            if keycode == 262:  # arrow right
                lin_vel_y = self.COMMANDS_RANGE_Y[0]
            if keycode == 81:  # a
                ang_vel = self.COMMANDS_RANGE_THETA[1]
            if keycode == 69:  # e
                ang_vel = self.COMMANDS_RANGE_THETA[0]
            if keycode == 80:  # p
                self.phase_frequency_factor += 0.1
            if keycode == 59:  # m
                self.phase_frequency_factor -= 0.1
        else:
            neck_pitch = 0
            head_pitch = 0
            head_yaw = 0
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

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def _latest_onnx(self):
        files = glob.glob(os.path.join(self.watch_dir, "*.onnx"))
        return max(files, key=os.path.getmtime) if files else None

    def _maybe_reload_policy(self):
        """--watch_dir: when training writes a newer checkpoint, swap the policy
        and restart the robot from the home pose so you see the new behavior."""
        latest = self._latest_onnx()
        if latest is None or latest == self._loaded_onnx:
            return
        try:
            policy = OnnxInfer(latest, awd=True)
        except Exception as e:  # file may still be mid-write; retry next poll
            print(f"(waiting for {os.path.basename(latest)}: {e})")
            return
        self.policy = policy
        self._loaded_onnx = latest
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.gait_step = 0
        print(f"[reloaded] {os.path.basename(latest)}")

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=True,
                key_callback=self.key_callback,
            ) as viewer:
                # Show joint axes/actuator so joint articulation is visible
                # even though the servo/motor meshes are not modeled.
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                self.model.vis.scale.jointlength = 0.05
                self.model.vis.scale.jointwidth = 0.015
                if self.debug_vis:
                    print("[방향 표시] 몸통축: X빨강=전진 / Y초록=왼쪽 / Z파랑=위, 파란 화살표=명령속도, 하늘색=실제속도")
                counter = 0
                last_poll = 0.0
                while True:

                    step_start = time.time()
                    if self.watch_dir and step_start - last_poll > 2.0:
                        last_poll = step_start
                        self._maybe_reload_policy()

                    mujoco.mj_step(self.model, self.data)

                    counter += 1

                    if counter % self.decimation == 0:
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )

                            self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                    if self.debug_vis and counter % self.decimation == 0:
                        with viewer.lock():
                            viewer.user_scn.ngeom = 0
                            debug_vis.draw(
                                viewer.user_scn, self.data, self._imu_id,
                                self.commands, self.get_gyro(self.data)[2],
                            )
                        if step_start - self._last_print > 1.0:
                            self._last_print = step_start
                            lv = debug_vis.local_velocity(self.data, self._imu_id)
                            print(
                                f"cmd(전진X,좌Y)=({self.commands[0]:+.2f},{self.commands[1]:+.2f})  "
                                f"실제 몸통좌표 속도=({lv[0]:+.2f},{lv[1]:+.2f})"
                            )
                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, default=None,
        help="policy to start with; optional when --watch_dir is given (waits for the first .onnx)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/dcubot/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument("--no_debug_vis", action="store_true", default=False)
    parser.add_argument(
        "--watch_dir",
        type=str,
        default=None,
        help="keep the viewer open and hot-swap to the newest .onnx written here",
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=0.0,
        help="starting forward command, m/s (trained range: 0.05-0.10). "
        "Arrow keys still work, but they jump straight to +-0.15, not a fine adjustment.",
    )

    args = parser.parse_args()

    onnx_path = args.onnx_model_path
    if onnx_path is None:
        if not args.watch_dir:
            parser.error("give -o <policy.onnx> or --watch_dir <dir>")
        while True:  # training may not have written its first checkpoint yet
            files = glob.glob(os.path.join(args.watch_dir, "*.onnx"))
            if files:
                onnx_path = max(files, key=os.path.getmtime)
                break
            print(f"waiting for the first .onnx in {args.watch_dir} ...")
            time.sleep(10)
    mjinfer = MjInfer(args.model_path, onnx_path, args.standing)
    mjinfer.watch_dir = args.watch_dir
    mjinfer.debug_vis = not args.no_debug_vis
    mjinfer.commands[0] = args.vx
    mjinfer.run()
