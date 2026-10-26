"""Interactive MuJoCo demo: stand -> auto-push -> (policy flails while
falling) -> reset to standing -> repeat, forever. For recording a loop that
visually shows the push-recovery behavior without having to manually
Ctrl+drag the robot over and over.

Usage:
    .venv/bin/python tools/demo_push_reset_loop.py -o <ckpt.onnx> [--push_mag 0.5]
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from playground.dcubot.mujoco_infer_standing import StandingInfer


class PushResetLoop(StandingInfer):
    def __init__(self, model_path, onnx_model_path, push_mag, settle_s,
                 reset_tilt_deg, max_wait_s, push_dir_deg, viz_amplify=1.0):
        super().__init__(model_path, onnx_model_path)
        self.push_mag = push_mag
        self.settle_s = settle_s
        self.reset_tilt_deg = reset_tilt_deg
        self.max_wait_s = max_wait_s
        self.push_dir_deg = push_dir_deg  # None = random each cycle
        # Cosmetic ONLY: measured real action magnitude during a fall is
        # small (~0.15-0.25 mean, ~5-8deg joint deviation at this training
        # stage) -- this multiplies the commanded excursion for a more
        # visible demo clip. It is NOT the policy's real trained output
        # scale; don't present amplified footage as literal behavior.
        self.viz_amplify = viz_amplify
        self.rng = np.random.default_rng()

    def do_reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.last_action[:] = 0
        self.last_last_action[:] = 0
        self.last_last_last_action[:] = 0
        self.qpos0 = self.get_actuator_joints_qpos(self.data.qpos).copy()

    def apply_push(self):
        theta = (np.radians(self.push_dir_deg) if self.push_dir_deg is not None
                 else self.rng.uniform(0, 2 * np.pi))
        self.data.qvel[0] += self.push_mag * np.cos(theta)
        self.data.qvel[1] += self.push_mag * np.sin(theta)
        print(f"[push] magnitude={self.push_mag} m/s, dir={np.degrees(theta):.0f}deg", flush=True)

    def get_tilt_deg(self):
        g = self.get_gravity(self.data)
        return np.degrees(np.arccos(np.clip(g[2], -1.0, 1.0)))

    def run(self):
        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=True, show_right_ui=True,
        ) as viewer:
            print(f"[안내] {self.settle_s}초 서있다가 자동으로 밀고, 쓰러지면(또는 "
                  f"{self.max_wait_s}초 지나면) 자동 리셋합니다. 반복됩니다.", flush=True)
            self.do_reset()
            counter = 0
            phase = "settling"  # settling -> pushed -> (fallen or timed out) -> reset
            phase_start = time.time()
            last_status = 0.0
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
                    self.data.ctrl = self.default_actuator + action * self.action_scale * self.viz_amplify
                viewer.sync()

                now = time.time()
                elapsed = now - phase_start
                tilt = self.get_tilt_deg()

                if phase == "settling" and elapsed > self.settle_s:
                    self.apply_push()
                    phase = "pushed"
                    phase_start = now
                elif phase == "pushed":
                    if tilt > self.reset_tilt_deg:
                        print(f"[쓰러짐] tilt={tilt:.0f}deg, {elapsed:.1f}초 후 -> 리셋", flush=True)
                        self.do_reset()
                        phase = "settling"
                        phase_start = now
                    elif elapsed > self.max_wait_s:
                        print(f"[버팀] {self.max_wait_s}초 동안 안 쓰러짐 -> 그냥 리셋하고 다음 라운드", flush=True)
                        self.do_reset()
                        phase = "settling"
                        phase_start = now

                if now - last_status > 1.0:
                    last_status = now
                    joint_dev_deg = np.degrees(np.max(np.abs(
                        self.get_actuator_joints_qpos(self.data.qpos) - self.qpos0)))
                    print(f"[상태] phase={phase} tilt={tilt:.0f}deg height={self.data.qpos[2]:.3f}m "
                          f"action평균|값|={np.abs(self.last_action).mean():.3f} "
                          f"관절변화(리셋대비 최대)={joint_dev_deg:.1f}deg", flush=True)

                dt_left = self.model.opt.timestep - (time.time() - step_start)
                if dt_left > 0:
                    time.sleep(dt_left)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="playground/dcubot/xmls/scene_flat_terrain.xml")
    parser.add_argument("--push_mag", type=float, default=0.5, help="push velocity kick, m/s")
    parser.add_argument("--push_dir_deg", type=float, default=None,
                         help="fixed push direction in degrees (0=+X); omit for a random direction each cycle")
    parser.add_argument("--settle_s", type=float, default=2.0, help="seconds standing before each push")
    parser.add_argument("--reset_tilt_deg", type=float, default=90.0,
                         help="tilt (deg) at which it's considered fallen and gets reset (90 = actual training fall/termination threshold)")
    parser.add_argument("--max_wait_s", type=float, default=5.0,
                         help="reset anyway after this long post-push even if it never falls")
    parser.add_argument("--viz_amplify", type=float, default=1.0,
                         help="COSMETIC multiplier on commanded joint excursion for a more visible "
                         "demo clip -- NOT the policy's real trained magnitude, do not caption "
                         "amplified footage as literal behavior")
    args = parser.parse_args()
    PushResetLoop(args.model_path, args.onnx_model_path, args.push_mag, args.settle_s,
                  args.reset_tilt_deg, args.max_wait_s, args.push_dir_deg, args.viz_amplify).run()
