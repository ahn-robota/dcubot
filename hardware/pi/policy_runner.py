"""Autonomous sim2real policy runner: runs a trained ONNX policy on the
real robot, closed-loop, locally on the Pi at the exact training control
rate (50Hz). See docs/SIM_TO_REAL.md.

Unlike hardware/pi/bridge_server.py (human teleop/visualization), this is
NOT remote-controlled frame-by-frame: an optional laptop only sends
low-rate velocity commands + e-stop, and watches read-only telemetry in
the same wire format bridge_server.py uses, so hardware/pc/twin_viewer.py's
Mirror mode can watch the walk with no changes.

--mock: synthetic sensors (hardware/pi/sensors.py's MockSensors), so the
observation construction and control-loop timing can be verified with no
robot and no real servo writes. --dry-run: real sensors, but torque/servo
writes are no-ops (same meaning as in bridge_server.py).

Observation/action construction mirrors playground/dcubot/mujoco_infer.py's
MjInfer exactly -- see docs/SIM_TO_REAL.md section 3/4 for the field-by-
field mapping. Deliberately does NOT reuse hardware/pi/safety.py's
clamp_and_slew: that's for Command mode's human-set joint targets, whereas
here the speed-limit clip is training's own motor_targets logic (accuracy,
not independently-added safety).
"""
import argparse
import math
import socket
import sys
import time
from pathlib import Path
from threading import Lock, Thread

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw_config import load as load_hw_config  # noqa: E402
from protocol import DEFAULT_PORT, FAULT_COMMS_ERROR, make_telemetry, pack, unpack  # noqa: E402
from safety import TorqueGate, load_home_ctrl, load_joint_limits  # noqa: E402
from sensors import MockSensors, RealSensors  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from playground.common.onnx_infer import OnnxInfer  # noqa: E402
from playground.dcubot import policy_constants  # noqa: E402
from playground.dcubot.symmetry import SymmetricPolicy  # noqa: E402

SCENE_MJCF_PATH = REPO_ROOT / "playground/dcubot/xmls/scene_flat_terrain.xml"
DCUBOT_MJCF_PATH = REPO_ROOT / "playground/dcubot/xmls/dcubot.xml"
CONTROL_HZ = 1.0 / policy_constants.CONTROL_DT

# Looser than bridge_server.py's FAULT_TIMEOUT_S=0.3: losing the
# supervisory laptop link doesn't need to be instantly dangerous the way
# losing a teleoperation link does, since the policy itself is a standing/
# walking controller, not an open-loop remote control. Cutting torque
# while it's standing unsupported is more dangerous than just standing
# still -- see docs/SIM_TO_REAL.md section 5 (confirmed with the user).
CMD_TIMEOUT_S = 1.0

# A transient servo-bus hiccup (DxlTimeoutError etc.) shouldn't crash the
# whole control loop and leave torque on with nothing updating it -- that's
# worse than a clean shutdown. One or two in a row are tolerated (that tick
# is just skipped, servos hold their last goal); this many in a row force
# torque off.
COMMS_ERROR_LIMIT = 3

# Right when torque turns on, the policy can command all 13 joints toward
# its standing pose at once, at training's full rate (MAX_MOTOR_VELOCITY,
# ~300 deg/s) -- a real hardware test saw a USB+I2C comms glitch exactly
# at that moment (see docs/HARDWARE_SETUP.md's "알려진 문제" entry); a
# brief, large simultaneous current draw is the suspected cause (confirmed
# NOT an undervoltage event via vcgencmd, so likely a short current/noise
# transient too brief for that to catch). For this long after torque
# turns on, slew is capped at the gentler rate below instead of training's
# full rate -- steady-state control fidelity is unaffected once it elapses.
SOFT_START_DURATION_S = 0.5
SOFT_START_MAX_VELOCITY_DEG = 60.0  # matches Command mode's own default (--max-rate-deg)


class PolicyRunner:
    def __init__(self, args):
        self.args = args
        self.config = load_hw_config()
        self.joint_names = policy_constants.ACTUATOR_ORDER  # the flat-array order the ONNX expects

        if args.mock:
            self.sensors = MockSensors(self.joint_names)
        else:
            self.sensors = RealSensors(self.joint_names, self.config, dry_run=args.dry_run, device=args.device)

        self.joint_limits = load_joint_limits(DCUBOT_MJCF_PATH)
        self.default_actuator = np.array(load_home_ctrl(SCENE_MJCF_PATH))

        self.policy = None
        if not args.hold:
            policy = OnnxInfer(args.onnx, awd=True, num_threads=1)
            if args.symmetric or args.symmetric_trained:
                policy = SymmetricPolicy(policy, pre_tanh=args.symmetric_trained)
            self.policy = policy

        self._reset_history()
        self.gate = TorqueGate()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", args.port))
        self.sock.settimeout(0.01)
        self._lock = Lock()
        self._last_cmd = None
        self._last_cmd_recv_t = None
        self._last_cmd_seq = None
        self._max_cmd_seq_recv = None
        self._client_addr = None
        self._seq = 0
        self._have_network_command = False

        self.velocity_command = [args.vx, args.vy, args.wz]
        self._loop_hz = 0.0
        self._consecutive_errors = 0
        self._last_error_print_t = 0.0
        self._soft_start_until = 0.0

    def _reset_history(self):
        """Fresh start whenever torque newly turns on, so the first frame
        doesn't feed a stale pre-disable action history into the policy."""
        self._soft_start_until = time.time() + SOFT_START_DURATION_S
        self.last_action = np.zeros(13)
        self.last_last_action = np.zeros(13)
        self.last_last_last_action = np.zeros(13)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.gait_step = 0

    def _recv_loop(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                cmd = unpack(data)
            except (ValueError, UnicodeDecodeError):
                continue
            with self._lock:
                self._client_addr = addr
                self._last_cmd = cmd
                self._last_cmd_recv_t = time.time()
                self._have_network_command = True
                seq = cmd.get("seq")
                self._last_cmd_seq = seq
                if seq is not None and (self._max_cmd_seq_recv is None or seq > self._max_cmd_seq_recv):
                    self._max_cmd_seq_recv = seq

    def _gait_phase(self):
        phase = (self.gait_step * policy_constants.CONTROL_DT / policy_constants.GAIT_PERIOD) % 1.0
        return np.array([math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase)])

    def _build_obs(self, gyro, accel, joint_pos, joint_vel, foot_contact, command):
        """Exactly MjInfer.get_obs's "state" (playground/dcubot/
        mujoco_infer.py), field for field -- see docs/SIM_TO_REAL.md
        section 3. No noise (inference, not training)."""
        joint_pos_arr = np.array([joint_pos[n] for n in self.joint_names])
        joint_vel_arr = np.array([joint_vel[n] for n in self.joint_names])
        contacts = np.array([foot_contact.get("left", False), foot_contact.get("right", False)])
        obs = np.concatenate(
            [
                np.array(gyro),
                np.array(accel),
                np.array(command),
                joint_pos_arr - self.default_actuator,
                joint_vel_arr * policy_constants.DOF_VEL_SCALE,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                self._gait_phase(),
            ]
        )
        self.gait_step += 1
        return obs

    def _policy_command(self):
        vx, vy, wz = self.velocity_command
        # Head commands fixed at 0 for now -- see docs/SIM_TO_REAL.md
        # section 3 ("처음엔 머리 3개는 0 고정").
        return [vx, vy, wz, 0.0, 0.0, 0.0]

    def _on_comms_error(self, e):
        """A transient servo-bus error (DxlTimeoutError etc.) must not
        crash the control loop -- that would leave torque on with nothing
        updating it, worse than a clean shutdown. Tolerate a few in a row
        (this tick is just skipped; servos hold their last goal); force
        torque off on sustained failure.

        A sustained failure (port/USB actually down, not just one slow
        servo) means every further read/write keeps failing too -- so this
        prints at most once/second (not once per 20ms tick) and attempts
        disable_torque exactly once (== not >=), not forever on a dead
        port."""
        self._consecutive_errors += 1
        now = time.time()
        if now - self._last_error_print_t > 1.0:
            print(f"policy_runner: servo comms error ({e}) -- {self._consecutive_errors} in a row")
            self._last_error_print_t = now
        if self._consecutive_errors == COMMS_ERROR_LIMIT:
            print("policy_runner: too many comms errors in a row -- forcing torque OFF")
            try:
                self.sensors.disable_torque()
            except Exception as e2:
                print(f"policy_runner: disable_torque also failed ({e2}) -- "
                      "bus looks down, not retrying every tick")
            self.gate.enabled = False
            self.gate.fault = FAULT_COMMS_ERROR

    def _step(self):
        try:
            quat, gyro, accel, joint_pos, joint_vel, foot_contact, servo_read_ms = self.sensors.read()
        except Exception as e:
            self._on_comms_error(e)
            return
        self._consecutive_errors = 0

        with self._lock:
            last_cmd = self._last_cmd
            last_cmd_seq = self._last_cmd_seq
            last_cmd_recv_t = self._last_cmd_recv_t
            max_cmd_seq_recv = self._max_cmd_seq_recv
            client_addr = self._client_addr
            have_network_command = self._have_network_command

        request = bool(last_cmd.get("torque_request")) if last_cmd else False
        estop_now = bool(last_cmd.get("estop")) if last_cmd else False
        if last_cmd and last_cmd.get("velocity_command") is not None:
            self.velocity_command = last_cmd["velocity_command"]

        timed_out = (
            have_network_command
            and last_cmd_recv_t is not None
            and time.time() - last_cmd_recv_t > CMD_TIMEOUT_S
        )
        if timed_out:
            # Decay the command, NOT the torque -- see CMD_TIMEOUT_S above.
            self.velocity_command = [0.0, 0.0, 0.0]

        try:
            # watchdog_timed_out=False always: a comms timeout never faults
            # torque here (only e-stop does), unlike bridge_server.py's Bridge.
            start_rad = self.gate.update(request, estop_now, False, self.sensors)
            if start_rad is not None:
                self._reset_history()
                print("policy_runner: torque ON")

            if self.gate.enabled and self.args.hold:
                # --hold: isolates "torque on, write a target every tick"
                # from "...and also run ONNX inference every tick" -- see
                # docs/HARDWARE_SETUP.md's investigation. No policy, no
                # obs/action bookkeeping, just hold the home pose.
                self.sensors.write_targets(dict(zip(self.joint_names, self.default_actuator.tolist())))
            elif self.gate.enabled:
                command = self._policy_command()
                obs = self._build_obs(gyro, accel, joint_pos, joint_vel, foot_contact, command)
                action = self.policy.infer(obs)

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = np.array(action).copy()

                if time.time() < self._soft_start_until:
                    max_velocity = math.radians(SOFT_START_MAX_VELOCITY_DEG)
                else:
                    max_velocity = policy_constants.MAX_MOTOR_VELOCITY
                max_step = max_velocity * policy_constants.CONTROL_DT
                motor_targets = self.default_actuator + self.last_action * policy_constants.ACTION_SCALE
                motor_targets = np.clip(
                    motor_targets, self.prev_motor_targets - max_step, self.prev_motor_targets + max_step
                )
                self.motor_targets = motor_targets
                self.prev_motor_targets = motor_targets.copy()

                self.sensors.write_targets(dict(zip(self.joint_names, motor_targets.tolist())))
        except Exception as e:
            self._on_comms_error(e)
            return
        self._consecutive_errors = 0

        hold_ms = (time.time() - last_cmd_recv_t) * 1000.0 if last_cmd_recv_t is not None else None
        self._seq += 1
        msg = make_telemetry(
            seq=self._seq,
            t_pi=time.time(),
            loop_hz=round(self._loop_hz, 1),
            servo_read_ms=round(servo_read_ms, 2),
            imu_quat_wxyz=quat,
            imu_gyro=gyro,
            imu_accel=accel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            foot_contact=foot_contact,
            torque_enabled=self.gate.enabled,
            fault=self.gate.fault,
            echo_cmd_seq=last_cmd_seq,
            echo_cmd_t_pc=(last_cmd or {}).get("t_pc"),
            last_cmd_seq_recv=max_cmd_seq_recv,
            hold_ms=round(hold_ms, 2) if hold_ms is not None else None,
        )
        if client_addr is not None:
            self.sock.sendto(pack(msg), client_addr)

    def run(self):
        Thread(target=self._recv_loop, daemon=True).start()
        mode = "mock" if self.args.mock else "real"
        if self.args.dry_run:
            mode += ", dry-run"
        print(
            f"policy_runner: listening on UDP :{self.args.port} ({mode}), "
            f"control {CONTROL_HZ:.0f}Hz, onnx={self.args.onnx}"
        )
        print("Send torque_request=true (network) to start -- torque stays OFF until then.")

        period = policy_constants.CONTROL_DT
        hz_window = []
        last_loop_t = time.time()
        last_print = time.time()

        while True:
            loop_start = time.time()
            dt = loop_start - last_loop_t
            hz_window.append(dt)
            hz_window = hz_window[-50:]
            last_loop_t = loop_start
            mean_dt = sum(hz_window) / len(hz_window) if hz_window else 0.0
            self._loop_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0

            self._step()

            if time.time() - last_print > 1.0:
                print(
                    f"loop_hz={self._loop_hz:.1f} torque_enabled={self.gate.enabled} "
                    f"fault={self.gate.fault} velocity_command={[round(v, 3) for v in self.velocity_command]}"
                )
                last_print = time.time()

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="trained policy checkpoint (not needed with --hold)")
    parser.add_argument(
        "--hold", action="store_true",
        help="torque on, hold the home pose, no ONNX inference at all -- isolates "
        "'torque on + writing a target every tick' from 'and also running "
        "inference every tick', see docs/HARDWARE_SETUP.md",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mock", action="store_true", help="synthetic data, no hardware libraries needed")
    parser.add_argument("--dry-run", action="store_true", help="read real sensors but never write to servos")
    parser.add_argument("--symmetric", action="store_true", help="average the policy with its left/right mirror")
    parser.add_argument(
        "--symmetric_trained", action="store_true",
        help="policy was trained with runner.py --symmetric_policy: rebuild it exactly",
    )
    parser.add_argument(
        "--device", default="/dev/ttyACM0",
        help="servo bus serial device -- if the USB adapter disconnects/reconnects it can "
        "come back as ttyACM1 etc.; check `dmesg | tail` and pass the new path here",
    )
    parser.add_argument("--vx", type=float, default=0.0, help="local fallback forward command, m/s")
    parser.add_argument("--vy", type=float, default=0.0, help="local fallback strafe command, m/s")
    parser.add_argument("--wz", type=float, default=0.0, help="local fallback turn command, rad/s")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.hold and not args.onnx:
        parser.error("--onnx is required unless --hold is given")
    PolicyRunner(args).run()


if __name__ == "__main__":
    main()
