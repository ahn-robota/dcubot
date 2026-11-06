"""Hardware bridge: streams IMU/servo/foot-switch telemetry to the PC viewer
at a fixed rate over UDP, and tracks the latest joint targets/torque/e-stop
request it has received back.

Runs on the Raspberry Pi normally. --mock generates synthetic data instead
of touching real hardware, so the protocol and viewer can be developed and
tested without the robot (no pypot/BNO055/GPIO imports happen in that mode).

See docs/DIGITAL_TWIN.md for the full design.
"""
import argparse
import socket
import sys
import time
from pathlib import Path
from threading import Lock, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw_config import load as load_hw_config, servo_ids  # noqa: E402
from protocol import (  # noqa: E402
    DEFAULT_PORT,
    FAULT_COMMS_ERROR,
    FAULT_TIMEOUT_S,
    make_telemetry,
    pack,
    unpack,
)
from safety import TorqueGate, clamp_and_slew, load_joint_limits  # noqa: E402

# A transient servo-bus hiccup (DxlTimeoutError etc.) shouldn't crash the
# whole bridge and leave torque on with nothing updating it. One or two in
# a row are tolerated (that tick is just skipped); this many in a row
# force torque off. See hardware/pi/policy_runner.py's identical guard.
COMMS_ERROR_LIMIT = 3
from sensors import (  # noqa: E402
    MockSensors,
    RealSensors,
    gravity_to_roll_pitch_deg,
    quat_to_roll_pitch_deg,
)

LOOP_HZ = 50.0
MJCF_PATH = Path(__file__).resolve().parents[2] / "playground/dcubot/xmls/dcubot.xml"
DEFAULT_MAX_RATE_DEG_PER_S = 60.0


class Bridge:
    def __init__(self, args):
        self.args = args
        self.config = load_hw_config()
        self.joint_names = list(servo_ids(self.config).keys())

        if args.mock:
            self.sensors = MockSensors(self.joint_names)
        else:
            self.sensors = RealSensors(self.joint_names, self.config, dry_run=args.dry_run, device=args.device)

        self.joint_limits = load_joint_limits(MJCF_PATH)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", args.port))
        self.sock.settimeout(0.01)

        self._lock = Lock()
        self._last_cmd = None
        self._last_cmd_recv_t = None
        self._last_cmd_seq = None  # most recently *received* command's seq
        self._max_cmd_seq_recv = None  # highest seq ever received (reorder-safe)
        self._client_addr = None

        self._seq = 0
        self.gate = TorqueGate()
        self._consecutive_errors = 0
        self._last_error_print_t = 0.0
        self._last_applied = {}  # {name: rad} -- slew-rate baseline while torque is on

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
                seq = cmd.get("seq")
                self._last_cmd_seq = seq
                if seq is not None and (self._max_cmd_seq_recv is None or seq > self._max_cmd_seq_recv):
                    self._max_cmd_seq_recv = seq

    def _watchdog_timed_out(self):
        with self._lock:
            last_recv_t = self._last_cmd_recv_t
        return last_recv_t is not None and time.time() - last_recv_t > FAULT_TIMEOUT_S

    def _update_torque(self, last_cmd, dt):
        """Drives hardware/pi/safety.py's shared TorqueGate, then (only
        while it's enabled) clamps the commanded joint_target to the
        MJCF's own joint range and slew-rate-limits it from whatever was
        actually applied last frame before writing it to the servos."""
        request = bool(last_cmd.get("torque_request")) if last_cmd else False
        estop_now = bool(last_cmd.get("estop")) if last_cmd else False

        start_rad = self.gate.update(request, estop_now, self._watchdog_timed_out(), self.sensors)
        if start_rad is not None:
            self._last_applied = start_rad
            print("bridge: torque ON")

        if self.gate.enabled and last_cmd:
            target = clamp_and_slew(
                last_cmd.get("joint_target") or {},
                self._last_applied,
                self.joint_limits,
                self.args.max_rate_deg,
                dt,
            )
            self._last_applied = target
            self.sensors.write_targets(target)

    def _on_comms_error(self, e):
        """See hardware/pi/policy_runner.py's identical method: a
        sustained failure (port/USB actually down) means every further
        read/write keeps failing too, so this prints at most once/second
        and attempts disable_torque exactly once (== not >=), not forever
        on a dead port."""
        self._consecutive_errors += 1
        now = time.time()
        if now - self._last_error_print_t > 1.0:
            print(f"bridge: servo comms error ({e}) -- {self._consecutive_errors} in a row")
            self._last_error_print_t = now
        if self._consecutive_errors == COMMS_ERROR_LIMIT:
            print("bridge: too many comms errors in a row -- forcing torque OFF")
            try:
                self.sensors.disable_torque()
            except Exception as e2:
                print(f"bridge: disable_torque also failed ({e2}) -- bus looks down, not retrying every tick")
            self.gate.enabled = False
            self.gate.fault = FAULT_COMMS_ERROR

    def run(self):
        Thread(target=self._recv_loop, daemon=True).start()
        mode = "mock" if self.args.mock else "real"
        if self.args.dry_run:
            mode += ", dry-run"
        print(f"bridge: listening on UDP :{self.args.port} ({mode})")

        period = 1.0 / LOOP_HZ
        hz_window = []
        last_loop_t = time.time()

        while True:
            loop_start = time.time()
            dt = loop_start - last_loop_t
            hz_window.append(dt)
            hz_window = hz_window[-50:]
            last_loop_t = loop_start
            mean_dt = sum(hz_window) / len(hz_window) if hz_window else 0.0
            loop_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0  # dt==0 is possible on the very first tick

            read_start = time.time()
            try:
                quat, gyro, accel, joint_pos, joint_vel, foot_contact, servo_read_ms = self.sensors.read()
            except Exception as e:
                self._on_comms_error(e)
                time.sleep(max(0.0, period - (time.time() - loop_start)))
                continue
            if servo_read_ms is None:
                servo_read_ms = (time.time() - read_start) * 1000.0

            with self._lock:
                last_cmd = self._last_cmd
                last_cmd_seq = self._last_cmd_seq
                last_cmd_recv_t = self._last_cmd_recv_t
                max_cmd_seq_recv = self._max_cmd_seq_recv
                client_addr = self._client_addr

            try:
                self._update_torque(last_cmd, dt if dt > 0 else period)
            except Exception as e:
                self._on_comms_error(e)
                time.sleep(max(0.0, period - (time.time() - loop_start)))
                continue
            self._consecutive_errors = 0

            # How long we've been sitting on the command we're about to
            # echo -- the viewer subtracts this from its RTT measurement.
            hold_ms = (time.time() - last_cmd_recv_t) * 1000.0 if last_cmd_recv_t is not None else None

            self._seq += 1
            msg = make_telemetry(
                seq=self._seq,
                t_pi=time.time(),
                loop_hz=round(loop_hz, 1),
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

            if self._seq % 50 == 0:
                roll_q, pitch_q = quat_to_roll_pitch_deg(quat)
                roll_g, pitch_g = gravity_to_roll_pitch_deg(accel)
                cal = self.sensors.calibration_status()
                cal_str = f" cal(sys/gyro/accel/mag)={cal}" if cal is not None else ""
                print(
                    f"loop_hz={loop_hz:.1f} servo_read_ms={servo_read_ms:.2f} fault={self.gate.fault} | "
                    f"roll/pitch from quat=({roll_q:+.1f},{pitch_q:+.1f}) "
                    f"from gravity=({roll_g:+.1f},{pitch_g:+.1f}) deg "
                    f"gravity_base={[round(a, 2) for a in accel]}{cal_str}"
                )

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mock", action="store_true", help="synthetic data, no hardware libraries needed")
    parser.add_argument("--dry-run", action="store_true", help="read real sensors but never write to servos")
    parser.add_argument(
        "--max-rate-deg", type=float, default=DEFAULT_MAX_RATE_DEG_PER_S,
        help="Command-mode joint-target slew-rate limit, deg/s",
    )
    parser.add_argument(
        "--device", default="/dev/ttyACM0",
        help="servo bus serial device -- if the USB adapter disconnects/reconnects it can "
        "come back as ttyACM1 etc.; check `dmesg | tail` and pass the new path here",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    Bridge(args).run()


if __name__ == "__main__":
    main()
