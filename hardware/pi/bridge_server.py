"""Hardware bridge: streams IMU/servo/foot-switch telemetry to the PC viewer
at a fixed rate over UDP, and tracks the latest joint targets/torque/e-stop
request it has received back.

Runs on the Raspberry Pi normally. --mock generates synthetic data instead
of touching real hardware, so the protocol and viewer can be developed and
tested without the robot (no pypot/BNO055/GPIO imports happen in that mode).

See docs/DIGITAL_TWIN.md for the full design.
"""
import argparse
import math
import socket
import sys
import time
from pathlib import Path
from threading import Lock, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw_config import load as load_hw_config, servo_ids  # noqa: E402
from protocol import (  # noqa: E402
    DEFAULT_PORT,
    FAULT_ESTOP,
    FAULT_TIMEOUT,
    FAULT_TIMEOUT_S,
    make_telemetry,
    pack,
    unpack,
)

LOOP_HZ = 50.0


class MockSensors:
    """Synthetic IMU/servo/foot-switch data -- no hardware libraries, so
    this runs on any machine. Values are plausible-looking, not physical."""

    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.t0 = time.time()

    def read(self):
        t = time.time() - self.t0
        joint_pos = {name: 0.15 * math.sin(t + i) for i, name in enumerate(self.joint_names)}
        joint_vel = {name: 0.15 * math.cos(t + i) for i, name in enumerate(self.joint_names)}
        quat = [1.0, 0.0, 0.0, 0.0]
        gyro = [0.0, 0.0, 0.0]
        accel = [0.0, 0.0, 9.8]
        foot_contact = {"left": int(t) % 2 == 0, "right": int(t) % 2 == 1}
        servo_read_ms = 0.1
        return quat, gyro, accel, joint_pos, joint_vel, foot_contact, servo_read_ms


class Bridge:
    def __init__(self, args):
        self.args = args
        self.config = load_hw_config()
        self.joint_names = list(servo_ids(self.config).keys())

        if args.mock:
            self.sensors = MockSensors(self.joint_names)
        else:
            raise NotImplementedError(
                "Real-hardware sensor reading (IMU/servos/foot switches) "
                "lands in the Mirror-mode step -- see docs/DIGITAL_TWIN.md "
                "step 2. Use --mock for now."
            )

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
        self.fault = None
        self.torque_enabled = False  # real torque-on handling lands with safety.py

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

    def _check_watchdog(self):
        with self._lock:
            last_recv_t = self._last_cmd_recv_t
            last_cmd = self._last_cmd
        now = time.time()
        if last_cmd is not None and last_cmd.get("estop"):
            self.fault = FAULT_ESTOP
        elif last_recv_t is not None and now - last_recv_t > FAULT_TIMEOUT_S:
            self.fault = FAULT_TIMEOUT
        if self.fault is not None:
            self.torque_enabled = False  # real ramp-down lands with safety.py

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
            hz_window.append(loop_start - last_loop_t)
            hz_window = hz_window[-50:]
            last_loop_t = loop_start
            loop_hz = 1.0 / (sum(hz_window) / len(hz_window)) if hz_window else 0.0

            read_start = time.time()
            quat, gyro, accel, joint_pos, joint_vel, foot_contact, servo_read_ms = self.sensors.read()
            if servo_read_ms is None:
                servo_read_ms = (time.time() - read_start) * 1000.0

            self._check_watchdog()

            with self._lock:
                last_cmd = self._last_cmd
                last_cmd_seq = self._last_cmd_seq
                last_cmd_recv_t = self._last_cmd_recv_t
                max_cmd_seq_recv = self._max_cmd_seq_recv
                client_addr = self._client_addr

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
                torque_enabled=self.torque_enabled,
                fault=self.fault,
                echo_cmd_seq=last_cmd_seq,
                echo_cmd_t_pc=(last_cmd or {}).get("t_pc"),
                last_cmd_seq_recv=max_cmd_seq_recv,
                hold_ms=round(hold_ms, 2) if hold_ms is not None else None,
            )
            if client_addr is not None:
                self.sock.sendto(pack(msg), client_addr)

            if self._seq % 50 == 0:
                print(f"loop_hz={loop_hz:.1f} servo_read_ms={servo_read_ms:.2f} fault={self.fault}")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mock", action="store_true", help="synthetic data, no hardware libraries needed")
    parser.add_argument("--dry-run", action="store_true", help="read real sensors but never write to servos")
    return parser


def main():
    args = build_arg_parser().parse_args()
    Bridge(args).run()


if __name__ == "__main__":
    main()
