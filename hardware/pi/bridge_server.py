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
from hw_config import inverted_joints, load as load_hw_config, servo_ids  # noqa: E402
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


class RealSensors:
    """Reads the real BNO055 IMU + 13 STS3215 servos + (if wired) 2 foot
    switches. See docs/HARDWARE_SETUP.md for the physical setup this
    assumes. Joint velocity is finite-differenced from consecutive position
    reads rather than read from the servo bus, since get_present_position
    is the only read method already confirmed working (hardware/pi/setup/).

    NOTE: the IMU quaternion's field order (w,x,y,z) is written per
    Adafruit's usual convention but is NOT confirmed against this exact
    library version -- the first real run should sanity-check it (hold the
    robot level; printed imu_quat_wxyz should read close to [1,0,0,0]). If
    it doesn't, swap the reorder in _read_imu().
    """

    def __init__(self, joint_names, config):
        import numpy as np
        import board
        import busio
        import adafruit_bno055
        from pypot.feetech import FeetechSTS3215IO

        self.np = np
        self.joint_names = joint_names
        ids_by_name = servo_ids(config)
        self.ids = [ids_by_name[n] for n in joint_names]
        self.inverted = inverted_joints(config)

        i2c = busio.I2C(board.SCL, board.SDA)
        i2c_addr = int(config["imu"]["i2c_address"], 16)
        self.imu = adafruit_bno055.BNO055_I2C(i2c, address=i2c_addr)
        self._warned_no_fusion = False

        r_base_chip = np.array(config["imu"]["R_base_chip"], dtype=float)
        self.r_base_chip = r_base_chip
        # R_chip_base = R_base_chip^T (a rotation matrix's inverse is its
        # transpose) as a quaternion -- see _read_imu() for how it's used.
        q_mount = _mat_to_quat(r_base_chip)
        self.q_chip_base = np.array([q_mount[0], -q_mount[1], -q_mount[2], -q_mount[3]])

        self.servo_io = FeetechSTS3215IO("/dev/ttyACM0")

        self._prev_pos = None
        self._prev_t = None

        self.foot_buttons = None
        try:
            from gpiozero import Button

            fs = config["foot_switches"]
            self.foot_buttons = {
                "left": Button(fs["left"]["gpio"], pull_up=True),
                "right": Button(fs["right"]["gpio"], pull_up=True),
            }
        except Exception as e:
            print(
                f"bridge: foot switches unavailable ({e}) -- reporting not-pressed. "
                "Wiring is still unconfirmed, see docs/HARDWARE_SETUP.md."
            )

    def _read_imu(self):
        np = self.np
        quat = self.imu.quaternion  # (w, x, y, z) per Adafruit's docs -- verify on hardware
        gyro = self.imu.gyro  # rad/s, chip frame
        accel = self.imu.acceleration  # m/s^2 (includes gravity), chip frame

        if quat[0] is None:
            if not self._warned_no_fusion:
                print("bridge: IMU quaternion is None (not in a fusion mode, or still "
                      "calibrating) -- reporting identity until it recovers.")
                self._warned_no_fusion = True
            quat = (1.0, 0.0, 0.0, 0.0)
        q_chip = np.array(quat, dtype=float)
        # R_world_base = R_world_chip @ R_chip_base (chip's own reported
        # orientation, composed with the fixed mount) -- NOT a similarity
        # transform (q_mount*q_chip*q_mount_conj), which was the bug: that
        # reports R_base_chip itself (a 180-degree flip) when the chip
        # reads back exactly what it would when the robot is level, which
        # is precisely what showed up as the robot rendering upside down.
        q_base = _quat_mul_np(q_chip, self.q_chip_base)

        gyro_chip = np.array([0.0, 0.0, 0.0] if None in gyro else gyro, dtype=float)
        accel_chip = np.array([0.0, 0.0, 9.8] if None in accel else accel, dtype=float)
        gyro_base = self.r_base_chip @ gyro_chip
        accel_base = self.r_base_chip @ accel_chip
        return q_base.tolist(), gyro_base.tolist(), accel_base.tolist()

    def read(self):
        quat, gyro, accel = self._read_imu()

        read_start = time.time()
        positions_deg = self.servo_io.get_present_position(self.ids)
        servo_read_ms = (time.time() - read_start) * 1000.0

        now = time.time()
        joint_pos, joint_vel = {}, {}
        for name, raw_deg in zip(self.joint_names, positions_deg):
            sign = -1.0 if name in self.inverted else 1.0
            rad = math.radians(sign * raw_deg)
            joint_pos[name] = rad
            prev = (self._prev_pos or {}).get(name)
            dt = now - self._prev_t if self._prev_t is not None else None
            joint_vel[name] = (rad - prev) / dt if prev is not None and dt else 0.0
        self._prev_pos = joint_pos
        self._prev_t = now

        if self.foot_buttons is not None:
            foot_contact = {side: btn.is_pressed for side, btn in self.foot_buttons.items()}
        else:
            foot_contact = {"left": False, "right": False}

        return quat, gyro, accel, joint_pos, joint_vel, foot_contact, servo_read_ms


def _quat_mul_np(a, b):
    """Hamilton product of two (w,x,y,z) quaternions. Plain numpy (no
    mujoco) -- the Pi's venv has pypot/adafruit, not the full mujoco/
    playground stack, and this is the only quaternion math the bridge
    needs."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    import numpy as np

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_to_roll_pitch_deg(q):
    """Standard aerospace (ZYX) quaternion -> roll/pitch, in degrees.
    Diagnostic only -- just so it's directly comparable to the
    gravity-vector estimate below; this is not the yaw-zero decomposition
    the viewer uses for rendering. Plain math (no numpy), so --mock keeps
    working with no third-party dependency at all."""
    w, x, y, z = q
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    return roll, pitch


def _gravity_to_roll_pitch_deg(accel_base):
    """roll/pitch implied directly by the measured gravity-reaction vector
    in base frame (mount-transformed only, no quaternion/fusion involved).
    Should closely match _quat_to_roll_pitch_deg(quat) while stationary --
    both describe the same physical tilt from two independent paths. Real
    divergence is expected only under actual acceleration (walking), not
    while held still, so a mismatch while still means the quaternion-side
    transform (mount math or field order) is still wrong."""
    ax, ay, az = accel_base
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
    return roll, pitch


def _mat_to_quat(r):
    """3x3 rotation matrix -> (w,x,y,z) quaternion (Shepperd's method)."""
    import numpy as np

    m = r
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


class Bridge:
    def __init__(self, args):
        self.args = args
        self.config = load_hw_config()
        self.joint_names = list(servo_ids(self.config).keys())

        if args.mock:
            self.sensors = MockSensors(self.joint_names)
        else:
            self.sensors = RealSensors(self.joint_names, self.config)

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
                roll_q, pitch_q = _quat_to_roll_pitch_deg(quat)
                roll_g, pitch_g = _gravity_to_roll_pitch_deg(accel)
                print(
                    f"loop_hz={loop_hz:.1f} servo_read_ms={servo_read_ms:.2f} fault={self.fault} | "
                    f"roll/pitch from quat=({roll_q:+.1f},{pitch_q:+.1f}) "
                    f"from gravity=({roll_g:+.1f},{pitch_g:+.1f}) deg "
                    f"gravity_base={[round(a, 2) for a in accel]}"
                )

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
