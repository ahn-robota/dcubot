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
    FAULT_ZERO_OFFSET,
    make_telemetry,
    pack,
    unpack,
)
from safety import clamp_and_slew, enable_torque_safely, load_joint_limits  # noqa: E402

LOOP_HZ = 50.0
MJCF_PATH = Path(__file__).resolve().parents[2] / "playground/dcubot/xmls/dcubot.xml"
DEFAULT_MAX_RATE_DEG_PER_S = 60.0


class MockSensors:
    """Synthetic IMU/servo/foot-switch data -- no hardware libraries, so
    this runs on any machine. Values are plausible-looking, not physical.

    While "torque" is off, joints idle on a sine wave (just something that
    visibly moves in Mirror mode). While "on", they instead lag-track
    whatever write_targets() last received, approximating a real servo's
    response -- so Command mode's ghost-vs-commanded comparison has
    something meaningful to show even with no hardware at all.
    """

    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.t0 = time.time()
        self.torque_on = False
        self.pos = dict.fromkeys(joint_names, 0.0)
        self.commanded = dict(self.pos)

    def read(self):
        t = time.time() - self.t0
        if self.torque_on:
            for name in self.joint_names:
                target = self.commanded.get(name, self.pos[name])
                self.pos[name] += (target - self.pos[name]) * 0.15
            joint_pos = dict(self.pos)
            joint_vel = dict.fromkeys(self.joint_names, 0.0)
        else:
            joint_pos = {name: 0.15 * math.sin(t + i) for i, name in enumerate(self.joint_names)}
            joint_vel = {name: 0.15 * math.cos(t + i) for i, name in enumerate(self.joint_names)}
            self.pos = dict(joint_pos)  # so enable_torque() starts smoothly from "current"
        quat = [1.0, 0.0, 0.0, 0.0]
        gyro = [0.0, 0.0, 0.0]
        accel = [0.0, 0.0, 9.8]
        foot_contact = {"left": int(t) % 2 == 0, "right": int(t) % 2 == 1}
        servo_read_ms = 0.1
        return quat, gyro, accel, joint_pos, joint_vel, foot_contact, servo_read_ms

    def calibration_status(self):
        return None

    def enable_torque(self):
        self.torque_on = True
        self.commanded = dict(self.pos)
        return True, "mock", dict(self.pos)

    def disable_torque(self):
        self.torque_on = False

    def write_targets(self, targets_rad):
        self.commanded.update(targets_rad)


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

    def __init__(self, joint_names, config, dry_run=False):
        import numpy as np
        import board
        import busio
        import adafruit_bno055
        from pypot.feetech import FeetechSTS3215IO

        self.np = np
        self.joint_names = joint_names
        self.dry_run = dry_run
        ids_by_name = servo_ids(config)
        self.ids_by_name = ids_by_name
        self.ids = [ids_by_name[n] for n in joint_names]
        self.inverted = inverted_joints(config)
        if dry_run:
            print("bridge: --dry-run -- torque/servo-write calls are no-ops below")

        i2c = busio.I2C(board.SCL, board.SDA)
        i2c_addr = int(config["imu"]["i2c_address"], 16)
        self.imu = adafruit_bno055.BNO055_I2C(i2c, address=i2c_addr)
        # IMUPLUS: gyro+accel fusion only, no magnetometer. The policy
        # never uses absolute yaw (only gyro/accel/gravity), and NDOF's
        # magnetometer was picking up interference from the servos/
        # battery/laptop and jumping yaw by a large offset while
        # stationary -- see docs/HARDWARE_SETUP.md.
        fusion_mode = config["imu"].get("fusion_mode", "IMUPLUS")
        self.imu.mode = getattr(adafruit_bno055, f"{fusion_mode}_MODE")
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

    def calibration_status(self):
        return self.imu.calibration_status  # (sys, gyro, accel, mag), each 0-3

    def _deg_to_rad_by_name(self, positions_deg):
        """{id: deg} (servo_io's own order/keys) -> {name: rad} (sign
        un-inverted back to sim convention)."""
        by_id = {}
        for name, id_ in self.ids_by_name.items():
            if id_ in positions_deg:
                sign = -1.0 if name in self.inverted else 1.0
                by_id[name] = math.radians(sign * positions_deg[id_])
        return by_id

    def enable_torque(self):
        """(ok, info, start_positions_rad). See hardware/pi/safety.py for
        the zero-offset check and the "start the goal at the current
        position" trick. --dry-run skips the real hardware call but still
        exercises the rest of the Command-mode pipeline against real
        reads."""
        if self.dry_run:
            positions_deg = dict(zip(self.ids, self.servo_io.get_present_position(self.ids)))
            return True, "dry-run (not actually enabled)", self._deg_to_rad_by_name(positions_deg)
        try:
            positions_deg = enable_torque_safely(self.servo_io, self.ids)
        except RuntimeError as e:
            return False, str(e), {}
        return True, "ok", self._deg_to_rad_by_name(positions_deg)

    def disable_torque(self):
        if self.dry_run:
            return
        self.servo_io.disable_torque(self.ids)

    def write_targets(self, targets_rad):
        if self.dry_run:
            return
        goal_deg = {}
        for name, rad in targets_rad.items():
            id_ = self.ids_by_name.get(name)
            if id_ is None:
                continue
            sign = -1.0 if name in self.inverted else 1.0
            goal_deg[id_] = sign * math.degrees(rad)
        self.servo_io.set_goal_position(goal_deg)


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
            self.sensors = RealSensors(self.joint_names, self.config, dry_run=args.dry_run)

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
        self.fault = None
        self.torque_enabled = False
        self._last_applied = {}  # {name: rad} -- slew-rate baseline while torque is on
        self._prev_torque_request = False  # edge-detect, not level-detect (see _update_torque)

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
        """Only sets self.fault -- _update_torque() is what actually acts
        on it (disabling torque, latching). Keeping the two separate is
        what makes a fault "sticky": once set, nothing here clears it."""
        with self._lock:
            last_recv_t = self._last_cmd_recv_t
            last_cmd = self._last_cmd
        now = time.time()
        if last_cmd is not None and last_cmd.get("estop"):
            self.fault = FAULT_ESTOP
        elif last_recv_t is not None and now - last_recv_t > FAULT_TIMEOUT_S:
            self.fault = FAULT_TIMEOUT

    def _update_torque(self, last_cmd, dt):
        """The whole torque state machine in one place: a latched fault
        forces torque off and stays off until the viewer explicitly
        requests torque again (no auto re-arm on reconnect -- see
        docs/DIGITAL_TWIN.md step 4); otherwise torque_request's rising/
        falling edge drives enable/disable, and while on, the commanded
        joint_target is clamped to the MJCF's own joint range and
        slew-rate-limited from whatever was actually applied last frame
        before being written to the servos."""
        request = bool(last_cmd.get("torque_request")) if last_cmd else False
        estop_now = bool(last_cmd.get("estop")) if last_cmd else False
        # Edge, not level: during a comms timeout, self._last_cmd is a
        # STALE packet that never gets replaced (nothing new is arriving),
        # so its torque_request can sit at True indefinitely. Treating
        # that as a fresh "re-request" cleared the fault every single
        # loop tick, right back to enabled, the instant the timeout set
        # it -- the watchdog never had a visible effect. A real re-request
        # is a False->True transition.
        fresh_request = request and not self._prev_torque_request
        self._prev_torque_request = request

        if estop_now:
            # Absolute override: can't be bypassed by also requesting
            # torque in the same packet (self.fault is already set to
            # FAULT_ESTOP by _check_watchdog for as long as this holds).
            if self.torque_enabled:
                self.sensors.disable_torque()
                self.torque_enabled = False
            return

        if self.fault is not None:
            if fresh_request:
                # The user pressed the retry/torque button again -- try
                # again below instead of just sitting stuck forever.
                self.fault = None
            elif self.torque_enabled:
                self.sensors.disable_torque()
                self.torque_enabled = False
                return
            else:
                return

        if request and not self.torque_enabled:
            ok, info, start_rad = self.sensors.enable_torque()
            if ok:
                self.torque_enabled = True
                self._last_applied = start_rad
                print(f"bridge: torque ON ({info})")
            else:
                self.fault = FAULT_ZERO_OFFSET
                print(f"bridge: torque-on REJECTED ({info})")
                return
        elif not request and self.torque_enabled:
            self.sensors.disable_torque()
            self.torque_enabled = False
            print("bridge: torque OFF (requested)")

        if self.torque_enabled and last_cmd:
            target = clamp_and_slew(
                last_cmd.get("joint_target") or {},
                self._last_applied,
                self.joint_limits,
                self.args.max_rate_deg,
                dt,
            )
            self._last_applied = target
            self.sensors.write_targets(target)

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

            self._update_torque(last_cmd, dt if dt > 0 else period)

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
                cal = self.sensors.calibration_status()
                cal_str = f" cal(sys/gyro/accel/mag)={cal}" if cal is not None else ""
                print(
                    f"loop_hz={loop_hz:.1f} servo_read_ms={servo_read_ms:.2f} fault={self.fault} | "
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
    return parser


def main():
    args = build_arg_parser().parse_args()
    Bridge(args).run()


if __name__ == "__main__":
    main()
