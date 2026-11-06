"""IMU/servo/foot-switch reading, split out of bridge_server.py so
hardware/pi/policy_runner.py (the autonomous sim2real runner) can use the
exact same sensor layer instead of a second copy. See docs/DIGITAL_TWIN.md
and docs/SIM_TO_REAL.md.

MockSensors needs no third-party packages at all. RealSensors needs
pypot/adafruit-circuitpython-bno055/gpiozero, imported lazily inside
__init__ so importing this module (or running --mock) never requires them.
"""
import math
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw_config import inverted_joints, servo_ids  # noqa: E402
from safety import enable_torque_safely  # noqa: E402


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

    def __init__(self, joint_names, config, dry_run=False, device="/dev/ttyACM0"):
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
            print("sensors: --dry-run -- torque/servo-write calls are no-ops below")

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

        self.servo_io = FeetechSTS3215IO(device)

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
                f"sensors: foot switches unavailable ({e}) -- reporting not-pressed. "
                "Wiring is still unconfirmed, see docs/HARDWARE_SETUP.md."
            )

    def _read_imu(self):
        np = self.np
        quat = self.imu.quaternion  # (w, x, y, z) per Adafruit's docs -- verify on hardware
        gyro = self.imu.gyro  # rad/s, chip frame
        accel = self.imu.acceleration  # m/s^2 (includes gravity), chip frame

        if quat[0] is None:
            if not self._warned_no_fusion:
                print("sensors: IMU quaternion is None (not in a fusion mode, or still "
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
        exercises the rest of the pipeline against real reads."""
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
    playground stack, and this is the only quaternion math needed here."""
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


def quat_to_roll_pitch_deg(q):
    """Standard aerospace (ZYX) quaternion -> roll/pitch, in degrees.
    Diagnostic only -- just so it's directly comparable to the
    gravity-vector estimate below; this is not any particular renderer's
    yaw-zero decomposition. Plain math (no numpy), so --mock keeps working
    with no third-party dependency at all."""
    w, x, y, z = q
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    return roll, pitch


def gravity_to_roll_pitch_deg(accel_base):
    """roll/pitch implied directly by the measured gravity-reaction vector
    in base frame (mount-transformed only, no quaternion/fusion involved).
    Should closely match quat_to_roll_pitch_deg(quat) while stationary --
    both describe the same physical tilt from two independent paths. Real
    divergence is expected only under actual acceleration (walking), not
    while held still, so a mismatch while still means the quaternion-side
    transform (mount math or field order) is still wrong."""
    ax, ay, az = accel_base
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
    return roll, pitch
