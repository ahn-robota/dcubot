"""Wire protocol shared by hardware/pi/bridge_server.py and
hardware/pc/twin_viewer.py -- the one place packet fields are defined, so
both ends agree without hand-duplicating field names. UDP, one JSON
document per packet (well under the Wi-Fi MTU at 13 joints).

See docs/DIGITAL_TWIN.md for the full design and rationale.
"""
import json

DEFAULT_PORT = 8420
COMMAND_HZ = 20.0  # the viewer sends at this rate even with nothing new to say
FAULT_TIMEOUT_S = 0.3

FAULT_TIMEOUT = "timeout"
FAULT_ESTOP = "estop"
FAULT_ZERO_OFFSET = "zero_offset_exceeded"


def pack(msg: dict) -> bytes:
    return json.dumps(msg).encode("utf-8")


def unpack(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def make_telemetry(
    seq,
    t_pi,
    loop_hz,
    servo_read_ms,
    imu_quat_wxyz,
    imu_gyro,
    imu_accel,
    joint_pos,
    joint_vel,
    foot_contact,
    torque_enabled,
    fault,
    echo_cmd_seq,
    echo_cmd_t_pc,
    last_cmd_seq_recv,
    servo_temp_c=None,
    servo_voltage_v=None,
):
    msg = {
        "seq": seq,
        "t_pi": t_pi,
        "loop_hz": loop_hz,
        "servo_read_ms": servo_read_ms,
        "imu_quat_wxyz": list(imu_quat_wxyz),
        "imu_gyro": list(imu_gyro),
        "imu_accel": list(imu_accel),
        "joint_pos": dict(joint_pos),
        "joint_vel": dict(joint_vel),
        "foot_contact": dict(foot_contact),
        "torque_enabled": bool(torque_enabled),
        "fault": fault,
        "echo_cmd_seq": echo_cmd_seq,
        "echo_cmd_t_pc": echo_cmd_t_pc,
        "last_cmd_seq_recv": last_cmd_seq_recv,
    }
    if servo_temp_c is not None:
        msg["servo_temp_c"] = dict(servo_temp_c)
    if servo_voltage_v is not None:
        msg["servo_voltage_v"] = dict(servo_voltage_v)
    return msg


def make_command(seq, t_pc, torque_request, estop, joint_target):
    return {
        "seq": seq,
        "t_pc": t_pc,
        "torque_request": bool(torque_request),
        "estop": bool(estop),
        "joint_target": dict(joint_target),
    }


class SeqLossTracker:
    """Sliding-window loss-rate estimate from a stream of packet sequence
    numbers, independent of which direction it's used for."""

    def __init__(self, window=100):
        self.window = window
        self._seen = set()
        self._max_seq = None

    def observe(self, seq):
        if seq is None:
            return
        if self._max_seq is None or seq > self._max_seq:
            self._max_seq = seq
        self._seen.add(seq)
        lo = self._max_seq - self.window
        self._seen = {s for s in self._seen if s > lo}

    def loss_rate(self):
        if self._max_seq is None or self._max_seq < 1:
            return 0.0
        lo = max(0, self._max_seq - self.window + 1)
        expected = self._max_seq - lo + 1
        seen_in_window = sum(1 for s in self._seen if s >= lo)
        return 1.0 - seen_in_window / expected
