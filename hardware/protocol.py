"""Wire protocol shared by hardware/pi/bridge_server.py and
hardware/pc/twin_viewer.py -- the one place packet fields are defined, so
both ends agree without hand-duplicating field names. UDP, one JSON
document per packet (well under the Wi-Fi MTU at 13 joints).

See docs/DIGITAL_TWIN.md for the full design and rationale.
"""
import json
import time

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
    hold_ms=None,
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
        # Time the bridge held the echoed command before this reply went
        # out (receive-thread latency + wait for the next 50Hz tick) --
        # the viewer subtracts this so RTT reflects network time only, not
        # how long the Pi happened to sit on the packet. See
        # docs/DIGITAL_TWIN.md.
        "hold_ms": hold_ms,
    }
    if servo_temp_c is not None:
        msg["servo_temp_c"] = dict(servo_temp_c)
    if servo_voltage_v is not None:
        msg["servo_voltage_v"] = dict(servo_voltage_v)
    return msg


def make_command(seq, t_pc, torque_request, estop, joint_target, velocity_command=None):
    msg = {
        "seq": seq,
        "t_pc": t_pc,
        "torque_request": bool(torque_request),
        "estop": bool(estop),
        "joint_target": dict(joint_target),
    }
    if velocity_command is not None:
        # [vx, vy, wz], m/s and rad/s -- hardware/pi/policy_runner.py's
        # autonomous walker only (bridge_server.py's Command mode ignores
        # it). See docs/SIM_TO_REAL.md.
        msg["velocity_command"] = list(velocity_command)
    return msg


class SeqLossTracker:
    """Time-windowed loss-rate estimate from a stream of packet sequence
    numbers, independent of which direction it's used for.

    loss_rate() returns None (not enough data yet) until `warmup_s` has
    passed since the first observed packet -- otherwise the first couple
    of packets alone (e.g. seq 1 of 2 seen) read as a large, meaningless
    loss spike right at startup.
    """

    def __init__(self, window_s=5.0, warmup_s=1.0):
        self.window_s = window_s
        self.warmup_s = warmup_s
        self._samples = []  # [(t, seq), ...], newest last
        self._first_t = None

    def observe(self, seq):
        if seq is None:
            return
        now = time.time()
        if self._first_t is None:
            self._first_t = now
        self._samples.append((now, seq))
        lo = now - self.window_s
        self._samples = [(t, s) for t, s in self._samples if t >= lo]

    def loss_rate(self):
        if self._first_t is None or time.time() - self._first_t < self.warmup_s:
            return None
        if len(self._samples) < 2:
            return None
        seqs = [s for _, s in self._samples]
        span = max(seqs) - min(seqs) + 1
        if span <= 0:
            return None
        return 1.0 - len(set(seqs)) / span
