"""Shared safety logic for anything that turns servo torque on: the
zero-offset sanity check and the "start the goal at wherever the servo
already is" trick that lets torque turn on without a sudden jump. Used by
both hold_zero.py (standalone CLI tool) and bridge_server.py's Command-mode
torque handling, so this isn't implemented twice.

Also has the MJCF joint-limit loader and the per-frame clamp+slew-rate
limiter used while torque is on. See docs/DIGITAL_TWIN.md.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import FAULT_ESTOP, FAULT_TIMEOUT, FAULT_ZERO_OFFSET  # noqa: E402

ZERO_OFFSET_LIMIT_DEG = 15.0


def check_near_zero(positions_deg, limit_deg=ZERO_OFFSET_LIMIT_DEG):
    """{id: deg} of positions that exceed the zero-offset safety margin.
    Empty dict means every joint is close enough to 0 to proceed."""
    return {i: p for i, p in positions_deg.items() if abs(p) > limit_deg}


TORQUE_ENABLE_STAGGER_S = 0.1


def enable_torque_safely(servo_io, ids, limit_deg=ZERO_OFFSET_LIMIT_DEG, stagger_s=TORQUE_ENABLE_STAGGER_S):
    """Reads current position, refuses if any joint is too far from zero,
    otherwise sets the goal to the current position (so torque doesn't
    yank the servo toward some stale registered goal) and enables torque
    ONE SERVO AT A TIME, `stagger_s` apart -- not all 13 simultaneously.

    Real hardware repeatedly showed a USB+I2C comms glitch within one
    control tick of torque turning on (see docs/HARDWARE_SETUP.md's
    "해결된 문제" entry). Raised from 0.01s (0.13s total) to 0.1s (1.3s
    total) on 2026-11-11 as a guess at current-surge mitigation; the
    real cause turned out (2026-11-13) to be a disconnected ground wire
    on the servo power barrel jack, unrelated to stagger timing. Left at
    0.1s anyway since spreading enable current over time is still
    reasonable and the wider gap is harmless.

    Returns {id: current_deg} on success. Raises RuntimeError with the
    offending {id: deg} on failure -- nothing is enabled in that case.
    """
    import time

    positions = dict(zip(ids, servo_io.get_present_position(ids)))
    bad = check_near_zero(positions, limit_deg)
    if bad:
        raise RuntimeError(f"zero_offset_exceeded: {bad}")
    servo_io.set_goal_position(positions)
    for id_ in ids:
        servo_io.enable_torque([id_])
        time.sleep(stagger_s)
    return positions


def load_joint_limits(mjcf_path):
    """{joint_name: (lo, hi)} in radians, straight from the MJCF's own
    <joint range="lo hi"/> attributes -- no mujoco import needed (the Pi's
    venv doesn't have it), and no separate copy of the numbers to drift
    out of sync with the actual model."""
    root = ET.parse(mjcf_path).getroot()
    limits = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        rng = joint.get("range")
        if name is not None and rng is not None:
            lo, hi = (float(v) for v in rng.split())
            limits[name] = (lo, hi)
    return limits


def load_home_ctrl(scene_mjcf_path):
    """List of 13 floats (radians), in <actuator> declaration order, from
    the scene MJCF's <key name="home" ctrl="..."/> -- the home-pose target
    each actuator's position controller holds at rest. Plain ElementTree,
    no mujoco import needed. Note: this is scene_flat_terrain.xml (which
    <include>s dcubot.xml and adds the keyframe), not dcubot.xml itself --
    load_joint_limits() reads the latter, this reads the former."""
    root = ET.parse(scene_mjcf_path).getroot()
    for key in root.iter("key"):
        if key.get("name") == "home":
            return [float(v) for v in key.get("ctrl").split()]
    raise ValueError(f"no <key name=\"home\"> in {scene_mjcf_path}")


class TorqueGate:
    """The torque on/off/e-stop/fault state machine, shared by
    bridge_server.py's Command mode and hardware/pi/policy_runner.py so
    this isn't reimplemented a second time -- it's subtler than it looks
    (see git history: a stale repeated torque_request was once treated as
    a fresh "retry" every tick, defeating the watchdog; e-stop could be
    bypassed by also requesting torque in the same packet; and e-stop
    release needs to arm the edge-detector since torque_request commonly
    never actually drops to False while e-stop is asserted).

    `sensors` needs .enable_torque() -> (ok, info, start_positions_rad)
    and .disable_torque() (hardware/pi/sensors.py's MockSensors/
    RealSensors both provide these).
    """

    def __init__(self):
        self.enabled = False
        self.fault = None
        self._prev_request = False

    def update(self, request, estop_now, watchdog_timed_out, sensors):
        """Call once per control loop tick. Returns the positions (rad) to
        seed a slew-rate baseline from, on the frame torque actually turns
        on; None otherwise."""
        request = bool(request)

        if estop_now:
            # Absolute override: can't be bypassed by also requesting
            # torque in the same packet. Also forces the edge-detector's
            # baseline to False, so the first request==True sample seen
            # once estop releases counts as fresh (torque_request commonly
            # stays True continuously through an e-stop -- it's a separate
            # flag -- so without this there'd never be a false->true edge
            # to clear the fault on).
            self.fault = FAULT_ESTOP
            self._prev_request = False
            if self.enabled:
                sensors.disable_torque()
                self.enabled = False
            return None

        if watchdog_timed_out:
            self.fault = FAULT_TIMEOUT

        # Edge, not level: a stale repeated request (e.g. nothing new
        # arriving during a timeout) must not keep re-clearing the fault.
        fresh_request = request and not self._prev_request
        self._prev_request = request

        if self.fault is not None:
            if fresh_request:
                self.fault = None  # explicit retry
            elif self.enabled:
                sensors.disable_torque()
                self.enabled = False
                return None
            else:
                return None

        if request and not self.enabled:
            ok, info, start_rad = sensors.enable_torque()
            if ok:
                self.enabled = True
                return start_rad
            self.fault = FAULT_ZERO_OFFSET
            return None
        if not request and self.enabled:
            sensors.disable_torque()
            self.enabled = False
        return None


def clamp_and_slew(target, prev_applied, limits, max_rate_deg_per_s, dt):
    """target/prev_applied: {name: radians}. limits: {name: (lo, hi)}
    radians. Returns the value to actually write this frame: clamped to
    the joint's own range, then slew-rate-limited from whatever was
    actually applied last frame (not from `target`, so a step command
    ramps in smoothly instead of snapping)."""
    import math

    max_step = math.radians(max_rate_deg_per_s) * dt
    out = {}
    for name, val in target.items():
        lo, hi = limits.get(name, (-1e9, 1e9))
        val = max(lo, min(hi, val))
        prev = prev_applied.get(name, val)
        step = max(-max_step, min(max_step, val - prev))
        out[name] = prev + step
    return out
