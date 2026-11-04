"""Shared safety logic for anything that turns servo torque on: the
zero-offset sanity check and the "start the goal at wherever the servo
already is" trick that lets torque turn on without a sudden jump. Used by
both hold_zero.py (standalone CLI tool) and bridge_server.py's Command-mode
torque handling, so this isn't implemented twice.

Also has the MJCF joint-limit loader and the per-frame clamp+slew-rate
limiter used while torque is on. See docs/DIGITAL_TWIN.md.
"""
import xml.etree.ElementTree as ET

ZERO_OFFSET_LIMIT_DEG = 15.0


def check_near_zero(positions_deg, limit_deg=ZERO_OFFSET_LIMIT_DEG):
    """{id: deg} of positions that exceed the zero-offset safety margin.
    Empty dict means every joint is close enough to 0 to proceed."""
    return {i: p for i, p in positions_deg.items() if abs(p) > limit_deg}


def enable_torque_safely(servo_io, ids, limit_deg=ZERO_OFFSET_LIMIT_DEG):
    """Reads current position, refuses if any joint is too far from zero,
    otherwise sets the goal to the current position (so torque doesn't
    yank the servo toward some stale registered goal) and enables torque.

    Returns {id: current_deg} on success. Raises RuntimeError with the
    offending {id: deg} on failure -- nothing is enabled in that case.
    """
    positions = dict(zip(ids, servo_io.get_present_position(ids)))
    bad = check_near_zero(positions, limit_deg)
    if bad:
        raise RuntimeError(f"zero_offset_exceeded: {bad}")
    servo_io.set_goal_position(positions)
    servo_io.enable_torque(ids)
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
