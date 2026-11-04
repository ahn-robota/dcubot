"""Shared loader for dcubot_hw_config.json -- the single source of truth for
servo IDs, direction inversion, and the IMU axis transform. hardware/pi and
hardware/pc scripts import this instead of hardcoding the same values.

See docs/HARDWARE_SETUP.md for where these values come from.
"""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "dcubot_hw_config.json"


def load():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _joints(config):
    """(name, servo_dict) pairs, skipping the "_comment" metadata key."""
    return [(n, s) for n, s in config["servos"].items() if not n.startswith("_")]


def servo_ids(config=None):
    """{joint_name: id}, in the order declared in the config file."""
    config = config or load()
    return {name: s["id"] for name, s in _joints(config)}


def inverted_joints(config=None):
    """Set of joint names whose sign must be flipped on the real servo."""
    config = config or load()
    return {name for name, s in _joints(config) if s["invert"]}
