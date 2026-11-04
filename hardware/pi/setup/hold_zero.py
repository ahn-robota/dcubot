import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw_config import servo_ids  # noqa: E402
from safety import enable_torque_safely  # noqa: E402

from pypot.feetech import FeetechSTS3215IO

io = FeetechSTS3215IO("/dev/ttyACM0")
ids = list(servo_ids().values())
print("start:", {i: round(p, 1) for i, p in zip(ids, io.get_present_position(ids))})
try:
    pos = enable_torque_safely(io, ids)
except RuntimeError as e:
    print("ABORT:", e)
    raise SystemExit
for k in range(1, 41):
    a = 1 - k/40
    io.set_goal_position({i: pos[i]*a for i in ids})
    time.sleep(0.05)
time.sleep(3)
print("held:", dict(zip(ids, [round(p,1) for p in io.get_present_position(ids)])))
io.disable_torque(ids)
print("torque off")
