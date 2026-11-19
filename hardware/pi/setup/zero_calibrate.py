"""Feetech STS3215 영점(0점) 재보정 -- 서보 자신의 EEPROM 보정 레지스터를 다시 굽는다.
dcubot_hw_config.json의 zero_offset_deg는 기록용일 뿐 코드에서 읽지 않는다(소프트웨어
옵셋은 항상 0) -- 실제 0점은 서보 EEPROM에 물리적으로 저장되어 있다 (docs/HARDWARE_SETUP.md
"0점 보정" 참고).

원래 셋업 때(2026-11) 쓴 것과 같은 방법: 토크 끄고, 관절을 손으로 로봇의 URDF 0 자세로
맞춘 다음, Torque_Enable 레지스터에 128(0/1이 아닌 특수값)을 써서 "지금 위치 = 0도"로
EEPROM에 보정값을 다시 계산해 넣는다. pypot에는 이 트릭을 감싸는 별도 메서드가 없어서
`io._set_torque_enable`(비공개지만 enable_torque/disable_torque와 동일하게
_set_control_value로 레지스터를 직접 쓰는 제너릭 세터)을 그대로 쓴다 -- 2026-11-18
Pi에서 inspect.getsource로 확인.

Usage:
    python zero_calibrate.py                        # 13개 전부
    python zero_calibrate.py --joint head_pitch      # 이것만
    python zero_calibrate.py --joint head_pitch right_knee
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw_config import servo_ids  # noqa: E402

from pypot.feetech import FeetechSTS3215IO

TORQUE_ENABLE_CALIBRATE = 128  # 0/1이 아닌 이 값이 "현재 위치를 0으로" 트리거

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="/dev/ttyACM0")
parser.add_argument(
    "--joint", nargs="+", default=None,
    help="다시 구울 관절 이름 (기본: dcubot_hw_config.json의 13개 전부)",
)
args = parser.parse_args()

all_ids = servo_ids()
names = args.joint or list(all_ids)
targets = {name: all_ids[name] for name in names}
ids = list(targets.values())

io = FeetechSTS3215IO(args.port)

print(f"대상: {targets}")

print("EEPROM 잠금 해제...")
io.set_lock({i: 0 for i in ids})

print("토크 끄는 중 (손으로 움직일 수 있게)...")
io.disable_torque(ids)

print("로봇을 URDF 0 자세로 맞추세요. 실시간으로 현재 각도 보여줍니다. "
      "자세 다 잡히면 Ctrl+C.")
try:
    while True:
        positions = io.get_present_position(ids)
        line = "  ".join(f"{n}={p:+.1f}" for n, p in zip(targets, positions))
        print(f"\r{line}   ", end="", flush=True)
        time.sleep(0.3)
except KeyboardInterrupt:
    print()

positions = dict(zip(ids, io.get_present_position(ids)))
print("\n이 값을 0도로 굽습니다 (되돌릴 수 없음):")
for name, i in targets.items():
    print(f"  {name} (id {i}): 현재 {positions[i]:+.1f}도 -> 0.0도로")
if input("'yes' 입력하면 진행: ").strip() != "yes":
    print("취소함. 아무것도 쓰지 않았습니다.")
    sys.exit(0)

io._set_torque_enable({i: TORQUE_ENABLE_CALIBRATE for i in ids})
time.sleep(0.5)

print("EEPROM 다시 잠금...")
io.set_lock({i: 1 for i in ids})

print("\n확인:")
for name, i in targets.items():
    p = io.get_present_position([i])[0]
    print(f"  {name} (id {i}): 이제 {p:+.1f}도")
