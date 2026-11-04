import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hw_config import load, servo_ids, inverted_joints  # noqa: E402

from pypot.feetech import FeetechSTS3215IO

io = FeetechSTS3215IO("/dev/ttyACM0")

# UI 프롬프트 텍스트일 뿐 하드웨어 상수가 아니므로 config에는 넣지 않음.
PROMPTS = {
    "right_hip_yaw": "오른발 끝을 안쪽(왼쪽)으로",
    "right_hip_roll": "오른다리를 바깥쪽(오른쪽)으로",
    "right_hip_pitch": "오른다리를 앞으로 들기",
    "right_knee": "오른 정강이를 앞으로",
    "right_ankle": "오른발 끝을 위로",
    "left_hip_yaw": "왼발 끝을 안쪽(오른쪽)으로",
    "left_hip_roll": "왼다리를 바깥쪽(왼쪽)으로",
    "left_hip_pitch": "왼다리를 앞으로 들기",
    "left_knee": "왼 정강이를 앞으로",
    "left_ankle": "왼발 끝을 위로",
    "neck_pitch": "목을 뒤로 젖히기",
    "head_pitch": "머리를 뒤로 젖히기",
    "head_yaw": "얼굴을 로봇 오른쪽으로",
}

config = load()
ids = servo_ids(config)
inverted = inverted_joints(config)

res = []
for name, i in ids.items():
    input(f"[{name}] {PROMPTS[name]} 약 20도 움직이고 Enter: ")
    v = io.get_present_position([i])[0]
    observed = "REVERSED" if v < -5 else ("OK" if v > 5 else "too small")
    expected = "REVERSED" if name in inverted else "OK"
    flag = "" if observed == expected else "  <-- 설정 파일(dcubot_hw_config.json)과 다름!"
    print(f"  {name}: {v:+.1f} deg -> {observed} (설정 파일 기준: {expected}){flag}")
    res.append((name, round(v, 1), observed))
print("\nSUMMARY:")
for r in res:
    print(r)
