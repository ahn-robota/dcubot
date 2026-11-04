# 디지털 트윈 테스트 도구 (설계)

실물 디쿠봇과 시뮬레이션(MuJoCo) 로봇을 실시간으로 동기화해서 테스트하는 도구. 이 문서는 **설계안**이고, 구현은 사용자 확인 후 진행한다.

## 1. 전체 구조

```
라즈베리파이 (hardware/pi/bridge_server.py)          노트북 (hardware/pc/twin_viewer.py)
┌─────────────────────────────┐                   ┌──────────────────────────────┐
│ IMU(BNO055) ─┐              │   UDP, JSON       │  MuJoCo passive viewer        │
│ 서보 13개   ─┼─ 50Hz 폴링 ──┼──── 텔레메트리 ───▶│   (Mirror/Command 공용 3D)   │
│ 발 스위치 2 ─┘   (축 변환,  │   (포트 하나,     │   Control 패널(슬라이더) +    │
│              방향 반전 적용)│    sendto 응답)   │   고스트/발 스위치 화살표     │
│                             │◀── 명령/하트비트──┤   (터미널에 Hz/RTT/손실률)    │
│  워치독: 0.3s 끊기면 토크 OFF│   (20Hz, 항상)    │                               │
└─────────────────────────────┘                   └──────────────────────────────┘
```

- 하나의 UDP 소켓을 브리지가 고정 포트(예: 8420)에서 열어두고, 뷰어가 보낸 패킷의 발신 주소를 "마지막 뷰어 주소"로 기억해서 그 주소로 `sendto`한다. 포트 하나로 양방향 — 포트 2개 관리할 필요 없음.
- 브리지는 수신 전용 데몬 스레드(`Queue(maxsize=1)`, 최신 패킷만 유지 — Open Duck 런타임의 `imu_client.py`와 같은 패턴) + 메인 스레드의 고정 50Hz 루프(센서 읽기 → 전송 → 큐에서 최신 명령 꺼내 적용 → 워치독 체크)로 구성한다.

### 왜 UDP인가 / Open Duck 런타임 재사용 안 한 이유

- 같은 Wi-Fi에서 50Hz 상태 스트리밍 + 저지연 명령 전달에는 TCP의 순서보장/재전송이 오히려 지연을 늘릴 수 있어 UDP가 적합하다 (패킷 하나 잃어도 다음 패킷이 곧 오므로 상태값은 최신 것만 쓰면 됨).
- Open Duck 런타임의 `scripts/imu_server.py`/`imu_client.py`를 확인해봤는데: TCP + `pickle` + 길이 프레이밍 없는 `recv(1024)` (메시지가 쪼개지거나 합쳐질 수 있음) + 서버가 연결 종료 후 소켓을 닫아버려 재접속이 실질적으로 깨지는 버그가 있었다. `pickle`을 신뢰 안 되는 소스에서 역직렬화하는 것도 안전하지 않다. 그래서 그대로 가져다 쓰지 않고, 아래 2번의 자체 JSON/UDP 프로토콜을 쓴다. 재사용할 가치가 있는 부분(큐 기반 수신 스레드 패턴)만 패턴으로 가져온다.
- JSON을 쓰는 이유: 패킷 하나가 Wi-Fi MTU(1472바이트)보다 훨씬 작아서(아래 2번 필드 기준 약 300바이트) 바이너리로 줄일 실익이 적고, 디버깅(그냥 `nc -u`나 Wireshark로 봐도 읽힘)과 필드 추가가 훨씬 쉽다. 성능이 문제가 되면 나중에 `struct`로 바꿀 수 있다.
- 프로토콜 정의(필드 이름, 버전)는 `hardware/protocol.py` 하나에 두고 `bridge_server.py`/`twin_viewer.py` 둘 다 이걸 import한다 (양쪽에 같은 필드 이름을 따로 적지 않음).

## 2. 데이터 형식

### 텔레메트리 (Pi → PC, 50Hz)

```json
{
  "seq": 12345,
  "t_pi": 1731000000.123,
  "loop_hz": 49.8,
  "servo_read_ms": 4.2,
  "imu_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "imu_gyro": [0.0, 0.0, 0.0],
  "imu_accel": [0.0, 0.0, 9.8],
  "joint_pos": {"right_hip_yaw": 0.01, "...": "..."},
  "joint_vel": {"right_hip_yaw": 0.0, "...": "..."},
  "foot_contact": {"left": false, "right": false},
  "torque_enabled": false,
  "fault": null,
  "echo_cmd_seq": 601,
  "echo_cmd_t_pc": 1731000000.098,
  "last_cmd_seq_recv": 601,
  "servo_temp_c": {"right_hip_yaw": 32.0, "...": "..."},
  "servo_voltage_v": {"right_hip_yaw": 7.3, "...": "..."}
}
```

- IMU 값·조인트 값은 **축 변환·방향 반전을 이미 적용한 뒤**의 값 — 즉 시뮬레이션/MJCF 좌표계 그대로. PC 쪽은 하드웨어의 축 뒤집힘을 몰라도 된다 (`hardware/dcubot_hw_config.json`의 `R_base_chip`/`invert`는 브리지 안에서만 쓰인다).
- `joint_pos`/`joint_vel`/`joint_target`은 조인트 이름 → 값 딕셔너리로, 순서 문제가 생기지 않게 한다 (리스트+순서 암묵 규약보다 안전).
- `loop_hz`/`servo_read_ms`: 브리지 메인 루프의 실측 주기와 서보 13개 동시읽기(sync read, `io.get_present_position(ids)`가 이미 쓰는 방식)에 걸린 시간 — 1초 이동평균, 매 텔레메트리 패킷에 포함. 50Hz가 안 나오면 여기서 바로 보인다 (아래 "브리지 루프 실측" 참고).
- `seq`: 텔레메트리 자체의 일련번호. 뷰어가 기대 카운터 대비 빠진 번호 비율로 **Pi→PC 손실률**을 계산.
- `echo_cmd_seq`/`echo_cmd_t_pc`: 브리지가 가장 최근에 **수신한** 명령 패킷의 `seq`/`t_pc`를 그대로 돌려줌 → 뷰어가 `now_pc - echo_cmd_t_pc`로 **왕복 지연(RTT)**을 계산(두 기기 시계가 달라도 됨, PC 자기 시계만 씀). `echo_cmd_seq`는 그 RTT가 "몇 번째로 보낸 명령"에 대한 것인지 식별.
- `last_cmd_seq_recv`: 브리지가 지금까지 받은 명령 중 가장 큰 `seq` — 뷰어가 자기가 보낸 seq 카운터와 비교해서 **PC→Pi 손실률**도 계산(양방향 모두 측정).
- `servo_temp_c`/`servo_voltage_v`: pypot이 지원하면 포함, 아니면 필드 생략(선택 항목으로 취급, 매 프레임 추가 왕복 없이 같은 패킷에 끼워 보냄).
- `fault`: `null` 또는 `"timeout"`/`"estop"`/`"zero_offset_exceeded"` 등 — 래치된 고장 상태(아래 4번).

### 명령/하트비트 (PC → Pi)

뷰어가 **항상 20Hz로** 보낸다 (Mirror 모드에서도, 슬라이더를 안 움직여도) — 브리지 워치독이 쓸 "뷰어가 살아있다"는 신호가 필요하기 때문.

```json
{
  "seq": 601,
  "t_pc": 1731000000.098,
  "torque_request": false,
  "estop": false,
  "joint_target": {"right_hip_yaw": 0.0, "...": "..."}
}
```

- `seq`: 뷰어 쪽 송신 일련번호(1씩 증가) — 위 `last_cmd_seq_recv`/`echo_cmd_seq`와 짝이 되어 손실률·RTT 계산에 쓰임.
- `torque_request`: Command 모드 토크 ON/OFF 버튼 상태.
- `estop`: Space 키가 눌리면 `true` (아래 4번).
- `joint_target`: Command 모드일 때만 의미 있음(슬라이더 값). Mirror 모드에서는 현재 측정값을 그대로 돌려보내 토크가 꺼진 상태에서는 아무 효과가 없게 한다.

### 브리지 루프 실측 (Hz / sync read 시간)

`bridge_server.py`는 매 루프마다 전체 주기와 "서보 13개 읽기"에 걸린 시간을 재서 1초 이동평균을 `loop_hz`/`servo_read_ms`로 텔레메트리에 싣고 stdout에도 로그한다. 지금 쓰는 스크립트들(`hold_zero.py` 등)이 이미 `io.get_present_position(ids)`에 ID 리스트를 한 번에 넘기는 걸로 봐서 pypot이 STS3215의 sync read(여러 서보를 한 패킷으로 동시 읽기)를 쓰고 있을 가능성이 높지만, 실측값은 실제 라즈베리파이+서보 13개에 연결해봐야 나온다 — Mirror 모드 실물 검증 단계에서 같이 보고한다. 50Hz가 안 나오면: ① sync read가 실제로 쓰이는지 pypot 쪽 확인, ② 안 되면 폴링 주기를 낮추고 그 사실을 `loop_hz`로 투명하게 보여주는 선에서 멈춘다(정책 실행 주기 결정은 이 수치를 보고 사용자가 판단).

## 3. 좌표 변환 (`hardware/dcubot_hw_config.json` 재사용)

- IMU: 브리지가 BNO055 raw 벡터/쿼터니언에 `R_base_chip`을 적용해서 base 좌표계로 변환한 뒤 전송.
- 서보: `invert`가 `true`인 3개 조인트(`left_hip_yaw`, `left_hip_roll`, `head_yaw`)는 읽을 때 각도·속도에 −1, 쓸 때 목표각에 −1을 적용. `dcubot_hw_config.json`을 한 번만 읽고 양방향 변환 함수를 `hardware/hw_config.py`에 추가해서 쓴다 (지금 `servo_ids`/`inverted_joints`가 있는 그 파일).
- 새 설정 값을 스크립트에 다시 하드코딩하지 않는다 — 1·2단계에서 지킨 원칙을 그대로 따른다.

## 4. 안전 로직 (브리지 쪽)

| 상황 | 동작 |
|---|---|
| 시작 시 | 토크 OFF. `fault=null` |
| `torque_request: true` 수신 | `hold_zero.py`와 같은 절차 재사용(공유 함수로 분리 예정, 5번 참고): 모든 조인트가 0점 ±15° 안인지 확인 → 아니면 거부(`fault="zero_offset_exceeded"`) → 현재 위치를 목표로 토크 ON → 그 다음 프레임부터 `joint_target`을 향해 서서히 이동(슬루 제한, 아래) |
| 매 프레임 `joint_target` 적용 | ① MJCF 자체의 조인트 limit으로 clamp(값을 따로 복붙하지 않고 `dcubot.xml`을 한 번 로드해서 그 범위를 읽어 씀) → ② 이전에 실제로 보낸 목표각 기준 슬루 레이트 제한(기본 초당 60°, CLI로 조절) → ③ 서보에 쓰기 |
| **통신 0.3초 이상 끊김** (마지막 명령 수신 이후) | **토크 OFF** (아래 "왜 OFF인가" 참고), `fault="timeout"`으로 래치 |
| `estop: true` 수신 | 즉시 토크 OFF, `fault="estop"`으로 래치 |
| `fault`가 래치된 상태 | 통신이 복구되거나 `estop`이 풀려도 **자동으로 토크를 다시 켜지 않음** — 뷰어에서 사용자가 명시적으로 토크 ON을 다시 눌러야 `fault`가 풀리고 재시도 가능. (끊겼다가 복구된 순간 자동으로 다시 토크가 들어오는 게 더 위험하다고 판단) |
| `--dry-run` | 위 로직은 다 돌지만 실제 서보 `set_goal_position`/`enable_torque` 호출을 하지 않음(읽기는 정상 — IMU/조인트/발 스위치 실측값을 그대로 전송). 연결 테스트용 |
| `--mock` | 실제 하드웨어 라이브러리(pypot/adafruit-BNO055/GPIO)를 **import하지 않고**, 합성 데이터(사인파 관절 각도, 고정 자세 IMU, 토글되는 가짜 발 스위치)를 같은 프로토콜로 송신. 노트북에서 라즈베리파이 없이 혼자 실행 가능 |

**왜 타임아웃에 "토크 OFF"를 제안하는가**: "자세 유지"는 뷰어가 멈추거나 네트워크가 끊긴 그 순간의 목표를 영원히 붙잡고 있는 것인데, 그 목표가 전환 동작 중간의 애매한 값일 수도 있고, 사용자가 로봇을 들고 있다가 손을 놓는 등 상황이 바뀌었을 수도 있다. 반면 토크 OFF는 그냥 멈추는 것이라 결과를 예측하기 쉽고, 이미 "시작 시 토크 OFF"·"E-stop도 토크 OFF"와 같은 원칙이라 로직이 하나로 통일된다. 다만 로봇이 서 있는 상태에서 토크가 꺼지면 넘어지므로, Command 모드 테스트는 항상 사람이 들고 있거나 받칠 수 있는 상태에서 하는 걸 전제로 한다(지금 실행 순서 5번에도 "들어 올린 상태" 명시).

## 5. 중복 구현 방지

- `hold_zero.py`의 "현재 위치 목표 → 토크 ON → 서서히 이동" 로직을 `hardware/pi/safety.py`로 뽑아서 `hold_zero.py`와 `bridge_server.py`가 같이 쓴다 (구현 단계에서 `hold_zero.py`를 이 공유 함수를 호출하는 형태로 바꿈 — 동작은 동일, 코드만 한 곳으로).
- MJCF 조인트 limit은 `playground/dcubot/xmls/dcubot.xml`을 그대로 로드해서 읽는다(별도 숫자로 다시 적지 않음).
- 서보 ID·방향 반전·IMU 변환은 기존 `hardware/dcubot_hw_config.json`/`hardware/hw_config.py` 그대로 재사용.
- 와이어 프로토콜은 `hardware/protocol.py` 하나에만 정의.

## 6. 화면 구성 (노트북) — 창 하나, MuJoCo 기본 뷰어만

사용자 제안대로, matplotlib 창을 따로 띄우지 않고 **MuJoCo 기본 뷰어(`launch_passive`) 하나**로 전부 처리하는 쪽을 1순위로 검토한다. 의존성 추가가 없고, `replay_directions_mujoco.py`/`mujoco_infer.py`가 이미 쓰는 패턴들의 조합이라 구현 리스크가 작다.

- **Control 패널(오른쪽, 기본 UI)**: `playground/dcubot/xmls/dcubot.xml`의 13개 `<position>` 액추에이터(이름이 조인트 이름과 동일, `inheritrange="1"`로 슬라이더 범위가 곧 조인트 limit)가 `show_right_ui=True`(기본값, `replay_directions_mujoco.py`가 `False`로 끄는 그 패널)일 때 자동으로 슬라이더로 나타난다. 매 프레임 `d.ctrl`을 읽으면 그게 곧 "목표각" — 따로 슬라이더 UI를 만들 필요가 없다. Command 모드에서만 이 값을 `joint_target`으로 실어 보낸다.
  - **검증 필요**: 이 패널이 `launch_passive`에서도 그대로 나오고, 슬라이더를 움직이면 파이썬에서 읽는 `d.ctrl`이 실제로 갱신되는지는 GUI 조작이 필요해서 내(Claude Code)가 터미널에서 직접 확인할 수 없다 — 아래 "검증 요청"에서 사용자가 짧게 확인.
  - 안 된다면: matplotlib Slider(이미 의존성 있음)로 대체 — 설계는 그대로, UI 창만 하나 추가.
- **키보드(`key_callback`, `mujoco_infer.py`와 같은 패턴)**:
  - `Space`: E-stop 토글 (다음 프레임부터 명령 패킷 `estop: true`).
  - `T`: 토크 ON/OFF 요청 토글 (`torque_request`).
  - `M`: Mirror ↔ Command 모드 전환.
- **3D 장면**: `scene_flat_terrain.xml` 재사용, 매 프레임 qpos/ctrl을 직접 갱신(물리 step 없이 teleport, `replay_directions_mujoco.py`와 동일한 방식):
  - **Mirror 모드**: 몸통 **위치는 공중 고정**(home 자세의 z 높이로 고정, xy는 0) — IMU는 위치를 주지 않으므로 뜨는 느낌을 만들지 않는다. 방향은 "기울기(roll/pitch)는 그대로, yaw는 연결 시점 기준 상대값"으로 구성한다: 연결 첫 프레임의 `imu_quat_wxyz`에서 `scipy.spatial.transform.Rotation`으로 Z축 기준 yaw0를 뽑아두고, 이후 매 프레임 측정 quat에서 yaw를 `yaw_now - yaw0`로 바꿔서 재구성한 쿼터니언을 base에 쓴다. (BNO055가 켤 때마다 임의의 heading을 0으로 잡기 때문에 raw yaw를 그대로 쓰면 로봇이 매번 이상한 방향으로 서 있게 됨 — roll/pitch는 중력 기준이라 이 문제가 없음.) 조인트는 `joint_pos`를 그대로 qpos에.
  - **Command 모드**: "명령"(불투명, `d.ctrl`=슬라이더 값으로 그려지는 기본 모델) + "실제"(반투명 고스트, 받은 `joint_pos`) 두 인스턴스를 `render_checkpoints.py`가 이미 쓴 `mjv_updateScene`(1번째)+`mjv_addGeoms`(나머지, alpha 낮춰서) 합성 기법으로 한 장면에.
  - **발 스위치**: 각 발 site 위에 작은 화살표/콘 geom을 `mjv_addGeoms`로 추가, `foot_contact`에 따라 표시·색(녹/적) 변경. MJCF 자체는 안 건드림.
  - **상태 텍스트**: MuJoCo 뷰어에 자유 텍스트 오버레이 API가 없어서, 지연(RTT)/Hz/손실률(양방향)/`loop_hz`/`servo_read_ms`/`fault` 상태는 터미널에 ~2Hz로 한 줄씩 갱신 출력(목표각 vs 실제각은 Command 모드에서 Control 패널 슬라이더=목표, 고스트=실제로 이미 시각적으로 보이므로 터미널엔 요약 숫자만).

### 6.1 검증 요청 (GUI 조작이 필요해서 사용자가 확인)

아래를 실행하고, Control 탭에서 액추에이터 슬라이더를 하나 움직였을 때 터미널에 찍히는 `d.ctrl` 값이 실제로 바뀌는지, 그리고 그 관절이 화면에서 움직이는지 봐주세요. 실행 경로는 노트북 저장소 루트(`rl_training`) 기준입니다.

```bash
uv run python -c "
import mujoco, mujoco.viewer, time
m = mujoco.MjModel.from_xml_path('playground/dcubot/xmls/scene_flat_terrain.xml')
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
with mujoco.viewer.launch_passive(m, d) as viewer:
    t0 = time.time()
    while viewer.is_running():
        mujoco.mj_step(m, d)
        viewer.sync()
        if time.time() - t0 > 0.2:
            print([round(x, 3) for x in d.ctrl])
            t0 = time.time()
        time.sleep(0.002)
"
```

- 되면: 위 설계(Control 패널 슬라이더 그대로 사용) 확정.
- 안 되면(패널이 없거나 `d.ctrl`이 안 바뀌면): matplotlib Slider로 대체 — 이미 문서에 대안으로 적어둠.

**검증 완료** (사용자 실행, 2026-11-05): Control 패널 슬라이더로 `d.ctrl`이 바뀌고 시뮬 관절도 같이 움직임 확인. Command 모드는 이 방식(Control 패널)으로 구현됨 — `hardware/pc/twin_viewer.py`의 `M` 키로 Mirror↔Command 전환, `T` 키로 토크 요청 토글. 안전 로직(0점 체크, 클램프+슬루 제한, e-stop/워치독 fault 래치)은 `hardware/pi/safety.py` + `bridge_server.py`에 구현, mock 대상 스크립트 테스트로 7개 시나리오 확인(토크 on/off, MJCF 범위 클램프, e-stop, fault 래치, 재요청 시 해제, 워치독 타임아웃).

## 7. 실행 순서 (최종, 구현 후 그대로 문서에 유지)

1. 노트북·라즈베리파이를 같은 Wi-Fi(핫스팟)에 연결.
2. 라즈베리파이에서 브리지 먼저 실행:
   ```bash
   cd ~/dcubot/hardware/pi
   source ~/venv/bin/activate
   python bridge_server.py [--dry-run]
   ```
3. 노트북에서 뷰어 실행, `--pi-host`로 주소 지정(코드에 하드코딩하지 않음):
   ```bash
   uv run python hardware/pc/twin_viewer.py --pi-host <pi-host-or-ip> [--ping] [--mock]
   ```
4. 연동 확인 (각 단계를 따로 실행해서 확인):
   1. **연결 테스트**: 토크 OFF, 화면 없음. 라즈베리파이 `bridge_server.py --dry-run`, 노트북 `twin_viewer.py --ping`(Hz/지연/손실만 출력하고 종료) — 그리고 `twin_viewer.py --mock`으로 라즈베리파이 없이 노트북 혼자서도 같은 걸 확인.
   2. **Mirror**: 토크 OFF 상태에서 실물을 기울이거나 관절을 손으로 움직이면 화면 모델이 따라오는지, 발 스위치 화살표가 밟을 때 바뀌는지 확인.
   3. **Command**: 로봇을 들어 올린 상태에서 토크 ON → 관절 하나씩, 작은 각도로 슬라이더를 움직이며 시뮬/실물/고스트 일치 확인.
5. 종료: 노트북 앱에서 토크 OFF → 앱 종료 → 브리지 종료(Ctrl+C, 종료 시에도 토크 OFF) → 라즈베리파이 `sudo shutdown -h now` → 배터리 OFF.

## 8. 파일 구성

| 경로 | 역할 |
|---|---|
| `hardware/protocol.py` | 와이어 프로토콜(필드 정의, 패킹/파싱) — pi/pc 공용 |
| `hardware/pi/safety.py` | 토크 ON 안전 램프, limit clamp, 슬루 제한 — `hold_zero.py`/`bridge_server.py` 공용 |
| `hardware/pi/bridge_server.py` | 50Hz 센서 읽기+전송, 명령 수신+적용, 워치독. `--dry-run`/`--mock` |
| `hardware/pc/twin_viewer.py` | MuJoCo 뷰어 창 하나(Control 패널 슬라이더, 고스트, 발 스위치 화살표, 터미널 상태 로그). `--pi-host`, `--ping`, `--mock` |

## 9. 구현 순서 제안

1. `hardware/protocol.py` + `bridge_server.py --mock` + `twin_viewer.py --ping`(화면 없음) → 노트북 혼자 localhost로 왕복 확인.
2. `twin_viewer.py` Mirror 모드(3D + 발 스위치 화살표) → 먼저 `--mock`, 그 다음 실제 `bridge_server.py --dry-run`(라즈베리파이, 실제 IMU·조인트 값).
3. `hardware/pi/safety.py` 분리 + `hold_zero.py`를 그걸 쓰도록 변경(동작 동일 확인) → Command 모드(슬라이더, 토크 ON/OFF, 고스트, 워치독/E-stop) → `--mock`으로 먼저, 그 다음 실물(들어 올린 상태).

각 단계 구현 후: 노트북 `--mock` 검증 → push → 라즈베리파이 쪽 실행 명령(`git pull` → `bridge_server.py` 실행)과 노트북 실행 명령을 순서대로 드림 → 사용자가 실물로 확인한 결과를 알려주면 다음 단계로 진행.
