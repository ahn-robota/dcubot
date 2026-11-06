# 실기 추론 러너 (설계)

학습된 ONNX 정책을 실제 로봇 위에서 자율로 돌리는 도구. 디지털 트윈(`hardware/pi/bridge_server.py`, `hardware/pc/twin_viewer.py`)은 사람이 슬라이더로 조종/관찰하는 도구이고, 이 문서는 그와 다른 **자율 주행 러너**를 설계한다. 이 문서는 설계안이고, 구현은 사용자 확인 후 진행한다.

## 1. 전체 구조

```
라즈베리파이: hardware/pi/policy_runner.py        노트북 (선택, 감시용)
┌─────────────────────────────────┐              ┌───────────────────────────┐
│ 50Hz 루프 (제어 주기 = 학습과 동일│  텔레메트리(hardware/protocol.py  │  twin_viewer.py --pi-host │
│  sim_dt*decimation=0.02s):      │  그대로, bridge_server.py와 같은  │  (Mirror 모드만, 자율 보행 │
│  센서 읽기(RealSensors 재사용)  │─────포맷) ───────────────────────▶│  관찰용. Command 모드/슬라 │
│  obs 94차원 구성(MjInfer와 동일)│                                   │  이더는 안 씀 — 여긴 사람이│
│  ONNX 추론 (onnxruntime)        │◀── 명령(속도 3개 + e-stop만) ─────┤  조종하는 게 아니라서)     │
│  motor_targets 계산+속도 제한   │   (낮은 주기, 안전 필수 아님)      └───────────────────────────┘
│  서보 쓰기(safety.py 재사용)    │
│  토크 ON 시 0점 체크(safety.py) │
└─────────────────────────────────┘
```

- **라즈베리파이가 전부 로컬로 닫힌 루프를 돈다** (센서 읽기 → 추론 → 서보 쓰기까지 네트워크 왕복 없음). Wi-Fi 지연/지터가 50Hz 제어 루프에 끼면 안 되기 때문 — 디지털 트윈의 Command 모드(사람이 매 프레임 명령을 보내는 원격조종)와는 근본적으로 다른 구조.
- 노트북은 **감시자**: 속도 명령(저주기, 안전에 필수 아님 — 받지 못하면 그냥 직전 값 유지 또는 정지 명령으로 감쇠)과 e-stop만 보내고, 기존 텔레메트리 포맷을 그대로 받아서 `twin_viewer.py`의 Mirror 모드로 걷는 모습을 구경할 수 있다(코드 수정 없이 재사용).

## 2. 코드 재사용 / 리팩터

| 가져올 것 | 현재 위치 | 조치 |
|---|---|---|
| `MockSensors`/`RealSensors` | `hardware/pi/bridge_server.py` | **`hardware/pi/sensors.py`로 분리** — `bridge_server.py`와 `policy_runner.py`가 같이 씀 (지금은 bridge_server.py에만 있어서 그대로 두면 150줄 가까이 복붙하게 됨) |
| 0점 체크 + 토크 ON 트릭 | `hardware/pi/safety.py` | 그대로 재사용 (`enable_torque_safely`) |
| 와이어 프로토콜 | `hardware/protocol.py` | 그대로 재사용. `make_command`에 **선택 필드 `velocity_command: [vx, vy, wz]`** 추가 (`bridge_server.py`의 Command 모드는 이 필드를 그냥 무시 — 해롭지 않음) |
| `OnnxInfer` | `playground/common/onnx_infer.py` | 그대로 import (onnxruntime만 필요, jax/mujoco 없음 — 라즈베리파이에서 바로 됨) |
| `mirror_obs`/`mirror_joints` (`--symmetric`) | `playground/dcubot/symmetry.py` | 그대로 import (numpy만 필요) |
| 추론 상수(action_scale 등) | `playground/dcubot/joystick.py`에 하드코딩돼 있음, `mujoco_infer.py`도 따로 하드코딩 | **`playground/dcubot/policy_constants.py` 신규** (순수 파이썬/숫자, jax·mujoco 의존성 없음) 로 뽑아서 양쪽이 같은 곳을 보게 함 — `mujoco_infer.py`도 이걸 쓰도록 리팩터(지금처럼 두 군데 따로 적혀 있으면 나중에 한쪽만 바뀌는 사고가 난다) |
| 홈 포즈 `default_actuator` (13개 ctrl 값) | MJCF 키프레임 (`model.keyframe("home").ctrl`) | 라즈베리파이엔 mujoco가 없으므로, `safety.py`의 MJCF-직접-파싱 방식(ElementTree)을 확장해서 `<key ctrl="...">`도 같이 읽는 `load_home_ctrl(mjcf_path)` 추가 |

## 3. 관측값(94차원) 구성 — `MjInfer.get_obs`(`playground/dcubot/mujoco_infer.py`)와 정확히 동일해야 함

| 순서 | 항목 | 차원 | 실기 소스 |
|---|---|---|---|
| 1 | gyro | 3 | `RealSensors`의 `gyro_base` (rad/s, base 프레임) — 노이즈 없음(추론 시엔 노이즈 안 씀) |
| 2 | accelerometer | 3 | `accel_base` (m/s², base 프레임) — **+1.3 보정 없음**: `mujoco_infer.py` 자체 주석대로 학습 때도 그 보정이 JAX 불변성 때문에 실제로는 적용 안 됐어서, 원본 그대로 써야 함 |
| 3 | command | 6 | `[vx, vy, wz, neck_pitch, head_pitch, head_yaw]` — 처음엔 머리 3개는 0 고정(머리 명령은 1차 구현 범위 밖), 이동 3개는 노트북이 보낸 `velocity_command` |
| 4 | joint_angles − default | 13 | `joint_pos`(이미 방향 보정됨) − `default_actuator`, **액추에이터 선언 순서**(아래 5번) 그대로 |
| 5 | joint_vel × 0.05 | 13 | `joint_vel`(`RealSensors`에서 위치 차분) × `DOF_VEL_SCALE` |
| 6-8 | last_action, last_last_action, last_last_last_action | 13×3 | 러너가 직접 보관(스텝마다 한 칸씩 밀기), `mujoco_infer.py`와 동일 |
| 9 | motor_targets | 13 | **지금 막 계산한 목표**(실측값 아님) — 시뮬 충실도를 위해 측정값이 아니라 우리가 보낸 타겟을 그대로 넣음 |
| 10 | contacts | 2 | `foot_contact["left"], foot_contact["right"]` |
| 11 | gait_phase | 2 | `[cos(2π·phase), sin(2π·phase)]`, `phase=(gait_step·0.02/0.4) % 1` |

**액추에이터 순서** (`dcubot.xml`의 `<actuator>` 선언 순서, `policy_constants.py`에 고정 리스트로 둠): `left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle, neck_pitch, head_pitch, head_yaw`.

## 4. 행동(모터 타겟) 계산 — `MjInfer.run()`과 동일

```python
action = policy.infer(obs)            # ONNX, --symmetric이면 SymmetricPolicy로 감쌈
last_last_last_action = last_last_action.copy()
last_last_action = last_action.copy()
last_action = action.copy()

motor_targets = default_actuator + action * ACTION_SCALE   # 0.25
motor_targets = np.clip(
    motor_targets,
    prev_motor_targets - MAX_MOTOR_VELOCITY * CONTROL_DT,   # 5.24 rad/s * 0.02s
    prev_motor_targets + MAX_MOTOR_VELOCITY * CONTROL_DT,
)
prev_motor_targets = motor_targets.copy()
```

이 속도-제한 클립은 **학습 때 그대로 있던 로직**이라 안전장치가 아니라 정확도 문제(시뮬과 다르게 움직이면 sim2real 갭이 생김) — `safety.clamp_and_slew`(Command 모드용, MJCF 범위+사용자 지정 속도)와는 별개로 둔다. MJCF 범위 클램프는 생략 가능(학습된 정책 자체가 그 범위를 벗어나는 action을 거의 안 내지만, 안전망으로 `safety.load_joint_limits`로 한 번 더 clip하는 것도 비용이 거의 없으니 추가).

## 5. 안전 (Command 모드와 다른 점 — 확인 필요)

| 상황 | Command 모드(기존) | 정책 러너(이 설계) |
|---|---|---|
| 시작 | 토크 OFF | 토크 OFF (동일) |
| 토크 ON | 사람이 `T` | 사람이 명시적으로 시작 신호(노트북에서 "시작" 명령 또는 라즈베리파이 콘솔 입력) → `safety.enable_torque_safely`로 0점 체크 (동일 로직 재사용) |
| 명령 연결 끊김 | **0.3초 안에 토크 OFF + fault 래치** (사람이 매 프레임 조종하는 원격조종이라, 끊기면 즉시 멈춰야 함) | **제안: 더 느슨하게** — 정책 자체가 "서서 버티는" 닫힌 루프라, 노트북이 끊겨도 로봇이 당장 위험해지진 않음. 속도 명령 갱신이 **1초 이상** 없으면 속도 명령만 0(제자리 걸음/정지 자세)으로 감쇠, 토크는 유지. 토크를 끄는 건 로봇이 서 있는 상태에서 더 위험할 수 있음(멈춰 서 있다가 갑자기 주저앉음) |
| E-stop | 즉시 토크 OFF + fault 래치 | 동일(즉시 토크 OFF, 래치) — 이건 Command 모드와 똑같이 유지 |

**확정** (사용자, 2026-11-06): 보행 테스트는 바닥에 세워두고 하므로, 끊김 시 토크 OFF가 더 위험함(서 있다가 바로 넘어짐) — 위 "토크 유지 + 속도 명령만 0으로 감쇠" 그대로 간다.

## 6. 파일 구성

| 경로 | 역할 |
|---|---|
| `hardware/pi/sensors.py` | `MockSensors`/`RealSensors` (bridge_server.py에서 분리) |
| `hardware/pi/policy_runner.py` | 50Hz 자율 루프: 센서 읽기 → obs 구성 → ONNX 추론 → motor_targets → 서보 쓰기. `--mock`(하드웨어 없이), `--symmetric`, `--onnx <path>` |
| `playground/dcubot/policy_constants.py` | action_scale, dof_vel_scale, gait_period, max_motor_velocity, control_dt, 커맨드 범위, 액추에이터 순서 — 숫자만, jax/mujoco 없음 |
| `playground/dcubot/mujoco_infer.py` | (리팩터) 하드코딩된 상수를 `policy_constants.py` 참조로 교체 (동작은 동일) |
| `hardware/pi/requirements.txt` | 변화 없음 예상 (onnxruntime·numpy 이미 있음) |

## 7. 테스트 순서 (기존 디지털 트윈과 같은 리듬)

1. `policy_runner.py --mock`: 가짜 센서 + 실제 ONNX로 94차원 obs가 정확한 형태/순서로 나가는지, 루프가 50Hz로 도는지 확인 (서보 없음, 노트북 혼자).
2. 실물, **로봇을 들어 올린 상태**, 토크 ON, 정지 명령(속도 0)으로 먼저 — 다리가 그냥 떨지 않고 서 있는 자세를 잡는지 확인. 노트북 `twin_viewer.py --pi-host ...`로 구경.
3. 로봇을 바닥에 세우고(사람이 받칠 수 있는 상태) 정지 명령 → 서 있는지 확인.
4. 작은 전진 속도 명령 → 걷는지 확인.

## 8. 구현 순서 제안

1. `hardware/pi/sensors.py` 분리(동작 동일 확인) + `policy_constants.py` 생성 + `mujoco_infer.py` 리팩터(기존 검증 문서 수치와 비교해서 동일한지 확인).
2. `hardware/pi/policy_runner.py` 뼈대 + `--mock` (obs 구성·타이밍 검증, 서보 없음).
3. 실물 연동(2번 항목의 안전 설계 확인 후) — 1,2,3,4단계 실행 순서대로.
