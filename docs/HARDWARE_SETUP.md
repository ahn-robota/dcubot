# 실기 하드웨어 셋업

디쿠봇 실기(라즈베리파이 + 서보 13개 + IMU + 발 스위치) 셋업 절차와 결과. 2026-11-04~05에 확인한 내용이며, 처음부터 다시 할 때 그대로 따라 할 수 있도록 적는다.

**비밀번호, IP 주소, 호스트 이름은 이 문서에 쓰지 않는다.** 아래 모든 명령은 `<pi-user>@<pi-host>`를 사용자의 실제 라즈베리파이 사용자 이름/호스트(또는 IP)로 바꿔서 쓴다.

## 라즈베리파이

| 항목 | 값 |
|---|---|
| 보드 | Raspberry Pi Zero 2 W |
| OS | Raspberry Pi OS (Legacy, 64-bit) Lite = Debian Bookworm, aarch64, Python 3.11 (Open Duck 런타임 호환 때문에 Trixie 대신 선택) |
| 설정 | SSH 켬, I2C 켬 (`sudo raspi-config nonint do_i2c 0`), `sudo apt install i2c-tools python3-venv python3-pip git` |
| 파이썬 | `~/venv` 가상환경, `~/Open_Duck_Mini_Runtime`(v2 브랜치) `pip install -e .` 설치 (pypot 5.0.2, rustypot, onnxruntime 1.18.1, adafruit-circuitpython-bno055 5.4.13) |
| 전원 | 배터리 → UBEC 5V → GPIO 2·6번 핀. **USB 전원과 동시 연결 금지** |
| 서보 어댑터 연결 | Waveshare Bus Servo Adapter (A) ↔ Pi 데이터 USB 포트, OTG 젠더(USB 2.0, 480 Mbps) → `/dev/ttyACM0` |
| 서보 전원 | 어댑터 DC 플러그(7.4 V). 케이블 연결·분리 시 DC 플러그만 뽑으면 Pi는 켜진 채 유지 |

## IMU (BNO055)

- 배선: VIN→1번(3.3V), GND→9번, SDA→3번, SCL→5번
- **I2C 주소 0x29** (`i2cdetect -y 1`로 확인). Open Duck 기본값과 다를 수 있으니 런타임에서 주소를 지정해야 한다.
- **융합 모드: `IMUPLUS`** (자이로+가속도 융합, 지자기 미사용). 기본 `NDOF` 모드(지자기 포함)로 테스트했을 때 가만히 있는데도 yaw가 한 번씩 크게 튀는 문제가 있었다 — 서보·배터리·노트북 등 주변 자기 간섭으로 추정. 정책은 절대 yaw를 전혀 쓰지 않고 자이로·가속도(중력 방향)만 쓰므로 지자기가 필요 없어서, 실기 런타임도 `IMUPLUS`를 기본값으로 한다 (`hardware/dcubot_hw_config.json`의 `imu.fusion_mode`).

### 장착 방향 (중력 측정으로 확정)

| 칩 축 | 로봇(base) 축 |
|---|---|
| +X | +Y (왼쪽) |
| +Y | +X (앞) |
| +Z | −Z (아래, 칩이 뒤집혀 장착) |

- 변환식: `v_base = (v_chip_y, v_chip_x, -v_chip_z)`
- 회전행렬 `R_base_chip = [[0,1,0],[1,0,0],[0,0,-1]]`, det = +1
- 측정값
  - 똑바로: gravity(0.0, −0.3, −9.8)
  - 앞으로 숙임: (0.0, −3.0, −9.3)
  - 왼쪽으로 기울임: (−5.3, 0.0, −8.3)
- **이전에 가정했던 "칩 Z 위, 칩 +X = 오른쪽"은 틀렸다.** 다른 문서에서 이 가정을 봤다면 위 표로 바꾼다.
- 시뮬레이션의 IMU site는 그대로 둔다(base 축과 같은 방향). 변환은 실기 런타임에서만 한다.
- 가속도 부호: BNO055는 정지 상태에서 위를 향한 축이 +9.8로 나온다. 위 변환을 적용하면 base z = +9.8. MuJoCo accelerometer 규약(정지 시 +g)과 같은 부호다.

## 서보 (STS3215 × 13)

### ID 표

Open Duck v2 런타임과 같은 번호 체계 (head_roll=33은 디쿠봇에 없음).

| 관절 | ID | 관절 | ID |
|---|---|---|---|
| right_hip_yaw | 10 | left_hip_yaw | 20 |
| right_hip_roll | 11 | left_hip_roll | 21 |
| right_hip_pitch | 12 | left_hip_pitch | 22 |
| right_knee | 13 | left_knee | 23 |
| right_ankle | 14 | left_ankle | 24 |
| neck_pitch | 30 | head_pitch | 31 |
| head_yaw | 32 | | |

### ID 설정 방법

- 조립된 상태라서 Open Duck `scripts/configure_motor.py`에서 마지막 `set_goal_position(0)` 줄만 지운 사본(`configure_motor_nomove.py`, [`hardware/pi/setup/`](../../hardware/pi/setup/) 참고)을 썼다.
- 데이지 체인을 하나씩 늘려가며 ID 1 서보만 찾아 바꾸는 방식.
- 레지스터 값: P 32, I 0, D 0, 가속도 0, mode 0(위치)
- left_ankle은 처음 ID가 2였다. 스크립트가 스캔으로 찾아서 설정했다.

### 0점 보정 (하드웨어)

- 혼(horn)이 임의 각도로 조립돼 있었다. 6개 서보가 ±150~178°(wrap 경계 근처)였다.
- 토크를 끄고 로봇을 **URDF 0 자세**로 맞춘 다음, 13개에 `Torque_Enable = 128`을 써서 현재 위치를 0°로 보정했다.
- 전원을 껐다 켠 뒤에도 0.0° 유지를 확인했고, EEPROM은 다시 잠궜다(lock=1). 소프트웨어 옵셋은 0이다.

저장된 offset (deg):

| ID | offset | ID | offset |
|---|---|---|---|
| 10 | −106.86 | 20 | 165.93 |
| 11 | −41.63 | 21 | −148.18 |
| 12 | −2.68 | 22 | 29.85 |
| 13 | 151.78 | 23 | 178.07 |
| 14 | 79.43 | 24 | −26.77 |
| 30 | 34.68 | | |
| 31 | 159.6 | | |
| 32 | 63.16 | | |

### 방향 확인

관절을 손으로 MJCF 축 기준 + 방향으로 약 20° 움직여서 측정.

- **같음 (10개)**: ID 10~14, 22, 23, 24, 30, 31
- **반대 (3개)**: ID 20 (left_hip_yaw), 21 (left_hip_roll), 32 (head_yaw)
- 원인: MJCF는 좌우 yaw·roll 축을 거울 대칭으로 정의했는데, 서보는 좌우 같은 방향으로 장착돼 있다.
- 해결: **실기 런타임에서만** 부호를 반전한다. 명령은 `q_servo = −q_sim`, 읽기는 각도·속도 모두 −1. 시뮬레이션과 정책은 그대로 둔다.

### 토크 유지 테스트

절차: 0° ±15° 안인지 확인 → 현재 위치를 목표로 토크 켜기 → 2초에 걸쳐 0°로 이동 → 3초 유지.

- 결과: 대부분 ±0.4°. hip_roll만 +1.5°(11) / −1.4°(21) 정상상태 오차(눕힌 자세에서 다리 무게 영향으로 추정).
- ±15° 밖이면 실행을 중단하는 안전장치 동작도 확인함.

## 발 스위치 (SS-10)

- 계획: 왼쪽 GPIO22(15번 핀), 오른쪽 GPIO27(13번 핀), COM→GND, NO→GPIO, 내부 pull-up(밟으면 0)
- 현재 배선 중, **미확인**.

## 설정 값의 단일 소스

위 ID 표, 방향 반전, IMU 주소·축 변환은 [`hardware/dcubot_hw_config.json`](../../hardware/dcubot_hw_config.json)에 모아뒀다. 하드웨어를 다루는 모든 스크립트는 이 파일을 읽어야 하며, 같은 값을 다시 하드코딩하지 않는다.

## 배포 방식

- 라즈베리파이는 public 저장소를 **HTTPS로 pull만** 한다. push는 노트북에서만 한다.
- 처음 한 번:
  ```bash
  cd ~ && git clone https://github.com/ahn-robota/dcubot.git
  ```
- 업데이트할 때:
  ```bash
  cd ~/dcubot && git pull
  ```
- 라즈베리파이에서는 코드를 직접 고치지 않는다.
- 저장소 루트는 `~/dcubot`이다 (`git clone`이 그 이름으로 만든다. 노트북의 `DCUBOT/rl_training` 같은 하위 경로가 **없다**).
- 실행은 항상 기존 `~/venv`(Open Duck 런타임 설치됨, pypot 포함)를 활성화한 뒤 한다. 시스템 `python3`에는 pypot이 없어서 바로 실행하면 `ModuleNotFoundError`가 난다. 추가 패키지는 `hardware/pi/requirements.txt`로 관리: 디지털 트윈 브리지(`bridge_server.py`, 실물 모드)의 발 스위치 읽기에 `gpiozero`가 필요 — `~/venv`에서 `pip install -r hardware/pi/requirements.txt`.

  ```bash
  cd ~/dcubot/hardware/pi/setup
  source ~/venv/bin/activate
  python check_directions.py          # 조인트별로 실측 방향을 설정 파일(invert)과 비교
  python hold_zero.py                 # 0점 근처에서 토크 ON -> 0도로 복귀 -> 유지
  python configure_motor_nomove.py --id <새 ID>   # 새 서보에 ID만 부여 (움직이지 않음)
  ```

### (선택) 노트북 → 라즈베리파이 SSH 키 등록

등록하면 노트북에서 비밀번호 없이 원격 명령을 보낼 수 있다(예: `ssh <pi-user>@<pi-host> "cd ~/dcubot && git pull"`).

```bash
# 노트북에서, 키가 없다면 먼저 생성
ssh-keygen -t ed25519 -C "dcubot-laptop"

# 공개키를 라즈베리파이로 전송 (비밀번호 1회 입력)
ssh-copy-id <pi-user>@<pi-host>
```

키 등록은 사용자가 직접 한다. 비밀번호와 개인키는 출력하지 않는다.
