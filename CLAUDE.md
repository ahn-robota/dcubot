# CLAUDE.md

두 PC(노트북, 실습실 PC)의 Claude Code가 공유하는 프로젝트 안내서. 작업 시작 전에 읽고, 진행 상황 섹션은 단계가 끝날 때마다 갱신한다.

**이 저장소는 public이다.** 비밀번호, IP 주소, 호스트 이름, Tailscale·AnyDesk 정보, 계정 정보는 이 파일과 커밋 어디에도 넣지 않는다.

## 프로젝트 개요

- 디쿠봇: 대구가톨릭대 마스코트 이족보행 로봇. Open Duck Mini 기반.
- MuJoCo Playground(JAX/Brax/MJX) + PPO로 전방향(omnidirectional) 보행 정책을 학습한다.
- 원칙: 원본 `open_duck_mini_v2` 학습 파이프라인 구조를 그대로 따르고, 로봇 모델(MJCF), 레퍼런스 모션, 로봇 형상에 묶인 값만 바꾼다. 원본과 다른 점은 모두 [`docs/CHANGES_FROM_ORIGINAL.md`](docs/CHANGES_FROM_ORIGINAL.md)에 있다.

## 폴더 구조

| 경로 | 역할 |
|---|---|
| `playground/dcubot/` | 디쿠봇 환경: `joystick.py`(보상·관측·명령 범위), `runner.py`(학습 진입점), `randomize.py`(도메인 랜덤화), `mujoco_infer.py`(키보드 조종 뷰어), `xmls/`(MJCF) |
| `playground/dcubot/data/` | 레퍼런스 모션. `polynomial_coefficients_omnidirectional_headv2_deg9.pkl` 사용 중 (머리 수정 반영). `_newurdf_deg9.pkl`(머리 수정 전)·`_omnidirectional.pkl`(구모델)은 비교용으로 남겨둠 |
| `playground/common/` | 원본 Open Duck 공용 코드: 보상, 레퍼런스 모션, ONNX 내보내기, 공용 러너 |
| `robot_description/`, `tools/` | 최종 URDF와 MJCF 생성기(`tools/gen_dcubot_mjcf.py`) |
| `scripts/` | 평가·검증: `eval_command_tracking.py`(명령 따라가기 점수), `render_checkpoints.py`(체크포인트 영상), `replay_reference.py`(레퍼런스 점검) 등 |
| `docs/` | 학습 절차, 원본과 달라진 점, 실험 기록 |
| `checkpoints/` | 학습 결과(`checkpoints/<실험이름>/`). git에 올리지 않음 |

## 작업 규칙

- 한 단계씩 진행하고, 단계마다 결과를 보고한 뒤 사용자 확인을 받고 넘어간다.
- sudo가 필요한 명령은 실행하지 말고 사용자에게 명령어만 준다.
- 개발자처럼 파일을 정리한다. 필요 없는 파일은 만들지도 남기지도 않고, 같은 기능을 중복 구현하지 않는다.
- 원본 Open Duck의 보상 가중치와 PPO 설정은 승인 없이 바꾸지 않는다. 원본과 다르게 바꾼 것은 반드시 `docs/CHANGES_FROM_ORIGINAL.md`에 이유와 함께 기록한다.
- 의존성은 git에 있는 `uv.lock` 기준으로 `uv sync`해서 맞춘다. 버전을 임의로 바꾸지 않는다(`playground==0.0.5`, `jax[cuda12]==0.9.0`, `jaxlib==0.9.0` 고정). 의존성을 바꿔야 하면 사용자 승인 후 `pyproject.toml`과 `uv.lock`을 같이 커밋한다.
- 레퍼런스 모션(`playground/dcubot/data/*.pkl`) 재생성 여부는 사용자가 결정한다.

## 두 PC 작업 방식

- 노트북 = 코드 개발·짧은 검증, 실습실 PC = 본 학습.
- 두 PC 모두 같은 경로에서 작업한다: `~/dcubot/DCUBOT/rl_training` (저장소 루트).
- 작업 시작 전 `git pull`, 끝나면 커밋·push. 두 PC에서 같은 파일을 동시에 고치지 않는다.
- 체크포인트, 로그, `.venv`는 git에 올리지 않고 각 PC에서 따로 관리한다.
- PC마다 다른 값은 코드에 박지 않고 실행 인자나 환경변수로 넘긴다: `--num_envs`, `XLA_PYTHON_CLIENT_MEM_FRACTION`.

  | PC | `--num_envs` | `XLA_PYTHON_CLIENT_MEM_FRACTION` |
  |---|---|---|
  | 노트북 (RTX 4050 6GB) | 1024 | 0.95 |
  | 실습실 PC (RTX 3080 10GB) | 1024 (2048은 메모리 부족) | 0.85 (0.95는 cuSolver 오류) |

## 자주 쓰는 명령 (저장소 루트에서)

학습 절차 전체(단계별 명령 범위, 체크포인트 고르는 법): [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md)

```bash
uv sync                                              # uv.lock 기준 환경 구성
uv run python -c "import jax; print(jax.devices())"  # CudaDevice 확인

# 학습 (--num_envs, MEM_FRACTION은 위 표의 PC별 값, 명령 범위는 단계별)
XLA_PYTHON_CLIENT_MEM_FRACTION=<비율> uv run python -u -m playground.dcubot.runner \
  --task flat_terrain_train --num_envs <N> \
  --output_dir checkpoints/<실험이름> --num_timesteps <스텝> --num_evals <N> \
  --lin_vel_x <MIN> <MAX> --lin_vel_y <MIN> <MAX> --ang_vel_yaw <MIN> <MAX> \
  [--restore_checkpoint_path $PWD/checkpoints/<이전 실험>/<체크포인트>]   # 이어서 학습 (절대경로)

# 평가: 명령 따라가기 (1.0=완벽, 0=제자리). --env N은 로봇 N대·밀기 포함 (GPU 사용)
uv run python scripts/eval_command_tracking.py [--env 64] --scale 0.75 --turn 0.15 [--symmetric] <policy.onnx ...>

# 키보드 조종 뷰어 (↑↓ 전후, ←→ 옆걸음, Q/E 회전, Space 정지)
uv run python -m playground.dcubot.mujoco_infer -o <policy.onnx> \
  --lin_vel_x -0.15 0.2 --lin_vel_y -0.12 0.12 --ang_vel_yaw -0.15 0.15 --symmetric --heading_hold 1.0

uv run tensorboard --logdir checkpoints/<실험이름>
```

짧은 학습은 `--num_evals`를 낮춰야 목표 스텝 근처에서 멈춘다(이유는 `playground/dcubot/runner.py`의 `DCUBOT_NUM_EVALS` 주석).

## 검증 기준

노트북 검증 결과(obs·action 차원, 레퍼런스 재생, 짧은 검증 학습 곡선)와 실행 명령: [`docs/VERIFICATION_2026-10-27.md`](docs/VERIFICATION_2026-10-27.md). 새 PC나 환경 변경 후에는 같은 명령으로 돌려서 이 문서와 비교한다.

## 현재 진행 상황과 다음 할 일

- [x] 원점 재시작(원본 `open_duck_mini_v2` 구조), 두 PC 환경 구성·검증 ([`docs/VERIFICATION_2026-10-27.md`](docs/VERIFICATION_2026-10-27.md))
- [x] 구모델 50M 본 학습 `omni_1024_50m`: 명령을 따라 걷지 않고 제자리에 서 있음 ([`docs/RESULTS_omni_1024_50m.md`](docs/RESULTS_omni_1024_50m.md))
- [x] 실습실 PC 자동 실험(2026-10-28~29): 발 관련 보상 복원 + 방향별 단계 학습으로 처음 걸음. 구모델 최선 `cur5` 7.2M ([`docs/AUTO_EXPERIMENTS_2026-10-28.md`](docs/AUTO_EXPERIMENTS_2026-10-28.md))
- [x] 최종 URDF 리모델(노트북, 2026-11-02~03): 외피·머리·배터리 반영 2.072 kg, 레퍼런스 모션·MJCF 재생성. 레퍼런스 생성기는 별도 저장소 ([`docs/GIT_ORGANIZATION.md`](docs/GIT_ORGANIZATION.md))
- [x] **최종 URDF로 처음부터 학습 (실습실 PC, 2026-11-03 02:10~11:20): 전진·후진·좌우·좌우회전 모두 명령을 따라 걷는 정책**
  - 정책: `nm5` 3.6M + 좌우 대칭 추론(`--symmetric`). 실습실 PC `checkpoints/nm5/2026_11_03_090557_3604480.onnx` (git에 없음)
  - 64대 학습환경 평가(밀기 포함): 여섯 방향 모두 명령의 0.59~0.84배, 8초 내 넘어짐 ~31%. 밀기 없으면 넘어지지 않음
  - 함께 고친 것: 서보 kp/forcerange를 원본 BAM 실측값(17.8/3.35)으로, `randomize.py` 몸통 ID 버그, 몸통 무게중심 흔들기 ±5→±1 cm
  - 재현 절차: [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md)
- [x] **머리 형상 수정 반영 (노트북, 2026-11-04, 브랜치 `model/head-v2`)**: 실물 조립 후 머리 형상 소폭 변경, 질량 2.072→2.075 kg
  - head_1만 변경 확인(질량 221.0→224.0g, 높이 91→56mm, 부피 −14.3%), 나머지 링크·조인트는 전부 동일 (cos=1.000 교차검증)
  - 전체 CoM 이동 3.52mm (X −3.34, Y ~0, Z −1.13) — 5mm 미만 기준으로는 재생성 불필요했지만, 사용자가 Placo에서 회전 동작을 새 머리 모델로 직접 보고 싶다고 해서 **레퍼런스 모션도 재생성함**: `polynomial_coefficients_omnidirectional_headv2_deg9.pkl` (격자·설정은 기존과 동일, degree=9, 0% 깨짐 확인). `joystick.py`가 이걸 가리킴
  - kp=17.8/forcerange=3.35, IMU·발 site, home 높이 전부 변화 없음 확인
  - 자세한 내용: [`docs/CHANGES_FROM_ORIGINAL.md`](docs/CHANGES_FROM_ORIGINAL.md) "Head v2"
- [x] **200M 장기 학습 (실습실 PC, 2026-11-04 03:00 ~ 11-06 03:20, 구간 long1~long14)** ([`docs/AUTO_EXPERIMENTS_2026-10-28.md`](docs/AUTO_EXPERIMENTS_2026-10-28.md) "Long run")
  - 최종 정책: `checkpoints/long14/2026_11_06_000746_7864320.onnx` (실습실 PC에만 있음). 좌우 대칭으로 학습(`--symmetric_policy`)했으므로 실행할 때 `--symmetric_trained`
  - 64대 학습환경 평가(넓힌 랜덤화·밀기 포함): 여섯 방향 명령의 0.76~0.98배, `|r-1|` 0.11 (`nm5`는 0.29). 몸통 앞 기울기 약 8° (`nm5` 약 11°). 밀기를 받으면 8초 안에 약 33% 넘어짐(변화 없음)
  - 원본과 달라진 점: `CHANGES_FROM_ORIGINAL.md` #12~#17 (회전 보상 저역통과, 레퍼런스를 실제 속도로 조회·보간, 몸통 기울기 벌점(후진 중 제외), action_rate -1.0, 랜덤화 확대, 최고 속도 0.2 m/s)
  - 레퍼런스 모션은 재생성하지 않기로 함(사용자 결정): 옆걸음과 회전(최대 0.11 rad/s)은 Placo 기준 동작의 한계에 묶여 있음
- [x] **실기 하드웨어 (노트북, 2026-11-04~05, 브랜치 `hardware/bringup`, master에 아직 병합 안 됨)**: 라즈베리파이 Zero 2 W, 서보 13개 ID·0점·방향, IMU(BNO055) 장착 방향, 디지털 트윈 도구. 자세한 내용은 그 브랜치의 `docs/HARDWARE_SETUP.md`, `docs/DIGITAL_TWIN.md`, 값은 `hardware/dcubot_hw_config.json`
  - 정책을 실물에서 돌릴 때 학습 쪽과 맞춰야 할 것:
    - 제어 주기 50 Hz(`ctrl_dt` 0.02), 목표각 = 기본자세 + 행동 × 0.25 rad, 목표각 변화는 5.24 rad/s로 제한
    - 관측(94)에 발 접촉 2개가 들어감 → 발 스위치가 필요함(브랜치 기준 배선 미확인)
    - 서보 3개 부호 반전(left_hip_yaw, left_hip_roll, head_yaw)과 IMU 축 변환은 실기 런타임에서만 함. 시뮬레이션·정책은 그대로
    - 서보 레지스터 P 32는 Open Duck 설정 스크립트와 같은 값이고, 시뮬레이션 kp 17.8도 Open Duck 값(BAM 식별)이다. P를 바꾸면 시뮬레이션과 달라지므로 그대로 둔다
    - 장기 학습 정책은 좌우 대칭 추론이 필요: `mujoco_infer.py`의 `SymmetricPolicy(pre_tanh=True)`
- [ ] **다음**:
  1. 실물 로봇 무게중심 확인: 전원을 켜고 관절 0°로 세워 앞으로 쏠리는지, 전체 무게(모델 2.075 kg), 라즈베리파이·전자부품 위치. 모델과 다르면 URDF 질량·무게중심을 고치고 최종 정책에서 이어서 학습
  2. 실물 실행기(노트북·라즈베리파이)에 위 항목을 반영하고, 최종 정책을 디지털 트윈으로 먼저 확인
  3. 최종 정책 파일은 git에 없으므로 노트북으로 복사: `scp <실습실PC>:~/dcubot/DCUBOT/rl_training/checkpoints/long14/2026_11_06_000746_7864320.onnx .`
