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
| `playground/dcubot/` | 디쿠봇 환경: `joystick.py`(보상·관측·명령 범위), `runner.py`(학습 진입점), `xmls/`(MJCF) |
| `playground/dcubot/data/` | 레퍼런스 모션. `polynomial_coefficients_omnidirectional.pkl` 하나만 사용 |
| `playground/common/` | 원본 Open Duck 공용 코드: 보상, 레퍼런스 모션, 도메인 랜덤화, ONNX 내보내기, 공용 러너 |
| `scripts/` | 검증·평가 스크립트 (레퍼런스 점검/재생, 방향 데모, 외란 평가) |
| `docs/` | 원본과 달라진 점, 검증 결과 |
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

```bash
uv sync                                              # uv.lock 기준 환경 구성
uv run python -c "import jax; print(jax.devices())"  # CudaDevice 확인

# 학습 (--num_envs, MEM_FRACTION은 위 표의 PC별 값)
XLA_PYTHON_CLIENT_MEM_FRACTION=<비율> uv run python -u -m playground.dcubot.runner \
  --task flat_terrain_train --num_envs <N> \
  --output_dir checkpoints/<실험이름> --num_timesteps <스텝>

# 이어서 학습: 위 명령에 --restore_checkpoint_path checkpoints/<실험이름>/<체크포인트> 추가
uv run tensorboard --logdir checkpoints/<실험이름>
```

짧은 학습은 `--num_evals`를 낮춰야 목표 스텝 근처에서 멈춘다(이유는 `playground/dcubot/runner.py`의 `DCUBOT_NUM_EVALS` 주석).

## 검증 기준

노트북 검증 결과(obs·action 차원, 레퍼런스 재생, 짧은 검증 학습 곡선)와 실행 명령: [`docs/VERIFICATION_2026-10-27.md`](docs/VERIFICATION_2026-10-27.md). 새 PC나 환경 변경 후에는 같은 명령으로 돌려서 이 문서와 비교한다.

## 현재 진행 상황과 다음 할 일

- [x] 원점 재시작: 원본 `open_duck_mini_v2` 구조로 `joystick.py` 재작성, 전방향 레퍼런스 모션 연결
- [x] 노트북 검증: 레퍼런스 재생, 짧은 전방향 검증 학습(3.6M 스텝) 통과
- [x] `uv.lock` 고정, `--num_envs` 인자, 전진 전용 pkl 삭제
- [x] 실습실 PC 환경 구성 및 노트북 검증 결과와 비교 (일치, `docs/VERIFICATION_2026-10-27.md` 하단)
- [x] 실습실 PC `num_envs` 결정: 1024 (2048은 메모리 부족, 자세한 내용은 `docs/VERIFICATION_2026-10-27.md` 하단)
- [x] 실습실 PC 1024 처리량 측정: 768 steps/s (노트북 495) → 50M 약 18시간
- [ ] 실습실 PC 본 학습 **진행 중**: `omni_1024_50m`, 2026-10-27 15:21 시작, 예상 종료 10-28 오전 9~10시
  - 설정: 50M 스텝, `--num_envs 1024`, `--num_evals` 기본 30, MEM_FRACTION 0.85, 그 외는 노트북 검증과 동일
  - 실습실 PC tmux 세션 `train`: 창 `train`(학습), 창 `tensorboard`(포트 6006)
  - 로그·체크포인트: `checkpoints/omni_1024_50m/` (`train.log`에 STEP 줄)
