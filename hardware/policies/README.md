# 실기 배포 정책

실기(`hardware/pi/policy_runner.py`)에 지금 올리는 **현재 배포 정책 하나만** 여기 둔다. `checkpoints/`(학습 중 스냅샷 수십~수백 개)는 여전히 git에 올리지 않는다 — 이 폴더는 그와 별개로, "지금 로봇에 실제로 쓰는 것" 하나만 추적한다. `playground/dcubot/data/*.pkl`(레퍼런스 모션)을 git에 올리는 것과 같은 이유다.

- 파일명은 `current.onnx`로 고정. 새 정책을 배포할 때는 덮어쓴다(이전 버전은 git 기록에 남는다).
- ONNX 하나당 수백 KB~1MB 정도라 git에 올려도 문제없다.

## 배포할 때 (실습실 PC)

```bash
cd ~/dcubot/DCUBOT/rl_training   # 저장소 루트
cp checkpoints/<실험이름>/<체크포인트>.onnx hardware/policies/current.onnx
git add hardware/policies/current.onnx
git commit -m "deploy: <실험이름>/<체크포인트> as current.onnx"
git push
```

## 받을 때 (라즈베리파이, 노트북)

```bash
git pull
```

**정책과 기본 자세(xml)는 짝이다.** `policy_runner.py`는 기본 자세를 `playground/dcubot/xmls/scene_flat_terrain.xml`의 home에서 읽는다. 정책을 학습한 때와 home이 다르면(예: 2026-11-12 long17부터 몸통을 세운 새 home) 같은 커밋에서 xml도 함께 바꾼다.

그 다음 `hardware/pi/policy_runner.py --onnx hardware/policies/current.onnx ...`로 실행한다.

**long18부터는 목을 고정해서 학습했다.** `policy_runner.py`가 기본으로 neck_pitch 목표를 home에 고정한다(`--fix_neck`). 목을 고정하지 않고 학습한 long17 이전 정책은 `--no-fix_neck`으로 실행한다.

**2026-11-16: long22 25.6M 시험 배포.** 좌우 다리 충돌(master #25)과 다리 간격 벌점(#26)으로 학습한 정책. 서 있을 때 제자리걸음이 long18보다 심하다(발 들림 시간 88% vs 17%). 문제가 있으면 long18 12.8M으로 되돌린다: `git show 4c78f4a:hardware/policies/current.onnx > hardware/policies/current.onnx`.

**2026-11-16: long25 8.5M 시험 배포.** 서 있을 때 제자리걸음 없음(시뮬), V자(hip_yaw -10도)는 남아 있음. xml에 발 충돌 hull과 발–정강이 충돌 쌍 추가. 되돌리기: long22 `git show 5e0432f:hardware/policies/current.onnx > hardware/policies/current.onnx`.

**2026-11-16: long26 17.0M 시험 배포.** hip_yaw 관절 범위를 ±4도로 좁혀 학습(master #29). xml의 hip_yaw 범위도 ±4도로 바뀌어서 `policy_runner`/`bridge_server`가 로봇에서도 ±4도로 자른다. 시뮬 hip_yaw 약 -4도(long25는 -10도). 되돌리기: long25 `git show ee819f4:hardware/policies/current.onnx > hardware/policies/current.onnx` + `git checkout ee819f4 -- playground/dcubot/xmls/`.

**2026-11-17: long26 25.6M(최종)으로 교체.** 17.0M보다 회전이 좋고(1.07/0.97 vs 0.92/0.84) 서 있을 때 발 들림 0%(17.0M 7.5%), 옆걸음은 약함(0.40/0.47 vs 0.49/0.55). xml(±4도)은 그대로. 17.0M으로 되돌리기: `git show 5e47081:hardware/policies/current.onnx > hardware/policies/current.onnx`.
