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
