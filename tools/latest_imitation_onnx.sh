#!/bin/bash
# Pick the newest DCUBOT imitation checkpoint (ONNX is exported at every eval) and print the viewer command.
cd "$(dirname "$0")/.."
DIR=${1:-checkpoints_dcubot_imitation}
LATEST=$(ls -t "$DIR"/*.onnx 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
  echo "No ONNX yet in $DIR (first eval is around 0.67M steps)."
  exit 1
fi
.venv/bin/python -c "
import onnxruntime as ort, sys
s = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
print('check:', [(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])
" "$LATEST" || exit 1
STEP=$(basename "$LATEST" .onnx | awk -F_ '{print $NF}')
echo "latest: $LATEST  (step $STEP, $(python3 -c "print(f'{$STEP/2e7*100:.1f}%')" 2>/dev/null) of 20M)"
echo "view with:"
echo "  cd ~/dcubot/Open_Duck_Playground && .venv/bin/python -m playground.dcubot.mujoco_infer -o $LATEST"
echo "or keep the viewer open and auto-switch to each new checkpoint:"
echo "  cd ~/dcubot/Open_Duck_Playground && .venv/bin/python -m playground.dcubot.mujoco_infer --watch_dir $DIR"
