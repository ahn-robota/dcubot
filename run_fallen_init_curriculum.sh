#!/bin/bash
# Step 2: fallen-init curriculum, piecewise phases resuming from the Step 1
# checkpoint. Each phase raises fallen_init_config.prob (standing-mode envs
# only) and trains ~1,146,880 steps (7 of brax's 163,840-step chunks,
# num_evals=2 -> single real chunk run), then runs eval_fallen_recovery.py
# against the fresh checkpoint before starting the next phase. Stops early
# if a phase's eval flags barbie_doll_suspected.
set -e
cd /home/ahn/dcubot/DCUBOT/rl_training
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

OUT_DIR=$(pwd)/checkpoints_standing_joystick_merge
LOG=train_fallen_init_curriculum.log
EVAL_DIR=eval_results
mkdir -p "$EVAL_DIR"

# orbax's restore requires an absolute checkpoint path (relative path raised
# "Checkpoint path should be absolute" on this orbax version).
RESTORE=${OUT_DIR}/2026_10_18_104047_2129920
PROBS=(0.05 0.10 0.15 0.20)
STEPS=1146880

echo "$(date +%T) === fallen-init curriculum start, restore from $RESTORE ===" >> "$LOG"

for PROB in "${PROBS[@]}"; do
  echo "$(date +%T) === phase prob=$PROB, restore=$RESTORE ===" | tee -a "$LOG"

  BEFORE=$(find "$OUT_DIR" -maxdepth 1 -type d -name '2*_[0-9]*' | sort | tail -1)

  .venv/bin/python -u -m playground.dcubot.runner \
    --task flat_terrain_train \
    --output_dir "$OUT_DIR" \
    --num_timesteps "$STEPS" \
    --num_evals 2 \
    --restore_checkpoint_path "$RESTORE" \
    --fallen_init_prob "$PROB" >> "$LOG" 2>&1

  # Newest checkpoint dir (with matching .onnx) not equal to what existed before this phase.
  NEW_CKPT=$(find "$OUT_DIR" -maxdepth 1 -type d -name '2*_[0-9]*' | sort | tail -1)
  if [ "$NEW_CKPT" == "$BEFORE" ]; then
    echo "$(date +%T) !!! phase prob=$PROB produced no new checkpoint dir, aborting" | tee -a "$LOG"
    exit 1
  fi
  ONNX="${NEW_CKPT}.onnx"
  echo "$(date +%T) phase prob=$PROB done -> $ONNX" | tee -a "$LOG"

  TAG="fallen_recovery_prob${PROB}"
  echo "$(date +%T) === eval_fallen_recovery.py (tag=$TAG) ===" | tee -a "$LOG"
  .venv/bin/python tools/eval_fallen_recovery.py -o "$ONNX" \
    --out_dir "$EVAL_DIR" --tag "$TAG" >> "$LOG" 2>&1

  BARBIE=$(.venv/bin/python -c "import json; print(json.load(open('${EVAL_DIR}/${TAG}.json'))['barbie_doll_suspected'])")
  echo "$(date +%T) phase prob=$PROB barbie_doll_suspected=$BARBIE" | tee -a "$LOG"
  if [ "$BARBIE" == "True" ]; then
    echo "$(date +%T) !!! BARBIE DOLL SUSPECTED at prob=$PROB -- stopping curriculum early for review" | tee -a "$LOG"
    exit 2
  fi

  RESTORE="$NEW_CKPT"
done

echo "$(date +%T) === fallen-init curriculum complete, final checkpoint: $RESTORE ===" | tee -a "$LOG"
