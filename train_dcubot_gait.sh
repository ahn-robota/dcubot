#!/bin/bash
# dcubot gait training (phase obs + feet_gait). Uses the low-poly mesh task so
# MJX fits in the 6GB GPU; JAX may take 95% of GPU memory (TF export runs on CPU).
cd /home/ahn/dcubot/DCUBOT/rl_training
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
OUT=${1:-checkpoints_dcubot_gait}
STEPS=${2:-20000000}
.venv/bin/python -u -m playground.dcubot.runner --task flat_terrain_train \
  --output_dir "$OUT" --num_timesteps "$STEPS" > train_gait.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  echo "$(date +%T) rss_MB=$(ps -o rss= -p $PID | awk '{print int($1/1024)}') gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')" >> train_monitor.log
  sleep 30
done
wait $PID; echo "$(date +%T) EXIT code=$?" >> train_monitor.log
