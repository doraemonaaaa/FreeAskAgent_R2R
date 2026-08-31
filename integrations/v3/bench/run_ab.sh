#!/usr/bin/env bash
# usage: run_ab.sh <name> <VLN_SPATIAL_MEMORY 0|1> <habitat_gpu>
set -u
name=$1; sp=$2; gpu=$3
IFS="," read -ra gpus <<< "$gpu"   # e.g. 4,5 -> ranks round-robin over the cards
S=${AB_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)/outputs/experiments}
out=$S/ab/$name; mkdir -p $out
cd /data/pengyh/workspace/FreeAskAgent_R2R
for rank in 0 1 2 3 4 5 6 7; do
  g=${gpus[$((rank % ${#gpus[@]}))]}
  ( VLN_SPATIAL_MEMORY=$sp VLLM_BASE_URL=${VLLM_URL:-http://127.0.0.1:8100/v1} CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    /data/pengyh/miniconda3/envs/habitat/bin/python integrations/v3/run_habitat.py --split val_unseen \
    --episode-ids @integrations/v3/eval_sets/val_unseen_40.txt --world-size 8 --rank $rank --max-steps 150 --camera-pitch-deg ${CAMERA_PITCH_DEG:--15} --debug-memory ${EXTRA_ARGS:-} \
    --gpu-id 0 --actor-python /data/pengyh/workspace/FreeAskAgent/.venv/bin/python --model-path vllm-${MODEL_NAME:-qwen3-vl-8b} \
    --output-dir $out > $out/rank_$rank.log 2>&1 ) &
done
wait
grep -h "^rank=" $out/rank_*.log | grep -c "id="
