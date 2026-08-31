#!/usr/bin/env bash
# Full R2R-CE val_unseen (1839 episodes) with the -15 deg camera pitch, Qwen3-VL-4B, every episode recorded.
#   SERVERS: "gpu:port ..." vLLM 4B servers to (re)use; RANK_GPUS: habitat GPU per rank, comma list (its length = world size)
set -u
ROOT=/data/pengyh/workspace/FreeAskAgent_R2R; M=/data/pengyh/workspace/FreeAskAgent/models
NAME=${NAME:-full_val_unseen_pitch15_4b}; OUT=$ROOT/outputs/$NAME; mkdir -p $OUT/logs $OUT/videos $OUT/rank_json
SERVERS=${SERVERS:-"2:8300 3:8301 0:8302"}
RANK_GPUS=${RANK_GPUS:-"4,4,4,4,4,4,5,5,5,5,5,5,7,7,7,7,7,7,6,6,6,2,2,3"}
PITCH=${PITCH:--15}; MAX_STEPS=${MAX_STEPS:-150}
cd $ROOT
# 1. servers
ports=()
for s in $SERVERS; do g=${s%%:*}; p=${s##*:}; ports+=($p)
  if ! curl -s --max-time 2 http://127.0.0.1:$p/v1/models | grep -q qwen3-vl-4b; then
    GPU=$g TP=1 PORT=$p GPU_UTIL=${GPU_UTIL:-0.28} SERVED_NAME=qwen3-vl-4b MODEL_PATH=$M/Qwen3-VL-4B-Instruct setsid nohup integrations/v3/serve_vllm.sh > $OUT/logs/vllm_gpu${g}_$p.log 2>&1 &
  fi
done
for p in "${ports[@]}"; do for i in $(seq 1 240); do curl -s --max-time 2 http://127.0.0.1:$p/v1/models | grep -q qwen3-vl-4b && break; sleep 5; done; echo "server $p ready $(date)"; done
# 2. ranks
IFS="," read -ra gpus <<< "$RANK_GPUS"; W=${#gpus[@]}; echo "world size $W, servers ${ports[*]}"
for rank in $(seq 0 $((W-1))); do
  g=${gpus[$rank]}; p=${ports[$((rank % ${#ports[@]}))]}
  ( VLN_SPATIAL_MEMORY=1 VLLM_BASE_URL=http://127.0.0.1:$p/v1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    /data/pengyh/miniconda3/envs/habitat/bin/python integrations/v3/run_habitat.py --split val_unseen --episodes 0 \
    --world-size $W --rank $rank --max-steps $MAX_STEPS --camera-pitch-deg $PITCH --debug-memory \
    --gpu-id 0 --actor-python /data/pengyh/workspace/FreeAskAgent/.venv/bin/python --model-path vllm-qwen3-vl-4b \
    --record-video --video-dir $OUT/videos --output-dir $OUT/rank_json \
    > $OUT/logs/rank_$(printf %02d $rank).log 2>&1 ) &
  sleep 2
done
wait
echo "ALL RANKS EXITED $(date)"
