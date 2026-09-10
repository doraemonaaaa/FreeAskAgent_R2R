#!/usr/bin/env bash
# Full R2R-CE val_unseen (1839 episodes) with the -15 deg camera pitch, Qwen3-VL-4B, every episode recorded.
#   SERVERS: "gpu:port ..." vLLM 4B servers to (re)use; RANK_GPUS: habitat GPU per rank, comma list (its length = world size)
set -u
ROOT=/data/pengyh/workspace/FreeAskAgent_R2R; FREEASK=/data/pengyh/workspace/FreeAskAgent; M=$FREEASK/models
NAME=${NAME:-full_val_unseen_pitch15_4b}; OUT=$ROOT/outputs/$NAME; mkdir -p $OUT/logs $OUT/videos $OUT/rank_json $OUT/evidence
SERVERS=${SERVERS:-"2:8300 3:8301"}
RANK_GPUS=${RANK_GPUS:-"4,4,5,5,6,6,7,7"}
PITCH=${PITCH:--15}; MAX_STEPS=${MAX_STEPS:-150}
JOYAI_MODEL_NAME=${JOYAI_MODEL_NAME:-jdopensource/JoyAI-VL-Interaction}
# Two endpoints keep eight ranks at four clients per JoyAI server. Override on
# shared hosts, for example JOYAI_SERVERS="1:7060 0:7061".
JOYAI_SERVERS=${JOYAI_SERVERS:-"1:7060 0:7061"}
cd $ROOT
# 1. JoyAI Captioner service
joyai_bases=()
for s in $JOYAI_SERVERS; do
  jg=${s%%:*}; jp=${s##*:}; jb="http://127.0.0.1:$jp/v1"; joyai_bases+=("$jb")
  if ! curl -s --max-time 2 "$jb/models" | grep -q "$JOYAI_MODEL_NAME"; then
    setsid nohup env JOYAI_PORT="$jp" JOYAI_GPU="$jg" \
      JOYAI_GPU_MEMORY_UTILIZATION="${JOYAI_GPU_MEMORY_UTILIZATION:-0.28}" \
      JOYAI_MAX_MODEL_LEN="${JOYAI_MAX_MODEL_LEN:-16384}" \
      "$FREEASK/scripts/start_joyai_captioner.sh" \
      > "$OUT/logs/joyai_gpu${jg}_${jp}.log" 2>&1 &
  fi
done
for jb in "${joyai_bases[@]}"; do
  joyai_ready=0
  for i in $(seq 1 240); do
    if curl -s --max-time 2 "$jb/models" | grep -q "$JOYAI_MODEL_NAME"; then joyai_ready=1; break; fi
    sleep 5
  done
  if [[ "$joyai_ready" != 1 ]]; then echo "JoyAI Captioner failed to start on $jb" >&2; exit 1; fi
  echo "JoyAI Captioner ready on $jb $(date)"
done

# 2. planning/waypoint servers
ports=()
for s in $SERVERS; do g=${s%%:*}; p=${s##*:}; ports+=($p)
  if ! curl -s --max-time 2 http://127.0.0.1:$p/v1/models | grep -q qwen3-vl-4b; then
    GPU=$g TP=1 PORT=$p GPU_UTIL=${GPU_UTIL:-0.28} SERVED_NAME=qwen3-vl-4b MODEL_PATH=$M/Qwen3-VL-4B-Instruct setsid nohup integrations/v3/serve_vllm.sh > $OUT/logs/vllm_gpu${g}_$p.log 2>&1 &
  fi
done
for p in "${ports[@]}"; do for i in $(seq 1 240); do curl -s --max-time 2 http://127.0.0.1:$p/v1/models | grep -q qwen3-vl-4b && break; sleep 5; done; echo "server $p ready $(date)"; done
# 3. ranks
IFS="," read -ra gpus <<< "$RANK_GPUS"; W=${#gpus[@]}; echo "world size $W, servers ${ports[*]}"
for rank in $(seq 0 $((W-1))); do
  g=${gpus[$rank]}; p=${ports[$((rank % ${#ports[@]}))]}
  jb=${joyai_bases[$((rank % ${#joyai_bases[@]}))]}
  ( VLN_SPATIAL_MEMORY=1 VLLM_BASE_URL=http://127.0.0.1:$p/v1 JOYAI_API_BASE="$jb" JOYAI_EVIDENCE_DIR="$OUT/evidence/rank_$(printf %02d $rank)" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    /data/pengyh/miniconda3/envs/habitat/bin/python integrations/v3/run_habitat.py --split val_unseen --episodes 0 \
    --world-size $W --rank $rank --max-steps $MAX_STEPS --camera-pitch-deg $PITCH --debug-memory \
    --gpu-id 0 --actor-python /data/pengyh/workspace/FreeAskAgent/.venv/bin/python --model-path vllm-qwen3-vl-4b \
    --record-video --video-dir $OUT/videos --output-dir $OUT/rank_json \
    > $OUT/logs/rank_$(printf %02d $rank).log 2>&1 ) &
  sleep 2
done
wait
echo "ALL RANKS EXITED $(date)"
