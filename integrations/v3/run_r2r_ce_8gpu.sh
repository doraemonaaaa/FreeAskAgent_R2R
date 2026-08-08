#!/usr/bin/env bash
# Full val_unseen R2R-CE evaluation for vln_agent_3, one shard per GPU.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
habitat_python="${HABITAT_PYTHON:-/data/pengyh/miniconda3/envs/habitat/bin/python}"
actor_python="${ACTOR_PYTHON:-/data/pengyh/workspace/FreeAskAgent/.venv/bin/python}"
model_path="${MODEL_PATH:-/data/pengyh/workspace/FreeAskAgent/models/Qwen3-VL-8B-Instruct}"

split="${SPLIT:-val_unseen}"
scene_id="${SCENE_ID:-all}"
episodes="${EPISODES:-0}"
max_steps="${MAX_STEPS:-200}"
world_size="${WORLD_SIZE:-8}"
record_video="${RECORD_VIDEO:-true}"
run_id="${R2R_RUN_ID:-vln-agent-3-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-${root_dir}/outputs/vln_agent_3_8gpu/${run_id}}"
video_dir="${VIDEO_DIR:-${root_dir}/videos/vln_agent_3_8gpu/${run_id}}"

runner_args=(
  --split "${split}"
  --scene-id "${scene_id}"
  --episodes "${episodes}"
  --max-steps "${max_steps}"
  --world-size "${world_size}"
  --model-path "${model_path}"
  --actor-python "${actor_python}"
  --output-dir "${output_dir}"
)
if [[ "${record_video}" == "true" ]]; then
  runner_args+=(--record-video --video-dir "${video_dir}")
fi

for gpu in $(seq 0 $((world_size - 1))); do
  if ! nvidia-smi --id="${gpu}" --query-gpu=name \
      --format=csv,noheader >/dev/null 2>&1; then
    echo "GPU ${gpu} is unavailable; WORLD_SIZE=${world_size}." >&2
    exit 1
  fi
done

mkdir -p "${output_dir}"
pids=()

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

for rank in $(seq 0 $((world_size - 1))); do
  (
    echo "rank=${rank} physical_gpu=${rank}"
    # Habitat sees one GPU as logical 0. The actor worker receives the
    # physical rank explicitly because the v3 runner creates a fresh process.
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${rank}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
      "${habitat_python}" "${root_dir}/integrations/v3/run_habitat.py" \
      --rank "${rank}" \
      --gpu-id "${rank}" \
      "${runner_args[@]}"
  ) >"${output_dir}/rank_${rank}.log" 2>&1 &
  pids+=("$!")
done

echo "Started ${world_size} vln_agent_3 ranks."
echo "Logs: ${output_dir}/rank_{0..$((world_size - 1))}.log"
failed_ranks=()
for rank in $(seq 0 $((world_size - 1))); do
  if ! wait "${pids[$rank]}"; then
    failed_ranks+=("${rank}")
    echo "rank=${rank} failed; other ranks will continue." >&2
  fi
done

if ((${#failed_ranks[@]} > 0)); then
  echo "Failed ranks: ${failed_ranks[*]}" >&2
  echo "Successful rank outputs were preserved in ${output_dir}." >&2
  exit 1
fi

"${habitat_python}" \
  "${root_dir}/integrations/aggregate_r2r_ce_results.py" \
  "${output_dir}" --world-size "${world_size}" \
  | tee "${output_dir}/aggregate.log"

trap - EXIT INT TERM
echo "Completed: ${output_dir}"
