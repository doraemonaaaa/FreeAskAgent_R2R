#!/usr/bin/env bash
# Evaluate R2R-CE with one independent v1 model worker per GPU.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="/data/pengyh/miniconda3/envs/habitat/bin/python"

# Edit these settings, then run: bash integrations/v1/run_r2r_ce_8gpu.sh
split="val_unseen"
scene_id="all"
episodes=0             # 0 evaluates every episode in the selected split.
max_steps=500
record_video=false     # Set true to save RGB + TopDownMap videos.
video_dir="${root_dir}/videos/r2r_8gpu"
output_dir="${root_dir}/outputs/r2r_ce_8gpu"
artifact_dir="${output_dir}/artifacts"
run_id="${R2R_RUN_ID:-r2r-8gpu-$(date -u +%Y%m%dT%H%M%SZ)}"
memory_mode="${MEMORY_MODE:-temporal}"

runner_args=(
  --split "${split}"
  --scene-id "${scene_id}"
  --episodes "${episodes}"
  --max-steps "${max_steps}"
  --artifact-dir "${artifact_dir}"
  --run-id "${run_id}"
  --memory-mode "${memory_mode}"
)
if [[ "${record_video}" == "true" ]]; then
  runner_args+=(--record-video --video-dir "${video_dir}")
fi

for gpu in $(seq 0 7); do
  if ! nvidia-smi --id="${gpu}" --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo "GPU ${gpu} is unavailable; this launcher requires GPUs 0-7." >&2
    exit 1
  fi
done

stop_previous_run() {
  local pattern="${root_dir}/integrations/v1/(r2r_ce_example|vln_agent_worker)\\.py"
  if ! pgrep -f "${pattern}" >/dev/null; then
    return
  fi

  echo "Stopping previous R2R evaluation processes..."
  pkill -TERM -f "${pattern}" || true
  for _ in $(seq 1 10); do
    pgrep -f "${pattern}" >/dev/null || return
    sleep 1
  done
  echo "Force-stopping remaining R2R evaluation processes..."
  pkill -KILL -f "${pattern}" || true
}

stop_previous_run

mkdir -p "${output_dir}"
rm -f "${output_dir}"/rank_{0,1,2,3,4,5,6,7}.json
pids=()

cleanup() {
  local status=$?
  if ((status != 0)); then
    echo "8-GPU evaluation stopped with exit status ${status}." >&2
  fi
  for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT INT TERM

for rank in $(seq 0 7); do
  # The worker inherits this rank's CUDA_VISIBLE_DEVICES. Do not pass
  # --gpu-id: resetting it to 0 would make every worker select physical GPU 0.
  (
    echo "rank=${rank} CUDA_VISIBLE_DEVICES=${rank}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${rank}" \
      "${python_bin}" "${root_dir}/integrations/v1/r2r_ce_example.py" \
      --rank "${rank}" --world-size 8 --output-dir "${output_dir}" \
      "${runner_args[@]}"
  ) >"${output_dir}/rank_${rank}.log" 2>&1 &
  pids+=("$!")
done

echo "Started 8 ranks; logs: ${output_dir}/rank_{0..7}.log"
for pid in "${pids[@]}"; do
  wait "${pid}"
done
"${python_bin}" "${root_dir}/integrations/aggregate_r2r_ce_results.py" \
  "${output_dir}" --world-size 8
trap - EXIT INT TERM
