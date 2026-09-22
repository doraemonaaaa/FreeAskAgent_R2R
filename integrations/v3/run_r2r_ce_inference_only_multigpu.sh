#!/usr/bin/env bash
# v19 inference-only R2R-CE evaluation. Every model/agent/captioner setting
# comes from integrations/v3/config.yaml (override the file with V3_CONFIG=...).
# Only run-shape knobs are taken from the environment:
#   SPLIT EPISODES EPISODE_IDS MAX_STEPS RANK_GPUS   (default: the yaml eval/runner sections)
#   ACTION_SPACE=waypoint|discrete                   (default: the yaml navigation section)
#   TRACE_JSONL=1 EVIDENCE_ARCHIVE_DIR VLN_FROZEN_PLAN_FILE R2R_RUN_ID OUTPUT_DIR
# Any other exported VLN_*/JOYAI_*/CAPTIONER_* variable still wins over the
# yaml (run_habitat.py only fills unset variables), which is how A/B scripts
# flip one switch without editing the file.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
freeask_dir="${FREEASK_DIR:-/data/pengyh/workspace/FreeAskAgent}"
habitat_python="${HABITAT_PYTHON:-/data/pengyh/miniconda3/envs/habitat/bin/python}"
actor_python="${ACTOR_PYTHON:-${freeask_dir}/.venv/bin/python}"
config_file="${V3_CONFIG:-${root_dir}/integrations/v3/config.yaml}"
# Every endpoint is loopback; a shell-level HTTP proxy would answer 502 for them.
export no_proxy='*' NO_PROXY='*'

# CFG_* : resolved model roles (CFG_<ROLE>_MODEL, CFG_<ROLE>_URLS) and eval defaults.
eval "$("${habitat_python}" "${root_dir}/integrations/v3/run_config.py" --config "${config_file}" --shell)"

split="${SPLIT:-${CFG_EVAL_SPLIT:-val_unseen}}"
episodes="${EPISODES:-${CFG_EVAL_EPISODES:-0}}"
episode_ids="${EPISODE_IDS:-${CFG_EVAL_EPISODE_SET:+@${root_dir}/${CFG_EVAL_EPISODE_SET}}}"
max_steps="${MAX_STEPS:-${CFG_RUNNER_MAX_STEPS:-500}}"
rank_gpus_csv="${RANK_GPUS:-${CFG_EVAL_RANK_GPUS:-4,5,6,7}}"
trace_jsonl="${TRACE_JSONL:-0}"
action_space="${ACTION_SPACE:-}"
case "${action_space}" in
  ""|waypoint|discrete) ;;
  *) echo "ACTION_SPACE must be waypoint or discrete, not '${action_space}'." >&2; exit 2 ;;
esac
frozen_plan_file="${VLN_FROZEN_PLAN_FILE:-}"
evidence_archive_dir="${EVIDENCE_ARCHIVE_DIR:-}"
gpu_monitor_interval_s="${GPU_MONITOR_INTERVAL_S:-10}"
worker_stagger_s="${WORKER_STAGGER_S:-2}"
run_id="${R2R_RUN_ID:-inference-only-v19-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-${root_dir}/outputs/${run_id}}"

IFS=',' read -r -a rank_gpus <<<"${rank_gpus_csv}"
world_size="${#rank_gpus[@]}"
if ((world_size < 1)); then
  echo "RANK_GPUS (or eval.rank_gpus in ${config_file}) must name at least one GPU." >&2
  exit 2
fi
if ((world_size > 8)) && [[ "${ALLOW_ENDPOINT_OVERSUBSCRIPTION:-0}" != "1" ]]; then
  echo "Refusing ${world_size} rollout workers: the current v19 topology was validated with at most eight ranks." >&2
  echo "Use more model replicas or set ALLOW_ENDPOINT_OVERSUBSCRIPTION=1 after a measured capacity test." >&2
  exit 2
fi

mkdir -p "${output_dir}"
if compgen -G "${output_dir}/rank_*.log" >/dev/null; then
  echo "Refusing to overwrite an existing run: ${output_dir}" >&2
  exit 2
fi

check_model() {
  local url="$1" expected="$2"
  "${actor_python}" - "$url" "$expected" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1].rstrip('/') + '/models', timeout=5) as response:
    served = {item["id"] for item in json.load(response)["data"]}
if sys.argv[2] not in served:
    raise SystemExit(f"{sys.argv[1]} serves {sorted(served)}, not {sys.argv[2]!r}")
PY
}
for role in OBSERVER DECISION PLANNER ACTOR SOM; do
  model_var="CFG_${role}_MODEL"; urls_var="CFG_${role}_URLS"
  [[ -n "${!model_var:-}" ]] || continue
  IFS=',' read -r -a urls <<<"${!urls_var}"
  for url in "${urls[@]}"; do check_model "$url" "${!model_var}"; done
  echo "${role,,}: ${!model_var} @ ${!urls_var}"
done

cp "${config_file}" "${output_dir}/config.yaml"
"${habitat_python}" - "${output_dir}/manifest.json" "${config_file}" <<PY
import json, pathlib, sys, hashlib
sys.path.insert(0, "${root_dir}/integrations/v3")
from run_config import RunConfig
resolved = RunConfig.load(sys.argv[2]).manifest()
frozen = "${frozen_plan_file}"
resolved.update({
  "mode": "inference_only_completion_confirmation",
  "split": "${split}", "episodes": int("${episodes}"),
  "episode_ids": "${episode_ids}", "max_steps": int("${max_steps}"),
  "world_size": ${world_size}, "rank_gpus": "${rank_gpus_csv}".split(','),
  "trace_jsonl": "${trace_jsonl}" == "1",
  "evidence_archive_dir": "${evidence_archive_dir}" or None,
  "frozen_plan_file": frozen or None,
  "frozen_plan_sha256": hashlib.sha256(pathlib.Path(frozen).read_bytes()).hexdigest() if frozen else None,
  "scheduling": "${world_size} sticky rollout workers; rank r uses base_urls[r % n] of each role",
  "gpu_monitor_interval_s": float("${gpu_monitor_interval_s}"),
})
pathlib.Path(sys.argv[1]).write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n")
PY

pids=()
monitor_pid=""
stop_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill -TERM "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
  fi
}
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  stop_monitor
  exit "$status"
}
trap cleanup EXIT INT TERM

if command -v nvidia-smi >/dev/null 2>&1; then
  (
    echo "timestamp,gpu_index,utilization_gpu_percent,utilization_memory_percent,memory_used_mib"
    while true; do
      timestamp="$(date --iso-8601=seconds)"
      nvidia-smi \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used \
        --format=csv,noheader,nounits |
        while IFS= read -r sample; do echo "${timestamp},${sample}"; done
      sleep "${gpu_monitor_interval_s}"
    done
  ) >"${output_dir}/gpu_utilization.csv" 2>"${output_dir}/gpu_monitor.log" &
  monitor_pid="$!"
fi

for rank in $(seq 0 $((world_size - 1))); do
  gpu="${rank_gpus[$rank]}"
  (
    extra_args=()
    if [[ "${trace_jsonl}" == "1" ]]; then
      extra_args+=(--trace-jsonl "${output_dir}/rank_${rank}_trace.jsonl")
    fi
    if [[ "${action_space}" == "discrete" ]]; then
      extra_args+=(--discrete-action-space)
    elif [[ "${action_space}" == "waypoint" ]]; then
      extra_args+=(--no-discrete-action-space)
    fi
    echo "rank=${rank} physical_gpu=${gpu} config=${config_file}"
    # Model endpoints, captioner switches and agent flags are filled from the
    # yaml inside run_habitat.py (per rank); only run-shape values are passed here.
    if [[ -n "${evidence_archive_dir}" ]]; then export JOYAI_EVIDENCE_DIR="${evidence_archive_dir}"; fi
    if [[ -n "${frozen_plan_file}" ]]; then export VLN_FROZEN_PLAN_FILE="${frozen_plan_file}"; fi
    PYTHONPATH="${freeask_dir}:${root_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True \
      "${habitat_python}" "${root_dir}/integrations/v3/run_habitat.py" \
      --config "${config_file}" \
      --split "${split}" --episodes "${episodes}" --max-steps "${max_steps}" \
      ${episode_ids:+--episode-ids "${episode_ids}"} \
      --world-size "${world_size}" --rank "${rank}" \
      --actor-python "${actor_python}" \
      --output-dir "${output_dir}" "${extra_args[@]}"
  ) >"${output_dir}/rank_${rank}.log" 2>&1 &
  pids+=("$!")
  sleep "${worker_stagger_s}"
done

printf '%s\n' "${pids[@]}" >"${output_dir}/rank_pids.txt"
failed=()
for rank in $(seq 0 $((world_size - 1))); do
  if ! wait "${pids[$rank]}"; then failed+=("$rank"); fi
done
if ((${#failed[@]})); then
  echo "Failed ranks: ${failed[*]}" | tee "${output_dir}/failure.log" >&2
  exit 1
fi
stop_monitor

PYTHONPATH="${root_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
"${habitat_python}" "${root_dir}/integrations/aggregate_r2r_ce_results.py" \
  "${output_dir}" --world-size "${world_size}" | tee "${output_dir}/aggregate.log"
trap - EXIT INT TERM
echo "Completed: ${output_dir}"
