#!/usr/bin/env bash
# Start one model service exactly as integrations/v3/config.yaml describes it.
#   integrations/v3/serve_service.sh qwen3-vl-8b      # -> serve_vllm.sh
#   integrations/v3/serve_service.sh joyai            # -> FreeAskAgent/scripts/start_joyai_captioner.sh
# Variables already exported by the caller win over the yaml (GPU=3 ... to
# move a service). DRY_RUN=1 prints the resolved environment and command.
set -euo pipefail
service="${1:?usage: serve_service.sh <service name from config.yaml>}"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
freeask_dir="${FREEASK_DIR:-/data/pengyh/workspace/FreeAskAgent}"
habitat_python="${HABITAT_PYTHON:-/data/pengyh/miniconda3/envs/habitat/bin/python}"
config_file="${V3_CONFIG:-${root_dir}/integrations/v3/config.yaml}"

kind="$("${habitat_python}" - "${config_file}" "${service}" <<'PY'
import sys, yaml
print((yaml.safe_load(open(sys.argv[1]))["services"][sys.argv[2]]).get("kind", "vllm"))
PY
)"
while IFS= read -r line; do
  key="${line%%=*}"
  if [[ -z "${!key:-}" ]]; then
    # Values are shell-quoted by run_config.py; eval strips the quoting.
    eval "export ${line}"
  fi
done < <("${habitat_python}" "${root_dir}/integrations/v3/run_config.py" --config "${config_file}" --serve "${service}")

if [[ "${kind}" == "joyai" ]]; then
  script="${freeask_dir}/scripts/start_joyai_captioner.sh"
else
  script="${root_dir}/integrations/v3/serve_vllm.sh"
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  env | grep -E '^(SERVED_NAME|PORT|MODEL_PATH|GPU|GPU_UTIL|TP|MAX_MODEL_LEN|MAX_NUM_SEQS|MAX_IMAGES|JOYAI_[A-Z_]+)=' | sort
  echo "exec ${script}"
  exit 0
fi
exec "${script}"
