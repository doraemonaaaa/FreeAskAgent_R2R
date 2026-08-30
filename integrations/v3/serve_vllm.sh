#!/usr/bin/env bash
# Serve Qwen3-VL for the vln_agent_4 workers through vLLM's OpenAI API.
#
#   GPU=0 PORT=8100 integrations/v3/serve_vllm.sh
#
# Then run the evaluator with:
#   VLLM_BASE_URL=http://127.0.0.1:8100/v1 ... run_habitat.py --model-path vllm-qwen3-vl-8b
#
# Prefix caching is on: every request shares the long system prompt and the
# per-stage header, so only the frames and the tail text are re-prefilled.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
agent_root="${AGENT_ROOT:-/data/pengyh/workspace/FreeAskAgent}"
vllm_bin="${VLLM_BIN:-${agent_root}/.venv-vllm/bin/vllm}"
model_path="${MODEL_PATH:-${agent_root}/models/Qwen3-VL-8B-Instruct}"
served_name="${SERVED_NAME:-qwen3-vl-8b}"
gpu="${GPU:-0}"
port="${PORT:-8100}"
# Fraction of the whole card: weights are ~17 GB, the rest is KV cache.
gpu_util="${GPU_UTIL:-0.30}"
max_len="${MAX_MODEL_LEN:-8192}"
max_images="${MAX_IMAGES:-16}"
max_seqs="${MAX_NUM_SEQS:-16}"
# Tensor parallel size: GPU may be a comma list (e.g. GPU=2,3 TP=2) for a
# checkpoint that does not fit one card, such as Qwen3-VL-32B-Instruct-FP8.
tp="${TP:-1}"
# Some checkpoints (InternVL) ship their own modelling code.
trust_remote="${TRUST_REMOTE_CODE:-0}"
extra_args=()
if [[ "${trust_remote}" == "1" ]]; then extra_args+=(--trust-remote-code); fi
# VLLM_EXTRA_ARGS: extra CLI flags, e.g. --mm-processor-kwargs '{"max_dynamic_patch": 1}' for InternVL
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then eval "extra_args+=(${VLLM_EXTRA_ARGS})"; fi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu}" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
exec "${vllm_bin}" serve "${model_path}" \
  --served-model-name "${served_name}" \
  --host 127.0.0.1 --port "${port}" \
  --dtype bfloat16 \
  --tensor-parallel-size "${tp}" \
  --max-model-len "${max_len}" \
  --max-num-seqs "${max_seqs}" \
  --limit-mm-per-prompt "{\"image\": ${max_images}}" \
  --gpu-memory-utilization "${gpu_util}" \
  --enable-prefix-caching \
  --no-enable-log-requests \
  "${extra_args[@]}"
