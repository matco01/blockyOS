#!/usr/bin/env bash
# Starts llama-server for the configured model. Runs under systemd (blocky-llm.service).
#
# Placement (which parts go on the GPU vs system RAM) is left to llama.cpp's
# --fit, which measures free memory and also handles MoE models (keeping
# attention on the GPU and spilling experts to RAM). We only cap the context.
#
# Settings (in /etc/blocky/blocky.env):
#   BLOCKY_MODEL        path to the .gguf to run (default: first one in the models dir)
#   BLOCKY_CTX          context window in tokens (default 16384)
#   BLOCKY_MAX_VRAM_MB  pretend the GPU only has this much memory (to test weaker cards)
#   BLOCKY_KV           KV cache type: f16 | q8_0 | q4_0 (default q8_0)
#   BLOCKY_NGL          GPU layers: auto (let --fit decide) or a number
#   BLOCKY_THREADS      CPU threads: auto or a number
# The Models page writes /opt/blocky/data/llm.env, which systemd loads after blocky.env.
set -euo pipefail

LLAMA_SERVER="${BLOCKY_LLAMA_SERVER:-/opt/blocky/llama.cpp/bin/llama-server}"
export LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MODELS_DIR="${BLOCKY_MODELS_DIR:-/opt/blocky/models}"
MODEL="${BLOCKY_MODEL:-$(ls -1 "$MODELS_DIR"/*.gguf 2>/dev/null | head -1 || true)}"
CTX="${BLOCKY_CTX:-16384}"

if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "No .gguf model found (BLOCKY_MODEL=${BLOCKY_MODEL:-unset}, dir=$MODELS_DIR)" >&2
  exit 1
fi

fit_args=(--fit on)
vram_mb=0
if command -v nvidia-smi >/dev/null 2>&1; then
  vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)
fi
if [[ -n "${BLOCKY_MAX_VRAM_MB:-}" && "$vram_mb" -gt "$BLOCKY_MAX_VRAM_MB" ]]; then
  # Leave the difference unused, so the model is placed as if the card were smaller.
  fit_args+=(--fit-target "$(( vram_mb - BLOCKY_MAX_VRAM_MB ))")
  vram_mb=$BLOCKY_MAX_VRAM_MB
fi

threads=$(( $(nproc) / 2 )); (( threads < 1 )) && threads=1   # physical cores ≈ half of logical
[[ "${BLOCKY_THREADS:-auto}" =~ ^[0-9]+$ ]] && threads=$BLOCKY_THREADS

KV="${BLOCKY_KV:-q8_0}"
[[ "$KV" =~ ^(f16|q8_0|q4_0)$ ]] || KV=q8_0

gpu_args=()
if [[ "${BLOCKY_NGL:-auto}" =~ ^[0-9]+$ ]]; then
  gpu_args=(--n-gpu-layers "$BLOCKY_NGL")
fi

echo "blocky-llm: model=$(basename "$MODEL") ($(( $(stat -c %s "$MODEL") / 1024 / 1024 )) MB) vram=${vram_mb} MB ctx=$CTX kv=$KV ngl=${BLOCKY_NGL:-auto} threads=$threads"

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size "$CTX" \
  "${fit_args[@]}" \
  "${gpu_args[@]}" \
  --threads "$threads" \
  --flash-attn auto \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  --jinja \
  --log-verbosity "${BLOCKY_LOG_VERBOSITY:-4}" \
  --no-webui
