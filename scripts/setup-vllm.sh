#!/bin/bash
# vLLM as a second inference framework for Experiments (its own environment; it brings its own PyTorch).
# Needs an NVIDIA GPU with compute capability 7.0 or newer (RTX 20xx and later). GTX 10xx (Pascal) isn't supported by vLLM.
# Run as the blocky user. Safe to run again.
set -euo pipefail
VENV=/opt/blocky/vllm-venv
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q vllm
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
