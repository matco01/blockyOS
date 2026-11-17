#!/bin/bash
# TensorRT-LLM as a fourth inference framework for Experiments (its own environment; brings its own PyTorch).
# Uses the PyTorch backend of trtllm-serve, so Hugging Face checkpoints are served directly without an engine build.
# Needs a recent NVIDIA GPU and driver; check TensorRT-LLM's support matrix for your GPU, CUDA and Python versions.
# Run as the blocky user. Safe to run again.
set -euo pipefail
VENV=/opt/blocky/trtllm-venv
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
# TensorRT-LLM needs MPI headers/libraries for its distributed runtime (sudo apt install libopenmpi-dev).
"$VENV/bin/pip" install -q --extra-index-url https://pypi.nvidia.com tensorrt_llm
"$VENV/bin/python" -c "import tensorrt_llm, torch; print('tensorrt_llm', tensorrt_llm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
test -x "$VENV/bin/trtllm-serve" && echo "trtllm-serve ready"
