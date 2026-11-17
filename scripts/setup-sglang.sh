#!/bin/bash
# SGLang as a third inference framework for Experiments (its own environment; it brings its own PyTorch).
# Runs Hugging Face checkpoints like vLLM; useful to compare serving frameworks on identical tests.
# Needs a recent NVIDIA GPU (check SGLang's documentation for the supported compute capabilities).
# Run as the blocky user. Safe to run again.
set -euo pipefail
VENV=/opt/blocky/sglang-venv
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "sglang[all]"
"$VENV/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
