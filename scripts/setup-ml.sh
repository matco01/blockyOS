#!/bin/bash
# The PyTorch side of the lab (separate from the backend's own environment):
#   - lm-evaluation-harness (the field's standard eval tool), run against Blocky's model servers
#   - transformers / peft for activations and LoRA fine-tuning of Hugging Face checkpoints
#   - llama.cpp's convert_hf_to_gguf.py requirements, to bring fine-tuned models back as GGUF
# PyTorch is installed from the CUDA 12.6 wheels: they still support Pascal GPUs (GTX 10xx) as well as newer ones.
# Run as the blocky user (no sudo needed). Safe to run again.
set -euo pipefail
PREFIX=/opt/blocky
VENV="$PREFIX/ml-venv"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
echo "==> PyTorch (CUDA 12.6 wheels)"
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu126
echo "==> transformers, peft, datasets, lm-eval, gguf tools"
"$VENV/bin/pip" install -q transformers peft accelerate datasets safetensors sentencepiece "lm-eval[api]" gguf numpy
SITE=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
echo "$REPO_DIR/client" > "$SITE/blocky-client.pth"
"$VENV/bin/python" -c "import torch, transformers, peft, lm_eval; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('transformers', transformers.__version__, 'peft', peft.__version__)"
