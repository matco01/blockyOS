#!/usr/bin/env bash
# blockyOS installer. Run as root on a fresh Ubuntu/Debian server (or in WSL for dev):
#   sudo ./scripts/install.sh
# Safe to re-run: every step checks whether it's already done.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX=/opt/blocky
LLAMA_SRC=/home/blocky/llama.cpp

[[ $EUID -eq 0 ]] || { echo "Run with sudo"; exit 1; }

echo "==> User and folders"
id blocky >/dev/null 2>&1 || useradd -m -s /bin/bash blocky
mkdir -p "$PREFIX/models" "$PREFIX/data" /etc/blocky
# On the device the code is copied here; in WSL dev it's a link to the Windows folder.
[[ -e "$PREFIX/app" ]] || ln -s "$REPO_DIR" "$PREFIX/app"
chown -R blocky:blocky "$PREFIX/models" "$PREFIX/data"

echo "==> System packages"
# bubblewrap = sandbox for running model-written code in the coding tests (no network, read-only).
# python3-numpy: used by HumanEval+/MBPP+ test scripts inside the sandbox.
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git curl python3-venv libcurl4-openssl-dev bubblewrap python3-numpy >/dev/null

echo "==> llama.cpp"
if [[ ! -x "$PREFIX/llama.cpp/bin/llama-server" ]]; then
  [[ -d "$LLAMA_SRC" ]] || sudo -u blocky git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
  CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON)
  if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH=/usr/local/cuda/bin:$PATH
    # 61 = GTX 10xx (Pascal), 75 = RTX 20xx, 86 = RTX 30xx, 89 = RTX 40xx
    CMAKE_ARGS+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${BLOCKY_CUDA_ARCHS:-61;75;86;89}")
    echo "    building with CUDA"
  else
    echo "    no CUDA toolkit found, building CPU-only"
  fi
  sudo -u blocky env PATH="$PATH" cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" "${CMAKE_ARGS[@]}" >/dev/null
  sudo -u blocky env PATH="$PATH" cmake --build "$LLAMA_SRC/build" --config Release -j"$(nproc)" --target llama-server llama-bench
  # Copy the server, benchmark tool and their shared libraries.
  mkdir -p "$PREFIX/llama.cpp/bin"
  cp -a "$LLAMA_SRC/build/bin/." "$PREFIX/llama.cpp/bin/"
fi

echo "==> Python environment"
[[ -d "$PREFIX/venv" ]] || python3 -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install -q -r "$REPO_DIR/backend/requirements.txt"

echo "==> Permissions"
# The web app may start/stop/restart the model server (for model switching and
# benchmarks) and read its log. Nothing else.
cat > /etc/sudoers.d/blocky-services <<'EOF'
blocky ALL=(root) NOPASSWD: /usr/bin/systemctl start blocky-llm, /usr/bin/systemctl stop blocky-llm, /usr/bin/systemctl restart blocky-llm
EOF
chmod 440 /etc/sudoers.d/blocky-services
visudo -cf /etc/sudoers.d/blocky-services >/dev/null
usermod -aG systemd-journal blocky

echo "==> Services"
[[ -f /etc/blocky/blocky.env ]] || cat > /etc/blocky/blocky.env <<'EOF'
# blockyOS settings. Restart services after editing:
#   sudo systemctl restart blocky-llm blocky-web
# BLOCKY_MODEL=/opt/blocky/models/your-model.gguf
# BLOCKY_CTX=8192
# BLOCKY_MAX_VRAM_MB=4096    # simulate a smaller GPU
EOF
cp "$REPO_DIR"/scripts/systemd/*.service /etc/systemd/system/
chmod +x "$REPO_DIR/scripts/start-llm.sh" 2>/dev/null || true
systemctl daemon-reload
systemctl enable --now blocky-llm blocky-web
systemctl restart blocky-llm blocky-web

echo
echo "Done. Open http://localhost:8000"
