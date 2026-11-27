#!/usr/bin/env bash
# blockyOS in one command, on Ubuntu 24.04 (a dedicated machine, a GPU server, or WSL2 on a Windows PC):
#
#   curl -fsSL https://raw.githubusercontent.com/matco01/blockyOS/main/scripts/get.sh | sudo bash
#
# It installs what's missing (NVIDIA driver and CUDA toolkit when there's an NVIDIA GPU, build tools, Node.js), gets the
# code, builds llama.cpp for this machine's GPU, builds the web UI, starts the services and downloads a small starter
# model. Running it again updates Blocky. It stops and says so when it needs a reboot (after a new GPU driver).
#
# Options (environment variables, e.g. `curl … | sudo BLOCKY_NO_MODEL=1 bash`):
#   BLOCKY_DIR=/opt/blocky/src   where the code goes (ignored when run from a checkout)
#   BLOCKY_REF=main              branch or tag
#   BLOCKY_NO_CUDA=1             build for the CPU only, even with an NVIDIA GPU
#   BLOCKY_NO_DRIVER=1           never install an NVIDIA driver
#   BLOCKY_NO_MODEL=1            don't download the starter model
#   BLOCKY_DRY_RUN=1             show what would happen; change nothing
set -euo pipefail

REPO_URL=https://github.com/matco01/blockyOS
REF=${BLOCKY_REF:-main}
DRY=${BLOCKY_DRY_RUN:-0}
STARTER_REPO=unsloth/Qwen3.5-0.8B-GGUF
STARTER_FILE=Qwen3.5-0.8B-Q8_0.gguf

say() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
run() { if [[ $DRY == 1 ]]; then printf '    [dry run] %s\n' "$*"; else "$@"; fi; }
die() { printf '\nblockyOS: %s\n' "$*" >&2; exit 1; }
apt_install() { DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@" >/dev/null; }
# The web UI is built in a temporary copy: the checkout may live on another system's disk (WSL: a Windows folder whose
# node_modules holds Windows builds of the tools), so its node_modules is never touched. Only the finished dist/ is copied.
build_ui() {
  local tmp; tmp=$(mktemp -d)
  tar -C "$1/frontend" --exclude=node_modules --exclude=dist -cf - . | tar -C "$tmp" -xf -
  (cd "$tmp" && npm ci --no-audit --no-fund --loglevel=error && npm run build --silent)
  rm -rf "$1/frontend/dist" && cp -r "$tmp/dist" "$1/frontend/dist" && rm -rf "$tmp"
}

[[ $EUID -eq 0 ]] || die "Run as root: curl -fsSL … | sudo bash"
[[ -r /etc/os-release ]] && . /etc/os-release
[[ "${ID:-}" == ubuntu ]] || note "Not Ubuntu (${PRETTY_NAME:-unknown}): Blocky is made for Ubuntu 24.04; continuing anyway."
ARCH=$(uname -m)
WSL=0; grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && WSL=1

say "Checking this machine"
note "System: ${PRETTY_NAME:-unknown}, $ARCH$([[ $WSL == 1 ]] && echo ', WSL2')"
if [[ ! -d /run/systemd/system ]]; then
  if [[ $WSL == 1 ]]; then
    run bash -c 'grep -q "^systemd=true" /etc/wsl.conf 2>/dev/null || printf "[boot]\nsystemd=true\n" >> /etc/wsl.conf'
    die "WSL needs systemd for Blocky's services; it's now turned on. In Windows PowerShell run: wsl --shutdown
Then open this Linux again and run the same command."
  fi
  die "This system runs without systemd. Use the Docker image instead (see docs/SETUP.md)."
fi

say "Base packages"
run apt-get update -qq
run apt_install git curl ca-certificates pciutils build-essential cmake python3-venv
python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' || die "Blocky needs Python 3.12 or newer (Ubuntu 24.04 has it)."

# ---------- GPU ----------
NVIDIA=0
if [[ $WSL == 1 ]]; then
  [[ -x /usr/lib/wsl/lib/nvidia-smi ]] && NVIDIA=1   # the Windows driver provides the GPU
elif lspci 2>/dev/null | grep -qi 'vga.*nvidia\|3d.*nvidia'; then
  NVIDIA=1
fi
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
if [[ $NVIDIA == 1 && ${BLOCKY_NO_CUDA:-0} != 1 ]]; then
  say "NVIDIA GPU"
  if ! "$SMI" >/dev/null 2>&1; then
    [[ $WSL == 1 ]] && die "Windows has no working NVIDIA driver for WSL. Install the latest NVIDIA driver on Windows, then run this again."
    [[ ${BLOCKY_NO_DRIVER:-0} == 1 ]] && die "No working NVIDIA driver, and BLOCKY_NO_DRIVER=1. Install one, reboot, run this again."
    note "No working driver: installing the recommended one."
    run apt_install ubuntu-drivers-common
    run ubuntu-drivers install
    printf '\nThe NVIDIA driver is installed. Reboot (sudo reboot), then run the same command again to finish.\n'
    exit 0
  fi
  note "$("$SMI" --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    # CUDA 12.x: CUDA 13 dropped GTX 10xx (Pascal) cards.
    case "$WSL:$ARCH" in
      1:x86_64) REPO=wsl-ubuntu/x86_64 ;;
      0:x86_64) REPO=ubuntu2404/x86_64 ;;
      0:aarch64) REPO=ubuntu2404/sbsa ;;
      *) die "No CUDA package for $ARCH here; rerun with BLOCKY_NO_CUDA=1 for a CPU build." ;;
    esac
    note "Installing the CUDA toolkit 12.9 (a few GB)."
    run curl -fsSLo /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/cuda-keyring_1.1-1_all.deb"
    run dpkg -i /tmp/cuda-keyring.deb
    run apt-get update -qq
    run apt_install cuda-toolkit-12-9
  fi
  # Compile only for the GPUs in this machine (much faster than every generation).
  ARCHS=$("$SMI" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' .' | sort -u | paste -sd';' || true)
  export BLOCKY_CUDA_ARCHS=${BLOCKY_CUDA_ARCHS:-${ARCHS:-61;75;86;89}}
  note "Building for GPU architecture(s) $BLOCKY_CUDA_ARCHS"
else
  say "No NVIDIA GPU in use: building for the CPU"
  export PATH=$(printf '%s' "$PATH" | sed 's#/usr/local/cuda/bin:##g')
fi

# ---------- Code ----------
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)
if [[ -n "$HERE" && -f "$HERE/install.sh" && -d "$HERE/../backend" ]]; then
  SRC=$(cd "$HERE/.." && pwd)
  say "Using the code in $SRC"
else
  SRC=${BLOCKY_DIR:-/opt/blocky/src}
  if [[ -d "$SRC/.git" ]]; then
    say "Updating the code in $SRC"
    run git -C "$SRC" fetch --depth 1 origin "$REF"
    run git -C "$SRC" checkout -q FETCH_HEAD
  else
    say "Getting the code ($REF) into $SRC"
    run git clone --depth 1 --branch "$REF" "$REPO_URL" "$SRC"
  fi
fi

say "Installing Blocky"
run bash "$SRC/scripts/install.sh"

# ---------- Web UI ----------
say "Building the web UI"
if ! command -v npm >/dev/null 2>&1 || (( $(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0) < 18 )); then
  run apt_install nodejs npm
fi
run build_ui "$SRC"
run systemctl restart blocky-web

# ---------- Starter model ----------
if [[ ${BLOCKY_NO_MODEL:-0} != 1 ]] && ! compgen -G "/opt/blocky/models/*.gguf" >/dev/null; then
  say "Downloading a small starter model ($STARTER_FILE, about 0.8 GB)"
  run sudo -u blocky curl -fL --retry 3 -o "/opt/blocky/models/$STARTER_FILE.part" "https://huggingface.co/$STARTER_REPO/resolve/main/$STARTER_FILE"
  run sudo -u blocky mv "/opt/blocky/models/$STARTER_FILE.part" "/opt/blocky/models/$STARTER_FILE"
  run systemctl restart blocky-llm
fi

say "Done"
note "Open http://localhost:8000 on this machine. It is reachable from this machine only; Settings → Remote access"
note "lets other devices in, with a token."
[[ $WSL == 1 ]] && note "WSL stops this Linux when no window is attached: keep a 'wsl -d <name>' window open (docs/SETUP.md)."
note "Update later by running the same command again."
