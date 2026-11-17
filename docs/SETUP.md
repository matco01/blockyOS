# Setting up Blocky from scratch

Everything needed to rebuild a working Blocky from this repository.

**Requirements:** Linux (Ubuntu 24.04 recommended, a dedicated box or WSL2), **Python 3.12 or newer** (Ubuntu 24.04's
`python3`), Node.js for building the UI. The backend and the installer refuse older Python versions with a clear message.

## A. On a Windows PC (development, via WSL2)

Nothing here touches Windows itself: Blocky lives in its own Linux, stored wherever you choose.

1. **Create the Linux** (PowerShell, pick a drive with 30+ GB free):
   ```powershell
   wsl --install Ubuntu-24.04 --location D:\wsl\ubuntu --name blocky --no-launch
   ```
2. **Create the `blocky` user** and make it the default:
   ```powershell
   wsl -d blocky -u root -- bash -c "useradd -m -s /bin/bash -G sudo blocky; echo 'blocky ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/blocky; printf '[boot]\nsystemd=true\n[user]\ndefault=blocky\n' > /etc/wsl.conf"
   wsl --terminate blocky
   ```
3. **NVIDIA GPU only: install the CUDA toolkit** (WSL edition; your Windows NVIDIA driver provides the GPU). Stay on **CUDA 12.x**: CUDA 13 dropped GTX 10xx (Pascal) cards.
   ```bash
   cd /tmp && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-9
   ```
4. **Install Blocky** (the repo lives on Windows, e.g. `C:\dev\blockyOS`):
   ```bash
   cd /mnt/c/dev/blockyOS && sudo BLOCKY_CUDA_ARCHS='61;75;86;89' ./scripts/install.sh
   ```
   `BLOCKY_CUDA_ARCHS` lists the GPU generations to compile for (61 = GTX 10xx, 75 = RTX 20xx, 86 = RTX 30xx, 89 = RTX 40xx). Fewer = faster build.
5. **Build the web UI** (Windows, needs Node.js): `cd frontend; npm install; npm run build`
6. **Get a model**: open http://localhost:8000 → Models → pick one, or Hugging Face → paste a repo.

WSL stops Linux when no terminal is attached. Keep `wsl -d blocky` open while using Blocky.

## B. On a dedicated machine (Ubuntu Server 24.04)

1. Install Ubuntu Server 24.04 (minimal), with the NVIDIA driver if there's an NVIDIA GPU:
   `sudo ubuntu-drivers install` (use the **580** branch for GTX 10xx cards).
2. Install the CUDA toolkit 12.x from NVIDIA's Ubuntu repository (as in step A3, but with the `ubuntu2404` repo instead of `wsl-ubuntu`).
3. `git clone <this repo> && cd blockyOS && sudo ./scripts/install.sh`
4. Build the UI (`frontend/`, needs Node.js) or copy a built `frontend/dist` over.

## C. Optional research extras (both setups)

### Several machines

- **llama.cpp across machines (RPC):** on each worker, Settings → "Multi-machine: lend this GPU" → On. It listens on
  localhost by default; from the controller, tunnel to it (`ssh -N -L 50052:localhost:50052 user@worker`) and enter
  `localhost:50052` under Experiments → advanced settings → RPC workers. "Listen on the network" exposes `rpc-server`
  without authentication or encryption; use it only on a network you trust. Builds made before this version need
  `sudo ./scripts/install.sh` again (it now builds `rpc-server`).
- **vLLM across machines:** start a Ray cluster yourself (`ray start --head` on this machine, `ray start
  --address=<this machine>:6379` on the others, all with the same vLLM version and model path), then set Pipeline stages.
- **Training across machines:** Internals → Full training → Machines > 1, with this machine's network address. Every
  machine needs the ML environment and the same checkpoint and data paths; the running job shows the command for the
  others. torchrun opens a port on that address while the job runs.


- **PyTorch lab + lm-evaluation-harness** (Internals page, lm-eval tasks in experiments), a few GB:
  `bash scripts/setup-ml.sh` (as the `blocky` user; installs into `/opt/blocky/ml-venv`, PyTorch CUDA 12.6 wheels so Pascal GPUs still work)
- **Notebooks** (JupyterLab on 127.0.0.1:8888 with a token, the `blocky` Python client importable):
  `sudo bash scripts/setup-notebooks.sh`
- **vLLM** as a second framework in Experiments (NVIDIA RTX 20-series or newer): `bash scripts/setup-vllm.sh`
  (as the `blocky` user; its own environment in `/opt/blocky/vllm-venv`)
- **SGLang** as a third framework for Hugging Face checkpoints: `bash scripts/setup-sglang.sh`
  (as the `blocky` user; its own environment in `/opt/blocky/sglang-venv`). Check SGLang's documentation for supported GPUs.
- **TensorRT-LLM** as a fourth framework (PyTorch backend, serves checkpoints without an engine build):
  `sudo apt install libopenmpi-dev` then `bash scripts/setup-trtllm.sh` (as the `blocky` user; `/opt/blocky/trtllm-venv`).
  Check NVIDIA's support matrix for your GPU, driver, CUDA and Python versions first.
- **More than one GPU:** nothing to install. Experiments then offer a GPU count and, for llama.cpp, the split mode
  (layer = pipeline, row = tensor parallel). vLLM and SGLang use the GPU count as their tensor-parallel size.
- **GPU kernel profiles** use NVIDIA Nsight Systems (`nsys`, part of the CUDA toolkit). Kernel tracing works on a regular
  Linux install, not inside WSL.
- **Custom engines** (llama.cpp builds) and **weight variants** need nothing extra: they're built from the UI into
  `/opt/blocky/engines` and the models folder. Building an engine runs that repository's code, so only add ones you trust.

## Remote access

The web service binds to `127.0.0.1` by default. An SSH tunnel can reach it without exposing a network listener:
`ssh -L 8001:localhost:8000 user@device`, then open `http://localhost:8001`.

For direct access on a trusted network, enable remote access in Settings, add `UVICORN_HOST=0.0.0.0` to
`/etc/blocky/blocky.env`, and restart `blocky-web`. Remote clients still need the token. Removing that setting
and restarting restores localhost binding. Existing installations must install the updated service unit and run
`systemctl daemon-reload` before this default takes effect.

## Tests

- Backend: `python -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt && .venv/bin/python -m pytest backend/tests`
  (runs anywhere: no GPU, no models; everything happens in a temporary folder)
- Frontend: `cd frontend && npm test`
- GitHub Actions runs both, plus the type check and build, on every push (`.github/workflows/ci.yml`).

## What is NOT in this repository (on purpose)

| Thing | Where it lives | How to keep it |
|---|---|---|
| AI models (GBs) | `/opt/blocky/models` | Re-download from the Models page |
| Chats, memories, test & benchmark results | `/opt/blocky/data/blocky.db` | Back up that file (see below) |
| Downloaded benchmark sets | `/opt/blocky/data/benchmarks/` | Re-download from the Tests page |
| Runtime choices (model, context…) | `/opt/blocky/data/llm.env` | Recreated from the Models page |

Backup of your personal data (from Windows), safe while Blocky runs:
```powershell
wsl -d blocky -- bash /mnt/c/dev/blockyOS/scripts/backup.sh /mnt/c/Users/<you>/Documents
```
To restore: stop Blocky (`sudo systemctl stop blocky-web`), copy the backup over `/opt/blocky/data/blocky.db`, start it again.
