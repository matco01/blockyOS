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

Install Ubuntu Server 24.04 (minimal), then:

```bash
curl -fsSL https://raw.githubusercontent.com/matco01/blockyOS/main/scripts/get.sh | sudo bash
```

`scripts/get.sh` does every step: base packages; with an NVIDIA GPU the recommended driver (then it asks for a reboot;
run the same command again afterwards) and the CUDA toolkit 12.9 (12.x keeps GTX 10xx cards supported); the code in
`/opt/blocky/src`; `scripts/install.sh` with llama.cpp built only for the GPU generations in the machine; the web UI
(Ubuntu's Node.js, built in a temporary copy); a small starter model. Running it again updates Blocky. Options:
`BLOCKY_NO_CUDA=1` (CPU build), `BLOCKY_NO_DRIVER=1`, `BLOCKY_NO_MODEL=1`, `BLOCKY_REF=<branch or tag>`,
`BLOCKY_DRY_RUN=1` (show the plan, change nothing). In WSL it turns on systemd if needed and asks for `wsl --shutdown`.

The manual equivalent: the NVIDIA driver (`sudo ubuntu-drivers install`), the CUDA toolkit 12.x from NVIDIA's
`ubuntu2404` repository (as in step A3), `sudo ./scripts/install.sh` in a clone, and the UI (`frontend/`, Node.js 18+).

## Docker

`docker compose up -d` in a clone builds the image (CUDA 12.9, llama.cpp for GTX 10xx to RTX 40xx) and starts it with
the GPU; it needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
Without an NVIDIA GPU: `docker build --build-arg BASE=ubuntu:24.04 -t blockyos:cpu .` and
`docker run -d --network host -v blocky-models:/opt/blocky/models -v blocky-data:/opt/blocky/data blockyos:cpu`.

- The container uses the host's network on purpose: Blocky answers only requests from the machine itself (and other
  devices with the remote-access token), and inside a bridge network every request looks like it comes from outside.
- There's no systemd in a container, so Blocky runs the chat model server itself (`BLOCKY_SUPERVISE_LLM=1`); its log is
  in `/opt/blocky/data/llm.log`. Models and data live in the two volumes.
- Not in the image yet: the GPU clock lock helper (it needs root) and the PyTorch lab (`scripts/setup-ml.sh` isn't set
  up for containers). For training and the lab, use the full install.

## C. Optional research extras (both setups)

### Several machines

- **llama.cpp across machines (RPC):** on each worker, Settings → "Multi-machine: lend this GPU" → On. It listens on
  localhost by default; from the controller, tunnel to it (`ssh -N -L 50052:localhost:50052 user@worker`) and enter
  `localhost:50052` under Experiments → advanced settings → RPC workers. "Listen on the network" exposes llama.cpp's RPC server
  (`ggml-rpc-server`; `rpc-server` in older llama.cpp) without authentication or encryption; use it only on a network you trust. Builds made before this version need
  `sudo ./scripts/install.sh` again (it now builds the RPC server).
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
