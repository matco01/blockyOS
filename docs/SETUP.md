# Setting up Blocky from scratch

Everything needed to rebuild a working Blocky from this repository.

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

- **PyTorch lab + lm-evaluation-harness** (Internals page, lm-eval tasks in experiments), a few GB:
  `bash scripts/setup-ml.sh` (as the `blocky` user; installs into `/opt/blocky/ml-venv`, PyTorch CUDA 12.6 wheels so Pascal GPUs still work)
- **Notebooks** (JupyterLab on 127.0.0.1:8888 with a token, the `blocky` Python client importable):
  `sudo bash scripts/setup-notebooks.sh`
- **Custom engines** (llama.cpp builds) and **weight variants** need nothing extra: they're built from the UI into
  `/opt/blocky/engines` and the models folder. Building an engine runs that repository's code, so only add ones you trust.

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
