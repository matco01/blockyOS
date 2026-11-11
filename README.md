# blockyOS

A local AI appliance: Ubuntu Server + llama.cpp + a web UI. Turn it on, open the page, chat. Nothing leaves the device.

**New machine?** See [docs/SETUP.md](docs/SETUP.md) for the full from-scratch setup (WSL or a dedicated box), plus what isn't in git and how to back it up.

## Layout

| Path | What it is |
|---|---|
| `backend/app/` | FastAPI app: chat streaming, conversation storage (SQLite), system stats |
| `frontend/src/` | React UI (built into `frontend/dist`, served by the backend) |
| `scripts/start-llm.sh` | Starts llama-server with settings picked from the detected GPU/RAM |
| `scripts/systemd/` | Services that start everything at boot |
| `scripts/install.sh` | One installer for both dev (WSL) and the real device |

On the machine: models in `/opt/blocky/models`, data in `/opt/blocky/data`, settings in `/etc/blocky/blocky.env`.

## Lab features

| Page | What it does |
|---|---|
| **Models** | Metadata read from each GGUF header (arch, params, quant, layers, experts, trained ctx); fit estimate for this machine; switch model / context / KV cache type / GPU layers / threads (restarts the model server, rolls back on failure); Hugging Face browser + downloads |
| **Benchmarks** | `llama-bench` presets or custom sweeps (pp / tg / depth / repetitions / KV / flash-attn / ngl / threads / MoE offload); GPU power sampled every 0.5 s → tokens per joule; peak VRAM; hardware fingerprint saved per run; compare up to 4 runs (shareable `#/bench?compare=3,4`); CSV/JSON export |
| **Chat Inspector** | Sampling (temperature, top-p/k, min-p, penalties, max tokens, seed), presets, custom system prompt; per-answer TTFT, prompt/gen speed, cached tokens, context used |
| **System** | Live CPU/GPU/RAM/VRAM/power plus the model server log with filter |

The web app restarts `blocky-llm` through a narrow sudoers rule (`/etc/sudoers.d/blocky-services`); runtime choices are saved to `/opt/blocky/data/llm.env`.

## Everyday commands (on Windows, PowerShell)

```powershell
wsl -d blocky                      # open a Linux terminal (keeps the AI running while open)
# then open http://localhost:8000

wsl -d blocky -- sudo systemctl restart blocky-web   # after backend code changes
wsl -d blocky -- sudo systemctl restart blocky-llm   # after changing model/settings
wsl -d blocky -- journalctl -u blocky-llm -f         # watch the model server log
```

UI development with live reload: `cd frontend; npm run dev` → http://localhost:5173.
After UI changes, `npm run build` so the backend serves the new version.

## Removing everything (dev PC)

`wsl --unregister blocky` deletes the Linux install in `D:\wsl\ubuntu`. Windows is untouched.
