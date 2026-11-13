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
| **Experiments** (the research hub) | One model, one variable at a time. Each run profiles a full config: memory layout from llama.cpp's load log, MMLU-Pro (fixed 560 questions, 5-shot letter log-likelihood, Wilson CI, paired McNemar test vs baseline), llama-bench decode/prefill at 0 and 4k context (5 reps, median, robust spread, impossible samples rejected), perplexity + KL divergence vs the baseline's logits on WikiText-2, GPU/CPU telemetry. Hardware analysis: roofline vs spec-sheet peaks, per-token time split (VRAM reads / RAM reads / other), binding constraint, clocks, VRAM headroom. Provenance per run (llama.cpp commit, model/dataset SHA-256, driver/CUDA, exact commands). Run queue + sweeps (full grids, repeats), repeat-baseline session noise, CSV/JSON export |
| **Experiments → Engines** | Build any llama.cpp repo + branch/tag/commit (your fork, a PR, upstream master) for this GPU and use the engine as an experiment variable, A/B against stock with identical measurements. vLLM as a second framework for Hugging Face checkpoints (`scripts/setup-vllm.sh`); an engine-neutral serving-speed test makes frameworks comparable. Optional per-operation GPU kernel profile (Nsight Systems) |
| **Experiments → tests** | Every run also does a needle-in-a-haystack long-context test (2k…ctx × 5 depths); per experiment, add lm-evaluation-harness tasks (e.g. gsm8k, ifeval) run against each run's own server |
| **Weights → Layer sensitivity** | Quantize (or remove) one layer at a time and measure KL divergence vs the source, with a control noise floor; one click turns "protect the most sensitive layers" into a workshop recipe |
| **Internals** (PyTorch) | Hugging Face checkpoints; activations explorer (per-layer residual norms, layer deltas, logit lens, attention entropy), layer knockout, GPU time per layer (attention vs feed-forward, prefill and decode); LoRA / QLoRA fine-tuning → merge → GGUF (with recipe), then profile it in Experiments. Needs `scripts/setup-ml.sh` |
| **Notebooks** | JupyterLab on the box (127.0.0.1:8888, token) and the `blocky` Python client (`client/blocky.py`): queue runs and sweeps, build weights/engines, get every result as a pandas table, send experiments to MLflow or Weights & Biases (local/offline by default). Needs `sudo scripts/setup-notebooks.sh` |
| **Energy & cost** | Every run measures idle power (model loaded) and power during ~15 s of sustained generation: energy per token (gross and above idle), tokens per Wh. Sources: a wall-power smart plug (Shelly, Tasmota or any JSON URL, set in Settings), else GPU (nvidia-smi) and CPU package (RAPL) sensors. With prices set: electricity per 1M tokens and decode speed per device price |
| **Cheap-device simulation** | Target-device presets (single-board computer, mini PC, GPU-less laptop, 4/6 GB GPUs) or custom: CPU only (`-dev none`), cores pinned with `taskset`, RAM/VRAM capacity limits (runs that wouldn't fit fail), plus a projected decode speed at the target's spec-sheet memory bandwidth. Engines can be built for CUDA, Vulkan (AMD/Intel/integrated GPUs) or CPU only |
| **Internals → Technique playground** | Prototype a quantization idea in Python before writing a kernel: RTN (2–8 bit, sym/asym, group size), NF4, ternary, or your own `quantize(w, name)` function, plus simulated KV-cache bits. Fake quantization on a Hugging Face checkpoint, scored against the unmodified model: KL divergence (± SE, p99), same top token, perplexity, per-layer and per-matrix-type weight error, effective bits per weight |
| **Remote devices** | One Blocky profiles setups on other Blocky devices (Experiments → Run on another device) and imports the results next to local runs, each with its own hardware snapshot. Every Blocky answers only itself (loopback) until remote access is turned on in Settings; other devices then need its token (`Authorization: Bearer`, or `?key=` once in a browser). Over untrusted networks use an SSH tunnel |
| **AI assistants (MCP)** | Blocky is an MCP server at `http://localhost:8000/mcp`: connect Claude Code (`claude mcp add --transport http blocky http://localhost:8000/mcp`), Claude Desktop, editors or other MCP clients and let the assistant read results, queue runs/sweeps, build weight variants, scan layers and build engines. No delete tools; the assistant's keys stay in your client. Best of all, run a coding agent on the device in the `research/` workspace: it gets Blocky's MCP tools (`.mcp.json`), research instructions (`CLAUDE.md`/`AGENTS.md`), a shell, Python and PyTorch. Coding agents changing Blocky itself: see the root `AGENTS.md` |
| **Tests** | `backend/tests` (pytest) and `frontend/src/*.test.ts` (vitest); GitHub Actions runs them on every push |
| **Experiments → Canvas** | Infinite pan/zoom map of an experiment: baseline → changed setting → runs, draggable cards, sticky notes, head-to-head compare of any two runs; layout stored on the box |
| **Experiments → Weights workshop** | Build weight variants from a high-precision source with llama.cpp's tools: base quantization type, importance matrix (llama-imatrix on WikiText-2 *train*, separate from the eval text), per-tensor precision overrides, output/embedding types, layer pruning. Each variant gets a `.recipe.json` that ends up in the run's provenance |
| **Models** | Metadata read from each GGUF header (arch, params, quant, layers, experts, trained ctx); fit estimate for this machine; switch model / context / KV cache type / GPU layers / threads (restarts the model server, rolls back on failure); Hugging Face browser + downloads |
| **Benchmarks** | `llama-bench` presets or custom sweeps (pp / tg / depth / repetitions / KV / flash-attn / ngl / threads / MoE offload); GPU power sampled every 0.5 s → tokens per joule; peak VRAM; hardware fingerprint saved per run; compare up to 4 runs (shareable `#/bench?compare=3,4`); CSV/JSON export |
| **Chat Inspector** | Sampling (temperature, top-p/k, min-p, penalties, max tokens, seed), presets, custom system prompt; per-answer TTFT, prompt/gen speed, cached tokens, context used |
| **System** | Live CPU/GPU/RAM/VRAM/power plus the model server log with filter |
| **Quality tests** | Fresh questions generated at run time (can't be memorized), Blocky's quick checks, established benchmarks (GSM8K, ARC, MMLU-Pro, MATH-500, AIME, HumanEval(+), MBPP+, CRUXEval, LiveCodeBench) with "public since" contamination warnings, a memorization check (famous vs fresh), custom Hugging Face imports, and an OpenAI-compatible `/v1` endpoint for outside harnesses |
| **Models → Inside the model** | The real network structure read from the GGUF tensor table: every layer's attention / linear-attention / feed-forward / expert parts, sized by memory |
| **Models → Compression lab** | Compare quantizations of one model (perplexity, KL divergence, same-top-token %, speed, size) with llama-perplexity; make new quantizations locally with llama-quantize |
| **Speed tests → Many users / Draft speed-up** | Parallel throughput with llama-batched-bench; speculative decoding with a small same-family draft model |
| **Chat → Inspector → Show confidence** | Per-word probability heatmap with the alternatives the model almost picked |

Auto-start on Windows: copy `scripts/windows/blocky-autostart.vbs` into your Startup folder (Win+R → `shell:startup`).

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
