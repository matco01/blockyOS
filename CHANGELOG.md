# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org); the **measurement protocol** has its
own version (`PROTOCOL` in `backend/app/profile.py`, specified in [docs/PROTOCOL.md](docs/PROTOCOL.md)) and is listed
with each release. Results are only comparable within one protocol version.

## [Unreleased]

### Added
- **Projects, the new front door.** Name a project (a goal is optional), then work inside it: Overview (how to begin:
  an existing model, training from scratch or your own code; then where each model stands and what to do next), Build
  (make it smaller, fine-tune, train, your own code, new quantization methods), Versions (the experiment tree for this
  project's models) and Ship (use a version on this device, reproducibility bundle, results CSV). Existing experiments
  and code appear under "Not in a project yet" and can be moved in; deleting a project never deletes them. The sidebar
  is now Projects, Chat, Device and Help, with every earlier page under "All tools". API: `/api/projects`.
- **Bring your own files, from the browser.** A project has Data, Canvas and Tests tabs. Upload a `.gguf` model file
  (streamed to disk, checked as GGUF, never overwrites), training data (`.jsonl` checked line by line, or `.txt`;
  picked from a list in the training forms), and your code as a folder or a `.zip` (a GitHub "Download ZIP" works as
  is; archives can't write outside the workspace). After an upload, a "Start file" choice sets what runs.
- **Your own AI, end to end.** Code from a public Git link (shallow clone; Git LFS files are reported, not fetched).
  Each code workspace gets its own Python environment: one button installs `requirements.txt` (or the
  `pyproject.toml`/`setup.py` package), reusing the device's PyTorch and Transformers when the Python versions match.
  A Hugging Face-format model folder (config.json + safetensors) can be uploaded for fine-tuning, and a `.gguf` can be
  downloaded on the device from any https link.

- **One-command install and Docker.** `curl -fsSL …/scripts/get.sh | sudo bash` installs everything on Ubuntu 24.04
  (driver and CUDA 12.9 with an NVIDIA GPU, llama.cpp for this machine's GPU only, the web UI built from a temporary
  copy, a starter model), updates when run again, handles WSL's systemd switch and has a dry-run mode. A Dockerfile and
  `docker-compose.yml` (host network, volumes for models and data); without systemd, Blocky now runs the chat model
  server itself (`BLOCKY_SUPERVISE_LLM=1`, log in DATA_DIR/llm.log). CI builds the CPU image and checks that it serves
  the API and the UI.
- **Settings comparable with published numbers.** Perplexity/KLD can be scored on the whole text (the setting of
  published llama.cpp tables), with a warning about time and disk. One preset adds it together with lm-evaluation-harness
  GSM8K and IFEval on their full test sets. The built-in MMLU-Pro sample is labelled as a paired sample, not a
  leaderboard score. Published reference values are recorded from the instrument checks (or the API) instead of a
  hand-edited JSON file; a citation is required. `blocky measure --ppl-chunks full`.
- **The `blocky` command** (`blockyos-client` 0.3.0): `blocky measure base.gguf variant.gguf` queues a baseline,
  optional reruns and variants, waits and prints a table where every number has its spread or interval and accuracy
  differences their paired test; `results`, `check` (exit code 1 on a failed instrument check), `status`, `models`,
  `experiments`, `projects`; CSV/JSON output; remote devices with `--url`/`--token`. The Python client takes a token.
- **Ship: publish to Hugging Face with a measured model card.** Each measured version in a project can be used on the
  device, downloaded (model file, model card), or published to a Hugging Face repository (private by default): the
  model card is written only from the measurements (accuracy with its interval and paired test against the baseline,
  speed and perplexity with their spread, KL divergence, recipe, hashes, hardware, protocol), optionally with the
  reproducibility bundle. The license is detected from the file or chosen; licenses with conditions need the owner's
  confirmation. The token is stored like other secrets and never returned; publishing is owner-only (not via MCP).
- **Hardware lab, level 1: run a model on a simulated chip.** Analog in-memory computing (crossbar tiles of device pairs,
  DAC and ADC with calibrated ranges, phase-change memory programming noise, drift and read noise ported from IBM
  AIHWKit, stuck devices, conductance levels, drift compensation) or digital number formats (INT8, INT4, FP8 E4M3,
  MXFP4). Every decoder matrix runs on the chip; KL divergence, same top token and perplexity against the exact model,
  per chip and with the spread over independently simulated chips; output and weight error per matrix type and layer;
  sweeps of one setting with a chart; CSV with the chip description and seed. Two presets check the pipeline (KL
  divergence near 0). All tools → Hardware lab; API `/api/hwlab`; docs/HARDWARE_LAB.md. A new CI job runs the
  simulator's tests with CPU PyTorch.
- **GPU safety.** A monitor stops all running GPU work (experiments, training, code runs, benchmarks, scans, …) when
  the GPU stays above its limit or stops responding, and pauses GPU work until the owner resumes. Blocky also pauses
  when it restarts after GPU work was interrupted, so a crash or power loss never restarts heavy work by itself. Queued
  runs wait until the GPU has cooled. Limits come from the GPU (3 °C below its maximum operating temperature) or
  Settings → GPU safety; a banner shows the pause everywhere. AI assistants can't resume.

### Changed
- **Measurement protocol `blocky-profile/4`: one GPU layout per run.** llama-bench used its own fit with 512 MiB free,
  while the server and the memory analysis used another, so for models that didn't fit entirely on the GPU the speed
  came from a different layout than the one analysed (seen as decode speed above the bandwidth limit). Now the server
  fits once and llama-bench, llama-perplexity and kernel profiles reuse its GPU layer count (mixture-of-experts models
  fit again with the same margin). Results from `blocky-profile/3` are not comparable with new ones; start a new
  experiment and baseline. docs/PROTOCOL.md section 4.
- GPUs that drive a display keep 1 GiB more free memory (`BLOCKY_DISPLAY_RESERVE_MB`) in every llama.cpp tool Blocky
  starts. On a desktop PC, an experiment with a model close to the VRAM size froze the whole computer.
- Research runs: snapshots hold up to 20,000 files / 200 GB. Files of 1 MB and more are hard-linked instead of copied
  (no extra space or time) and their hashes are cached; source files are always replaced, never edited in place, so a
  run's snapshot keeps the content it was hashed with.
- AI assistants over MCP with full access: ChatGPT and Claude (web, desktop and phone) connect through the owner's HTTPS
  tunnel with OAuth 2.1 sign-in (dynamic client registration, PKCE, rotating refresh tokens, reuse detection). The owner
  approves each one with a one-time pairing code and chooses its access level (read, write, full); connections can be
  disconnected at any time. Settings → AI assistants; docs/AI_ASSISTANTS.md.
- MCP tools for everything: `list_api` (every endpoint), `api_get`, `api_call`, `api_delete` and `query_database`
  (read-only SQL over all stored data). Every tool is labelled read-only, write or destructive.
- Deleting through MCP is possible at the `full` access level (previously no MCP tool deleted anything). Assistants on
  this device get the local level, `full` by default. Secrets (tokens, API keys, the pairing code) and the endpoints
  that manage access are never available through MCP.
- Requests that arrive through a tunnel or proxy on this machine (forwarding headers) are no longer treated as coming
  from the device itself.

### Fixed
- CI failed at random: the display check's cache treated "never checked" as checked at boot time, so on a freshly
  booted machine (CI runners) it skipped the check for its first minute. Failing tests now also appear as annotations,
  and the workflow uses current action versions.
- Installing and the Docker image work with current llama.cpp again: it renamed its RPC server (`rpc-server` →
  `ggml-rpc-server`), which made the build stop; engine builds finished but without the RPC worker. Both names work
  now, and the RPC worker (optional, for multi-machine runs) can no longer fail the whole install; CI checks that it
  still gets built.
- A fresh install (and the Docker image) failed at startup: the service, running as user `blocky`, couldn't create
  `/opt/blocky/engines` and `/opt/blocky/hf` in the root-owned `/opt/blocky`. The installer and the image now create
  them for it (Docker keeps both in volumes), and a test checks that every folder the backend uses there is created.
- llama.cpp programs now look for their libraries next to themselves and nowhere else (linked with the search path
  `$ORIGIN`). Before, they looked in the build folder, which the Docker image and engine builds delete, so anything
  started without Blocky's library path failed (for example `llama-server --version` in the container). That search
  path also ended with an empty entry, which makes the loader try the current folder too. Builds made before this
  keep their old search path until they are rebuilt.
- Kernel profiles of a custom engine traced its llama-bench on the **stock** build's GPU libraries: nsys starts the
  program, and the runner chose the stock folder for wrappers. They now load the engine's own libraries. Only regular
  Linux installs were affected (under WSL, nsys records no kernels); re-run kernel profiles of custom engines.
- GPU clock lock helper (`blocky-gpu-clocks`) also finds `nvidia-smi` where WSL installs it; re-run `scripts/install.sh`.
- Speed: a llama-bench repetition with a negative duration (the clock stepped backwards, seen under WSL) is now dropped
  like other invalid samples. Before, it counted as a near-zero rate and pulled the median down. Protocol unchanged:
  this was already the specified rule; earlier stored results keep their values.

## [0.2.0] — 2026-11-17

Measurement protocol: `blocky-profile/3` (unchanged by default; new optional settings are recorded per run).

### Added
- Apache-2.0 license, NOTICE, CITATION.cff, contributing guide, code of conduct, issue and pull-request templates.
- docs/PROTOCOL.md: the full measurement specification.
- Instrument checks: baseline vs itself, answer and speed repeatability, quantization order, bandwidth limit, cited
  reference values (UI, API, MCP tool).
- Statistics: paired accuracy difference with a Newcombe interval; equivalence (TOST) against a preselected margin;
  counterbalanced AB/BA repeated validation (`paired-median-sign/2`); preregistration of hypotheses (append-only).
- Test settings: perplexity/KLD sample size (32–256 chunks), custom perplexity/KLD text, equivalence margin, GPU clock
  lock and cool-down (noise controls, recorded in provenance and enforced in comparisons).
- Reproducibility bundle per experiment (provenance, commands, raw outputs, instrument checks, protocol document,
  `reproduce.py` that re-runs and compares on another device).
- Engines: SGLang and TensorRT-LLM as checkpoint frameworks; multi-GPU settings (GPU count, llama.cpp split mode,
  tensor-parallel size).
- Several machines: llama.cpp RPC workers (with a Settings switch to lend this GPU, localhost-only by default), vLLM
  pipeline stages over Ray, multi-machine training through torchrun.
- Full training: fine-tuning all weights or pretraining (optionally fewer layers), held-out evaluation, FSDP across GPUs,
  results as new checkpoints with recipes; estimated model-FLOPs utilization.
- Python client packaged for PyPI (`blockyos-client`).
- Tests: complete experiments through the pipeline against a fake llama.cpp; CI on Python 3.12 and 3.13, lint (ruff),
  coverage.

### Changed
- Backend routes split into `backend/app/routes/` (route resolution verified unchanged); typed request bodies for the
  experiment endpoints (extra fields still accepted).
- Background jobs (downloads, weight builds, quantizations, remote-device jobs) persist across restarts; remote-device
  jobs resume.
- Python 3.12 or newer is required and checked by the backend and installer.

### Fixed
- Security: DNS rebinding (loopback requests naming another host now need the remote-access token) and cross-site
  writes (requests from another site's Origin are refused).
- Experiment runs failed on machines without `nvidia-smi` (CPU-only, AMD or Intel GPUs).
- Agent: Stop only cancels its own chat's approvals; approvals bind to the pending tool.
- The Python-client download path after the route split.

## [0.1.0] — 2026-11-15

First public snapshot: experiments with provenance and paired statistics, engines, weights workshop, layer scans,
PyTorch lab, quality tests, energy, cheap-device simulation, remote devices, agent and MCP server.
