# Changelog

All notable changes. Versions follow [semantic versioning](https://semver.org); the **measurement protocol** has its
own version (`PROTOCOL` in `backend/app/profile.py`, specified in [docs/PROTOCOL.md](docs/PROTOCOL.md)) and is listed
with each release. Results are only comparable within one protocol version.

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
