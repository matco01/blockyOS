# blockyOS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24324900.svg)](https://doi.org/10.5281/zenodo.24324900)
[![CI](https://github.com/matco01/blockyOS/actions/workflows/ci.yml/badge.svg)](https://github.com/matco01/blockyOS/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/blockyos-client?label=blockyos-client)](https://pypi.org/project/blockyos-client/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**A self-hosted laboratory for measuring how language models actually run on the hardware in front of you.**

blockyOS turns a machine running Ubuntu Server into a dedicated, private AI appliance. Switch it on, open its web page,
and you have a local chat assistant and a full research lab. The lab measures every trade-off of local inference with
a fixed, versioned protocol, using statistics that hold up in a paper:
- quantization,
- inference engine and engine version,
- context size and KV-cache precision,
- one GPU, several GPUs or several machines,
- fine-tuning and pretraining.

Nothing leaves the device unless you explicitly connect something that sends it out.

---

## Why it exists

Most numbers about local LLM inference come from a single benchmark run on an undocumented machine. Typically there
are a few prompts for "quality", speed without any measure of spread, and no record of the engine build. Such results
can't be compared or reproduced.

blockyOS is built around experiments instead. You pick one model, change one setting at a time, and every
configuration is profiled the same way:

| Measured | How |
|---|---|
| **Accuracy** | MMLU-Pro on a fixed set of 560 questions; paired McNemar test against the baseline, with a Newcombe interval and Holm correction |
| **Fidelity** | Perplexity, and KL divergence against the baseline's own logits on held-out text |
| **Long context** | Needle-in-a-haystack retrieval, from 2k tokens up to the full context, at 5 depths |
| **Speed** | llama-bench prefill and decode (repeated; median and robust spread), plus an engine-neutral serving test |
| **Energy** | Idle power and power under load, from a wall-power smart plug or GPU/CPU sensors; energy per token |
| **Hardware** | Memory layout, a roofline analysis against spec-sheet peaks, clocks, VRAM headroom |

Every run records its **provenance**:
- the protocol version and the engine commit,
- SHA-256 hashes of the model and datasets,
- driver and CUDA versions,
- the exact commands.

Comparisons between runs measured under different conditions are **refused, not shown**.

## Who it's for

- **Researchers** comparing quantization methods, engines or kernels, who need results they can defend and others can
  reproduce.
- **Engineers** choosing the right model and configuration for specific hardware, from a single-board computer to a
  multi-GPU server.
- **Anyone** who wants a private, local AI assistant on their own machine.

---

## Research-grade by design

- **Specified protocol.** Every number is produced as written in [docs/PROTOCOL.md](docs/PROTOCOL.md): commands,
  data, sample sizes, statistics, comparability rules and known limitations.
- **Instrument checks.** blockyOS tests itself on your machine:
  - a rerun of the baseline must give near-zero KL divergence and the same answers;
  - fewer bits per weight must not score better;
  - speed must stay under the memory-bandwidth limit.

  See [docs/VALIDATION.md](docs/VALIDATION.md).
- **Statistics.**
  - Paired tests and paired confidence intervals.
  - Equivalence testing (TOST) against a margin chosen in advance.
  - Counterbalanced repeated pairs (AB/BA) to confirm speed differences.
  - A planner that turns the measured noise into the number of runs needed.
- **Preregistration.** Hypotheses and analysis plans can be recorded before any data exist. They can't be edited
  afterwards, and runs are labelled confirmatory or exploratory.
- **Noise controls.** Optional GPU clock locking and a cool-down between runs, both recorded in provenance.
- **Reproducibility bundles.** Each experiment exports as one zip with a `reproduce.py` script. The bundle contains:
  - provenance and commands,
  - raw tool output,
  - the instrument-check report.

  `reproduce.py` re-runs every configuration on another blockyOS machine and compares the results.

---

## What you can do

### Measure
- **Experiments**: the research hub. It runs a queue and full parameter sweeps, measures session noise with repeated
  baselines, and exports CSV/JSON. The **Canvas** view is a pan/zoom map of an experiment.
- **Benchmarks**: llama-bench presets and custom sweeps, with power sampling and tokens per joule.
- **Quality tests**:
  - questions generated fresh at run time, which can't have been memorized;
  - established benchmarks (GSM8K, ARC, MMLU-Pro, MATH-500, HumanEval+, LiveCodeBench and others), with contamination
    warnings;
  - lm-evaluation-harness tasks;
  - an OpenAI-compatible `/v1` endpoint for outside tools.
- **Cheap-device simulation**: profile a configuration as if it ran on a weaker device by limiting cores, RAM and VRAM
  and going CPU-only. A projected speed is calculated from the target's memory bandwidth.

### Change
- **Engines**: build any llama.cpp fork, branch, tag or commit for this GPU (CUDA, Vulkan or CPU) and A/B it against
  stock. vLLM, SGLang and TensorRT-LLM run Hugging Face checkpoints.
- **Weights workshop**: build quantized variants with llama.cpp's tools. Options include:
  - an importance matrix,
  - per-tensor precision,
  - layer pruning.

  Each variant's recipe ends up in the run's provenance.
- **Layer sensitivity**: quantize or remove one layer at a time and measure the damage against a control noise floor.
- **Technique playground**: prototype a quantization method in Python before writing a kernel, either with reference
  methods (GPTQ, AWQ, RTN, NF4, ternary) or your own function. Then export a real GGUF and profile it.

### Train and inspect
- **PyTorch internals**:
  - an activations explorer and logit lens,
  - attention entropy,
  - layer knockout,
  - GPU time per layer.
- **Training**:
  - LoRA and QLoRA;
  - full fine-tuning;
  - pretraining from scratch;
  - FSDP across several GPUs, or several machines.

  Results become new checkpoints that go straight back into Experiments.

### Scale out
- **Several GPUs**: GPU count is a variable, with layer or row split in llama.cpp and tensor parallelism in
  vLLM/SGLang.
- **Several machines**:
  - llama.cpp RPC workers;
  - vLLM pipeline stages over Ray;
  - multi-machine training with torchrun.
- **Remote devices**: one blockyOS machine profiles configurations on others and imports the results.

### Automate
- **Python client**: `pip install blockyos-client`. Queue runs and sweeps, then get every result as a pandas table, or
  send it to MLflow or Weights & Biases.
- **JupyterLab** on the device.
- **MCP server** at `http://localhost:8000/mcp`. Claude Code, Claude Desktop and other assistants can read results
  and queue experiments under the same rules. There are no delete tools.
- **Research agent** in the UI. It works through a task with every lab tool, and you approve each code run. It runs on
  the local model, or on Claude with your own API key.

### Everyday use
- **Chat** with sampling controls, per-answer speed and timing, and a per-word confidence heatmap.
- **Models**: a Hugging Face browser and downloads, GGUF metadata, a fit estimate for this machine, and one-click
  model and context switching.
- **System**: live CPU, GPU, RAM, VRAM and power, plus the model-server log.

---

## How it works

```
Ubuntu Server 24.04  (dedicated machine, or WSL2 for development)
│
├── blocky-web   (systemd)  FastAPI backend + React UI on :8000, SQLite storage, MCP server
│   ├── experiment runs     llama-bench, llama-perplexity, engine servers
│   │                       (llama.cpp / vLLM / SGLang / TensorRT-LLM), started per run
│   └── PyTorch lab         separate environment for internals and training
├── blocky-llm   (systemd)  llama-server for chat, restarted on model switch
│
└── /opt/blocky/models   /opt/blocky/data   /etc/blocky/blocky.env
```

The installer sets up the whole machine:
- a dedicated service account,
- llama.cpp, built for your GPU,
- services that start at boot,
- narrow root permissions (restart the model server; optionally lock GPU clocks) and nothing more.

By default the web interface listens only on the machine itself. Network access must be switched on explicitly and
then needs a token. See [docs/SETUP.md](docs/SETUP.md#remote-access).

---

## Getting started

**Requirements:** Ubuntu 24.04 (dedicated machine or WSL2), Python 3.12+, and Node.js to build the UI. An NVIDIA GPU
is optional: CPU-only, AMD and Intel machines work through CPU or Vulkan builds.

**Dedicated machine**
```bash
git clone https://github.com/matco01/blockyOS && cd blockyOS
sudo ./scripts/install.sh
cd frontend && npm install && npm run build
```
Then open `http://localhost:8000` on the machine, go to **Models**, and download a model.

**Windows PC (development, through WSL2):** see [docs/SETUP.md](docs/SETUP.md#a-on-a-windows-pc-development-via-wsl2).
It covers:
- creating the WSL instance,
- installing the CUDA toolkit,
- auto-start,
- what isn't in git and how to back it up.

**Optional extras** have their own setup scripts in `scripts/`:
- the PyTorch lab and lm-evaluation-harness: `setup-ml.sh`;
- JupyterLab: `setup-notebooks.sh`;
- vLLM, SGLang and TensorRT-LLM: `setup-vllm.sh`, `setup-sglang.sh` and `setup-trtllm.sh`.

### Everyday commands (Windows, PowerShell)

```powershell
wsl -d blocky                                        # open a Linux terminal (keeps blockyOS running while open)
wsl -d blocky -- sudo systemctl restart blocky-web   # after backend code changes
wsl -d blocky -- sudo systemctl restart blocky-llm   # after changing model or settings
wsl -d blocky -- journalctl -u blocky-llm -f         # follow the model-server log
```

On a dedicated machine, drop the `wsl -d blocky --` prefix.

---

## Project status

blockyOS is **0.2.0, early research software**. Here is what has been verified so far:
- **Automated tests** cover the statistics, the API and complete experiments run against a simulated engine:
  - 205 backend tests;
  - 26 frontend tests;
  - CI on Python 3.12 and 3.13.
- **Not yet published:** a validation study on real GPUs, which compares blockyOS against published reference values
  and against itself across repeated sessions. Until it exists, run the instrument checks on your own machine before
  relying on results. The procedure is in [docs/VALIDATION.md](docs/VALIDATION.md).

Measurement-protocol versions are listed in the [CHANGELOG](CHANGELOG.md). Results are only comparable within one
protocol version.

---

## Repository layout

| Path | Contents |
|---|---|
| `backend/app/` | FastAPI backend: experiments, statistics, engines, training, MCP server |
| `backend/tests/` | pytest suite, including end-to-end runs against a fake llama.cpp |
| `frontend/src/` | React + TypeScript UI (built into `frontend/dist` and served by the backend) |
| `client/` | `blockyos-client` Python package |
| `scripts/` | Installer, optional setup scripts, systemd services, Windows auto-start |
| `research/` | Workspace for your own research projects and coding agents ([guide](docs/RESEARCH_PROJECTS.md)) |
| `docs/` | Setup, measurement protocol, validation, research projects |
| `paper/` | Software paper draft |

---

## Citing

If you use blockyOS in research, cite it with the DOI
[10.5281/zenodo.24324900](https://doi.org/10.5281/zenodo.24324900), or use **"Cite this repository"** on GitHub,
which reads [CITATION.cff](CITATION.cff). Include the version and the protocol string from your runs' provenance.

## Contributing and license

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).
Coding agents working on blockyOS itself should read [AGENTS.md](AGENTS.md).

Licensed under [Apache-2.0](LICENSE). Third-party components keep their own licenses; see [NOTICE](NOTICE).

## Removing it

- **WSL:** `wsl --unregister blocky` deletes the Linux instance and everything in it. Windows itself is untouched.
- **Dedicated machine:** blockyOS is meant to own the machine; reinstalling Ubuntu Server is the clean way to remove it.
