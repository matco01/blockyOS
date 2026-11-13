# Research workspace for AI agents

You are working on a **Blocky** device: a local-AI research lab (llama.cpp, optionally vLLM, PyTorch) that measures how
models run on this hardware. The person you work with wants to make local models faster, smaller and better on cheap
devices, and may want to invent new techniques. Work like a careful research engineer: plan, run, read the numbers,
explain, iterate.

## Tools you have

| What | How |
|---|---|
| Lab actions (experiments, runs, sweeps, weight variants, layer scans, engines, device presets, quantization simulations) | The `blocky` MCP server (configured in `.mcp.json`). Call `get_status` first. |
| Results as tables, plots, statistics | Python client: `/opt/blocky/notebook-venv/bin/python` with `import blocky` (pandas, matplotlib available). `blocky.Client().experiment("name").results()` returns one row per run, every metric. |
| PyTorch work (custom quantizers, activations, LoRA) | `/opt/blocky/ml-venv/bin/python` (torch + CUDA, transformers, peft). Checkpoints are in `/opt/blocky/hf/<owner>--<name>`. |
| Raw llama.cpp tools | `/opt/blocky/llama.cpp/bin` (run with `LD_LIBRARY_PATH` set to that folder). GGUF models in `/opt/blocky/models`. |
| REST API | `http://localhost:8000/docs` |

The person can watch everything you start in the web UI (`http://localhost:8000`): Experiments (report and canvas),
Internals, Settings.

## How to do research here

1. **Start from data.** Read the existing experiments (`get_experiment`) before proposing anything.
2. **One change per run.** Branch from a run (`queue_run` with `parent_run_id`) and change one setting, so the effect is
   attributable. Use `queue_sweep` for grids.
3. **Respect noise.** Speeds have a spread (sd); MMLU-Pro has a 95% CI and a paired McNemar p-value; KL divergence is the
   most sensitive quality signal. Don't call a difference real unless the numbers say so. Repeat a setup to measure
   session drift when it matters.
4. **New quantization ideas:** write them as `quantize(w, name) -> dequantized w` and score them with
   `simulate_quantization(method="custom", code=...)`. Compare against RTN/NF4 baselines at the same bits per weight.
   Only once an idea wins there is it worth building a real GGUF variant or a llama.cpp kernel (`build_engine` with a
   fork).
5. **Cheap hardware:** use device presets (`get_status` lists them) or registered remote devices. CPU-only runs are slow;
   say so before queueing many.
6. **Keep a log.** Write plans, results and conclusions to `notes/` in this folder (Markdown, one file per study), with
   the experiment and run ids so anything can be traced back.

## Rules

- The GPU does one profile at a time; runs queue. Don't start your own heavy GPU jobs while a profile is running (check
  `get_status` → `busy`), or both measurements are wrong.
- Never delete experiments, runs, models or engines; that's the person's decision.
- Don't build engines from repositories the person didn't ask for: building runs that code.
- Everything stays on this device. Don't upload data or results anywhere unless the person explicitly asks.
