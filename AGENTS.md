# Notes for AI coding agents

Blocky is a privacy-first local-AI system: llama.cpp (and optionally vLLM) serving models, a FastAPI backend and a React UI,
with a research lab for measuring how models run on the hardware they're on. It runs on Linux (a dedicated box or WSL2).

To *use* the lab (run experiments, read results), connect to the MCP server at `http://localhost:8000/mcp` instead of
editing code. This file is for changing Blocky itself.

## Layout

| Path | What |
|---|---|
| `backend/app/main.py` | FastAPI app: every REST endpoint, lifespan (DB init, profile queue worker, MCP session manager) |
| `backend/app/profile.py` | Experiments: run tree, config validation, the measurement protocol, statistics, analysis, import/export |
| `backend/app/mcp_server.py` | MCP tools for AI assistants (thin wrappers over the modules below) |
| `backend/app/engines.py`, `weights.py`, `scans.py`, `torchlab.py`, `kernels.py`, `quantlab.py` | Engine builds, weight workshop, layer scans, PyTorch jobs, nsys kernel profiles, quantization helpers |
| `backend/torch_tools/` | Scripts that run inside the ML venv (activations, LoRA/QLoRA) |
| `backend/tests/` | pytest suite (no GPU or models needed) |
| `frontend/src/` | React + TypeScript UI (Vite). `ExperimentsPage.tsx`, `ResearchCanvas.tsx`, `expmetrics.ts`, `GuidePage.tsx` |
| `client/blocky.py` | Python client used in notebooks |
| `scripts/` | Install and setup scripts (`install.sh`, `setup-ml.sh`, `setup-vllm.sh`, `setup-notebooks.sh`) |

## Checks before you finish

```bash
# backend (from backend/)
python -m pytest -q
# frontend (from frontend/)
npm test && npm run build
```

CI runs the same on every push (`.github/workflows/ci.yml`).

## Rules

- **Measurement changes are protocol changes.** If you change how a metric is measured, bump `PROTOCOL` in
  `profile.py` so results from different versions are never compared silently.
- **Report noise, not just numbers.** New metrics need a spread (sd / CI) and, where it applies, a significance test.
  The UI must not state conclusions the data doesn't support.
- **Generic product text.** The UI and docs are for every user and every device: never hard-code facts about one
  machine (a specific GPU, driver version or operating system quirk).
- **Local only by default.** Services bind to localhost. Don't add anything that sends data off the device or opens a
  port to the network without an explicit user setting.
- **No destructive actions from tools.** The MCP server and background jobs don't delete user data; deleting stays a
  user action in the UI.
- Keep it simple: plain modules and functions, SQLite via `db.connect()`, no new frameworks without a clear need.
