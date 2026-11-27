# Notes for AI coding agents

Blocky is a privacy-first local-AI system: llama.cpp (and optionally vLLM) serving models, a FastAPI backend and a React UI,
with a research lab for measuring how models run on the hardware they're on. It runs on Linux (a dedicated box or WSL2).

To *use* the lab (run experiments, read results), connect to the MCP server at `http://localhost:8000/mcp` instead of
editing code. This file is for changing Blocky itself.

## Layout

| Path | What |
|---|---|
| `backend/app/main.py` | FastAPI app wiring: lifespan (DB init, workers, MCP session manager), access control (loopback, Host, Origin), router order |
| `backend/app/routes/` | REST endpoints, one module per area (`experiments.py`, `chat.py`, `torch.py`, …); `common.py` holds shared helpers and app state |
| `backend/app/projects.py` | Projects: group experiments and code (research projects) by what is being made; never owns or deletes them |
| `backend/app/jobstore.py` | Saves in-memory background jobs (downloads, weight builds, quantizations, remote-device jobs) so they survive restarts |
| `backend/app/rpcworker.py` | This device as a llama.cpp RPC worker (multi-machine runs); off and localhost-only by default |
| `backend/torch_tools/train.py` | Full training (all weights, or pretraining) with a held-out evaluation; FSDP under torchrun |
| `backend/app/bundle.py`, `reproduce_template.py` | Reproducibility bundle per experiment, and the script inside it that re-runs and compares |
| `backend/tests/fake_llama.py`, `test_pipeline_e2e.py` | A fake llama.cpp and complete experiments through the real pipeline |
| `docs/PROTOCOL.md` | The measurement specification: update it with every measurement change |
| `backend/app/instrument.py` | Instrument checks: properties a correct measurement pipeline must show (baseline vs itself, quant order, bandwidth, cited values) |
| `backend/app/profile.py` | Experiments: run tree, config validation, the measurement protocol, statistics, analysis, import/export |
| `backend/app/mcp_server.py` | MCP tools for AI assistants (thin wrappers over the modules below) |
| `backend/app/engines.py`, `weights.py`, `scans.py`, `torchlab.py`, `kernels.py`, `quantlab.py` | Engine builds, weight workshop, layer scans, PyTorch jobs, nsys kernel profiles, quantization helpers |
| `backend/torch_tools/` | Scripts that run inside the ML venv (activations, LoRA/QLoRA) |
| `backend/tests/` | pytest suite (no GPU or models needed) |
| `frontend/src/` | React + TypeScript UI (Vite). `ProjectsPage.tsx` (front door), `ExperimentsPage.tsx`, `ResearchCanvas.tsx`, `expmetrics.ts`, `GuidePage.tsx` |
| `client/blocky.py` | Python client used in notebooks |
| `scripts/` | Install and setup scripts (`install.sh`, `setup-ml.sh`, `setup-vllm.sh`, `setup-sglang.sh`, `setup-trtllm.sh`, `setup-notebooks.sh`) |

## Checks before you finish

The backend needs Python 3.12 or newer.

```bash
# backend (from backend/)
python -m pytest -q
# lint (from the repository root)
ruff check backend client
# frontend (from frontend/)
npm test && npm run build
```

CI runs the same on every push (`.github/workflows/ci.yml`).

## Rules

- **Measurement changes are protocol changes.** If you change how a metric is measured, bump `PROTOCOL` in
  `profile.py` so results from different versions are never compared silently, and update docs/PROTOCOL.md and
  CHANGELOG.md. The end-to-end tests (fake llama.cpp) must keep passing; extend the fake when a tool's output changes.
- **Report noise, not just numbers.** New metrics need a spread (sd / CI) and, where it applies, a significance test.
  The UI must not state conclusions the data doesn't support.
- **Generic product text.** The UI and docs are for every user and every device: never hard-code facts about one
  machine (a specific GPU, driver version or operating system quirk).
- **Local only by default.** Services bind to localhost. Don't add anything that sends data off the device or opens a
  port to the network without an explicit user setting. Requests count as local only when they come from loopback *and*
  name this device in `Host` (DNS rebinding); writes from another site's `Origin` are refused (CSRF). Keep both checks.
- **New endpoints go in `backend/app/routes/<area>.py`.** Use `routes.common.state` for the llama-server client and
  background tasks, not `app.state`. A new area needs its router added to `ROUTERS` in `main.py`; order only matters where
  two paths could match the same URL.
- **Settings that change how a run is measured** (sample sizes, test sets) go into the run's provenance and into
  `compare()`'s mismatch list, so runs measured differently are never compared silently. New config keys are stored only
  when set, so older runs' configs stay equal to new ones (repeat detection depends on it).
- **Destructive actions need the owner's consent.** Background jobs and the in-UI agent don't delete user data. Through
  MCP, deleting is possible only at the `full` access level the owner chose (Settings → AI assistants), through the
  `api_delete` tool marked destructive. MCP never exposes secrets or the endpoints that manage access
  (`mcp_server.BLOCKED`, `SECRET_KEYS`, `aiaccess.SECRET_TABLES`); new secret settings or access endpoints must be added
  there.
- Keep it simple: plain modules and functions, SQLite via `db.connect()`, no new frameworks without a clear need.
