"""Blocky as an MCP server: AI assistants (ChatGPT, Claude, Claude Code, other MCP clients) see and use the whole lab.

Served at /mcp (Streamable HTTP). Assistants on this device connect directly and get the owner's local access level;
cloud assistants (ChatGPT, Claude on the web and phone) connect through the owner's HTTPS tunnel and sign in with OAuth,
each with the level the owner approved (aiaccess.py): read, write (no deleting) or full.

Two kinds of tools:
- Lab tools (get_status, queue_run, build_engine, ...) wrap the common research steps with guidance in their descriptions.
- General tools reach everything else: list_api describes every endpoint of Blocky's API, api_get / api_call /
  api_delete call any of them (each needs the matching level), and query_database runs read-only SQL on the database.
Secrets never leave through MCP: API keys, passwords and access tokens are removed, and the endpoints that manage
access itself (remote access, AI-assistant access, the agent's API key) can't be called through MCP.
"""
import asyncio
import contextvars
import functools
import inspect
import json
import sqlite3
import time

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import db, engines, lab, profile, scans, torchlab, weights, research_jobs

INSTRUCTIONS = """Blocky is a local-AI research lab running on the user's own hardware. You can measure and improve how
models run on this device.

How it works:
- An *experiment* is a tree of *runs*. Each run profiles one setup (model file, engine, KV cache, context size, GPU
  layers, VRAM limit, flash attention, threads) with an identical protocol: memory layout, MMLU-Pro (560 fixed
  questions, 5-shot top-k option-letter scoring), long-context needle test, engine-neutral serving throughput, llama-bench speed,
  perplexity + KL divergence vs the experiment's baseline, telemetry, and a roofline-style hardware analysis.
- Branch from a run: the new run = that run's exact setup + your changes. Change ONE thing at a time.
- A profile takes ~3-10 minutes and needs the GPU: runs wait in a queue and execute one after another. Use
  get_status / get_experiment to follow progress; don't poll more than every ~30 s.
- Read significance, not raw differences: speed spread is descriptive, not a standard error; MMLU-Pro has a 95% CI and a paired McNemar
  p-value vs baseline/parent; KL divergence is the most sensitive quality signal (0 = identical outputs).
- KL divergence / perplexity only compare models with the same tokenizer; vLLM runs don't have them.
- Protocol 3 serving throughput and energy include the full request, including prefill, and require actual server token counts.
- Different protocols or KL reference baselines cannot be compared. Create a fresh experiment when the measurement protocol changes.
- Hardware bottlenecks and weight-read time splits are estimates, not direct measurements of where time went.
- Weights: build variants with build_weight_variant (quantization type, importance matrix, per-tensor precision
  overrides as regexes, layer pruning) from a high-precision source; find sensitive layers with start_layer_scan.
- Energy: each run measures idle and generating power (wall meter if configured, else GPU/CPU sensors) -> energy per
  token; with prices set in Blocky's Settings, also electricity per 1M tokens and speed per device price.
- Cheap devices: queue_run(changes={'preset': ...}) simulates CPU-only boxes, fewer cores, less RAM/VRAM (presets in
  get_status). Capacity and cores are real limits; bandwidth isn't, so CPU presets add a projected decode speed.
- New techniques: simulate_quantization tries a quantization method (or your own Python) on a checkpoint and scores it
  against the unmodified model before anyone writes a kernel.
- Be economical: every run costs minutes of the user's machine (CPU-only runs: much longer). Plan, explain, then queue.
- Everything else in Blocky (chat history, benchmarks, quality tests, engines, models and downloads, research projects,
  the agent, energy, devices, settings): call list_api to find the endpoint, then api_get / api_call / api_delete.
  query_database answers questions across all stored data with read-only SQL (call it with no sql to see the tables).
- Your access level (read, write or full) is set by the owner; a refused call says which level it needs. Deleting is
  permanent: confirm with the user before api_delete.
"""

server = MCPServer(name="blocky", title="Blocky local-AI lab", instructions=INSTRUCTIONS, version="2")

# The access level of the assistant making the current request; set per request by the /mcp entry point in main.py.
# In-process callers (tests, the backend itself) get full access.
LEVEL: contextvars.ContextVar[str] = contextvars.ContextVar("blocky_mcp_level", default="full")
_RANK = {"read": 0, "write": 1, "full": 2}


def _denied(needed: str) -> str | None:
    have = LEVEL.get()
    if _RANK.get(have, -1) >= _RANK[needed]:
        return None
    return json.dumps({"error": f"This needs '{needed}' access; this assistant was given '{have}'. The owner can reconnect it with "
                                f"more access (Blocky: Settings, AI assistants)."})


def tool(kind: str, **kwargs):
    """Register an MCP tool that needs access level `kind` (read, write or full). The plain function stays importable."""
    annotations = ToolAnnotations(readOnlyHint=kind == "read", destructiveHint=kind == "full", openWorldHint=False)

    def register(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def checked(*a, **k):
                return _denied(kind) or await fn(*a, **k)
        else:
            @functools.wraps(fn)
            def checked(*a, **k):
                return _denied(kind) or fn(*a, **k)
        server.tool(annotations=annotations, **kwargs)(checked)
        return fn
    return register


@tool("read", description="List custom research projects and jobs. These run architecture/training/evaluation programs alongside the built-in lab. project_id=0 lists projects; otherwise lists recent jobs with declared metrics and evidence.")
def get_research_projects(project_id: int = 0) -> str:
    try:
        if not project_id:
            return json.dumps(research_jobs.projects(), default=str)
        return json.dumps([{"id": j["id"], "status": j["status"], "parent_id": j["parent_id"], "error": j["error"],
                            "spec": j["spec"], "metrics": [e for e in j["events"] if e.get("type") == "metric"][-20:]}
                           for j in research_jobs.jobs(project_id)[:20]], default=str)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@tool("read", description="List registered research methods selectable in Experiments using model='method:<id>'. Each pins tested source and evaluator settings and declares supported metric contracts. Registration itself is a user action in Research projects.")
def get_registered_methods() -> str:
    from . import methods
    return json.dumps(methods.list_methods())


@tool("read", description="Compare 2–8 research jobs or built-in profiles (use 'profile:<run id>'). Reports matching/mismatched dataset, task, measurement protocol and units. Does not infer significance from raw differences.")
def compare_research_jobs(run_ids: list[str]) -> str:
    try:
        return json.dumps(research_jobs.compare(run_ids), default=str)
    except (ValueError, TypeError) as e:
        return json.dumps({"error": str(e)})


def _energy(e: dict | None) -> dict | None:
    if not e or not e.get("primary"):
        return {"note": e.get("note") or e.get("error")} if e else None
    s = e["sources"][e["primary"]]
    return {"measured": e["primary_label"], "j_per_token": s["j_per_token"], "net_j_per_token_above_idle": s.get("net_j_per_token"),
            "tokens_per_wh": s["tokens_per_wh"], "load_w": s["load_w"], "idle_w": s.get("idle_w"), "gen_tps": e["tps"]}


def _compact_run(r: dict) -> dict:
    """The numbers that matter, small enough for a model's context."""
    x = r.get("results") or {}
    sp, sv, m, hw = x.get("speed") or {}, x.get("serving") or {}, x.get("mmlu") or {}, x.get("hw") or {}
    return {
        "id": r["id"], "label": r["label"], "parent_id": r.get("parent_id"), "status": r["status"], "phase": r.get("phase"), "error": r.get("error"),
        "config": r.get("config"),
        "speed": {k: sp.get(k) for k in ("tg", "tg_sd", "pp", "pp_sd", "tg_d4096", "pp_d4096")} if sp else None,
        "serving": {k: sv.get(k) for k in ("tg", "tg_sd", "pp", "ttft_ms")} if sv else None,
        "mmlu_pro": {"acc": m.get("acc"), "ci95": m.get("ci95"), "n": m.get("n")} if m else None,
        "long_context_by_length": (x.get("needle") or {}).get("by_length"),
        "ruler": {k: (x.get("ruler") or {}).get(k) for k in ("score", "effective_length", "by_length", "by_task")} if x.get("ruler") else None,
        "lm_eval": {t: v.get("metrics") for t, v in ((x.get("lm_eval") or {}).get("tasks") or {}).items()} or None,
        "kld": x.get("kld"), "kld_p99": x.get("kld_p99"), "same_top_pct": x.get("same_top"), "ppl": x.get("ppl"),
        "shared_evaluations": x.get("research_metrics"), "research_job_id": x.get("research_job_id"),
        "vram_model_mb": (hw.get("headroom") or {}).get("vram_model_mb"), "weights_on_cpu_gb": hw.get("weights_on_cpu_gb"),
        "decode_roof_util": (hw.get("decode") or {}).get("roof_util"), "limiter": hw.get("limiter"),
        "energy": _energy(x.get("energy")), "cost": x.get("cost"),
        "hardware": {"gpu": (x.get("hardware") or {}).get("gpu"), "device": (x.get("hardware") or {}).get("device")},
        "stats_vs_baseline_and_parent": r.get("stats"),
        "notes": [n for n in (x.get("token_note"), x.get("fidelity_note"), sv.get("note"), sv.get("method"), m.get("probability_note")) if n] or None,
    }


@tool("read", description="Overview of this Blocky device: hardware limits, model files, Hugging Face checkpoints, engines, experiments, and whether the GPU is busy.")
def get_status() -> str:
    return json.dumps({
        "busy": lab.busy,
        "hardware": profile.limits(),
        "models": [{"file": m["file"], "quant": m.get("quant"), "gb": round(m["size_bytes"] / 2**30, 2), "name": m.get("name"), "arch": m.get("architecture")}
                   for m in lab.list_models() if not m.get("error")],
        "checkpoints": [f"hf:{c['id']}" for c in torchlab.checkpoints()],
        "engines": [{"id": e["id"], "kind": e.get("kind"), "status": e["status"], "ref": e.get("ref")} for e in engines.list_engines()],
        "experiments": [{"id": e["id"], "name": e["name"], "model": e["model"], "runs": e.get("runs")} for e in profile.list_experiments()],
        "queued_or_running_runs": _active_runs(),
        "device_presets": {k: {"label": v["label"], **v["config"]} for k, v in profile.DEVICE_PRESETS.items()},
    }, default=str)


def _active_runs() -> list[dict]:
    with db.connect() as c:
        return [dict(r) for r in c.execute("SELECT id, experiment_id, label, status, phase FROM exp_runs WHERE status IN ('queued', 'running') ORDER BY id")]


@tool("read", description="All runs of one experiment with their key results (speed with spread, MMLU-Pro with CI, long context, KL divergence, memory, hardware limiter, significance vs baseline and parent).")
def get_experiment(experiment_id: int) -> str:
    e = profile.get_experiment(experiment_id)
    if not e:
        return json.dumps({"error": "Experiment not found"})
    return json.dumps({"id": e["id"], "name": e["name"], "baseline_model": e["model"], "protocol_extras": e.get("protocol"),
                       "runs": [_compact_run(r) for r in e["runs_list"]]}, default=str)


@tool("read", description="Instrument checks for an experiment: is the measurement pipeline behaving correctly on this machine? Baseline rerun vs "
                         "itself (KLD ≈ 0, same answers, speed drift), fewer bits never better KLD, speed under the bandwidth limit, and matches with "
                         "cited reference values. Each check is pass / warn / fail / not_run (with the runs to add). Check this before trusting comparisons.")
def get_instrument_check(experiment_id: int) -> str:
    from . import instrument
    e = profile.get_experiment(experiment_id)
    return json.dumps(instrument.check(e) if e else {"error": "Experiment not found"})


@tool("write", description="Preregister a hypothesis for an experiment before running it: primary_metric (mmlu, kld, same_top, ppl, tg, pp, srv_tg, "
                         "jpt, needle), claim (difference, equivalence, non_inferiority), direction, alpha, min_effect, equivalence_margin_pts (needed for "
                         "accuracy equivalence), analysis text. Append-only: runs queued afterwards are confirmatory, earlier ones exploratory.")
def preregister(experiment_id: int, hypothesis: str, primary_metric: str, claim: str = "difference", direction: str = "either",
                alpha: float = 0.05, min_effect: float | None = None, equivalence_margin_pts: float | None = None, analysis: str = "") -> str:
    try:
        return json.dumps(profile.preregister(experiment_id, {"hypothesis": hypothesis, "primary_metric": primary_metric, "claim": claim,
                                                              "direction": direction, "alpha": alpha, "min_effect": min_effect,
                                                              "equivalence_margin_pts": equivalence_margin_pts, "analysis": analysis}))
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("write", description="Create an experiment for a model (a GGUF file name, or 'hf:<checkpoint>' for vLLM). Then queue a baseline with queue_run (no parent).")
def create_experiment(model: str, name: str = "") -> str:
    try:
        return json.dumps({"experiment_id": profile.create_experiment(name or model, model)})
    except ValueError as e:
        return json.dumps({"error": str(e)})


@tool("write", description="Queue one profiled run. parent_run_id: the run to branch from (its exact setup + `changes`); omit for the baseline. "
                         "changes keys: model, engine, kv (f16|q8_0|q4_0), ctx, ngl ('auto' or a number), vram_limit_mb, fa (auto|on|off), threads, "
                         "device (gpu|cpu), cores (pin to N CPU threads), ram_limit_mb, preset (a cheap-device simulation from get_status; "
                         "it fixes the device settings, set preset to '' to customize them).")
def queue_run(experiment_id: int, changes: dict | None = None, parent_run_id: int | None = None, label: str = "") -> str:
    e = profile.get_experiment(experiment_id)
    if not e:
        return json.dumps({"error": "Experiment not found"})
    parent = next((r for r in e["runs_list"] if r["id"] == parent_run_id), None) if parent_run_id else None
    base_cfg = parent["config"] if parent else {}
    if str((changes or {}).get("model", "")).startswith("method:") and (changes or {}).get("model") != base_cfg.get("model"):
        base_cfg = {}
    elif base_cfg.get("engine") == "research" and (changes or {}).get("model") and not changes["model"].startswith("method:"):
        base_cfg = {}
    try:
        rid = profile.create_run(experiment_id, {"label": label, "config": {**base_cfg, **(changes or {})}, "parent_id": parent_run_id})
        return json.dumps({"run_id": rid, "queued": True})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("write", description="Queue a sweep: every combination of the given values (grid: {setting: [values]}), from a parent run or the baseline. Max 32 runs.")
def queue_sweep(experiment_id: int, grid: dict, parent_run_id: int | None = None, repeats: int = 1) -> str:
    try:
        return json.dumps({"run_ids": profile.create_sweep(experiment_id, {"grid": grid, "parent_id": parent_run_id, "repeats": repeats})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("write", description="Cancel a queued or running profile.")
def cancel_run(run_id: int) -> str:
    profile.cancel(run_id)
    return json.dumps({"ok": True})


@tool("write", description="Set the tests every run of an experiment gets (replaces the previous choice). mmlu_questions: 560 (default), 1400 or 3500 "
                         "(larger detects smaller accuracy differences; nested, so paired tests vs older runs still work). ruler_samples: 0 (off), or "
                         "samples per task and length for the RULER-style long-context suite (4 quick, 8 standard, 20 precise). lm-evaluation-harness "
                         "tasks (generative, e.g. 'gsm8k,ifeval') with a per-task limit, and/or a GPU kernel profile. ppl_chunks: perplexity/KLD "
                         "sample, 32 (default), 64, 128 or 256 chunks of 512 tokens; set it before the baseline (KLD needs the same text for every run). "
                         "fidelity_text: absolute path of a .txt on the device to score perplexity/KLD on instead of WikiText-2 (e.g. code or another "
                         "language). equivalence_margin_pts: accuracy margin for equivalence tests. lock_clocks_mhz / cooldown_s: noise controls.")
def set_experiment_tests(experiment_id: int, mmlu_questions: int = 560, ruler_samples: int = 0, lm_eval_tasks: str = "", lm_eval_limit: int = 100,
                         kernel_profile: bool = False, ppl_chunks: int = 32, fidelity_text: str = "", equivalence_margin_pts: float = 0.0,
                         lock_clocks_mhz: int = 0, cooldown_s: int = 0) -> str:
    try:
        return json.dumps(profile.set_protocol(experiment_id, {"lm_eval": {"tasks": lm_eval_tasks, "limit": lm_eval_limit} if lm_eval_tasks else None,
                                                               "kernels": kernel_profile, "mmlu_per_category": max(1, mmlu_questions // 14),
                                                               "ruler": {"samples": ruler_samples} if ruler_samples else None,
                                                               "ppl_chunks": ppl_chunks, "fidelity_text": fidelity_text,
                                                               "equivalence_margin_pts": equivalence_margin_pts or None,
                                                               "lock_clocks_mhz": lock_clocks_mhz or None, "cooldown_s": cooldown_s or None}))
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("read", description="Tensor groups of a GGUF model (name pattern, count, size, current types): use before choosing per-tensor overrides.")
def list_tensor_groups(model: str) -> str:
    try:
        return json.dumps(weights.tensor_groups(model)[:60], default=str)
    except (OSError, ValueError) as err:
        return json.dumps({"error": str(err)})


@tool("write", description="Build a weight variant from a high-precision GGUF source with llama-quantize: qtype (e.g. Q4_K_M, Q3_K_M, IQ4_XS), "
                         "imatrix (importance matrix), overrides {regex_pattern: tensor_type} (first match wins; e.g. {'blk\\\\.(0|23)\\\\.': 'q8_0', 'attn_v': 'q6_k'}), "
                         "output_type / embedding_type, prune (layer numbers). The result appears as a model file; profile it with queue_run(changes={'model': ...}).")
async def build_weight_variant(source: str, qtype: str, imatrix: bool = False, overrides: dict | None = None, output_type: str = "",
                               embedding_type: str = "", prune: list[int] | None = None, name: str = "") -> str:
    try:
        jid = weights.start({"source": source, "qtype": qtype, "imatrix": imatrix, "overrides": overrides or {}, "output_type": output_type or None,
                             "embedding_type": embedding_type or None, "prune": prune or [], "name": name})
        return json.dumps({"job_id": jid, "output_file": weights.jobs[jid]["out"]})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("write", description="Layer sensitivity scan: change one layer at a time (mode 'quantize' to qtype q2_k|q3_k|q4_k|iq4_xs, or 'prune' to remove it) and measure KL divergence vs the source. Slow on big models (~1-2 h for 9B).")
async def start_layer_scan(model: str, mode: str = "quantize", qtype: str = "q2_k") -> str:
    try:
        return json.dumps({"scan_id": scans.start({"model": model, "mode": mode, "qtype": qtype})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("read", description="Results of layer scans for a model (per-layer extra KL divergence over the control, most / least sensitive layers).")
def get_layer_scans(model: str) -> str:
    out = []
    for s in scans.list_scans(model)[:5]:
        r = s["results"]
        out.append({"id": s["id"], "mode": s["mode"], "qtype": s["qtype"], "status": s["status"], "step": s["step"],
                    "control_kld": (r.get("control") or {}).get("kld"), "most_sensitive": r.get("most_sensitive"), "least_sensitive": r.get("least_sensitive"),
                    "layers": [{"layer": x["layer"], "excess_kld": x["excess_kld"]} for x in r.get("layers", [])]})
    return json.dumps(out)


@tool("write", description="Build a llama.cpp engine from an https git repository and ref (branch, tag, commit) so it can be an experiment variable. "
                         "backend: auto | cuda | vulkan | cpu. "
                         "This compiles and runs that repository's code on the user's machine: only use repositories the user asked for or trusts.")
async def build_engine(repo: str, ref: str = "master", name: str = "", backend: str = "auto") -> str:
    try:
        return json.dumps({"engine_id": engines.add({"repo": repo, "ref": ref, "name": name, "backend": backend})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("read", description="Status of background jobs: weight builds, layer scans, engine builds, PyTorch jobs.")
def get_jobs() -> str:
    return json.dumps({
        "weights": [{k: j.get(k) for k in ("id", "out", "status", "step", "error")} for j in weights.list_jobs()[:10]],
        "scans": [{k: s.get(k) for k in ("id", "model", "mode", "status", "step", "error")} for s in scans.list_scans()[:10]],
        "engines": [{k: e.get(k) for k in ("id", "status", "step", "error", "commit_sha")} for e in engines.list_engines()],
        "torch": [{**{k: j.get(k) for k in ("id", "kind", "status", "step", "error")},
                   **({"results": {k: (j.get("results") or {}).get(k) for k in ("checkpoint", "gguf", "first_eval", "final_eval", "first_loss",
                                                                               "final_loss", "mfu_estimate", "other_nodes_command", "end")}}
                      if j.get("kind") == "train" else {})} for j in torchlab.jobs()[:10]],
    }, default=str)


@tool("write", description="Full training on a Hugging Face checkpoint: mode 'finetune' updates every weight; 'pretrain' starts from random weights "
                         "with the checkpoint's architecture and tokenizer (layers: optionally fewer). data: a Hugging Face dataset id (id:config) "
                         "or an absolute .txt/.jsonl path; chat-style rows train on answers, text rows are packed into max_len blocks. A held-out "
                         "share (eval_frac) gives loss/perplexity before, during and after. gpus > 1 shards with FSDP. Several machines: nnodes > 1 "
                         "plus master_addr (this machine's network address); the job's results then show the command to run on the others. "
                         "The trained model becomes a new checkpoint (hf:<name>) for Experiments; to_gguf also converts it. Follow with get_jobs.")
async def start_training(checkpoint: str, data: str, mode: str = "finetune", steps: int = 500, lr: float = 0.0, batch: int = 4, accum: int = 1,
                         max_len: int = 1024, rows: int = 5000, eval_frac: float = 0.02, eval_every: int = 100, layers: int = 0, gpus: int = 1,
                         nnodes: int = 1, master_addr: str = "", to_gguf: bool = False, name: str = "") -> str:
    try:
        return json.dumps({"job_id": torchlab.start_train({"checkpoint": checkpoint.removeprefix("hf:"), "data": data, "mode": mode, "steps": steps,
                                                           "lr": lr or None, "batch": batch, "accum": accum, "max_len": max_len, "rows": rows,
                                                           "eval_frac": eval_frac, "eval_every": eval_every, "layers": layers, "gpus": gpus,
                                                           "nnodes": nnodes, "master_addr": master_addr, "to_gguf": to_gguf, "name": name})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("write", description="Technique playground: simulate a quantization method on a Hugging Face checkpoint (fake quantization: quantize, then "
                         "dequantize) and measure KL divergence (± SE), same-top-token %, perplexity, and per-layer weight and output error vs the "
                         "unmodified model. method: gptq | awq (calibrated, AWQ-style scale search) | rtn | nf4 | ternary | custom (code = Python defining "
                         "quantize(w, name) or quantize(w, name, h) where h = mean x^T x of the layer's inputs; returns the dequantized w) | none. "
                         "grid: 'int' (bits 2-8, group 32|64|128|256, sym or asym) or 'q4_0' (llama.cpp's Q4_0 grid, so the exported GGUF matches the simulation). "
                         "kv_bits (2-8) also simulates a quantized KV cache. Compare methods at equal bits_per_weight. Follow with get_jobs, then "
                         "get_quantsim_results; export a winner with export_quantization. Custom code runs on the user's machine.")
async def simulate_quantization(checkpoint: str, method: str = "gptq", grid: str = "int", bits: int = 4, group: int = 128, sym: bool = True,
                                kv_bits: int = 0, include_head: bool = False, code: str = "", label: str = "") -> str:
    try:
        return json.dumps({"job_id": torchlab.start_quantsim({"checkpoint": checkpoint.removeprefix("hf:"), "method": method, "grid": grid, "bits": bits,
                                                             "group": group, "sym": sym, "kv_bits": kv_bits, "include_head": include_head,
                                                             "code": code, "label": label})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@tool("read", description="Results of technique-playground simulations (most recent first), for comparing methods on the same checkpoint.")
def get_quantsim_results(checkpoint: str = "") -> str:
    out = []
    for j in torchlab.jobs("quantsim"):
        if checkpoint and j["params"].get("checkpoint") != checkpoint.removeprefix("hf:"):
            continue
        r = j["results"] or {}
        out.append({"job_id": j["id"], "status": j["status"], "error": j.get("error"), "params": {k: v for k, v in j["params"].items() if k != "code"},
                    **{k: r.get(k) for k in ("bits_per_weight", "kld", "kld_sd", "kld_p99", "same_top", "ppl", "ppl_base", "weight_rel_error",
                                             "output_rel_error", "per_kind_out")}})
    return json.dumps(out[:20])


@tool("write", description="Turn a finished playground simulation into a real GGUF (gguf_type: Q4_0 | Q4_1 | Q5_0 | Q8_0 | F16) and profile it in Experiments "
                         "against the original checkpoint converted to F16 (experiment '<model> · playground'): real speed, memory, MMLU-Pro, KL divergence. "
                         "Q4_0-grid simulations exported as Q4_0 match the simulation; other combinations re-quantize. Takes minutes; follow with get_jobs / get_experiment.")
async def export_quantization(simulation_job_id: int, gguf_type: str = "Q4_0") -> str:
    try:
        return json.dumps({"job_id": torchlab.start_export(simulation_job_id, {"type": gguf_type, "profile": True})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


# ---------- General tools: every endpoint and all stored data ----------

APP = None  # the FastAPI app; main.py sets it
BLOCKED = ("/api/ai-access", "/api/remote-access", "/api/agent/key", "/mcp", "/oauth", "/.well-known", "/docs", "/redoc", "/openapi.json")
SECRET_KEYS = frozenset({"remote_token", "anthropic_api_key", "ai_pairing_code", "api_key", "password", "access_token", "refresh_token"})
MAX_CHARS = 60_000
API_TIMEOUT_S = 120


def _redact(v):
    if isinstance(v, dict):
        return {k: ("[hidden]" if k in SECRET_KEYS and x else _redact(x)) for k, x in v.items()}
    if isinstance(v, list):
        return [_redact(x) for x in v]
    return v


def _endpoints() -> list[dict]:
    out = []
    paths = (APP.openapi().get("paths") or {}) if APP else {}
    for path, ops in paths.items():
        if path.startswith(BLOCKED):
            continue
        for method, op in ops.items():
            params = [p["name"] + ("" if p.get("required") else "?") + f" ({p.get('in')})" for p in op.get("parameters", [])]
            about = (op.get("description") or "").strip().split("\n")[0] or op.get("summary", "")
            out.append({"method": method.upper(), "path": path, "about": about[:200], "params": params, "body": "requestBody" in op})
    return out


@tool("read", description="Describe Blocky's whole API: every endpoint with its method, path, parameters and what it does. "
      "Optional filter matches path or description (e.g. 'chat', 'bench', 'models', 'research', 'engines', 'evals', 'energy'). "
      "Then use api_get (GET), api_call (POST/PUT/PATCH) or api_delete (DELETE).")
def list_api(filter: str = "") -> str:
    eps = _endpoints()
    if filter:
        f = filter.lower()
        eps = [e for e in eps if f in e["path"].lower() or f in e["about"].lower()]
    return json.dumps({"count": len(eps), "endpoints": eps})


def _check_path(path: str) -> str | None:
    if not isinstance(path, str) or not path.startswith("/") or path.startswith("//") or ".." in path or "://" in path:
        return "path must be an API path like /api/experiments"
    bare = path.split("?")[0].rstrip("/")
    if not bare or bare.startswith(BLOCKED) or not bare.startswith("/api/"):
        return "This endpoint isn't available to AI assistants (it manages access or secrets, or isn't part of the API)."
    return None


async def _call(method: str, path: str, query: dict | None, body) -> str:
    if APP is None:
        return json.dumps({"error": "API not available"})
    if (err := _check_path(path)):
        return json.dumps({"error": err})
    import httpx
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=APP), base_url="http://localhost:8000") as c:
            r = await asyncio.wait_for(c.request(method, path, params=query or None, json=body), API_TIMEOUT_S)
    except asyncio.TimeoutError:
        return json.dumps({"error": f"No complete answer within {API_TIMEOUT_S} s. Long jobs: start them, then poll their status endpoint."})
    except Exception as e:  # noqa: BLE001 - report it to the assistant, don't break the MCP session
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    ctype = r.headers.get("content-type", "")
    out: dict = {"status": r.status_code}
    if "json" in ctype:
        try:
            text = json.dumps(_redact(r.json()), default=str)
        except ValueError:
            text = r.text
    elif ctype.startswith("text/"):
        text = r.text
    else:
        return json.dumps({**out, "content_type": ctype, "bytes": len(r.content),
                           "note": "A file, not text: open it in Blocky's web page or download it on the device."})
    if len(text) > MAX_CHARS:
        out["truncated"] = f"showing {MAX_CHARS} of {len(text)} characters; narrow the request (filters, limit, an id)"
        text = text[:MAX_CHARS]
    out["body"] = text
    return json.dumps(out)


@tool("read", description="GET any Blocky API endpoint (see list_api), e.g. /api/conversations, /api/models, /api/research/projects, "
      "/api/settings. query: URL parameters as an object. Secrets are removed from answers.")
async def api_get(path: str, query: dict | None = None) -> str:
    return await _call("GET", path, query, None)


@tool("write", description="POST, PUT or PATCH any Blocky API endpoint (see list_api): start benchmarks, downloads, quality tests, "
      "chats and research jobs, change settings, and so on. body: the JSON body. Long jobs return an id; poll their status with api_get. "
      "Some endpoints run code or use the GPU for a long time: tell the user what you're starting.")
async def api_call(method: str, path: str, body: dict | list | None = None, query: dict | None = None) -> str:
    method = (method or "").upper()
    if method not in ("POST", "PUT", "PATCH"):
        return json.dumps({"error": "method must be POST, PUT or PATCH (GET: api_get, DELETE: api_delete)"})
    return await _call(method, path, query, body)


@tool("full", description="DELETE through any Blocky API endpoint (see list_api): experiments, runs, models, conversations, engines, "
      "jobs and so on. Permanent: ask the user to confirm first.")
async def api_delete(path: str, query: dict | None = None) -> str:
    return await _call("DELETE", path, query, None)


def _authorizer(action, arg1, arg2, _db, _trigger):
    from .aiaccess import SECRET_TABLES
    if action == sqlite3.SQLITE_READ:
        if arg1 in SECRET_TABLES:
            return sqlite3.SQLITE_DENY
        if (arg1, arg2) == ("devices", "token"):
            return sqlite3.SQLITE_IGNORE  # reads as NULL
        return sqlite3.SQLITE_OK
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


@tool("read", description="Read-only SQL (SQLite) over all of Blocky's stored data: experiments and runs (exp_runs), conversations "
      "and messages, benchmarks, quality-test results, research jobs and more. Call with no sql to list the tables and columns. "
      "At most 500 rows; settings and credentials aren't readable here (use api_get /api/settings).")
def query_database(sql: str = "", max_rows: int = 200) -> str:
    from . import config
    from .aiaccess import SECRET_TABLES
    try:
        conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})
    try:
        if not (sql or "").strip():
            tables = {}
            for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall():
                if name not in SECRET_TABLES:
                    tables[name] = [r[1] for r in conn.execute("SELECT * FROM pragma_table_info(?)", (name,))]
            return json.dumps({"tables": tables})
        conn.set_authorizer(_authorizer)
        deadline = time.monotonic() + 10
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description or ()]
        limit = max(1, min(int(max_rows), 500))
        rows = cur.fetchmany(limit + 1)
        text = json.dumps({"columns": cols, "rows": [list(r) for r in rows[:limit]], "more_rows": len(rows) > limit}, default=str)
        if len(text) > MAX_CHARS:
            return json.dumps({"error": f"Answer too large ({len(text)} characters); select fewer columns or rows."})
        return text
    except (sqlite3.Error, sqlite3.Warning, ValueError, TypeError) as e:
        msg = str(e)
        if "not authorized" in msg or "prohibited" in msg:
            msg = "Not readable from MCP (settings and credentials). Use api_get /api/settings."
        elif "interrupted" in msg:
            msg = "The query took longer than 10 s; narrow it down."
        return json.dumps({"error": msg})
    finally:
        conn.close()
