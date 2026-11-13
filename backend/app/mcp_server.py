"""Blocky as an MCP server: lets an AI assistant you already use (Claude Code, Claude Desktop, other MCP clients) run
the lab: read results, queue runs and sweeps, build weight variants, scan layers, build engines.

Served by the backend at http://localhost:8000/mcp (Streamable HTTP, local only, like the rest of the API). The
assistant's model and API keys stay in the user's own client; Blocky never contacts an AI provider itself.
Nothing here deletes data: deleting experiments, runs, models or engines stays a human action in the UI.
"""
import json

from mcp.server.mcpserver import MCPServer

from . import db, engines, lab, profile, scans, torchlab, weights

INSTRUCTIONS = """Blocky is a local-AI research lab running on the user's own hardware. You can measure and improve how
models run on this device.

How it works:
- An *experiment* is a tree of *runs*. Each run profiles one setup (model file, engine, KV cache, context size, GPU
  layers, VRAM limit, flash attention, threads) with an identical protocol: memory layout, MMLU-Pro (560 fixed
  questions, 5-shot letter log-likelihood), long-context needle test, engine-neutral serving speed, llama-bench speed,
  perplexity + KL divergence vs the experiment's baseline, telemetry, and a roofline-style hardware analysis.
- Branch from a run: the new run = that run's exact setup + your changes. Change ONE thing at a time.
- A profile takes ~3-10 minutes and needs the GPU: runs wait in a queue and execute one after another. Use
  get_status / get_experiment to follow progress; don't poll more than every ~30 s.
- Read significance, not raw differences: speeds carry a spread (sd); MMLU-Pro has a 95% CI and a paired McNemar
  p-value vs baseline/parent; KL divergence is the most sensitive quality signal (0 = identical outputs).
- KL divergence / perplexity only compare models with the same tokenizer; vLLM runs don't have them.
- Weights: build variants with build_weight_variant (quantization type, importance matrix, per-tensor precision
  overrides as regexes, layer pruning) from a high-precision source; find sensitive layers with start_layer_scan.
- Energy: each run measures idle and generating power (wall meter if configured, else GPU/CPU sensors) -> energy per
  token; with prices set in Blocky's Settings, also electricity per 1M tokens and speed per device price.
- Cheap devices: queue_run(changes={'preset': ...}) simulates CPU-only boxes, fewer cores, less RAM/VRAM (presets in
  get_status). Capacity and cores are real limits; bandwidth isn't, so CPU presets add a projected decode speed.
- New techniques: simulate_quantization tries a quantization method (or your own Python) on a checkpoint and scores it
  against the unmodified model before anyone writes a kernel.
- Be economical: every run costs minutes of the user's machine (CPU-only runs: much longer). Plan, explain, then queue.
"""

server = MCPServer(name="blocky", title="Blocky local-AI lab", instructions=INSTRUCTIONS, version="1")


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
        "lm_eval": {t: v.get("metrics") for t, v in ((x.get("lm_eval") or {}).get("tasks") or {}).items()} or None,
        "kld": x.get("kld"), "kld_p99": x.get("kld_p99"), "same_top_pct": x.get("same_top"), "ppl": x.get("ppl"),
        "vram_model_mb": (hw.get("headroom") or {}).get("vram_model_mb"), "weights_on_cpu_gb": hw.get("weights_on_cpu_gb"),
        "decode_roof_util": (hw.get("decode") or {}).get("roof_util"), "limiter": hw.get("limiter"),
        "energy": _energy(x.get("energy")), "cost": x.get("cost"),
        "hardware": {"gpu": (x.get("hardware") or {}).get("gpu"), "device": (x.get("hardware") or {}).get("device")},
        "stats_vs_baseline_and_parent": r.get("stats"),
        "notes": [n for n in (x.get("token_note"), x.get("fidelity_note")) if n] or None,
    }


@server.tool(description="Overview of this Blocky device: hardware limits, model files, Hugging Face checkpoints, engines, experiments, and whether the GPU is busy.")
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


@server.tool(description="All runs of one experiment with their key results (speed with spread, MMLU-Pro with CI, long context, KL divergence, memory, hardware limiter, significance vs baseline and parent).")
def get_experiment(experiment_id: int) -> str:
    e = profile.get_experiment(experiment_id)
    if not e:
        return json.dumps({"error": "Experiment not found"})
    return json.dumps({"id": e["id"], "name": e["name"], "baseline_model": e["model"], "protocol_extras": e.get("protocol"),
                       "runs": [_compact_run(r) for r in e["runs_list"]]}, default=str)


@server.tool(description="Create an experiment for a model (a GGUF file name, or 'hf:<checkpoint>' for vLLM). Then queue a baseline with queue_run (no parent).")
def create_experiment(model: str, name: str = "") -> str:
    try:
        return json.dumps({"experiment_id": profile.create_experiment(name or model, model)})
    except ValueError as e:
        return json.dumps({"error": str(e)})


@server.tool(description="Queue one profiled run. parent_run_id: the run to branch from (its exact setup + `changes`); omit for the baseline. "
                         "changes keys: model, engine, kv (f16|q8_0|q4_0), ctx, ngl ('auto' or a number), vram_limit_mb, fa (auto|on|off), threads, "
                         "device (gpu|cpu), cores (pin to N CPU threads), ram_limit_mb, preset (a cheap-device simulation from get_status; "
                         "it fixes the device settings, set preset to '' to customize them).")
def queue_run(experiment_id: int, changes: dict | None = None, parent_run_id: int | None = None, label: str = "") -> str:
    e = profile.get_experiment(experiment_id)
    if not e:
        return json.dumps({"error": "Experiment not found"})
    parent = next((r for r in e["runs_list"] if r["id"] == parent_run_id), None) if parent_run_id else None
    base_cfg = parent["config"] if parent else {}
    try:
        rid = profile.create_run(experiment_id, {"label": label, "config": {**base_cfg, **(changes or {})}, "parent_id": parent_run_id})
        return json.dumps({"run_id": rid, "queued": True})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@server.tool(description="Queue a sweep: every combination of the given values (grid: {setting: [values]}), from a parent run or the baseline. Max 32 runs.")
def queue_sweep(experiment_id: int, grid: dict, parent_run_id: int | None = None, repeats: int = 1) -> str:
    try:
        return json.dumps({"run_ids": profile.create_sweep(experiment_id, {"grid": grid, "parent_id": parent_run_id, "repeats": repeats})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@server.tool(description="Cancel a queued or running profile.")
def cancel_run(run_id: int) -> str:
    profile.cancel(run_id)
    return json.dumps({"ok": True})


@server.tool(description="Set extra tests every run of an experiment gets: lm-evaluation-harness tasks (generative, e.g. 'gsm8k,ifeval') with a per-task limit, and/or a GPU kernel profile.")
def set_experiment_tests(experiment_id: int, lm_eval_tasks: str = "", lm_eval_limit: int = 100, kernel_profile: bool = False) -> str:
    try:
        return json.dumps(profile.set_protocol(experiment_id, {"lm_eval": {"tasks": lm_eval_tasks, "limit": lm_eval_limit} if lm_eval_tasks else None,
                                                               "kernels": kernel_profile}))
    except ValueError as err:
        return json.dumps({"error": str(err)})


@server.tool(description="Tensor groups of a GGUF model (name pattern, count, size, current types): use before choosing per-tensor overrides.")
def list_tensor_groups(model: str) -> str:
    try:
        return json.dumps(weights.tensor_groups(model)[:60], default=str)
    except (OSError, ValueError) as err:
        return json.dumps({"error": str(err)})


@server.tool(description="Build a weight variant from a high-precision GGUF source with llama-quantize: qtype (e.g. Q4_K_M, Q3_K_M, IQ4_XS), "
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


@server.tool(description="Layer sensitivity scan: change one layer at a time (mode 'quantize' to qtype q2_k|q3_k|q4_k|iq4_xs, or 'prune' to remove it) and measure KL divergence vs the source. Slow on big models (~1-2 h for 9B).")
async def start_layer_scan(model: str, mode: str = "quantize", qtype: str = "q2_k") -> str:
    try:
        return json.dumps({"scan_id": scans.start({"model": model, "mode": mode, "qtype": qtype})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@server.tool(description="Results of layer scans for a model (per-layer extra KL divergence over the control, most / least sensitive layers).")
def get_layer_scans(model: str) -> str:
    out = []
    for s in scans.list_scans(model)[:5]:
        r = s["results"]
        out.append({"id": s["id"], "mode": s["mode"], "qtype": s["qtype"], "status": s["status"], "step": s["step"],
                    "control_kld": (r.get("control") or {}).get("kld"), "most_sensitive": r.get("most_sensitive"), "least_sensitive": r.get("least_sensitive"),
                    "layers": [{"layer": x["layer"], "excess_kld": x["excess_kld"]} for x in r.get("layers", [])]})
    return json.dumps(out)


@server.tool(description="Build a llama.cpp engine from an https git repository and ref (branch, tag, commit) so it can be an experiment variable. "
                         "backend: auto | cuda | vulkan | cpu. "
                         "This compiles and runs that repository's code on the user's machine: only use repositories the user asked for or trusts.")
async def build_engine(repo: str, ref: str = "master", name: str = "", backend: str = "auto") -> str:
    try:
        return json.dumps({"engine_id": engines.add({"repo": repo, "ref": ref, "name": name, "backend": backend})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@server.tool(description="Status of background jobs: weight builds, layer scans, engine builds, PyTorch jobs.")
def get_jobs() -> str:
    return json.dumps({
        "weights": [{k: j.get(k) for k in ("id", "out", "status", "step", "error")} for j in weights.list_jobs()[:10]],
        "scans": [{k: s.get(k) for k in ("id", "model", "mode", "status", "step", "error")} for s in scans.list_scans()[:10]],
        "engines": [{k: e.get(k) for k in ("id", "status", "step", "error", "commit_sha")} for e in engines.list_engines()],
        "torch": [{k: j.get(k) for k in ("id", "kind", "status", "step", "error")} for j in torchlab.jobs()[:10]],
    }, default=str)


@server.tool(description="Technique playground: simulate a quantization method on a Hugging Face checkpoint (fake quantization: quantize, then "
                         "dequantize) and measure KL divergence, same-top-token %, perplexity and per-layer weight error vs the unmodified model. "
                         "method: rtn (bits 2-8, sym or asym) | nf4 | ternary | custom (code = Python defining quantize(w, name) -> dequantized w) | none. "
                         "group: 32|64|128|256 weights per scale. kv_bits (2-8) also simulates a quantized KV cache. Speed/memory aren't measured. "
                         "Follow with get_jobs, then get_quantsim_results. Custom code runs on the user's machine: only run code the user agreed to.")
async def simulate_quantization(checkpoint: str, method: str = "rtn", bits: int = 4, group: int = 128, sym: bool = True, kv_bits: int = 0,
                                include_head: bool = False, code: str = "", label: str = "") -> str:
    try:
        return json.dumps({"job_id": torchlab.start_quantsim({"checkpoint": checkpoint.removeprefix("hf:"), "method": method, "bits": bits, "group": group,
                                                             "sym": sym, "kv_bits": kv_bits, "include_head": include_head, "code": code, "label": label})})
    except ValueError as err:
        return json.dumps({"error": str(err)})


@server.tool(description="Results of technique-playground simulations (most recent first), for comparing methods on the same checkpoint.")
def get_quantsim_results(checkpoint: str = "") -> str:
    out = []
    for j in torchlab.jobs("quantsim"):
        if checkpoint and j["params"].get("checkpoint") != checkpoint.removeprefix("hf:"):
            continue
        r = j["results"] or {}
        out.append({"job_id": j["id"], "status": j["status"], "error": j.get("error"), "params": {k: v for k, v in j["params"].items() if k != "code"},
                    **{k: r.get(k) for k in ("bits_per_weight", "kld", "kld_sd", "kld_p99", "same_top", "ppl", "ppl_base", "weight_rel_error", "per_kind")}})
    return json.dumps(out[:20])
