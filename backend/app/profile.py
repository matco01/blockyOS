"""Experiments: baseline → change one thing → compare → find the bottleneck.

An experiment is one model you're studying. Each *run* profiles one
configuration (model file, context, KV cache, GPU layers, simulated VRAM,
flash attention, threads) and measures everything in one go, always the same way:

  memory   load time, VRAM/RAM breakdown, layers on GPU     (temporary llama-server, load log)
  quality  fixed fresh questions (same seed for every run)   (math, logic, coding)
  speed    llama-bench: prompt, generation, long context    (+ power sampling)
  fidelity perplexity on WikiText; KL divergence vs the experiment's baseline
  telemetry GPU load, memory-controller load, power, temperature, CPU, RAM
  bottleneck  what limits this setup, with numbers (e.g. % of the GPU's memory-speed ceiling)

Because every run uses identical questions and text, runs are directly comparable.
"""
import asyncio
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from . import arena, config, db, engines, evals, gguf, kernels, lab, power, quantlab, ruler

DIR = config.DATA_DIR / "experiments"
PORT = arena.TEMP_PORT
_cancel: set[int] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    name        TEXT NOT NULL,
    model       TEXT NOT NULL,
    seed        INTEGER NOT NULL,
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS exp_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    created_at     REAL NOT NULL,
    status         TEXT NOT NULL,
    label          TEXT NOT NULL,
    config         TEXT NOT NULL,
    results        TEXT NOT NULL,
    phase          TEXT,
    log            TEXT,
    error          TEXT
);
"""

# Memory bandwidth (GB/s) of common GPUs: the hard ceiling for generation speed.
GPU_BANDWIDTH = {
    "RTX 5090": 1792, "RTX 5080": 960, "RTX 5070 Ti": 896, "RTX 5070": 672, "RTX 5060 Ti": 448, "RTX 5060": 448,
    "RTX 4090": 1008, "RTX 4080 SUPER": 736, "RTX 4080": 717, "RTX 4070 Ti SUPER": 672, "RTX 4070 Ti": 504,
    "RTX 4070 SUPER": 504, "RTX 4070": 504, "RTX 4060 Ti": 288, "RTX 4060": 272,
    "RTX 3090 Ti": 1008, "RTX 3090": 936, "RTX 3080 Ti": 912, "RTX 3080": 760, "RTX 3070 Ti": 608, "RTX 3070": 448,
    "RTX 3060 Ti": 448, "RTX 3060": 360, "RTX 3050": 224,
    "RTX 2080 Ti": 616, "RTX 2080 SUPER": 496, "RTX 2080": 448, "RTX 2070 SUPER": 448, "RTX 2070": 448,
    "RTX 2060 SUPER": 448, "RTX 2060": 336, "GTX 1660 Ti": 288, "GTX 1660 SUPER": 336, "GTX 1660": 192, "GTX 1650": 128,
    "GTX 1080 Ti": 484, "GTX 1080": 320, "GTX 1070 Ti": 256, "GTX 1070": 256, "GTX 1060": 192, "GTX 1050 Ti": 112, "GTX 1050": 112,
}


def init() -> None:
    from . import validation
    from . import methods
    DIR.mkdir(parents=True, exist_ok=True)
    with db.connect() as c:
        c.executescript(SCHEMA)
        c.executescript(validation.SCHEMA)
        for column in ("validation", "protocol_snapshot"):
            if column not in [r[1] for r in c.execute("PRAGMA table_info(exp_runs)")]:
                c.execute(f"ALTER TABLE exp_runs ADD COLUMN {column} TEXT")
        cols = [r[1] for r in c.execute("PRAGMA table_info(experiments)")]
        if "canvas" not in cols:  # research canvas layout: card positions, notes, viewport
            c.execute("ALTER TABLE experiments ADD COLUMN canvas TEXT")
        if "protocol" not in cols:  # extra tests every run of the experiment gets (e.g. lm-eval tasks)
            c.execute("ALTER TABLE experiments ADD COLUMN protocol TEXT")
        if "parent_id" not in [r[1] for r in c.execute("PRAGMA table_info(exp_runs)")]:
            # The run tree: each run is derived from a parent run (its config + one change). NULL = from the baseline.
            c.execute("ALTER TABLE exp_runs ADD COLUMN parent_id INTEGER")
        # Queued runs survive a restart (the worker picks them up); a run that was mid-way is lost.
        c.execute("UPDATE exp_runs SET status = 'error', error = 'Interrupted by a restart' WHERE status = 'running'")
        c.execute("UPDATE exp_runs SET status = 'error', error = 'Interrupted while preparing a method run' WHERE status = 'preparing'")
    methods.init()


# ---------- Hardware limits ----------

# Peak dense matrix throughput (TFLOPS) from spec sheets: FP16 tensor cores with FP32 accumulate
# (Turing and newer GeForce), or plain FP32 for Pascal, which has no tensor cores. The compute roof for prompt reading.
GPU_TFLOPS = {
    "RTX 4090": (165.2, "FP16 tensor, FP32 acc."), "RTX 4080": (97.5, "FP16 tensor, FP32 acc."), "RTX 4070": (58.3, "FP16 tensor, FP32 acc."),
    "RTX 3090": (71.0, "FP16 tensor, FP32 acc."), "RTX 3080": (59.5, "FP16 tensor, FP32 acc."), "RTX 3070": (40.6, "FP16 tensor, FP32 acc."),
    "RTX 3060": (25.5, "FP16 tensor, FP32 acc."),
    "RTX 2080 Ti": (53.8, "FP16 tensor, FP32 acc."), "RTX 2080 SUPER": (44.6, "FP16 tensor, FP32 acc."), "RTX 2080": (40.3, "FP16 tensor, FP32 acc."),
    "RTX 2070 SUPER": (36.3, "FP16 tensor, FP32 acc."), "RTX 2070": (29.9, "FP16 tensor, FP32 acc."), "RTX 2060": (25.8, "FP16 tensor, FP32 acc."),
    "GTX 1080 Ti": (11.3, "FP32"), "GTX 1080": (8.9, "FP32"), "GTX 1070": (6.5, "FP32"), "GTX 1060": (4.4, "FP32"), "GTX 1050 Ti": (2.1, "FP32"),
}


def _lookup(table: dict, name: str | None):
    if not name:
        return None
    for key in sorted(table, key=len, reverse=True):
        if key.lower() in name.lower():
            return table[key]
    return None


def gpu_bandwidth(name: str | None) -> float | None:
    return _lookup(GPU_BANDWIDTH, name)


_ram_bw: float | None = None

# Read bandwidth with one process per two hardware threads, summed: what CPU inference actually does.
_RAM_BW_CODE = """
import numpy as np, time, os, json
from multiprocessing import Pool
def work(_):
    a = np.ones(32 * 2**20, dtype=np.float64); best = 0
    for _ in range(6):
        t = time.perf_counter(); a.sum(); dt = time.perf_counter() - t
        best = max(best, a.nbytes / dt / 1e9)
    return best
n = max(1, (os.cpu_count() or 2) // 2)
try:
    with Pool(n) as p:
        print(json.dumps({"gbps": sum(p.map(work, range(n))), "method": "numpy sum, one process per 2 threads"}))
except (OSError, EOFError):
    # No working multiprocessing (e.g. WSL1): threads (numpy releases the GIL while summing). Total bytes over wall time,
    # not a sum of per-thread rates, which overstates bandwidth when the threads don't actually overlap.
    from concurrent.futures import ThreadPoolExecutor
    arrays = [np.ones(32 * 2**20, dtype=np.float64) for _ in range(n)]
    def run(a):
        for _ in range(6):
            a.sum()
    t = time.perf_counter()
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(run, arrays))
    gbps = n * 6 * arrays[0].nbytes / (time.perf_counter() - t) / 1e9
    print(json.dumps({"gbps": gbps, "method": "numpy sum, one thread per 2 hardware threads, total bytes / wall time"}))
"""


def ram_bandwidth() -> float | None:
    """Measured multi-core system-RAM read bandwidth (GB/s), cached: the ceiling for layers on the CPU."""
    global _ram_bw
    cache = DIR / "ram_read_bandwidth.json"
    if _ram_bw is None and cache.is_file():
        _ram_bw = json.loads(cache.read_text()).get("gbps")
    if _ram_bw is None:
        # Needs numpy: the installer adds python3-numpy; the backend's own and the ML environment are fallbacks.
        for python in ("python3", sys.executable, str(config.DATA_DIR.parent / "ml-venv" / "bin" / "python")):
            try:
                out = subprocess.run([python, "-c", _RAM_BW_CODE], capture_output=True, text=True, timeout=120).stdout.strip()
                measured = json.loads(out.splitlines()[-1])
                _ram_bw = round(float(measured["gbps"]), 1)
            except (ValueError, KeyError, IndexError, OSError, subprocess.SubprocessError):
                continue
            cache.write_text(json.dumps({"gbps": _ram_bw, "measured_at": time.time(), "method": measured["method"]}))
            break
    return _ram_bw


_sm_max: float | None = None


def sm_clock_max() -> float | None:
    global _sm_max
    if _sm_max is None and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            _sm_max = float(out.strip().splitlines()[0])
        except (ValueError, IndexError, OSError, subprocess.SubprocessError):
            pass
    return _sm_max


def limits() -> dict:
    hw = lab.hardware()
    g = hw["gpus"][0] if hw["gpus"] else None
    tf = _lookup(GPU_TFLOPS, g["name"] if g else None)
    return {"gpu": g["name"] if g else None, "gpu_bandwidth_gbps": gpu_bandwidth(g["name"] if g else None),
            "gpu_tflops": tf[0] if tf else None, "gpu_tflops_kind": tf[1] if tf else None, "sm_clock_max": sm_clock_max(),
            "vram_mb": g["vram_mb"] if g else 0, "power_limit_w": g.get("power_limit_w") if g else None, "gpu_count": len(hw["gpus"]),
            "ram_bandwidth_gbps": ram_bandwidth(), "ram_mb": hw["ram_mb"], "cpu": hw["cpu"], "cpu_threads": hw["cpu_threads"],
            "power_source": "nvidia-smi" if g else None, "cpu_power_source": "RAPL" if _rapl_file() else None}


# ---------- Records ----------

def _exp(r) -> dict:
    d = dict(r)
    for k in ("canvas", "protocol"):
        try:
            d[k] = json.loads(d.get(k) or "null")
        except ValueError:
            d[k] = None
    return d


def save_canvas(exp_id: int, canvas: dict) -> None:
    blob = json.dumps(canvas)
    if len(blob) > 500_000:
        raise ValueError("Canvas too large")
    with db.connect() as c:
        c.execute("UPDATE experiments SET canvas = ? WHERE id = ?", (blob, exp_id))


def _run(r) -> dict:
    d = dict(r)
    d["config"] = json.loads(d["config"]); d["results"] = json.loads(d["results"])
    for key in ("validation", "protocol_snapshot"):
        d[key] = json.loads(d.get(key) or "null")
    return d


def list_experiments() -> list[dict]:
    with db.connect() as c:
        exps = [_exp(r) for r in c.execute("SELECT * FROM experiments ORDER BY id DESC")]
        for e in exps:
            e["runs"] = c.execute("SELECT COUNT(*) FROM exp_runs WHERE experiment_id = ?", (e["id"],)).fetchone()[0]
    return exps


PREREG_SCHEMA = """
CREATE TABLE IF NOT EXISTS preregistrations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    created_at    REAL NOT NULL,
    plan          TEXT NOT NULL,
    sha256        TEXT NOT NULL
);
"""
PREREG_METRICS = ("mmlu", "kld", "same_top", "ppl", "tg", "pp", "srv_tg", "jpt", "needle")


def preregister(exp_id: int, body: dict) -> dict:
    """Record a hypothesis and analysis plan before the data exists. Plans are append-only (no edits, no deletes): a
    changed plan is a new entry with its own time, and runs finished after a plan count as confirmatory for it."""
    import hashlib
    if not get_experiment(exp_id):
        raise ValueError("Experiment not found")
    hyp = str(body.get("hypothesis") or "").strip()
    if not 10 <= len(hyp) <= 4000:
        raise ValueError("Write the hypothesis (10-4000 characters)")
    metric = body.get("primary_metric")
    if metric not in PREREG_METRICS:
        raise ValueError(f"Primary metric: one of {', '.join(PREREG_METRICS)}")
    claim = body.get("claim") or "difference"
    if claim not in ("difference", "equivalence", "non_inferiority"):
        raise ValueError("Claim: difference, equivalence or non_inferiority")
    plan = {"hypothesis": hyp, "primary_metric": metric, "claim": claim,
            "direction": body.get("direction") if body.get("direction") in ("higher", "lower", "either") else "either",
            "alpha": float(body.get("alpha") or 0.05), "analysis": str(body.get("analysis") or "").strip()[:4000],
            "secondary_metrics": [m for m in (body.get("secondary_metrics") or []) if m in PREREG_METRICS][:8]}
    if not 0.001 <= plan["alpha"] <= 0.2:
        raise ValueError("Alpha: 0.001-0.2")
    for key, lo, hi in (("min_effect", 0.0, 1e6), ("equivalence_margin_pts", 0.1, 20)):
        if body.get(key) not in (None, ""):
            v = float(body[key])
            if not lo <= v <= hi:
                raise ValueError(f"{key}: {lo}-{hi}")
            plan[key] = v
    if claim in ("equivalence", "non_inferiority") and metric == "mmlu" and not plan.get("equivalence_margin_pts"):
        raise ValueError("An equivalence or non-inferiority claim on accuracy needs its margin (accuracy points) now")
    plan["protocol"] = PROTOCOL
    text = json.dumps(plan, sort_keys=True)
    with db.connect() as c:
        c.executescript(PREREG_SCHEMA)
        pid = c.execute("INSERT INTO preregistrations (experiment_id, created_at, plan, sha256) VALUES (?, ?, ?, ?)",
                        (exp_id, time.time(), text, hashlib.sha256(text.encode()).hexdigest())).lastrowid
    return {"id": pid, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def preregistrations(exp_id: int, runs: list[dict]) -> list[dict]:
    with db.connect() as c:
        c.executescript(PREREG_SCHEMA)
        rows = [dict(r) for r in c.execute("SELECT * FROM preregistrations WHERE experiment_id = ? ORDER BY id", (exp_id,))]
    out = []
    for r in rows:
        after = [x["id"] for x in runs if x["status"] == "done" and (x.get("created_at") or 0) > r["created_at"]]
        out.append({"id": r["id"], "created_at": r["created_at"], "sha256": r["sha256"], "plan": json.loads(r["plan"]),
                    "confirmatory_runs": after,
                    "note": "Runs queued after this plan are confirmatory for it; earlier runs are exploratory."})
    return out


def get_experiment(exp_id: int) -> dict | None:
    with db.connect() as c:
        e = c.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        if not e:
            return None
        runs = [_run(r) for r in c.execute("SELECT * FROM exp_runs WHERE experiment_id = ? ORDER BY id", (exp_id,))]
    from . import validation
    return {**enrich({**_exp(e), "runs_list": runs, "limits": limits(), "preregistrations": preregistrations(exp_id, runs)}),
            "validations": validation.list_batches(exp_id, runs)}


def create_experiment(name: str, model: str) -> int:
    import random
    from . import methods
    if model.startswith(methods.PREFIX):
        method = methods.get(int(model.removeprefix(methods.PREFIX)))
        with db.connect() as c:
            return c.execute("INSERT INTO experiments (created_at,name,model,seed) VALUES (?,?,?,?)", (time.time(), name or method["name"], model, random.randint(1, 2**31 - 1))).lastrowid
    model = HF_PREFIX + Path(model[len(HF_PREFIX):]).name if model.startswith(HF_PREFIX) else Path(model).name
    mp = model_path(model)
    if not ((mp / "config.json").is_file() if model.startswith(HF_PREFIX) else mp.is_file()):
        raise ValueError("Model not found")
    with db.connect() as c:
        return c.execute("INSERT INTO experiments (created_at, name, model, seed) VALUES (?, ?, ?, ?)",
                         (time.time(), (name or model.replace(".gguf", ""))[:80], model, random.randint(1, 2**31 - 1))).lastrowid


def delete_experiment(exp_id: int) -> None:
    with db.connect() as c:
        active = [r[0] for r in c.execute("SELECT id FROM exp_runs WHERE experiment_id=? AND status IN ('queued','running')", (exp_id,))]
    for rid in active:
        cancel(rid)
    with db.connect() as c:
        c.executescript(PREREG_SCHEMA)
        c.execute("DELETE FROM preregistrations WHERE experiment_id = ?", (exp_id,))  # deleting the experiment is the user's choice
        c.execute("DELETE FROM exp_runs WHERE experiment_id = ?", (exp_id,))
        c.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
    shutil.rmtree(DIR / str(exp_id), ignore_errors=True)


def delete_run(run_id: int) -> None:
    with db.connect() as c:
        r = c.execute("SELECT parent_id FROM exp_runs WHERE id = ? AND status NOT IN ('queued', 'running')", (run_id,)).fetchone()
        if not r:
            return
        # Keep the tree connected: the deleted run's children hang from its parent instead.
        c.execute("UPDATE exp_runs SET parent_id = ? WHERE parent_id = ?", (r["parent_id"], run_id))
        c.execute("DELETE FROM exp_runs WHERE id = ?", (run_id,))


def cancel(run_id: int) -> None:
    from . import research_jobs
    with db.connect() as c:
        linked = c.execute("SELECT job_id FROM method_profile_jobs WHERE run_id=?", (run_id,)).fetchone()
    if linked:
        research_jobs.cancel(linked[0])
    with db.connect() as c:
        queued = c.execute("UPDATE exp_runs SET status = 'cancelled' WHERE id = ? AND status = 'queued'", (run_id,)).rowcount
    if not queued:
        _cancel.add(run_id)


def cancel_queue(exp_id: int) -> int:
    with db.connect() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM exp_runs WHERE experiment_id=? AND status='queued'", (exp_id,))]
    for rid in ids:
        cancel(rid)
    return len(ids)


SWEEP_KEYS = ("model", "engine", "kv", "ctx", "vram_limit_mb", "ngl", "fa", "threads", "device", "cores", "ram_limit_mb", "preset", "gpus", "split_mode", "pp")
MAX_SWEEP = 32


def create_sweep(exp_id: int, body: dict) -> list[int]:
    """Queue every combination of the given values (a full grid), optionally repeated."""
    import itertools
    exp = get_experiment(exp_id)
    if not exp:
        raise ValueError("Experiment not found")
    grid = {k: list(dict.fromkeys(v)) for k, v in (body.get("grid") or {}).items() if k in SWEEP_KEYS and isinstance(v, list) and v}
    if not grid:
        raise ValueError("Pick at least one value to sweep")
    repeats = max(1, min(5, int(body.get("repeats") or 1)))
    done = [r for r in exp["runs_list"] if r["status"] == "done"]
    parent = next((r for r in exp["runs_list"] if r["id"] == int(body["parent_id"])), None) if body.get("parent_id") is not None else None
    if body.get("parent_id") is not None and parent is None:
        raise ValueError("The run to branch from isn't in this experiment")
    base = parent["config"] if parent else (done[0]["config"] if done else normalize_config(exp, {}))
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    if len(combos) * repeats > MAX_SWEEP:
        raise ValueError(f"That's {len(combos) * repeats} runs; the limit is {MAX_SWEEP} per sweep")
    # Validate first: a grid mixing engines and model formats contains pairs that can't run (vLLM + GGUF file).
    # Those are skipped; anything else invalid is an error.
    valid = []
    for combo in combos:
        try:
            valid.append(normalize_config(exp, {**base, **combo}))
        except ValueError as e:
            if "GGUF" not in str(e) and "checkpoint" not in str(e):
                raise
    if not valid:
        raise ValueError("None of these combinations can run: vLLM needs Hugging Face checkpoints, llama.cpp needs GGUF files")
    ids = []
    for rep in range(repeats):
        for cfg in valid:
            label = describe_change(base, cfg)
            if label == "Repeat of baseline" and parent:
                label = f"Repeat of {parent['label']}"[:60]
            if not done and not ids and not parent:
                label = baseline_label(cfg)  # the first run of a fresh experiment is its baseline, whatever its settings
            if repeats > 1:
                label = f"{label} · rep {rep + 1}"
            ids.append(create_run(exp_id, {"label": label, "config": cfg, "parent_id": parent["id"] if parent else None}))
    return ids


_chat_down = False


async def _restore_chat(llm: httpx.AsyncClient) -> None:
    global _chat_down
    _chat_down = False
    try:
        await lab.llm_service("start")
        await lab.wait_ready(llm, timeout=300)
    except Exception:  # noqa: BLE001
        pass


async def worker(llm: httpx.AsyncClient) -> None:
    """Runs queued profiles one after another, whenever the lab is free."""
    while True:
        try:
            await asyncio.sleep(2)
            if lab.busy or lab._lock.locked():
                continue
            with db.connect() as c:
                row = c.execute("SELECT id FROM exp_runs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
            if row:
                await run(row["id"], llm)
            elif _chat_down:  # the queue was cancelled between runs: bring chat back
                await _restore_chat(llm)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the worker must never die
            await asyncio.sleep(5)


def _save(run_id: int, **f) -> None:
    sets = ", ".join(f"{k} = ?" for k in f)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()]
    with db.connect() as c:
        c.execute(f"UPDATE exp_runs SET {sets} WHERE id = ?", (*vals, run_id))


DEFAULTS = {"ctx": 8192, "kv": "q8_0", "ngl": "auto", "vram_limit_mb": None, "fa": "auto", "threads": None, "engine": "stock",
            "device": "gpu", "cores": None, "ram_limit_mb": None, "preset": None}

# Cheap-device simulation. What's simulated: CPU-only execution, core count (pinned with taskset), RAM and VRAM capacity
# (a run that wouldn't fit fails). What isn't: per-core speed and memory bandwidth, so measured speeds are an upper
# bound; for memory-bound decode, results add a projection at the target's spec-sheet bandwidth.
DEVICE_PRESETS = {
    "sbc-8gb": {"label": "Single-board computer, 8 GB (Raspberry Pi 5 class)", "config": {"device": "cpu", "cores": 4, "threads": 4, "ram_limit_mb": 8192},
                "ram_bandwidth_gbps": 17.1, "memory": "LPDDR4X-4267, 32-bit"},
    "minipc-16gb": {"label": "Mini PC, 4 cores, 16 GB (Intel N100 class)", "config": {"device": "cpu", "cores": 4, "threads": 4, "ram_limit_mb": 16384},
                    "ram_bandwidth_gbps": 38.4, "memory": "DDR5-4800, single channel"},
    "laptop-16gb": {"label": "Laptop without a GPU, 8 threads, 16 GB", "config": {"device": "cpu", "cores": 8, "threads": 8, "ram_limit_mb": 16384},
                    "ram_bandwidth_gbps": 51.2, "memory": "DDR4-3200, dual channel"},
    "gpu-4gb": {"label": "Entry GPU, 4 GB VRAM", "config": {"device": "gpu", "vram_limit_mb": 4096}},
    "gpu-6gb": {"label": "Budget GPU, 6 GB VRAM", "config": {"device": "gpu", "vram_limit_mb": 6144}},
}
OS_RESERVE_MB = 700  # RAM a minimal Linux needs for itself on the simulated device


HF_PREFIX = "hf:"  # models given as "hf:<owner>--<name>" are Hugging Face checkpoints (for vLLM), others are GGUF files


def model_path(model: str) -> Path:
    if model.startswith(HF_PREFIX):
        from . import torchlab
        return torchlab.HF / Path(model[len(HF_PREFIX):]).name
    return lab.MODELS_DIR / Path(model).name


def checkpoint_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.glob("*.safetensors"))


def normalize_config(exp: dict, cfg: dict) -> dict:
    from . import methods
    model = str((cfg or {}).get("model") or exp["model"])
    if model.startswith(methods.PREFIX):
        return methods.normalize(model, cfg or {})
    preset = (cfg or {}).get("preset") or None
    if preset and preset not in DEVICE_PRESETS:
        raise ValueError(f"Unknown device preset (choose from {', '.join(DEVICE_PRESETS)})")
    # A preset defines the simulated device: its settings win over ones inherited from a parent run. To customize
    # device settings, clear the preset.
    out = {**DEFAULTS, "model": exp["model"], **{k: v for k, v in (cfg or {}).items() if v not in (None, "")},
           **(DEVICE_PRESETS[preset]["config"] if preset else {})}
    m = str(out["model"])
    out["model"] = HF_PREFIX + Path(m[len(HF_PREFIX):]).name if m.startswith(HF_PREFIX) else Path(m).name
    out["engine"] = str(out.get("engine") or "stock")
    hf = out["model"].startswith(HF_PREFIX)
    if hf and engines.kind(out["engine"]) == "llamacpp" and not (cfg or {}).get("engine"):
        out["engine"] = "vllm"  # a checkpoint with no engine chosen: the framework that runs checkpoints
    ckpt = engines.is_checkpoint_framework(out["engine"])
    fw = engines.CHECKPOINT_FRAMEWORKS.get(engines.kind(out["engine"]))
    if ckpt and not hf:
        raise ValueError(f"{fw} runs original Hugging Face checkpoints, not GGUF files: pick a checkpoint (Internals → Checkpoints)")
    if not ckpt and hf:
        raise ValueError("llama.cpp runs GGUF files: pick a GGUF model, or choose vLLM or SGLang as the engine for a checkpoint")
    if ckpt:
        requested = cfg or {}
        unsupported = [k for k in ("kv", "ngl", "fa") if requested.get(k) not in (None, "", "auto")]
        unsupported += [k for k in ("threads", "ram_limit_mb", "split_mode", "rpc") if out.get(k) not in (None, "", "auto", [])]
        if unsupported:
            raise ValueError(f"{fw} does not apply these settings in Blocky: {', '.join(unsupported)}. Use engine defaults (auto) or llama.cpp")
        out["kv"] = "auto"
    if not (model_path(out["model"]) / "config.json" if hf else model_path(out["model"])).is_file():
        raise ValueError("Model not found")
    out["ctx"] = max(1024, min(262144, int(out["ctx"])))
    if out["kv"] not in lab.KV_TYPES and not (ckpt and out["kv"] == "auto"):
        raise ValueError("KV cache must be f16, q8_0 or q4_0")
    if str(out["ngl"]) != "auto":
        out["ngl"] = max(0, min(999, int(out["ngl"])))
    if out["vram_limit_mb"]:
        out["vram_limit_mb"] = max(512, int(out["vram_limit_mb"]))
    if out["fa"] not in ("auto", "on", "off"):
        raise ValueError("Flash attention must be auto, on or off")
    if out["threads"]:
        out["threads"] = max(1, min(256, int(out["threads"])))
    if out["device"] not in ("gpu", "cpu"):
        raise ValueError("Device must be gpu or cpu")
    if out["device"] == "cpu" and ckpt:
        raise ValueError(f"CPU-only runs use llama.cpp; {fw} here needs a GPU")
    # Multiple GPUs. Only stored when set, so configs of runs made before these settings existed stay identical.
    if out.get("gpus") in (None, "", "auto"):
        out.pop("gpus", None)
    else:
        gpu_count = len(lab.hardware().get("gpus") or [])
        out["gpus"] = int(out["gpus"])
        if out["device"] == "cpu":
            raise ValueError("CPU-only runs don't use GPUs")
        if not 1 <= out["gpus"] <= max(1, gpu_count):
            raise ValueError(f"This device has {gpu_count} GPU(s): use 1-{max(1, gpu_count)}")
    if out.get("split_mode") in (None, "", "auto"):
        out.pop("split_mode", None)
    elif out["split_mode"] not in ("layer", "row", "none"):
        raise ValueError("Split mode must be auto, layer (pipeline: layers spread over GPUs), row (tensor parallel) or none (one GPU)")
    # Other machines: llama.cpp RPC workers (host:port of an rpc-server), or pipeline stages for vLLM / TensorRT-LLM.
    rpc = out.get("rpc")
    if rpc in (None, "", []):
        out.pop("rpc", None)
    else:
        items = [x.strip() for x in (rpc.split(",") if isinstance(rpc, str) else rpc) if str(x).strip()]
        if not 1 <= len(items) <= 16 or not all(isinstance(x, str) and _RPC.match(x) for x in items):
            raise ValueError("RPC workers: 1-16 addresses like 192.168.1.40:50052")
        out["rpc"] = items
    if out.get("pp") in (None, "", 1, "1", "auto"):
        out.pop("pp", None)
    else:
        out["pp"] = int(out["pp"])
        if not 2 <= out["pp"] <= 16:
            raise ValueError("Pipeline stages must be 1-16")
        if engines.kind(out["engine"]) not in ("vllm", "trtllm"):
            raise ValueError("Pipeline stages apply to vLLM (across machines with Ray) and TensorRT-LLM; llama.cpp uses RPC workers")
    if out["cores"]:
        out["cores"] = int(out["cores"])
        if not 1 <= out["cores"] <= (os.cpu_count() or 1):
            raise ValueError(f"This device has {os.cpu_count()} CPU threads: cores must be 1-{os.cpu_count()}")
    if out["ram_limit_mb"]:
        out["ram_limit_mb"] = max(1024, int(out["ram_limit_mb"]))
    engines.bin_dir(out["engine"])  # raises if unknown or not built
    return out


_RPC = re.compile(r"^[A-Za-z0-9.\-]{1,253}:\d{2,5}$|^\[[0-9A-Fa-f:]+\]:\d{2,5}$")


def create_run(exp_id: int, body: dict) -> int:
    exp = get_experiment(exp_id)
    if not exp:
        raise ValueError("Experiment not found")
    from . import methods
    if str((body.get("config") or {}).get("model") or exp["model"]).startswith(methods.PREFIX):
        return methods.queue_profile(exp_id, body, normalize_config(exp, body.get("config") or {}))
    baseline = next((r for r in exp["runs_list"] if r["status"] == "done"), None)
    prior_protocol = ((baseline or {}).get("results", {}).get("provenance") or {}).get("protocol")
    if prior_protocol and prior_protocol != PROTOCOL:
        raise ValueError("This experiment's baseline used a different measurement protocol. Create a new experiment and baseline for new runs")
    cfg = normalize_config(exp, body.get("config") or {})
    parent = None
    if body.get("parent_id") is not None:
        parent = next((r for r in exp["runs_list"] if r["id"] == int(body["parent_id"])), None)
        if not parent:
            raise ValueError("The run to branch from isn't in this experiment")
    ref = parent or (exp["runs_list"][0] if exp["runs_list"] else None)
    label = (body.get("label") or "").strip()[:60] or (baseline_label(cfg) if not ref else describe_change(ref["config"], cfg))
    if label == "Baseline":
        label = baseline_label(cfg)
    if label == "Repeat of baseline" and parent and parent["id"] != exp["runs_list"][0]["id"]:
        label = f"Repeat of {parent['label']}"[:60]
    with db.connect() as c:
        return c.execute("INSERT INTO exp_runs (experiment_id, created_at, status, label, config, results, parent_id) VALUES (?, ?, 'queued', ?, ?, '{}', ?)",
                         (exp_id, time.time(), label, json.dumps(cfg), parent["id"] if parent else None)).lastrowid


def baseline_label(cfg: dict) -> str:
    """'Baseline · Qwen3.5-9B-Q8_0' (+ any non-default settings): always say what the baseline actually is."""
    extra = describe_change({**DEFAULTS, "model": cfg["model"]}, cfg)
    return f"Baseline · {cfg['model'].removesuffix('.gguf')}{'' if extra == 'Repeat of baseline' else f' · {extra}'}"[:80]


def weights_label(model: str) -> str:
    """'Qwen3.5-9B-Q3_K_M-imat-custom.gguf' -> 'Q3_K_M-imat-custom' (the part that tells variants apart)."""
    stem = model.removesuffix(".gguf")
    m = re.search(r"-((?:UD-)?(?:I?Q\d|BF16|F16|F32).*)$", stem, re.I)
    return m.group(1) if m else stem


def describe_change(base: dict, cfg: dict) -> str:
    parts = []
    if cfg["model"] != base.get("model"):
        if cfg["model"].startswith(HF_PREFIX) or str(base.get("model") or "").startswith(HF_PREFIX):
            parts.append(cfg["model"].removesuffix(".gguf"))  # a checkpoint on one side: name it in full
        else:
            a, b = (gguf.metadata(lab.MODELS_DIR / m) for m in (base.get("model") or cfg["model"], cfg["model"]))
            # Same model, other weights -> "Q3_K_M-imat"; a different model altogether -> its full name.
            parts.append(weights_label(cfg["model"]) if a.get("name") == b.get("name") else cfg["model"].removesuffix(".gguf"))
    if cfg["kv"] != base.get("kv"):
        parts.append(f"KV {cfg['kv']}")
    if cfg["ctx"] != base.get("ctx"):
        parts.append(f"ctx {cfg['ctx'] // 1024}k")
    if str(cfg["ngl"]) != str(base.get("ngl")):
        parts.append(f"{cfg['ngl']} GPU layers")
    if cfg.get("vram_limit_mb") != base.get("vram_limit_mb") and not (cfg.get("preset") and cfg.get("preset") != base.get("preset")):
        parts.append(f"{(cfg['vram_limit_mb'] or 0) / 1024:.0f} GB GPU" if cfg.get("vram_limit_mb") else "full GPU")
    if cfg["fa"] != base.get("fa"):
        parts.append(f"flash-attn {cfg['fa']}")
    if (cfg.get("engine") or "stock") != (base.get("engine") or "stock"):
        parts.append(f"engine {cfg.get('engine') or 'stock'}")
    if cfg.get("gpus") != base.get("gpus"):
        parts.append(f"{cfg['gpus']} GPU{'s' if cfg['gpus'] > 1 else ''}" if cfg.get("gpus") else "all GPUs")
    if cfg.get("split_mode") != base.get("split_mode"):
        parts.append(f"split {cfg.get('split_mode') or 'auto'}")
    if cfg.get("rpc") != base.get("rpc"):
        parts.append(f"+{len(cfg['rpc'])} RPC worker{'s' if len(cfg['rpc']) > 1 else ''}" if cfg.get("rpc") else "no RPC workers")
    if cfg.get("pp") != base.get("pp"):
        parts.append(f"{cfg.get('pp') or 1} pipeline stages")
    new_preset = cfg.get("preset") and cfg.get("preset") != base.get("preset")
    if cfg.get("threads") != base.get("threads") and not new_preset:
        parts.append(f"{cfg['threads'] or 'auto'} threads")
    if new_preset:
        parts.append(DEVICE_PRESETS[cfg["preset"]]["label"].split(" (")[0])
    else:
        if (cfg.get("device") or "gpu") != (base.get("device") or "gpu"):
            parts.append("CPU only" if cfg.get("device") == "cpu" else "GPU")
        if cfg.get("cores") != base.get("cores"):
            parts.append(f"{cfg['cores']} cores" if cfg.get("cores") else "all cores")
        if cfg.get("ram_limit_mb") != base.get("ram_limit_mb"):
            parts.append(f"{cfg['ram_limit_mb'] / 1024:.0f} GB RAM" if cfg.get("ram_limit_mb") else "full RAM")
    return " · ".join(parts) or "Repeat of baseline"


# ---------- Running a profile ----------

def _placement_args(cfg: dict, vram_total: float, bench: bool = False, margin_mb: int = lab.FIT_MARGIN_MB,
                    gpu_layers: int | None = None) -> list[str]:
    """Placement flags for llama.cpp tools.

    The run's server fits the model with `margin_mb` of GPU memory left free; the other tools then get the layer count
    it chose (`gpu_layers`), so speed, perplexity and the memory analysis all describe the same layout.
    """
    threads = cfg.get("threads") or cfg.get("cores")
    if cfg.get("device") == "cpu":
        # No GPU at all (not even for prompt processing, which -ngl 0 alone would still offload).
        return ["-dev", "none", "-ngl", "0"] + (["-t", str(threads)] if threads else [])
    args = []
    if str(cfg["ngl"]) != "auto":
        args += ["-ngl", str(cfg["ngl"])]
    elif gpu_layers is not None:
        args += ["-ngl", str(gpu_layers)]
    fit = "-fitt" if bench else "--fit-target"
    if gpu_layers is not None and str(cfg["ngl"]) == "auto":
        pass  # the layout is already decided
    elif cfg.get("vram_limit_mb") and vram_total:
        args += [fit, str(max(margin_mb, int(vram_total - cfg["vram_limit_mb"])))]
    elif str(cfg["ngl"]) == "auto":
        args += [fit, str(margin_mb)]
    if threads:
        args += ["-t", str(threads)]
    if cfg.get("split_mode"):
        args += ["-sm", cfg["split_mode"]]
    if cfg.get("rpc"):
        args += ["--rpc", ",".join(cfg["rpc"])]
    return args


def check_rpc_workers(cfg: dict, timeout: float = 3) -> None:
    """Fail fast, with the address, when an RPC worker can't be reached (instead of a hang inside llama.cpp)."""
    import socket
    for addr in cfg.get("rpc") or []:
        host, _, port = addr.rpartition(":")
        try:
            socket.create_connection((host.strip("[]"), int(port)), timeout=timeout).close()
        except OSError as e:
            raise RuntimeError(f"RPC worker {addr} isn't reachable ({e.__class__.__name__}). Start it there "
                               "(Settings → Multi-machine on that device, or rpc-server) and check the address") from None


def checkpoint_server_cmd(fw: str, ebin: Path, cfg: dict, util: float) -> tuple[list[str], str | None]:
    """Command that serves a Hugging Face checkpoint as an OpenAI-compatible API named "blocky" on PORT, plus a note
    about anything this framework can't be told (shown with the run)."""
    model, tp, pp = str(model_path(cfg["model"])), str(cfg.get("gpus") or 1), int(cfg.get("pp") or 1)
    if fw == "vllm":
        cmd = [str(ebin / "vllm"), "serve", model, "--host", "127.0.0.1", "--port", str(PORT), "--served-model-name", "blocky",
               "--max-model-len", str(cfg["ctx"]), "--gpu-memory-utilization", f"{util:.6f}", "--max-num-seqs", "4",
               "--tensor-parallel-size", tp, "--default-chat-template-kwargs", '{"enable_thinking": false}']
        if pp > 1:  # stages on other machines: the Ray cluster the user started (ray start --head / --address)
            cmd += ["--pipeline-parallel-size", str(pp), "--distributed-executor-backend", "ray"]
        return cmd, None
    if fw == "sglang":
        return ([str(ebin / "python"), "-m", "sglang.launch_server", "--model-path", model, "--host", "127.0.0.1", "--port", str(PORT),
                 "--served-model-name", "blocky", "--context-length", str(cfg["ctx"]), "--mem-fraction-static", f"{util:.6f}",
                 "--max-running-requests", "4", "--tp-size", tp],
                "SGLang serves with the model's default chat template (no server-wide setting turns thinking off)")
    if fw == "trtllm":
        # PyTorch backend: serves the checkpoint directly, no separate engine build. Its KV-cache budget is a fraction of
        # the memory left after loading the weights, so the same number means something different than in vLLM.
        return ([str(ebin / "trtllm-serve"), model, "--host", "127.0.0.1", "--port", str(PORT), "--backend", "pytorch",
                 "--max_seq_len", str(cfg["ctx"]), "--max_batch_size", "4", "--tp_size", tp, "--pp_size", str(pp),
                 "--kv_cache_free_gpu_memory_fraction", f"{min(0.9, util):.4f}"],
                "TensorRT-LLM (PyTorch backend) uses the model's default chat template; its KV-cache budget is a share of the memory left after the weights")
    raise ValueError(f"Unknown checkpoint framework {fw}")


def _pin(cmd: list[str], cfg: dict) -> list[str]:
    """Restrict a command to the first N CPU cores (simulating a smaller CPU) and to the first N GPUs."""
    if cfg.get("cores"):
        cmd = ["taskset", "-c", f"0-{cfg['cores'] - 1}", *cmd]
    if cfg.get("gpus"):
        cmd = ["env", f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(cfg['gpus']))}", *cmd]
    return cmd


def gpus_used(cfg: dict, lim: dict) -> int:
    return 0 if cfg.get("device") == "cpu" else (cfg.get("gpus") or lim.get("gpu_count") or 1)


def parallel_gpus(cfg: dict, lim: dict) -> int:
    """GPUs that work on one token at the same time. Tensor parallelism (llama.cpp row split, vLLM/SGLang tensor-parallel)
    splits every layer, so their bandwidth adds up; layer (pipeline) split runs the GPUs one after another for one sequence."""
    n = gpus_used(cfg, lim)
    if n <= 1:
        return 1
    if engines.is_checkpoint_framework(cfg.get("engine")):
        return n if cfg.get("gpus") else 1  # vLLM/SGLang use tensor parallelism only when asked for more than one GPU
    return n if cfg.get("split_mode") == "row" else 1


def _cpu_clock_temp() -> dict:
    out = {}
    try:
        f = psutil.cpu_freq()
        if f and f.current:
            out["cpu_mhz"] = f.current
        temps = [t.current for ts in (psutil.sensors_temperatures() or {}).values() for t in ts if t.current]
        if temps:
            out["cpu_temp"] = max(temps)
    except (AttributeError, OSError, RuntimeError):
        pass
    return out


def _rapl_file() -> Path | None:
    """CPU package energy counter (bare-metal Linux on Intel and AMD Zen; not available inside WSL)."""
    p = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    try:
        return p if p.is_file() and os.access(p, os.R_OK) else None
    except OSError:
        return None


class Sampler:
    """GPU/CPU/RAM telemetry every 0.5 s, with phase tags so each phase can be summarized."""

    def __init__(self, meter: tuple[str, str] | None = None):
        self.meter = meter  # (url, json field) of a wall-power meter, if one is set
        self.samples: list[dict] = []
        self.phase = "idle"
        self._stop = asyncio.Event()
        self._task = None
        self._rapl = _rapl_file()
        self._last_energy: tuple[float, int] | None = None

    def _cpu_power(self) -> float | None:
        if not self._rapl:
            return None
        try:
            now, e = time.monotonic(), int(self._rapl.read_text())
        except (OSError, ValueError):
            return None
        prev, self._last_energy = self._last_energy, (now, e)
        if not prev or e < prev[1]:
            return None
        return (e - prev[1]) / 1e6 / (now - prev[0])

    async def _loop(self):
        psutil.cpu_percent(interval=None)
        http = httpx.AsyncClient() if self.meter else None
        try:
            await self._sample_until_stopped(http)
        finally:
            if http:
                await http.aclose()

    async def _sample_until_stopped(self, http):
        while not self._stop.is_set():
            s = {"t": time.monotonic(), "phase": self.phase, "cpu": psutil.cpu_percent(interval=None),
                 "ram_mb": (psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**20}
            if (cp := self._cpu_power()) is not None:
                s["cpu_power"] = cp
            s.update(_cpu_clock_temp())
            if http and (w := await power.read_wall(http, *self.meter)) is not None:
                s["wall_power"] = w
            code, out = await lab._run("nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu,memory.used,clocks.sm",
                                       "--format=csv,noheader,nounits", timeout=5)
            if code == 0 and out.strip():
                vals = [x.strip() for x in out.strip().splitlines()[0].split(",")]
                for k, v in zip(("gpu", "mem_bw", "power", "temp", "vram_mb", "sm_clock"), vals):
                    try:
                        s[k] = float(v)
                    except ValueError:
                        pass
            self.samples.append(s)
            try:
                await asyncio.wait_for(self._stop.wait(), 0.5)
            except asyncio.TimeoutError:
                pass

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    def summary(self, phase: str) -> dict:
        ss = [s for s in self.samples if s["phase"] == phase]
        if not ss:
            return {}
        def avg(k): v = [s[k] for s in ss if k in s]; return round(sum(v) / len(v), 1) if v else None  # noqa: E702
        def mx(k): v = [s[k] for s in ss if k in s]; return round(max(v), 1) if v else None  # noqa: E702
        return {"gpu_avg": avg("gpu"), "mem_bw_avg": avg("mem_bw"), "power_avg": avg("power"), "power_max": mx("power"),
                "temp_max": mx("temp"), "vram_max_mb": mx("vram_mb"), "cpu_avg": avg("cpu"), "ram_max_mb": mx("ram_mb"),
                "sm_clock_avg": avg("sm_clock"), "cpu_power_avg": avg("cpu_power"), "wall_power_avg": avg("wall_power"), "cpu_mhz_avg": avg("cpu_mhz"), "cpu_temp_max": mx("cpu_temp"), "seconds": round(ss[-1]["t"] - ss[0]["t"], 1),
                "samples": len(ss)}

    def series(self, every: int = 2) -> list[dict]:
        t0 = self.samples[0]["t"] if self.samples else 0
        return [{"t": round(s["t"] - t0, 1), "phase": s["phase"], **{k: s.get(k) for k in ("gpu", "mem_bw", "power", "temp", "vram_mb", "cpu", "sm_clock", "cpu_power", "wall_power", "cpu_mhz", "cpu_temp")}}
                for i, s in enumerate(self.samples) if i % every == 0]


def _parse_load_log(text: str) -> dict:
    gpu = {"model": 0.0, "context": 0.0, "compute": 0.0}
    host = {"model": 0.0, "context": 0.0, "compute": 0.0}
    layers = None
    for line in text.splitlines():
        if m := lab._BUF.search(line):
            device, kind, mib = m.group(1), m.group(2), float(m.group(3))
            slot = {"model": "model", "KV": "context", "RS": "context"}.get(kind, "compute")
            (gpu if lab._is_gpu(device) else host)[slot] += mib
        elif m := lab._OFFLOAD.search(line):
            layers = {"gpu": int(m.group(1)), "total": int(m.group(2))}
    return {"gpu": {k: round(v) for k, v in gpu.items()}, "ram": {k: round(v) for k, v in host.items()}, "layers": layers}


# ---------- Quality: MMLU-Pro, 5-shot, log-likelihood ----------
# The same protocol as the Open LLM Leaderboard (v2): the model never writes an answer; we read the
# probability it assigns to each option letter after "Answer:". No parsing errors, no sampling noise,
# 10 options (chance = 10%). Contamination doesn't matter here: every run of an experiment is the same
# model, so we measure the *difference* a change makes on identical questions.

MMLU_PER_CATEGORY = 40   # × 14 subjects = 560 questions
MMLU_SEED = 20240611     # fixed forever: every experiment uses the same questions, so results compare across experiments
LETTERS = "ABCDEFGHIJ"


MMLU_SIZES = (40, 100, 250)  # questions per subject: 560, 1400 or 3500 in total (an experiment's protocol picks one)


def mmlu_sample(seed: int = MMLU_SEED, per_category: int = MMLU_PER_CATEGORY) -> tuple[list[dict], list[dict]]:
    """Few-shot examples and the test questions.

    Larger sizes are nested: the questions are ordered in tiers (the first 40 per subject, then the next 60, ...), so the
    first 560 answers of any size are exactly the default 560 and paired tests between sizes use the common questions.
    """
    import random
    from . import benchmarks
    if not benchmarks.path("mmlu_pro").is_file():
        raise RuntimeError("MMLU-Pro isn't downloaded yet: get it in Quality tests → Benchmarks (one click).")
    qs = evals.bench_questions("mmlu_pro")
    by_cat: dict[str, list[dict]] = {}
    for q in qs:
        by_cat.setdefault(q.get("tag") or "other", []).append(q)
    rng = random.Random(seed)
    shots, pools = [], {}
    for cat in sorted(by_cat):
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        if len(shots) < 5:
            shots.append(pool.pop(MMLU_PER_CATEGORY))  # the few-shot example is never also a test question
        pools[cat] = pool
    sample, lo = [], 0
    for hi in [t for t in MMLU_SIZES if t <= per_category] or [per_category]:
        for cat in sorted(pools):
            sample += [{**q, "tag": cat} for q in pools[cat][lo:hi]]
        lo = hi
    return shots, sample


def _mc_block(q: dict, answer: bool) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    return f"Question: {q['q'].strip()}\nOptions:\n{opts}\nAnswer:" + (f" {q['answer']}\n\n" if answer else "")


async def _next_token_logprobs(client: httpx.AsyncClient, prompt: str, api: str) -> list[tuple[str, float]]:
    """Top next-token log-probabilities after `prompt`, from llama.cpp's /completion or an OpenAI-style /v1/completions."""
    if api == "openai":
        r = await client.post("/v1/completions", json={"model": "blocky", "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 20})
        r.raise_for_status()
        top = ((r.json()["choices"][0].get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
        return list(top.items())
    r = await client.post("/completion", json={"prompt": prompt, "n_predict": 1, "n_probs": 50, "temperature": 0, "cache_prompt": True})
    r.raise_for_status()
    top = (r.json().get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
    return [(t["token"], t["logprob"]) for t in top]


def letter_choice(top: list[tuple[str, float]], n_options: int, answer: str) -> tuple[str | None, float]:
    """Predicted letter and the probability of the correct one, renormalized over the option letters only."""
    import math
    lp: dict[str, float] = {}
    for tok, v in top:
        t = tok.strip()
        if len(t) == 1 and t in LETTERS[:n_options] and t not in lp:
            lp[t] = v
    if not lp:
        return None, 0.0
    m = max(lp.values())
    z = sum(math.exp(v - m) for v in lp.values())
    return max(lp, key=lp.get), (math.exp(lp[answer] - m) / z if answer in lp else 0.0)


async def mmlu_eval(client: httpx.AsyncClient, seed: int, should_stop, progress, api: str = "llamacpp", per_category: int = MMLU_PER_CATEGORY) -> dict:
    shots, sample = mmlu_sample(seed, per_category)
    prefix = "The following are multiple choice questions (with answers).\n\n" + "".join(_mc_block(s, True) for s in shots)
    correct, p_true, cats = [], [], {}
    for i, q in enumerate(sample):
        if should_stop():
            raise asyncio.CancelledError
        top = await _next_token_logprobs(client, prefix + _mc_block(q, False), api)
        pred, pt = letter_choice(top, len(q["options"]), q["answer"])
        if pred is None:
            raise RuntimeError("MMLU-Pro could not score an answer: the server returned no option-letter log probabilities")
        ok = int(pred == q["answer"])
        correct.append(ok); p_true.append(pt)
        c = cats.setdefault(q["tag"], [0, 0]); c[0] += ok; c[1] += 1
        if i % 20 == 19:
            progress(i + 1, len(sample), sum(correct))
    n, k = len(correct), sum(correct)
    return {"benchmark": "MMLU-Pro", "protocol": "5-shot, top-k option-letter scoring (not full option likelihood)", "n": n, "correct": k, "acc": round(k / n, 4),
            "probability_note": "Scores are renormalized over option letters in the server's returned shortlist; missing letters do not have measured probabilities",
            "ci95": [round(x, 4) for x in wilson(k, n)], "p_correct_mean": round(sum(p_true) / n, 4),
            "by_category": {c: {"correct": v[0], "total": v[1]} for c, v in sorted(cats.items())},
            "answers": "".join(str(x) for x in correct), "chance": 0.1}


# ---------- lm-evaluation-harness (optional, per experiment) ----------

ML_VENV = config.DATA_DIR.parent / "ml-venv"
_TASKS = re.compile(r"^[a-z0-9_.-]+(,[a-z0-9_.-]+)*$")


PPL_CHUNKS = 32                     # default: 32 × 512 tokens of WikiText-2 test (the protocol string's "c512 x32")
PPL_CHUNK_OPTIONS = (32, 64, 128, 256)


def ppl_chunks(exp: dict) -> int:
    return int((exp.get("protocol") or {}).get("ppl_chunks") or PPL_CHUNKS)


async def fidelity_text(exp: dict) -> Path:
    """The text perplexity and KL divergence are scored on: WikiText-2 (test) unless the experiment chose its own file."""
    custom = (exp.get("protocol") or {}).get("fidelity_text")
    if custom:
        p = Path(custom)
        if not p.is_file():
            raise RuntimeError(f"The experiment's perplexity/KLD text {custom} is missing")
        return p
    return await quantlab.ensure_text()


def set_protocol(exp_id: int, body: dict) -> dict:
    """Extra tests every run of this experiment gets. Today: lm-evaluation-harness tasks (generative, via chat)."""
    lm = body.get("lm_eval") or {}
    tasks = re.sub(r"\s+", "", str(lm.get("tasks") or "")).lower()
    proto: dict = {}
    if tasks:
        if not _TASKS.match(tasks):
            raise ValueError("Tasks: comma-separated lm-eval task names, e.g. gsm8k,ifeval")
        proto["lm_eval"] = {"tasks": tasks, "limit": max(1, min(5000, int(lm.get("limit") or 100)))}
    if int(body.get("mmlu_per_category") or MMLU_PER_CATEGORY) != MMLU_PER_CATEGORY:
        n = int(body["mmlu_per_category"])
        if n not in MMLU_SIZES:
            raise ValueError(f"MMLU-Pro size: {', '.join(str(x * 14) for x in MMLU_SIZES)} questions")
        proto["mmlu_per_category"] = n
    if body.get("ruler"):
        proto["ruler"] = {"samples": max(2, min(50, int((body["ruler"] or {}).get("samples") or 8)))}
    text = str(body.get("fidelity_text") or "").strip()
    if text:
        p = Path(text)
        if not p.is_absolute() or p.suffix != ".txt" or not p.is_file():
            raise ValueError("Perplexity/KLD text: an absolute path to an existing .txt file on this device")
        if p.stat().st_size < 50_000:
            raise ValueError("Perplexity/KLD text: use at least 50 KB of text (32 chunks of 512 tokens need roughly 70 KB)")
        proto["fidelity_text"] = str(p)
    margin = body.get("equivalence_margin_pts")
    if margin not in (None, "", 0):
        margin = float(margin)
        if not 0.1 <= margin <= 20:
            raise ValueError("Equivalence margin: 0.1-20 accuracy points")
        proto["equivalence_margin_pts"] = margin
    clock = body.get("lock_clocks_mhz")
    if clock not in (None, "", 0):
        clock = int(clock)
        if not 300 <= clock <= 4000:
            raise ValueError("GPU clock lock: 300-4000 MHz")
        proto["lock_clocks_mhz"] = clock
    cool = int(body.get("cooldown_s") or 0)
    if cool:
        if not 5 <= cool <= 600:
            raise ValueError("Cool-down: 5-600 seconds")
        proto["cooldown_s"] = cool
    if int(body.get("ppl_chunks") or PPL_CHUNKS) != PPL_CHUNKS:
        n = int(body["ppl_chunks"])
        if n not in PPL_CHUNK_OPTIONS:
            raise ValueError(f"Perplexity/KLD sample: {', '.join(str(x) for x in PPL_CHUNK_OPTIONS)} chunks of 512 tokens")
        proto["ppl_chunks"] = n
    if body.get("kernels"):
        proto["kernels"] = True  # GPU kernel profile with Nsight Systems (llama.cpp runs)
    with db.connect() as c:
        c.execute("UPDATE experiments SET protocol = ? WHERE id = ?", (json.dumps(proto), exp_id))
    return proto


async def lm_eval_run(lm: dict, out_dir: Path, log: list[str], run_id: int) -> dict:
    """lm_eval against this run's temporary llama-server (OpenAI-compatible chat endpoint), greedy, fixed seed."""
    exe = ML_VENV / "bin" / "lm_eval"
    if not exe.is_file():
        return {"error": "lm-evaluation-harness isn't installed (scripts/setup-ml.sh)"}
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [str(exe), "run", "--model", "local-chat-completions",
           "--model_args", f"model=blocky,base_url=http://127.0.0.1:{PORT}/v1/chat/completions,num_concurrent=1,max_retries=2,tokenized_requests=False,timeout=900",
           "--tasks", lm["tasks"], "--apply_chat_template", "--limit", str(lm["limit"]), "--seed", "0,0,0,0",
           "--gen_kwargs", "temperature=0", "--output_path", str(out_dir)]
    t0 = time.monotonic()
    await quantlab._proc(cmd, log, run_id, timeout=4 * 3600)
    files = sorted(out_dir.rglob("results_*.json"))
    if not files:
        return {"error": "lm-eval produced no results: " + " | ".join(log[-3:])[-300:], "command": " ".join(cmd)}
    data = json.loads(files[-1].read_text())
    tasks = {}
    for task, metrics in (data.get("results") or {}).items():
        vals = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and "stderr" not in k}
        errs = {k.replace("_stderr", ""): v for k, v in metrics.items() if "stderr" in k and isinstance(v, (int, float))}
        tasks[task] = {"metrics": vals, "stderr": errs}
    return {"tasks": tasks, "limit": lm["limit"], "seconds": round(time.monotonic() - t0), "version": (data.get("lm_eval_version") or data.get("git_hash")),
            "command": " ".join(cmd)}


# ---------- Long context: needle in a haystack ----------
# A secret code is hidden at 5 depths inside real text (WikiText-2) of growing length, up to the run's context size.
# Retrieval accuracy by length × depth shows where long-context recall breaks (KV-cache precision, context size,
# flash attention and the model itself all affect it). Fixed seed: identical haystacks and needles for every run.

NEEDLE_SEED = 1234
NEEDLE_DEPTHS = [0.05, 0.25, 0.5, 0.75, 0.95]
NEEDLE_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536]
_CITIES = ["Lisbon", "Oslo", "Nairobi", "Lima", "Hanoi", "Quito", "Riga", "Perth", "Dakar", "Bergen", "Porto", "Cusco", "Tallinn", "Hobart", "Valletta"]


async def chars_per_token(client: httpx.AsyncClient, api: str) -> float:
    """How many characters of WikiText one token covers for this model's tokenizer (to size prompts in tokens)."""
    sample = (await quantlab.ensure_text()).read_text()[:40000]
    body = {"model": "blocky", "prompt": sample} if api == "openai" else {"content": sample}
    r = await client.post("/tokenize", json=body)
    r.raise_for_status()
    d = r.json()
    return len(sample) / max(1, d.get("count") or len(d.get("tokens", [])))


async def needle_eval(client: httpx.AsyncClient, ctx: int, should_stop, progress, api: str = "llamacpp") -> dict:
    import random
    from . import generate
    text = (await quantlab.ensure_text()).read_text()
    chars_per_tok = await chars_per_token(client, api)
    lengths = [L for L in NEEDLE_LENGTHS if L + 600 <= ctx]
    rng = random.Random(NEEDLE_SEED)
    cells, total = [], len(lengths) * len(NEEDLE_DEPTHS)
    for L in lengths:
        n_chars = int(L * chars_per_tok)
        start = rng.randrange(0, max(1, len(text) - n_chars - 1))
        hay = text[start:start + n_chars]
        for depth in NEEDLE_DEPTHS:
            if should_stop():
                raise asyncio.CancelledError
            city, code = rng.choice(_CITIES), str(rng.randrange(1_000_000, 9_999_999))
            cut = hay.rfind(". ", 0, int(len(hay) * depth)) + 2 if depth > 0.05 else int(len(hay) * depth)
            doc = f"{hay[:cut]}The secret code for the {city} vault is {code}. {hay[cut:]}"
            q = f"{doc}\n\nQuestion: What is the secret code for the {city} vault? Reply with the number only."
            info = await generate.complete(client, [{"role": "user", "content": q}], think=False, sampling={"temperature": 0.0},
                                           seed=1, max_tokens=24, should_stop=should_stop)
            t = info.get("timings") or {}
            cells.append({"length": L, "depth": depth, "ok": code in re.sub(r"[^\d]", "", info.get("content") or ""),
                          "prompt_tokens": t.get("prompt_n"), "prefill_tps": round(t["prompt_per_second"], 1) if t.get("prompt_per_second") else None})
            progress(len(cells), total, sum(c["ok"] for c in cells))
    by_len = {str(L): round(sum(c["ok"] for c in cells if c["length"] == L) / len(NEEDLE_DEPTHS), 3) for L in lengths}
    pf = {str(L): next((c["prefill_tps"] for c in cells if c["length"] == L and c["prefill_tps"]), None) for L in lengths}
    return {"lengths": lengths, "depths": NEEDLE_DEPTHS, "cells": cells, "acc": round(sum(c["ok"] for c in cells) / len(cells), 4) if cells else None,
            "note": None if cells else "No test length fits this context; retrieval was not measured",
            "by_length": by_len, "prefill_tps_by_length": pf, "max_length": lengths[-1] if lengths else None}


# ---------- Serving speed: the same client-side measurement for every engine ----------
# llama-bench only exists for llama.cpp. To compare frameworks fairly, every run also measures speed the way an
# application sees it: through the server's OpenAI-compatible chat API, streaming, with a ~512-token prompt and
# 128 forced output tokens. Each repetition uses a different text slice, so no prompt cache can help.

SERVING_REPS = 5


def _usage_tokens(usage: dict, key: str) -> int | None:
    """A stream event is not a token. Only accept counts reported by the server."""
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


async def serving_bench(client: httpx.AsyncClient, api: str, should_stop, prompt_tokens: int = 512, gen_tokens: int = 128) -> dict:
    import random
    text = (await quantlab.ensure_text()).read_text()
    cpt = await chars_per_token(client, api)
    rng = random.Random(4321)
    pp, tg, ttft = [], [], []
    for rep in range(SERVING_REPS + 1):  # the first repetition is a warm-up and isn't counted
        if should_stop():
            raise asyncio.CancelledError
        n = int(prompt_tokens * cpt)
        start = rng.randrange(0, max(1, len(text) - n - 1))
        body = {"model": "blocky", "messages": [{"role": "user", "content": text[start:start + n] + "\n\nContinue this text."}],
                "max_tokens": gen_tokens, "temperature": 0, "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
        t0 = time.perf_counter()
        first = None
        usage = {}
        async with client.stream("POST", "/v1/chat/completions", json=body, timeout=httpx.Timeout(30, read=600)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if should_stop():
                    raise asyncio.CancelledError
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                usage = d.get("usage") or usage
                delta = ((d.get("choices") or [{}])[0].get("delta") or {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
        finished = time.perf_counter()
        if rep == 0 or first is None:
            continue
        out_tokens = _usage_tokens(usage, "completion_tokens")
        in_tokens = _usage_tokens(usage, "prompt_tokens")
        ttft.append((first - t0) * 1000)
        if in_tokens is not None and first > t0:
            pp.append(in_tokens / (first - t0))
        if out_tokens is not None and finished > t0:
            tg.append(out_tokens / (finished - t0))
    s_pp, s_tg, s_tt = robust_stats(pp), robust_stats(tg), robust_stats(ttft)
    return {"pp": s_pp["median"], "pp_sd": s_pp["sd"], "tg": s_tg["median"], "tg_sd": s_tg["sd"], "ttft_ms": s_tt["median"], "ttft_sd": s_tt["sd"],
            "reps": len(tg), "requested_reps": SERVING_REPS, "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
            "samples": {"pp": pp, "tg": tg, "ttft_ms": ttft}, "timing_scope": "request",
            "note": "Token rates unavailable for responses without server usage counts" if len(pp) < SERVING_REPS or len(tg) < SERVING_REPS else None,
            "method": "Output tokens / full request time (includes prefill); prompt tokens / time to first content chunk is a latency proxy, not isolated prefill. Server usage counts only; median of 5 after 1 warm-up"}


ENERGY_IDLE_S = 8
ENERGY_GEN_S = 15


async def sustained_generation(client: httpx.AsyncClient, tg_est: float | None, ctx: int, should_stop) -> tuple[int, float, float]:
    """Return actual output tokens and the full request interval, including prefill."""
    n = int(min(max(256, (tg_est or 30) * ENERGY_GEN_S), max(64, ctx - 256), 8192))
    body = {"model": "blocky", "messages": [{"role": "user", "content": "Write a long, detailed story about a lighthouse keeper."}],
            "max_tokens": n, "temperature": 0, "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    started = time.monotonic()
    usage = {}
    async with client.stream("POST", "/v1/chat/completions", json=body, timeout=httpx.Timeout(30, read=900)) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if should_stop():
                raise asyncio.CancelledError
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])
            except ValueError:
                continue
            usage = d.get("usage") or usage
    finished = time.monotonic()
    return _usage_tokens(usage, "completion_tokens") or 0, started, finished


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_p(a: str, b: str) -> float | None:
    """Exact two-sided McNemar test on paired right/wrong answers (same questions, two setups)."""
    import math
    if not a or len(a) != len(b):
        return None
    n01 = sum(1 for x, y in zip(a, b) if x == "1" and y == "0")
    n10 = sum(1 for x, y in zip(a, b) if x == "0" and y == "1")
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_diff_ci(a: str, b: str, z: float = 1.96) -> dict | None:
    """Accuracy difference B - A on the same questions, with Newcombe's hybrid score interval for paired proportions
    (Newcombe 1998, method 10: Wilson intervals of both accuracies combined with the observed correlation).
    Better coverage than the Wald interval when few answers flip, which is the usual case here."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    e = sum(1 for x, y in zip(a, b) if x == "1" and y == "1")
    f = sum(1 for x, y in zip(a, b) if x == "1" and y == "0")
    g = sum(1 for x, y in zip(a, b) if x == "0" and y == "1")
    h = n - e - f - g
    p1, p2 = (e + f) / n, (e + g) / n
    l1, u1 = wilson(e + f, n, z)
    l2, u2 = wilson(e + g, n, z)
    den = (e + f) * (g + h) * (e + g) * (f + h)
    phi = (e * h - f * g) / den ** 0.5 if den else 0.0
    d = p2 - p1
    lo = d - max(0.0, (p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2) ** 0.5
    hi = d + max(0.0, (u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2) ** 0.5
    return {"diff": round(d, 5), "ci95": [round(max(-1.0, lo), 5), round(min(1.0, hi), 5)], "n": n,
            "method": "Newcombe hybrid score (paired)"}


def equivalence(a: str, b: str, margin_pts: float | None = None) -> dict | None:
    """Two one-sided tests (TOST) for "B's accuracy is within ±margin of A's", at α = 0.05 each side.

    Equivalent to checking whether the 90% paired interval (Newcombe) lies inside (-margin, +margin). Also reports the
    smallest margin the data would support, which is informative without a preselected margin; a claim of equivalence
    should use a margin chosen before seeing the data (a preregistration, or the experiment's test settings).
    """
    ci = paired_diff_ci(a, b, z=1.6448536)
    if not ci:
        return None
    lo, hi = ci["ci95"]  # with z = 1.645 this is the 90% interval
    out = {"ci90_pts": [round(lo * 100, 3), round(hi * 100, 3)], "min_margin_pts": round(max(abs(lo), abs(hi)) * 100, 3),
           "method": "TOST (two one-sided tests, α = 0.05) via the 90% Newcombe paired interval"}
    if margin_pts:
        out.update(margin_pts=margin_pts, equivalent=(-margin_pts < lo * 100 and hi * 100 < margin_pts))
    return out


# ---------- Provenance: everything needed to reproduce (or question) a run ----------

PROTOCOL = "blocky-profile/4 (MMLU-Pro option logprobs · llama-bench p512 n128 d0,4096 r5 · PPL/KLD WikiText-2 c512 x32 · serving/energy: actual usage, full request interval · one layout per run: server fit, tools reuse its GPU layers)"
REPO = Path(__file__).resolve().parents[2]


def _sh(*cmd: str, timeout: int = 20, libdir: Path | None = None) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "LD_LIBRARY_PATH": str(libdir or lab.LLAMA_BIN)})
        return (p.stdout + "\n" + p.stderr).strip()  # llama.cpp prints --version to stderr
    except (OSError, subprocess.SubprocessError):
        return ""


def file_sha256(path: Path) -> str | None:
    """SHA-256 of a file, cached by (name, size, mtime) because models are GBs."""
    import hashlib
    try:
        st = path.stat()
    except OSError:
        return None
    cache_file = DIR / "hashes.json"
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        cache = {}
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 2**20), b""):
                h.update(chunk)
        cache[key] = h.hexdigest()
        cache_file.write_text(json.dumps(cache))
    return cache[key]


def _recipe(model: str) -> dict | None:
    """How a weights-workshop variant was made (source, imatrix, overrides), if it was made here."""
    p = (lab.MODELS_DIR / model).with_suffix(".recipe.json")
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None


def provenance(cfg: dict) -> dict:
    from . import benchmarks
    hw = lab.hardware()
    g = hw["gpus"][0] if hw["gpus"] else {}
    eb = engines.bin_dir(cfg.get("engine"))
    ckpt = engines.is_checkpoint_framework(cfg.get("engine"))
    ver = "" if ckpt else _sh(str(eb / "llama-server"), "--version", libdir=eb)
    m = re.search(r"version:\s*(\S+)\s*\((?:build\s*\d+,\s*commit\s*)?([0-9a-f]{6,})\)", ver)
    mp = model_path(cfg["model"])
    if cfg["model"].startswith(HF_PREFIX):
        import hashlib
        h = hashlib.sha256("".join(f"{f.name}:{file_sha256(f)}" for f in sorted(mp.glob("*.safetensors"))).encode()).hexdigest()
        model_info = {"file": cfg["model"], "sha256": h, "bytes": checkpoint_bytes(mp), "recipe": None, "format": "safetensors"}
    else:
        model_info = {"file": cfg["model"], "sha256": file_sha256(mp), "bytes": mp.stat().st_size, "recipe": _recipe(cfg["model"]), "format": "gguf"}
    smi = _sh("nvidia-smi", "-q")
    cuda_rt = re.search(r"CUDA (?:UMD )?Version\s*:\s*([\d.]+)", smi)
    try:
        cuda_tk = json.loads(Path("/usr/local/cuda/version.json").read_text())["cuda"]["version"]
    except (OSError, ValueError, KeyError):
        cuda_tk = None
    git = lambda *a: _sh("git", "-c", "safe.directory=*", "-C", str(REPO), *a)  # noqa: E731
    text = quantlab.TEXT
    return {
        "protocol": PROTOCOL,
        "engine": next((e for e in engines.list_engines() if e["id"] == (cfg.get("engine") or "stock")), {"id": cfg.get("engine")}),
        "framework": engines.kind(cfg.get("engine")) if ckpt else "llama.cpp",
        "gpus_all": [x.get("name") for x in hw["gpus"]],
        "distributed": ({"rpc_workers": cfg.get("rpc"), "pipeline_stages": cfg.get("pp")} if (cfg.get("rpc") or cfg.get("pp")) else None),
        "llama_cpp": ({"version": engines.framework_version(cfg.get("engine")), "commit": None,
                       "built_with": engines.CHECKPOINT_FRAMEWORKS[engines.kind(cfg.get("engine"))]} if ckpt else
                      {"version": m.group(1) if m else None, "commit": m.group(2) if m else None,
                       "built_with": next((l.strip() for l in ver.splitlines() if l.startswith("built with")), None)}),
        "model": model_info,
        "data": {"mmlu_pro_sha256": file_sha256(benchmarks.path("mmlu_pro")), "mmlu_seed": MMLU_SEED,
                 "wikitext_sha256": file_sha256(text) if text.is_file() else None},
        "gpu": {"name": g.get("name"), "driver": g.get("driver"), "vram_mb": g.get("vram_mb"), "pcie": g.get("pcie"),
                "cuda_driver_api": cuda_rt.group(1) if cuda_rt else None, "cuda_toolkit": cuda_tk},
        "host": {"cpu": hw["cpu"], "threads": hw["cpu_threads"], "ram_mb": hw["ram_mb"], "ram_gb": round(hw["ram_mb"] / 1024),
                 "os": hw["os"], "kernel": hw["kernel"], "wsl": hw["wsl"], "device": __import__("socket").gethostname()},
        "blocky": {"commit": git("rev-parse", "--short", "HEAD") or None,
                   "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# Fields that must match the baseline for a comparison to be clean.
PROV_KEYS = [("llama.cpp commit", ("llama_cpp", "commit")), ("GPU driver", ("gpu", "driver")), ("CUDA toolkit", ("gpu", "cuda_toolkit")),
             ("protocol", ("protocol",)), ("MMLU-Pro data", ("data", "mmlu_pro_sha256")), ("WSL", ("host", "wsl"))]
# Hardware: when these differ, the hardware itself is what's being compared (shown, but not a warning).
HW_KEYS = [("GPU", ("gpu", "name")), ("VRAM", ("gpu", "vram_mb")), ("CPU", ("host", "cpu")), ("RAM", ("host", "ram_gb")), ("Device", ("host", "device"))]


def hardware_snapshot() -> dict:
    """Everything about the hardware a run's numbers depend on, frozen at run time."""
    import socket
    s = dict(limits())
    s["compute_capability"] = engines._cuda_arch()
    s["device"] = socket.gethostname()
    s["captured_at"] = time.time()
    return s


def _get(d: dict, path: tuple):
    for k in path:
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d


def _cmdline(cmd: list[str]) -> str:
    import shlex
    return " ".join(shlex.quote(c) for c in cmd)


async def _bench(cfg: dict, vram_total: float, run_id: int, log: list[str], cmds: dict | None = None, **placement) -> dict:
    cmd = [str(engines.bin_dir(cfg.get("engine")) / "llama-bench"), "-m", str(lab.MODELS_DIR / cfg["model"]), "-p", "512", "-n", "128",
           "-d", "0,4096", "-r", "5", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"], "-o", "jsonl"] + _placement_args(cfg, vram_total, bench=True, **placement)
    cmd = _pin(cmd, cfg)
    if cmds is not None:
        cmds["speed"] = _cmdline(cmd)
    out = await quantlab._proc(cmd, log, run_id, timeout=900)
    _retain_output(run_id, "speed", out)
    return parse_bench(out)


def _retain_output(run_id: int, phase: str, output: str) -> None:
    with db.connect() as c:
        row = c.execute("SELECT experiment_id FROM exp_runs WHERE id = ?", (run_id,)).fetchone()
    if row:
        folder = DIR / str(row["experiment_id"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"run{run_id}-{phase}.txt").write_text(output, encoding="utf-8")


def evidence_outputs(run_id: int) -> dict:
    with db.connect() as c:
        row = c.execute("SELECT experiment_id, results FROM exp_runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        raise ValueError("Run not found")
    if json.loads(row["results"]).get("imported_from"):
        return {"outputs": {}, "note": "Raw output files were not included in the imported experiment"}
    folder = DIR / str(row["experiment_id"])
    outputs = {}
    for phase in ("speed", "fidelity", "log"):
        path = folder / f"run{run_id}-{phase}.txt"
        if path.is_file():
            with path.open(encoding="utf-8", errors="replace") as f:
                content = f.read(1_000_001)
            outputs[phase] = {"text": content[:1_000_000], "truncated": len(content) > 1_000_000}
    return {"outputs": outputs, "note": "Output retention applies to new runs; earlier results may contain summaries only"}


def robust_stats(raw: list[float]) -> dict:
    """Median and robust spread of repeated measurements.

    Only non-finite/non-positive samples are invalid. Slow or unusually fast positive repetitions remain visible:
    this summary cannot distinguish a timer bug from a real performance change. Spread = 1.4826 × MAD.
    """
    pos = sorted(s for s in raw if isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) and s > 0)
    samples = pos
    if not samples:
        return {"median": None, "sd": None, "dropped": len(raw)}
    med = statistics.median(samples)
    mad = statistics.median(abs(s - med) for s in samples)
    return {"median": round(med, 2), "sd": round(1.4826 * mad, 2), "dropped": len(raw) - len(samples)}


def parse_bench(out: str) -> dict:
    """llama-bench -o jsonl output -> {pp, tg, pp_d4096, tg_d4096} with *_sd, *_samples and *_dropped."""
    res = {}
    for line in out.splitlines():
        if line.startswith("{"):
            x = json.loads(line)
            key = ("pp" if x.get("n_prompt") and not x.get("n_gen") else "tg") + ("_d4096" if x.get("n_depth") else "")
            raw = x.get("samples_ts") or []
            ns = x.get("samples_ns") or []
            if len(ns) == len(raw):
                # A negative duration (clock stepped backwards, seen under WSL) arrives as uint64: ~1.8e19 ns, and a
                # positive but meaningless rate of ~1e-8 tok/s. The duration is the measurement: drop such samples.
                raw = [None if isinstance(n, int) and not 0 < n < 2**63 else s for s, n in zip(raw, ns)]
            st = robust_stats(raw)
            if st["dropped"]:
                res[key + "_dropped"] = st["dropped"]
            if st["median"] is not None:
                res[key], res[key + "_sd"] = st["median"], st["sd"]
                res[key + "_samples"] = [round(s, 2) if isinstance(s, (int, float)) and math.isfinite(s) else None for s in raw]
            elif not raw:
                avg, sd = x.get("avg_ts"), x.get("stddev_ts")
                res[key] = avg if isinstance(avg, (int, float)) and math.isfinite(avg) and avg > 0 else None
                res[key + "_sd"] = sd if isinstance(sd, (int, float)) and math.isfinite(sd) and sd >= 0 else None
            else:
                res[key] = res[key + "_sd"] = None
    return res


class _SkipRest(Exception):
    """Jump to the final analysis step (used when an engine can't run the remaining phases)."""


def _check_kld_reference(path: Path, baseline: dict | None, chunks: int = PPL_CHUNKS, text_sha256: str | None = None) -> None:
    try:
        ref = json.loads(path.with_suffix(".json").read_text())
    except (OSError, ValueError):
        ref = {}
    if not path.is_file() or not baseline or ref.get("run_id") != baseline["id"] or ref.get("protocol") != PROTOCOL:
        raise RuntimeError("Baseline logits are missing or belong to a different run/protocol. Create a new experiment and profile its baseline before comparing KL divergence")
    if int(ref.get("chunks") or PPL_CHUNKS) != chunks:
        raise RuntimeError(f"The baseline's logits cover {ref.get('chunks') or PPL_CHUNKS} perplexity chunks but this experiment now uses {chunks}. "
                           "Create a new experiment (or set the sample size back) so every run is scored on the same text")
    if text_sha256 and ref.get("text_sha256") and ref["text_sha256"] != text_sha256:
        raise RuntimeError("The baseline's logits were computed on a different perplexity/KLD text than this run uses. "
                           "Create a new experiment so every run is scored on the same text")


async def _stop_server(proc: asyncio.subprocess.Process | None) -> None:
    """Stop a model server and everything it started (vLLM runs its engine in child processes)."""
    import signal
    if not proc or proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    group = pgid == proc.pid  # started with its own session: stop the whole group
    try:
        os.killpg(pgid, signal.SIGTERM) if group else proc.terminate()
        await asyncio.wait_for(proc.wait(), 30)
    except asyncio.TimeoutError:
        os.killpg(pgid, signal.SIGKILL) if group else proc.kill()
        await proc.wait()
    except ProcessLookupError:
        pass
    if group:
        try:
            os.killpg(pgid, signal.SIGKILL)  # stragglers (e.g. vLLM's engine core) that outlived the parent
        except ProcessLookupError:
            pass


GPU_CLOCKS = "/usr/local/sbin/blocky-gpu-clocks"  # root-owned helper allowed by sudoers (install.sh): lock <MHz> | reset


async def noise_controls_on(exp: dict, cfg: dict) -> dict:
    """Optional, per experiment: wait for the GPU to cool down, then lock its clock so boost behavior doesn't add noise.

    Both are recorded in the run's provenance; a run whose clock lock failed says so (and doesn't compare with locked runs).
    """
    proto = exp.get("protocol") or {}
    out: dict = {"cooldown_s": proto.get("cooldown_s"), "clock_lock_requested_mhz": proto.get("lock_clocks_mhz"), "clock_lock_mhz": None}
    if proto.get("cooldown_s"):
        await asyncio.sleep(proto["cooldown_s"])
    code, temp = await lab._run("nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", timeout=10)
    out["start_temp_c"] = float(temp.split()[0]) if code == 0 and temp.strip()[:1].isdigit() else None
    mhz = proto.get("lock_clocks_mhz")
    if mhz and cfg.get("device") != "cpu":
        code, msg = await lab._run("sudo", "-n", GPU_CLOCKS, "lock", str(mhz), timeout=20)
        if code == 0:
            out["clock_lock_mhz"] = mhz
        else:
            out["clock_lock_error"] = (msg.strip() or f"exit code {code}")[-200:] + " (re-run scripts/install.sh to install the clock helper)"
    return out


async def noise_controls_off(nc: dict | None) -> None:
    if nc and nc.get("clock_lock_mhz"):
        await lab._run("sudo", "-n", GPU_CLOCKS, "reset", timeout=20)


async def run(run_id: int, llm: httpx.AsyncClient) -> None:
    with db.connect() as c:
        rec = _run(c.execute("SELECT * FROM exp_runs WHERE id = ?", (run_id,)).fetchone())
    from . import methods
    if rec["config"]["model"].startswith(methods.PREFIX):
        await methods.run_profile(rec, llm)
        return
    exp = get_experiment(rec["experiment_id"])
    cfg = rec["config"]
    snapshot = rec.get("protocol_snapshot")
    if snapshot:
        if snapshot["version"] != PROTOCOL:
            _save(run_id, status="error", error="Queued validation uses a different protocol; create a new batch")
            return
        exp["protocol"] = snapshot["tests"]
    lim = exp["limits"]
    vram_total = lim["vram_mb"] or 0
    res: dict = {"config": cfg, "validation": rec.get("validation")}
    log: list[str] = []
    status, error = "done", None
    exp_dir = DIR / str(exp["id"]); exp_dir.mkdir(parents=True, exist_ok=True)
    base_kld = exp_dir / "base.kld"
    is_baseline = not any(r["status"] == "done" and not r["config"]["model"].startswith(methods.PREFIX) for r in exp["runs_list"] if r["id"] != run_id)
    settings = db.get_settings()
    try:
        meter = power.meter_config(settings)
    except ValueError:
        meter = None
    sampler = Sampler(meter)
    proc = None

    def phase(name: str):
        sampler.phase = name
        _save(run_id, phase=name, results=res, log="\n".join(log[-40:]))

    async with lab._lock:
        lab.busy = "experiment"
        _save(run_id, status="running")
        try:
            res["provenance"] = await asyncio.to_thread(provenance, cfg)
            res["weight_summary"] = await asyncio.to_thread(_weights, model_path(cfg["model"]))
            res["hardware"] = await asyncio.to_thread(hardware_snapshot)
            res["provenance"]["extra_tests"] = {"needle": f"WikiText-2 haystacks {NEEDLE_LENGTHS[0]}..ctx tokens × depths {NEEDLE_DEPTHS}, seed {NEEDLE_SEED}",
                                                "lm_eval": (exp.get("protocol") or {}).get("lm_eval"),
                                                "mmlu_questions": 14 * ((exp.get("protocol") or {}).get("mmlu_per_category") or MMLU_PER_CATEGORY),
                                                "ppl_chunks": ppl_chunks(exp),
                                                "fidelity_text": (exp.get("protocol") or {}).get("fidelity_text") or "WikiText-2 test",
                                                "ruler": (exp.get("protocol") or {}).get("ruler")}
            ftext = await fidelity_text(exp)
            res["provenance"]["data"]["fidelity_text"] = {"name": (exp.get("protocol") or {}).get("fidelity_text") or "WikiText-2 test",
                                                          "sha256": file_sha256(ftext)}
            cmds: dict[str, str] = {}
            res["commands"] = cmds
            await lab.llm_service("stop")
            await asyncio.sleep(1)
            res["provenance"]["noise_controls"] = await noise_controls_on(exp, cfg)
            display = await asyncio.to_thread(lab.display_active)
            margin = lab.FIT_MARGIN_MB + (lab.DISPLAY_RESERVE_MB if display else 0)
            res["provenance"]["placement"] = {"fit_margin_mb": margin, "display_active": display}
            placement = {"margin_mb": margin}
            sampler.start()

            # 1) Load: start a server with exactly this configuration, time it, read the memory layout.
            phase("loading")
            ebin = engines.bin_dir(cfg.get("engine"))
            ckpt = engines.is_checkpoint_framework(cfg.get("engine"))
            fw = engines.kind(cfg.get("engine"))
            api = "openai" if ckpt else "llamacpp"
            logfile = exp_dir / f"run{run_id}-server.log"
            if ckpt:
                # vLLM, SGLang and TensorRT-LLM pre-allocate a fraction of each GPU: give them what's actually free (other
                # programs may hold some), or the simulated VRAM limit.
                await asyncio.sleep(1.5)
                used = (sampler.samples[-1].get("vram_mb") if sampler.samples else 0) or 0
                budget = min(vram_total - used - 300, cfg.get("vram_limit_mb") or vram_total)
                if not vram_total or budget <= 0:
                    raise RuntimeError(f"Cannot determine a positive GPU memory budget for {engines.CHECKPOINT_FRAMEWORKS[fw]}")
                util = min(0.92, budget / vram_total)
                cmd, note = checkpoint_server_cmd(fw, ebin, cfg, util)
                if note:
                    res["framework_note"] = note
                env = dict(os.environ)
            else:
                check_rpc_workers(cfg)
                cmd = [str(ebin / "llama-server"), "-m", str(model_path(cfg["model"])), "--host", "127.0.0.1",
                       "--port", str(PORT), "-c", str(cfg["ctx"]), "-np", "1", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"],
                       "--jinja", "--no-webui", "-lv", "4",
                       # Non-thinking protocol by default: answers land in the reply (not a hidden reasoning field), which
                       # outside harnesses like lm-eval grade. Requests can still turn thinking on explicitly.
                       "--chat-template-kwargs", '{"enable_thinking": false}'] + _placement_args(cfg, vram_total, **placement)
                env = {**os.environ, "LD_LIBRARY_PATH": str(ebin)}
            cmd = _pin(cmd, cfg)
            cmds["load_and_quality"] = _cmdline(cmd)
            t0 = time.monotonic()
            with open(logfile, "w") as lf:
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=lf, stderr=asyncio.subprocess.STDOUT, env=env, start_new_session=ckpt)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=httpx.Timeout(10, read=600)) as client:
                ready = asyncio.create_task(lab.wait_ready(client, timeout=1500 if ckpt else 300))
                died = asyncio.create_task(proc.wait())
                done, _ = await asyncio.wait({ready, died}, return_when=asyncio.FIRST_COMPLETED)
                for t in (ready, died):
                    if not t.done():
                        t.cancel()
                if not (ready in done and ready.result()):
                    tail = logfile.read_text(errors="replace").splitlines()[-3:]
                    raise RuntimeError("The model didn't load with this setup (likely out of memory): " + " | ".join(tail)[-300:])
                res["load_seconds"] = round(time.monotonic() - t0, 1)
                res["memory"] = ({"gpu": {}, "ram": {}, "layers": None, "note": f"{engines.CHECKPOINT_FRAMEWORKS[fw]} pre-allocates GPU memory; see VRAM used"}
                                 if ckpt else _parse_load_log(logfile.read_text(errors="replace")))
                await asyncio.sleep(1.5)
                res["memory"]["vram_used_mb"] = (sampler.samples[-1].get("vram_mb") if sampler.samples else None)
                if cfg.get("ram_limit_mb") and not ckpt:
                    need = sum((res["memory"].get("ram") or {}).values()) + OS_RESERVE_MB
                    res["memory"]["ram_needed_mb"] = round(need)
                    if need > cfg["ram_limit_mb"]:
                        raise RuntimeError(f"Doesn't fit a device with {cfg['ram_limit_mb'] / 1024:.0f} GB RAM: the model's CPU-side buffers "
                                           f"plus ~{OS_RESERVE_MB} MB for the system need {need / 1024:.1f} GB")

                # 2) Quality: MMLU-Pro, 5-shot option-letter scoring, the same questions (560 by default) for every run.
                phase("quality")
                def progress(i, n, k):
                    res["mmlu_progress"] = {"done": i, "total": n, "correct": k}
                    _save(run_id, results=res)
                res["mmlu"] = await mmlu_eval(client, MMLU_SEED, lambda: run_id in _cancel, progress, api,
                                              (exp.get("protocol") or {}).get("mmlu_per_category") or MMLU_PER_CATEGORY)
                res.pop("mmlu_progress", None)

                # 2b) Long context: needle in a haystack, up to this run's context size.
                phase("longctx")
                def nprog(i, n, k):
                    res["needle_progress"] = {"done": i, "total": n, "correct": k}
                    _save(run_id, results=res)
                res["needle"] = await needle_eval(client, cfg["ctx"], lambda: run_id in _cancel, nprog, api)
                res.pop("needle_progress", None)

                # 2b'') Optional: RULER-style long-context suite (harder than one needle).
                rp = (exp.get("protocol") or {}).get("ruler")
                if rp:
                    phase("ruler")
                    def rprog(i, n):
                        res["ruler_progress"] = {"done": i, "total": n}
                        _save(run_id, results=res)
                    res["ruler"] = await ruler.run(client, (await quantlab.ensure_text()).read_text(), await chars_per_token(client, api), cfg["ctx"],
                                                   rp["samples"], lambda: run_id in _cancel, rprog)
                    res.pop("ruler_progress", None)

                # 2b') Serving speed: the same client-side measurement for every engine (framework comparisons).
                phase("serving")
                res["serving"] = await serving_bench(client, api, lambda: run_id in _cancel)

                # 2b'') Energy: idle power with the model loaded, then power during sustained generation.
                phase("idle")
                t_idle = time.monotonic()
                await asyncio.sleep(ENERGY_IDLE_S)
                idle = [x for x in sampler.samples if x["phase"] == "idle" and x["t"] >= t_idle + 2]  # let boost clocks settle
                phase("energy")
                tokens, t1, t2 = await sustained_generation(client, (res["serving"] or {}).get("tg"), cfg["ctx"], lambda: run_id in _cancel)
                load = [x for x in sampler.samples if x["phase"] == "energy" and t1 <= x["t"] <= t2]
                res["energy"] = power.energy_summary(idle, load, tokens, t2 - t1)
                res["energy"]["method"] = "Sampled mean power over the full request, including prefill, divided by server-reported output tokens"
                res["energy"]["meter"] = bool(meter)

                # 2c) Optional: lm-evaluation-harness tasks chosen for this experiment, against this same server.
                lm = (exp.get("protocol") or {}).get("lm_eval")
                if lm and lm.get("tasks"):
                    phase("lm_eval")
                    res["lm_eval"] = await lm_eval_run(lm, exp_dir / f"run{run_id}-lm-eval", log, run_id)
            await _stop_server(proc)
            proc = None
            await asyncio.sleep(1)

            if ckpt:
                # llama-bench, Nsight kernel profiles and the llama-perplexity fidelity tests are llama.cpp tools.
                res["fidelity_note"] = (f"Perplexity and KL divergence come from llama.cpp's tools, so they aren't measured for "
                                        f"{engines.CHECKPOINT_FRAMEWORKS[fw]} runs; compare with the serving speed, MMLU-Pro, long-context and lm-eval results.")
                raise _SkipRest

            # The other tools reuse the layer split the server's fit chose, so every phase measures the same layout.
            # Mixture-of-experts models are fitted by moving experts, which a layer count can't reproduce: those keep
            # fitting with the same margin.
            layers = (res.get("memory") or {}).get("layers")
            moe = bool(gguf.metadata(model_path(cfg["model"])).get("experts"))
            if layers and not moe and str(cfg["ngl"]) == "auto" and cfg.get("device") != "cpu":
                placement["gpu_layers"] = layers["gpu"]
            res["provenance"]["placement"]["tool_gpu_layers"] = placement.get("gpu_layers")

            # 3) Speed: llama-bench with the same setup.
            phase("speed")
            res["speed"] = await _bench(cfg, vram_total, run_id, log, cmds, **placement)

            # 3b) Optional: GPU kernel profile (Nsight Systems), if this experiment asked for it.
            if (exp.get("protocol") or {}).get("kernels") and cfg.get("device") != "cpu":
                phase("kernels")
                kargs = ["-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"]] + _placement_args(cfg, vram_total, bench=True, **placement)
                res["kernels"] = await kernels.profile(ebin / "llama-bench", model_path(cfg["model"]), kargs, exp_dir / f"run{run_id}-kernels", log, run_id)

            # 4) Fidelity: perplexity, and KL divergence vs the experiment's baseline.
            phase("fidelity")
            text = await fidelity_text(exp)
            pcmd = [str(ebin / "llama-perplexity"), "-m", str(lab.MODELS_DIR / cfg["model"]), "-f", str(text), "-c", "512",
                    "--chunks", str(ppl_chunks(exp)), "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"]] + _placement_args(cfg, vram_total, **placement)
            pcmd = _pin(pcmd, cfg)
            base_run = next((r for r in exp["runs_list"] if r["status"] == "done" and r["id"] != run_id and not r["config"]["model"].startswith(methods.PREFIX)), None)
            same_tok = is_baseline or not base_run or (
                gguf.metadata(lab.MODELS_DIR / cfg["model"]).get("tokenizer") == gguf.metadata(lab.MODELS_DIR / base_run["config"]["model"]).get("tokenizer"))
            if not same_tok:
                # A different tokenizer splits text into different tokens: per-token KL divergence is undefined and
                # perplexity isn't comparable. Measure this model's own perplexity, and say so.
                cmds["fidelity"] = _cmdline(pcmd)
                out = await quantlab._proc(pcmd, log, run_id)
                res["ppl"] = quantlab._num(r"Final estimate: PPL = ([\d.]+)", out)
                res["ppl_sd"] = quantlab._num(r"Final estimate: PPL = [\d.]+\s*\+/-\s*([\d.]+)", out)
                res["kld"] = res["same_top"] = None
                res["token_note"] = "different tokenizer than the baseline: KL divergence and perplexity can't be compared"
            elif is_baseline:
                cmds["fidelity"] = _cmdline(pcmd + ["--kl-divergence-base", str(base_kld)])
                out = await quantlab._proc(pcmd + ["--kl-divergence-base", str(base_kld)], log, run_id)
                res["ppl"] = quantlab._num(r"Final estimate: PPL = ([\d.]+)", out)
                res["ppl_sd"] = quantlab._num(r"Final estimate: PPL = [\d.]+\s*\+/-\s*([\d.]+)", out)
                if res["ppl"] is None or not base_kld.is_file():
                    raise RuntimeError("The evaluator did not produce baseline perplexity and logits")
                base_kld.with_suffix(".json").write_text(json.dumps({"run_id": run_id, "protocol": PROTOCOL, "chunks": ppl_chunks(exp),
                                                                     "text_sha256": file_sha256(text)}))
                res["kld"], res["same_top"] = 0.0, 100.0
                res["kld_reference"] = True
            else:
                _check_kld_reference(base_kld, base_run, ppl_chunks(exp), file_sha256(text))
                cmds["fidelity"] = _cmdline(pcmd + ["--kl-divergence-base", str(base_kld), "--kl-divergence"])
                out = await quantlab._proc(pcmd + ["--kl-divergence-base", str(base_kld), "--kl-divergence"], log, run_id)
                res["ppl"] = quantlab._num(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", out)
                res["ppl_sd"] = quantlab._num(r"Mean PPL\(Q\)\s*:\s*[\d.]+\s*±\s*([\d.]+)", out)
                res["kld"] = quantlab._num(r"Mean\s+KLD:\s*(-?[\d.]+(?:e-?\d+)?)", out)
                res["kld_sd"] = quantlab._num(r"Mean\s+KLD:\s*[\d.]+\s*±\s*([\d.]+)", out)
                res["kld_p99"] = quantlab._num(r"99\.0%\s+KLD:\s*([\d.]+)", out)
                res["same_top"] = quantlab._num(r"Same top p:\s*([\d.]+)", out)
                res["same_top_sd"] = quantlab._num(r"Same top p:\s*[\d.]+\s*±\s*([\d.]+)", out)

            _retain_output(run_id, "fidelity", out)
            raise _SkipRest
        except _SkipRest:
            # 5) Telemetry + bottleneck analysis (every engine ends here).
            res["telemetry"] = {p: sampler.summary(p) for p in ("loading", "quality", "longctx", "ruler", "serving", "idle", "energy", "lm_eval", "speed", "kernels", "fidelity")}
            res["cost"] = power.costs(res.get("energy"), (res.get("speed") or res.get("serving") or {}).get("tg"),
                                      {k: settings.get(k) for k in ("device_price", "electricity_price_kwh", "currency")})
            res["series"] = sampler.series()
            res["hw"] = analyze(res, cfg, res.get("hardware") or lim)
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001
            status, error = "error", str(e)[:400]
        finally:
            await sampler.stop()
            await noise_controls_off(res.get("provenance", {}).get("noise_controls"))
            _retain_output(run_id, "log", "\n".join(log))
            await _stop_server(proc)
            if res.get("series") is None and sampler.samples:
                res["series"] = sampler.series()
            _save(run_id, status=status, error=error, results=res, phase=None, log="\n".join(log[-60:]))
            _cancel.discard(run_id)
            with db.connect() as c:
                more = c.execute("SELECT 1 FROM exp_runs WHERE status = 'queued' LIMIT 1").fetchone()
            if not more:  # bring chat back only after the last queued profile
                await _restore_chat(llm)
            else:
                global _chat_down
                _chat_down = True
            lab.busy = None


# ---------- Hardware analysis (roofline) ----------

def _weights(path: Path) -> dict:
    """Bytes and parameters touched per generated token (embedding lookup excluded; MoE: active experts only)."""
    if path.is_dir():  # a Hugging Face checkpoint (vLLM): 16-bit weights, so parameters ≈ bytes / 2
        b = checkpoint_bytes(path)
        return {"bytes": b, "params": b / 2, "embedding_bytes": 0}
    try:
        an = gguf.anatomy(path)
    except (OSError, ValueError):
        size = path.stat().st_size
        return {"bytes": size, "params": None, "embedding_bytes": 0}
    b, p = an["total_bytes"] - an["embedding"]["bytes"], an["total_params"] - an["embedding"]["params"]
    if an.get("experts") and an.get("experts_used"):
        keep = an["experts_used"] / an["experts"]
        eb = sum(L["parts"].get("experts", {}).get("bytes", 0) for L in an["layers"])
        ep = sum(L["parts"].get("experts", {}).get("params", 0) for L in an["layers"])
        b, p = b - eb * (1 - keep), p - ep * (1 - keep)
    return {"bytes": b, "params": p, "embedding_bytes": an["embedding"]["bytes"]}


def analyze(res: dict, cfg: dict, lim: dict) -> dict:
    """Measured numbers against the hardware's physical limits. Pure function of stored data, so it can be recomputed."""
    w = res.get("weight_summary")
    if not w:
        path = model_path(cfg["model"])
        if not (path.is_file() or (path / "config.json").is_file()):
            return {}
        w = _weights(path)
    mem = res.get("memory") or {}
    # llama-bench numbers when there are any; otherwise (vLLM) the engine-neutral serving measurement.
    serving = res.get("serving") or {}
    # Full-request throughput includes prefill and cannot describe isolated decode/compute utilization.
    speed = res.get("speed") or (serving if serving.get("timing_scope") != "request" else {})
    tel = (res.get("telemetry") or {}).get("speed") or (res.get("telemetry") or {}).get("serving") or {}
    tg, pp = speed.get("tg"), speed.get("pp")
    bw, ram_bw, peak_tf = lim.get("gpu_bandwidth_gbps"), lim.get("ram_bandwidth_gbps"), lim.get("gpu_tflops")
    par = parallel_gpus(cfg, lim)
    if par > 1:  # identical GPUs assumed (spec of the first one); tensor-parallel GPUs read their weight shards at the same time
        bw, peak_tf = (bw * par if bw else bw), (peak_tf * par if peak_tf else peak_tf)
    distributed = bool(cfg.get("rpc")) or int(cfg.get("pp") or 1) > 1
    if distributed:  # part of the model runs on other machines: this machine's limits don't bound the result
        bw = ram_bw = peak_tf = None

    gb = w["bytes"] / 1e9
    # Which weights live where, from the load log: host "model" buffers minus the embedding table (always on CPU, only looked up).
    cpu_gb = max(0.0, ((mem.get("ram") or {}).get("model", 0) * 2**20 - w["embedding_bytes"]) / 1e9) if mem else 0.0
    cpu_gb = gb if cfg.get("device") == "cpu" else min(cpu_gb, gb)
    gpu_gb = gb - cpu_gb
    flops = 2 * w["params"] if w["params"] else None

    out: dict = {"weights_gb_per_token": round(gb, 3), "weights_on_gpu_gb": round(gpu_gb, 3), "weights_on_cpu_gb": round(cpu_gb, 3),
                 "params_active_b": round(w["params"] / 1e9, 3) if w["params"] else None,
                 "peaks": {"vram_gbps": bw, "ram_gbps": ram_bw, "tflops": peak_tf, "tflops_kind": lim.get("gpu_tflops_kind"),
                           "sm_clock_mhz": lim.get("sm_clock_max"), "gpus": gpus_used(cfg, lim), "parallel_gpus": par}}
    if distributed:
        out["distributed_note"] = ("Part of the model runs on other machines (RPC workers or pipeline stages): the hardware limits of this "
                                   "machine don't apply, and network transfers are part of every token's time")

    if tg:
        t_gpu = gpu_gb / bw * 1000 if bw else (0.0 if not gpu_gb else None)
        t_cpu = cpu_gb / ram_bw * 1000 if ram_bw else (0.0 if not cpu_gb else None)
        t_meas = 1000 / tg
        roof_ms = t_gpu + t_cpu if t_gpu is not None and t_cpu is not None else None
        out["decode"] = {
            "ms_per_token": round(t_meas, 2),
            "effective_gbps": round(gb * tg, 1),
            "roof_tps": round(1000 / roof_ms, 1) if roof_ms else None,
            "roof_util": round(roof_ms / t_meas, 3) if roof_ms else None,
            "time_split_ms": {"vram_reads": round(t_gpu, 2) if t_gpu is not None else None, "ram_reads": round(t_cpu, 2) if t_cpu is not None else None,
                              "other": round(max(0.0, t_meas - (roof_ms or 0)), 2) if roof_ms else None},
            "achieved_tflops": round(flops * tg / 1e12, 3) if flops else None,
            "arithmetic_intensity": round(flops / w["bytes"], 2) if flops else None,
            "long_context_ratio": round(speed["tg_d4096"] / tg, 3) if speed.get("tg_d4096") else None,
        }
    cpu_only = cfg.get("device") == "cpu"
    if pp and flops:
        ai = flops * 512 / w["bytes"]  # one 512-token batch reads the weights once
        roof = min(peak_tf, (bw or 0) * ai / 1000) if (peak_tf and bw and not cpu_only) else None  # no CPU FLOPS table
        ach = flops * pp / 1e12
        out["prefill"] = {"achieved_tflops": round(ach, 2), "arithmetic_intensity": round(ai, 1),
                          "roof_tflops": round(roof, 2) if roof else None, "roof_util": round(ach / roof, 3) if roof else None,
                          "long_context_ratio": round(speed["pp_d4096"] / pp, 3) if speed.get("pp_d4096") else None}

    sm, smax = tel.get("sm_clock_avg"), lim.get("sm_clock_max")
    out["clocks"] = {"sm_avg_mhz": sm, "sm_max_mhz": smax, "ratio": round(sm / smax, 3) if (sm and smax) else None,
                     "temp_max_c": tel.get("temp_max"), "gpu_util_avg": tel.get("gpu_avg"), "mem_util_avg": tel.get("mem_bw_avg"),
                     "cpu_util_avg": tel.get("cpu_avg"), "power_avg_w": tel.get("power_avg"), "cpu_power_avg_w": tel.get("cpu_power_avg")}

    preset = DEVICE_PRESETS.get(cfg.get("preset") or "")
    if preset and preset.get("ram_bandwidth_gbps") and cpu_only and tg and ram_bw:
        # Decode reads every weight once per token: at the same share of bandwidth this CPU achieved, the target device
        # would run at (share x its bandwidth) / weight bytes. Prefill is compute-bound and can't be projected this way.
        share = min(1.0, gb * tg / ram_bw)
        out["projection"] = {"target": preset["label"], "target_ram_gbps": preset["ram_bandwidth_gbps"], "memory": preset.get("memory"),
                             "bandwidth_share": round(share, 3), "tg": round(share * preset["ram_bandwidth_gbps"] / gb, 2) if gb else None,
                             "method": "decode tok/s = (share of RAM bandwidth achieved here) x target spec-sheet bandwidth / weight GB per token"}
    elif preset and preset.get("ram_bandwidth_gbps") and cpu_only:
        # Say why instead of leaving the target-device estimate out silently.
        out["projection"] = {"target": preset["label"], "target_ram_gbps": preset["ram_bandwidth_gbps"], "tg": None,
                             "unavailable": "this machine's RAM bandwidth couldn't be measured (needs numpy: sudo apt install python3-numpy)"
                             if not ram_bw else "no decode speed was measured"}

    vram_used = mem.get("vram_used_mb")
    ctx_mb = (mem.get("gpu") or {}).get("context") or 0
    own = sum((mem.get("gpu") or {}).values()) if mem else 0  # this model's own GPU buffers, from the load log
    if lim.get("vram_mb") and (own or vram_used):
        sim = cfg.get("vram_limit_mb")
        cap = sim or lim["vram_mb"]
        # nvidia-smi's "used" includes other programs (e.g. a desktop session); with a simulated limit only our buffers count.
        free = (cap - own) if sim else (cap - (vram_used or own))
        per_tok = ctx_mb / cfg["ctx"] if (ctx_mb and cfg.get("ctx")) else None
        out["headroom"] = {"vram_capacity_mb": cap, "vram_model_mb": round(own), "vram_used_total_mb": vram_used,
                           "vram_free_mb": round(free), "kv_mb_per_1k_tokens": round(per_tok * 1024, 2) if per_tok else None,
                           "extra_context_tokens": int(free * 0.9 / per_tok) if (per_tok and free > 0 and cpu_gb < 0.05) else None}

    # The binding constraint for generation, decided from the numbers above.
    d = out.get("decode") or {}
    split = d.get("time_split_ms") or {}
    if d.get("roof_util") is not None and d["roof_util"] > 1:
        out["limiter"] = {"what": "Roofline estimate exceeded", "evidence": "Measured throughput exceeds this simplified weight-read model; its assumptions do not explain this run"}
    elif cpu_gb > 0.05 and split.get("ram_reads"):
        out["limiter"] = {"what": "Possible RAM bandwidth limit",
                          "evidence": f"The model estimates at least {split['ram_reads']:.1f} ms of RAM reads per token; total measured time is {d['ms_per_token']:.1f} ms. This is not a measured time breakdown"}
    elif d.get("roof_util") is not None and d["roof_util"] >= 0.6:
        out["limiter"] = {"what": "Possible VRAM bandwidth limit", "evidence": f"Estimated weight traffic is {d['effective_gbps']} GB/s against {bw} GB/s nominal peak; a kernel profile is needed to confirm the cause"}
    elif d.get("roof_util") is not None:
        out["limiter"] = {"what": "Bottleneck undetermined",
                          "evidence": f"{d['roof_util'] * 100:.0f}% of the estimated memory-bandwidth roof. Low utilization alone does not identify the bottleneck"}
    return out


def compare(base: dict | None, run: dict, margin_pts: float | None = None) -> dict:
    """Paired statistics of a run against the experiment's baseline."""
    if not base or base["id"] == run["id"]:
        return {}
    a, b = (base["results"].get("mmlu") or {}), (run["results"].get("mmlu") or {})
    out = {}
    pa, pb = base["results"].get("provenance"), run["results"].get("provenance")
    if pa and pb:
        out["env_mismatch"] = [name for name, path in PROV_KEYS if _get(pa, path) != _get(pb, path)]
        if (base["config"].get("engine") or "stock") != (run["config"].get("engine") or "stock"):
            out["env_mismatch"] = [x for x in out["env_mismatch"] if x != "llama.cpp commit"]  # the engine is the variable here
        if _get(pa, ("model", "file")) == _get(pb, ("model", "file")) and _get(pa, ("model", "sha256")) != _get(pb, ("model", "sha256")):
            out["env_mismatch"].append("model file changed on disk")
        # Different hardware is information, not a problem: it's what you compare when testing devices.
        out["hw_diff"] = [name for name, path in HW_KEYS
                          if _get(pa, path) is not None and _get(pb, path) is not None and _get(pa, path) != _get(pb, path)]
    elif pa or pb:
        out["env_mismatch"] = ["provenance missing"]
    if pa and pb:
        for name, path in (("MMLU-Pro seed", ("data", "mmlu_seed")), ("MMLU-Pro question count", ("extra_tests", "mmlu_questions"))):
            if _get(pa, path) != _get(pb, path):
                out["env_mismatch"].append(name)
        # Runs from before the setting existed used the default sample.
        if (_get(pa, ("extra_tests", "ppl_chunks")) or PPL_CHUNKS) != (_get(pb, ("extra_tests", "ppl_chunks")) or PPL_CHUNKS):
            out["env_mismatch"].append("perplexity/KLD sample size")
        if _get(pa, ("data", "fidelity_text", "sha256")) != _get(pb, ("data", "fidelity_text", "sha256")):
            out["env_mismatch"].append("perplexity/KLD text")
        if _get(pa, ("noise_controls", "clock_lock_mhz")) != _get(pb, ("noise_controls", "clock_lock_mhz")):
            out["env_mismatch"].append("GPU clock lock")
        if _get(pa, ("placement", "fit_margin_mb")) != _get(pb, ("placement", "fit_margin_mb")):
            out["env_mismatch"].append("GPU memory margin")
    if a.get("answers") and b.get("answers") and not out.get("env_mismatch"):
        out["mmlu_p"] = mcnemar_p(a["answers"], b["answers"])
        out["mmlu_diff"] = paired_diff_ci(a["answers"], b["answers"])
        out["mmlu_equiv"] = equivalence(a["answers"], b["answers"], margin_pts)
        out["mmlu_flips"] = {"lost": sum(1 for x, y in zip(a["answers"], b["answers"]) if x == "1" and y == "0"),
                             "gained": sum(1 for x, y in zip(a["answers"], b["answers"]) if x == "0" and y == "1")}
    return out


def enrich(exp: dict) -> dict:
    """Recompute analysis and statistics for every run, so old runs benefit from fixes too."""
    lim = exp["limits"]
    base = next((r for r in exp["runs_list"] if r["status"] == "done"), None)
    by_id = {r["id"]: r for r in exp["runs_list"]}
    # A preregistered margin wins over one set later in the test settings: it was fixed before the data existed.
    prereg = next((p["plan"] for p in reversed(exp.get("preregistrations") or []) if p["plan"].get("equivalence_margin_pts")), None)
    margin = (prereg or {}).get("equivalence_margin_pts") or (exp.get("protocol") or {}).get("equivalence_margin_pts")
    for r in exp["runs_list"]:
        if r["config"].get("engine") != "research" and any(x["config"].get("engine") == "research" for x in exp["runs_list"]):
            from . import research_jobs
            native = research_jobs.native_metrics(r["results"])
            if native:
                r["results"]["research_metrics"] = {e["name"]: e for e in native}
        # Runs made before the tree existed (no parent) were all derived from the baseline.
        if r.get("parent_id") is None and base and r["id"] != base["id"]:
            r["parent_id"] = base["id"]
        if r["status"] == "done":
            # Each run is judged against the hardware it actually ran on (its snapshot), not today's machine.
            if not r["results"].get("imported_from") and not r["config"].get("engine") == "research":
                analysis = analyze(r["results"], r["config"], r["results"].get("hardware") or lim)
                if analysis:
                    r["results"]["hw"] = analysis
            r["results"].pop("bottleneck", None)
            r["stats"] = compare(base, r, margin)
            if base:
                r["stats"]["baseline_id"] = base["id"]
            parent = by_id.get(r.get("parent_id"))
            if parent and base and parent["id"] != base["id"] and parent["status"] == "done":
                vp = compare(parent, r, margin)
                r["stats"]["vs_parent"] = {k: vp[k] for k in ("mmlu_p", "mmlu_diff", "mmlu_equiv", "mmlu_flips", "env_mismatch") if k in vp}
    # Many runs vs one baseline: Holm-adjusted p-values (family-wise error rate) next to the raw ones.
    tested = [r for r in exp["runs_list"] if (r.get("stats") or {}).get("mmlu_p") is not None]
    for r, p in zip(tested, holm([r["stats"]["mmlu_p"] for r in tested])):
        r["stats"]["mmlu_p_holm"] = p
    return exp


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        out[i] = round(run, 6)
    return out


# ---------- Export ----------

def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, list) or k == "answers":
            continue
        else:
            out[key] = v
    return out


# ---------- Moving results between Blocky devices ----------

def import_experiment(data: dict, into: int | None = None) -> dict:
    """Bring an experiment exported (JSON) from another Blocky device in: as a new experiment, or merged into `into`.

    Imported runs keep their own results, provenance and hardware snapshot, so they're judged against the hardware
    they ran on. They're read-only here (their model files may not exist on this device). KL divergence of imported
    runs was measured against their own experiment's baseline.
    """
    runs = [r for r in (data.get("runs_list") or []) if r.get("status") == "done" and isinstance(r.get("results"), dict)]
    if not runs:
        raise ValueError("No finished runs in that file. Export an experiment with Export → JSON on the other device.")
    src_limits = data.get("limits") or {}
    src_device = next((r["results"].get("hardware", {}).get("device") or (r["results"].get("provenance") or {}).get("host", {}).get("device")
                       for r in runs if r["results"].get("hardware") or r["results"].get("provenance")), None)
    src_gpu = (src_limits.get("gpu") or "").replace("NVIDIA GeForce ", "") or "other device"
    origin = {"device": src_device, "gpu": src_gpu, "experiment": data.get("name"), "imported_at": time.time()}
    with db.connect() as c:
        if into is None:
            exp_id = c.execute("INSERT INTO experiments (created_at, name, model, seed) VALUES (?, ?, ?, ?)",
                               (time.time(), f"{data.get('name') or 'Imported'} (from {src_device or src_gpu})"[:80], runs[0]["config"]["model"],
                                int(data.get("seed") or 0))).lastrowid
            local_root = None
            c.execute("UPDATE experiments SET protocol = ? WHERE id = ?", (json.dumps(data.get("protocol") or {}), exp_id))
        else:
            exp_id = into
            r0 = c.execute("SELECT id FROM exp_runs WHERE experiment_id = ? AND status = 'done' ORDER BY id LIMIT 1", (into,)).fetchone()
            if not c.execute("SELECT 1 FROM experiments WHERE id = ?", (into,)).fetchone():
                raise ValueError("Experiment not found")
            local_root = r0["id"] if r0 else None
        id_map: dict[int, int] = {}
        for r in sorted(runs, key=lambda x: x["id"]):
            res = dict(r["results"])
            res.setdefault("hardware", src_limits)  # exports from before per-run snapshots: use the export's hardware
            res["imported_from"] = origin
            # Job IDs belong to the source device; never link imported evidence to unrelated local files.
            if res.get("research_job_id"):
                res["source_research_job_id"] = res.pop("research_job_id")
            parent = id_map.get(r.get("parent_id")) if r.get("parent_id") is not None else None
            if parent is None and r.get("parent_id") is None and local_root is not None:
                parent = local_root  # the other device's baseline hangs under ours when merging
            label = r["label"] if into is None else f"{r['label']} @ {src_device or src_gpu}"[:60]
            new_id = c.execute("INSERT INTO exp_runs (experiment_id, created_at, status, label, config, results, parent_id) VALUES (?, ?, 'done', ?, ?, ?, ?)",
                               (exp_id, r.get("created_at") or time.time(), label, json.dumps(r["config"]), json.dumps(res), parent)).lastrowid
            id_map[r["id"]] = new_id
    return {"experiment_id": exp_id, "runs": len(id_map), "from": origin}


def export_csv(exp: dict) -> str:
    import csv
    import io
    rows = []
    for r in exp["runs_list"]:
        res = {k: v for k, v in r["results"].items() if k not in ("series", "config")}
        rows.append({"run_id": r["id"], "label": r["label"], "status": r["status"],
                     "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"])),
                     **_flat(r["config"], "config."), **_flat(res), **_flat(r.get("stats") or {}, "vs_baseline.")})
    cols: list[str] = []
    for row in rows:
        cols += [c for c in row if c not in cols]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    buf.write("\n" + "\n".join(f"# hardware.{k},{v}" for k, v in _flat(exp["limits"]).items()) + "\n")
    return buf.getvalue()
