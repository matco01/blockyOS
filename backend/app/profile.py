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
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import psutil

from . import arena, config, db, engines, evals, gguf, lab, quantlab

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
    DIR.mkdir(parents=True, exist_ok=True)
    with db.connect() as c:
        c.executescript(SCHEMA)
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
import numpy as np, time, os
from multiprocessing import Pool
def work(_):
    a = np.ones(32 * 2**20, dtype=np.float64); best = 0
    for _ in range(6):
        t = time.perf_counter(); a.sum(); dt = time.perf_counter() - t
        best = max(best, a.nbytes / dt / 1e9)
    return best
n = max(1, (os.cpu_count() or 2) // 2)
with Pool(n) as p:
    print(sum(p.map(work, range(n))))
"""


def ram_bandwidth() -> float | None:
    """Measured multi-core system-RAM read bandwidth (GB/s), cached: the ceiling for layers on the CPU."""
    global _ram_bw
    cache = DIR / "ram_read_bandwidth.json"
    if _ram_bw is None and cache.is_file():
        _ram_bw = json.loads(cache.read_text()).get("gbps")
    if _ram_bw is None:
        try:
            out = subprocess.run(["python3", "-c", _RAM_BW_CODE], capture_output=True, text=True, timeout=120).stdout.strip()
            _ram_bw = round(float(out), 1)
            cache.write_text(json.dumps({"gbps": _ram_bw, "measured_at": time.time(), "method": "numpy sum, one process per 2 threads"}))
        except (ValueError, OSError, subprocess.SubprocessError):
            return None
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
            "vram_mb": g["vram_mb"] if g else 0, "power_limit_w": g.get("power_limit_w") if g else None,
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
    return d


def list_experiments() -> list[dict]:
    with db.connect() as c:
        exps = [_exp(r) for r in c.execute("SELECT * FROM experiments ORDER BY id DESC")]
        for e in exps:
            e["runs"] = c.execute("SELECT COUNT(*) FROM exp_runs WHERE experiment_id = ?", (e["id"],)).fetchone()[0]
    return exps


def get_experiment(exp_id: int) -> dict | None:
    with db.connect() as c:
        e = c.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        if not e:
            return None
        runs = [_run(r) for r in c.execute("SELECT * FROM exp_runs WHERE experiment_id = ? ORDER BY id", (exp_id,))]
    return enrich({**_exp(e), "runs_list": runs, "limits": limits()})


def create_experiment(name: str, model: str) -> int:
    import random
    model = Path(model).name
    if not (lab.MODELS_DIR / model).is_file():
        raise ValueError("Model not found")
    with db.connect() as c:
        return c.execute("INSERT INTO experiments (created_at, name, model, seed) VALUES (?, ?, ?, ?)",
                         (time.time(), (name or model.replace(".gguf", ""))[:80], model, random.randint(1, 2**31 - 1))).lastrowid


def delete_experiment(exp_id: int) -> None:
    with db.connect() as c:
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
    with db.connect() as c:
        queued = c.execute("UPDATE exp_runs SET status = 'cancelled' WHERE id = ? AND status = 'queued'", (run_id,)).rowcount
    if not queued:
        _cancel.add(run_id)


def cancel_queue(exp_id: int) -> int:
    with db.connect() as c:
        return c.execute("UPDATE exp_runs SET status = 'cancelled' WHERE experiment_id = ? AND status = 'queued'", (exp_id,)).rowcount


SWEEP_KEYS = ("model", "engine", "kv", "ctx", "vram_limit_mb", "ngl", "fa", "threads")
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
    base = parent["config"] if parent else (done[0]["config"] if done else normalize_config(exp, {}))
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    if len(combos) * repeats > MAX_SWEEP:
        raise ValueError(f"That's {len(combos) * repeats} runs; the limit is {MAX_SWEEP} per sweep")
    ids = []
    for rep in range(repeats):
        for combo in combos:
            cfg = normalize_config(exp, {**base, **combo})
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


DEFAULTS = {"ctx": 8192, "kv": "q8_0", "ngl": "auto", "vram_limit_mb": None, "fa": "auto", "threads": None, "engine": "stock"}


def normalize_config(exp: dict, cfg: dict) -> dict:
    out = {**DEFAULTS, "model": exp["model"], **{k: v for k, v in (cfg or {}).items() if v not in (None, "")}}
    out["model"] = Path(str(out["model"])).name
    if not (lab.MODELS_DIR / out["model"]).is_file():
        raise ValueError("Model not found")
    out["ctx"] = max(1024, min(262144, int(out["ctx"])))
    if out["kv"] not in lab.KV_TYPES:
        raise ValueError("KV cache must be f16, q8_0 or q4_0")
    if str(out["ngl"]) != "auto":
        out["ngl"] = max(0, min(999, int(out["ngl"])))
    if out["vram_limit_mb"]:
        out["vram_limit_mb"] = max(512, int(out["vram_limit_mb"]))
    if out["fa"] not in ("auto", "on", "off"):
        raise ValueError("Flash attention must be auto, on or off")
    if out["threads"]:
        out["threads"] = max(1, min(256, int(out["threads"])))
    out["engine"] = str(out.get("engine") or "stock")
    engines.bin_dir(out["engine"])  # raises if unknown or not built
    return out


def create_run(exp_id: int, body: dict) -> int:
    exp = get_experiment(exp_id)
    if not exp:
        raise ValueError("Experiment not found")
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
        a, b = (gguf.metadata(lab.MODELS_DIR / m) for m in (base.get("model") or cfg["model"], cfg["model"]))
        # Same model, other weights -> "Q3_K_M-imat"; a different model altogether -> its full name.
        parts.append(weights_label(cfg["model"]) if a.get("name") == b.get("name") else cfg["model"].removesuffix(".gguf"))
    if cfg["kv"] != base.get("kv"):
        parts.append(f"KV {cfg['kv']}")
    if cfg["ctx"] != base.get("ctx"):
        parts.append(f"ctx {cfg['ctx'] // 1024}k")
    if str(cfg["ngl"]) != str(base.get("ngl")):
        parts.append(f"{cfg['ngl']} GPU layers")
    if cfg.get("vram_limit_mb") != base.get("vram_limit_mb"):
        parts.append(f"{(cfg['vram_limit_mb'] or 0) / 1024:.0f} GB GPU" if cfg.get("vram_limit_mb") else "full GPU")
    if cfg["fa"] != base.get("fa"):
        parts.append(f"flash-attn {cfg['fa']}")
    if (cfg.get("engine") or "stock") != (base.get("engine") or "stock"):
        parts.append(f"engine {cfg.get('engine') or 'stock'}")
    if cfg.get("threads") != base.get("threads"):
        parts.append(f"{cfg['threads'] or 'auto'} threads")
    return " · ".join(parts) or "Repeat of baseline"


# ---------- Running a profile ----------

def _placement_args(cfg: dict, vram_total: float, bench: bool = False) -> list[str]:
    args = []
    if str(cfg["ngl"]) != "auto":
        args += ["-ngl", str(cfg["ngl"])]
    if cfg.get("vram_limit_mb") and vram_total:
        args += ["-fitt" if bench else "--fit-target", str(max(0, int(vram_total - cfg["vram_limit_mb"])))]
    elif str(cfg["ngl"]) == "auto":
        args += ["-fitt", "512"] if bench else ["--fit", "on"]
    if cfg.get("threads"):
        args += ["-t", str(cfg["threads"])]
    return args


def _rapl_file() -> Path | None:
    """CPU package energy counter (bare-metal Linux on Intel and AMD Zen; not available inside WSL)."""
    p = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    try:
        return p if p.is_file() and os.access(p, os.R_OK) else None
    except OSError:
        return None


class Sampler:
    """GPU/CPU/RAM telemetry every 0.5 s, with phase tags so each phase can be summarized."""

    def __init__(self):
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
        while not self._stop.is_set():
            s = {"t": time.monotonic(), "phase": self.phase, "cpu": psutil.cpu_percent(interval=None),
                 "ram_mb": (psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**20}
            if (cp := self._cpu_power()) is not None:
                s["cpu_power"] = cp
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
                "sm_clock_avg": avg("sm_clock"), "cpu_power_avg": avg("cpu_power"), "seconds": round(ss[-1]["t"] - ss[0]["t"], 1),
                "samples": len(ss)}

    def series(self, every: int = 2) -> list[dict]:
        t0 = self.samples[0]["t"] if self.samples else 0
        return [{"t": round(s["t"] - t0, 1), "phase": s["phase"], **{k: s.get(k) for k in ("gpu", "mem_bw", "power", "temp", "vram_mb", "cpu", "sm_clock", "cpu_power")}}
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


def mmlu_sample(seed: int = MMLU_SEED) -> tuple[list[dict], list[dict]]:
    import random
    from . import benchmarks
    if not benchmarks.path("mmlu_pro").is_file():
        raise RuntimeError("MMLU-Pro isn't downloaded yet: get it in Quality tests → Benchmarks (one click).")
    qs = evals.bench_questions("mmlu_pro")
    by_cat: dict[str, list[dict]] = {}
    for q in qs:
        by_cat.setdefault(q.get("tag") or "other", []).append(q)
    rng = random.Random(seed)
    shots, sample = [], []
    for cat in sorted(by_cat):
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        sample += [{**q, "tag": cat} for q in pool[:MMLU_PER_CATEGORY]]
        if len(shots) < 5:
            shots.append(pool[MMLU_PER_CATEGORY])
    return shots, sample


def _mc_block(q: dict, answer: bool) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    return f"Question: {q['q'].strip()}\nOptions:\n{opts}\nAnswer:" + (f" {q['answer']}\n\n" if answer else "")


async def mmlu_eval(client: httpx.AsyncClient, seed: int, should_stop, progress) -> dict:
    import math
    shots, sample = mmlu_sample(seed)
    prefix = "The following are multiple choice questions (with answers).\n\n" + "".join(_mc_block(s, True) for s in shots)
    correct, p_true, cats = [], [], {}
    for i, q in enumerate(sample):
        if should_stop():
            raise asyncio.CancelledError
        r = await client.post("/completion", json={"prompt": prefix + _mc_block(q, False), "n_predict": 1, "n_probs": 50,
                                                   "temperature": 0, "cache_prompt": True})
        r.raise_for_status()
        top = (r.json().get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
        lp = {}
        for t in top:
            tok = t["token"].strip()
            if len(tok) == 1 and tok in LETTERS[:len(q["options"])] and tok not in lp:
                lp[tok] = t["logprob"]
        if lp:
            m = max(lp.values())
            z = sum(math.exp(v - m) for v in lp.values())
            pred = max(lp, key=lp.get)
            pt = math.exp(lp[q["answer"]] - m) / z if q["answer"] in lp else 0.0
        else:
            pred, pt = None, 0.0
        ok = int(pred == q["answer"])
        correct.append(ok); p_true.append(pt)
        c = cats.setdefault(q["tag"], [0, 0]); c[0] += ok; c[1] += 1
        if i % 20 == 19:
            progress(i + 1, len(sample), sum(correct))
    n, k = len(correct), sum(correct)
    return {"benchmark": "MMLU-Pro", "protocol": "5-shot, letter log-likelihood", "n": n, "correct": k, "acc": round(k / n, 4),
            "ci95": [round(x, 4) for x in wilson(k, n)], "p_correct_mean": round(sum(p_true) / n, 4),
            "by_category": {c: {"correct": v[0], "total": v[1]} for c, v in sorted(cats.items())},
            "answers": "".join(str(x) for x in correct), "chance": 0.1}


# ---------- lm-evaluation-harness (optional, per experiment) ----------

ML_VENV = config.DATA_DIR.parent / "ml-venv"
_TASKS = re.compile(r"^[a-z0-9_.-]+(,[a-z0-9_.-]+)*$")


def set_protocol(exp_id: int, body: dict) -> dict:
    """Extra tests every run of this experiment gets. Today: lm-evaluation-harness tasks (generative, via chat)."""
    lm = body.get("lm_eval") or {}
    tasks = re.sub(r"\s+", "", str(lm.get("tasks") or "")).lower()
    proto: dict = {}
    if tasks:
        if not _TASKS.match(tasks):
            raise ValueError("Tasks: comma-separated lm-eval task names, e.g. gsm8k,ifeval")
        proto["lm_eval"] = {"tasks": tasks, "limit": max(1, min(5000, int(lm.get("limit") or 100)))}
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


async def needle_eval(client: httpx.AsyncClient, ctx: int, should_stop, progress) -> dict:
    import random
    from . import generate
    text = (await quantlab.ensure_text()).read_text()
    sample = text[:40000]
    r = await client.post("/tokenize", json={"content": sample})
    r.raise_for_status()
    chars_per_token = len(sample) / max(1, len(r.json().get("tokens", [])))
    lengths = [L for L in NEEDLE_LENGTHS if L + 600 <= ctx]
    rng = random.Random(NEEDLE_SEED)
    cells, total = [], len(lengths) * len(NEEDLE_DEPTHS)
    for L in lengths:
        n_chars = int(L * chars_per_token)
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
    return {"lengths": lengths, "depths": NEEDLE_DEPTHS, "cells": cells, "acc": round(sum(c["ok"] for c in cells) / max(1, len(cells)), 4),
            "by_length": by_len, "prefill_tps_by_length": pf, "max_length": lengths[-1] if lengths else None}


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


# ---------- Provenance: everything needed to reproduce (or question) a run ----------

PROTOCOL = "blocky-profile/2 (MMLU-Pro 560q 5-shot loglik · llama-bench p512 n128 d0,4096 r5 · PPL/KLD WikiText-2 c512 x32)"
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
    ver = _sh(str(eb / "llama-server"), "--version", libdir=eb)
    m = re.search(r"version:\s*(\S+)\s*\((?:build\s*\d+,\s*commit\s*)?([0-9a-f]{6,})\)", ver)
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
        "llama_cpp": {"version": m.group(1) if m else None, "commit": m.group(2) if m else None,
                      "built_with": next((l.strip() for l in ver.splitlines() if l.startswith("built with")), None)},
        "model": {"file": cfg["model"], "sha256": file_sha256(lab.MODELS_DIR / cfg["model"]),
                  "bytes": (lab.MODELS_DIR / cfg["model"]).stat().st_size,
                  "recipe": _recipe(cfg["model"])},
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


async def _bench(cfg: dict, vram_total: float, run_id: int, log: list[str], cmds: dict | None = None) -> dict:
    cmd = [str(engines.bin_dir(cfg.get("engine")) / "llama-bench"), "-m", str(lab.MODELS_DIR / cfg["model"]), "-p", "512", "-n", "128",
           "-d", "0,4096", "-r", "5", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"], "-o", "jsonl"] + _placement_args(cfg, vram_total, bench=True)
    if cmds is not None:
        cmds["speed"] = _cmdline(cmd)
    out = await quantlab._proc(cmd, log, run_id, timeout=900)
    return parse_bench(out)


def robust_stats(raw: list[float]) -> dict:
    """Median and robust spread of repeated measurements.

    Impossible repetitions (0, or a timer glitch far from the rest) are dropped: anything more than 50% away from the
    median. Spread = 1.4826 × median absolute deviation (≈ σ for normal noise, but one outlier can't inflate it).
    """
    pos = sorted(s for s in raw if s and s > 0)
    m0 = pos[len(pos) // 2] if pos else 0
    samples = [s for s in pos if 0.5 * m0 <= s <= 1.5 * m0]
    if not samples:
        return {"median": None, "sd": None, "dropped": len(raw)}
    med = samples[len(samples) // 2]
    mad = sorted(abs(s - med) for s in samples)[len(samples) // 2]
    return {"median": round(med, 2), "sd": round(1.4826 * mad, 2), "dropped": len(raw) - len(samples)}


def parse_bench(out: str) -> dict:
    """llama-bench -o jsonl output -> {pp, tg, pp_d4096, tg_d4096} with *_sd, *_samples and *_dropped."""
    res = {}
    for line in out.splitlines():
        if line.startswith("{"):
            x = json.loads(line)
            key = ("pp" if x.get("n_prompt") and not x.get("n_gen") else "tg") + ("_d4096" if x.get("n_depth") else "")
            raw = x.get("samples_ts") or []
            st = robust_stats(raw)
            if st["dropped"]:
                res[key + "_dropped"] = st["dropped"]
            if st["median"] is not None:
                res[key], res[key + "_sd"] = st["median"], st["sd"]
                res[key + "_samples"] = [round(s, 2) for s in raw]
            else:
                res[key], res[key + "_sd"] = round(x.get("avg_ts") or 0, 2), round(x.get("stddev_ts") or 0, 2)
    return res


async def run(run_id: int, llm: httpx.AsyncClient) -> None:
    with db.connect() as c:
        rec = _run(c.execute("SELECT * FROM exp_runs WHERE id = ?", (run_id,)).fetchone())
    exp = get_experiment(rec["experiment_id"])
    cfg = rec["config"]
    lim = exp["limits"]
    vram_total = lim["vram_mb"] or 0
    res: dict = {"config": cfg}
    log: list[str] = []
    status, error = "done", None
    exp_dir = DIR / str(exp["id"]); exp_dir.mkdir(parents=True, exist_ok=True)
    base_kld = exp_dir / "base.kld"
    is_baseline = not any(r["status"] == "done" for r in exp["runs_list"] if r["id"] != run_id)
    sampler = Sampler()
    proc = None

    def phase(name: str):
        sampler.phase = name
        _save(run_id, phase=name, results=res, log="\n".join(log[-40:]))

    async with lab._lock:
        lab.busy = "experiment"
        _save(run_id, status="running")
        try:
            res["provenance"] = await asyncio.to_thread(provenance, cfg)
            res["hardware"] = await asyncio.to_thread(hardware_snapshot)
            res["provenance"]["extra_tests"] = {"needle": f"WikiText-2 haystacks {NEEDLE_LENGTHS[0]}..ctx tokens × depths {NEEDLE_DEPTHS}, seed {NEEDLE_SEED}",
                                                "lm_eval": (exp.get("protocol") or {}).get("lm_eval")}
            cmds: dict[str, str] = {}
            res["commands"] = cmds
            await lab.llm_service("stop")
            await asyncio.sleep(1)
            sampler.start()

            # 1) Load: start a server with exactly this configuration, time it, read the memory layout.
            phase("loading")
            ebin = engines.bin_dir(cfg.get("engine"))
            logfile = exp_dir / f"run{run_id}-server.log"
            cmd = [str(ebin / "llama-server"), "-m", str(lab.MODELS_DIR / cfg["model"]), "--host", "127.0.0.1",
                   "--port", str(PORT), "-c", str(cfg["ctx"]), "-np", "1", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"],
                   "--jinja", "--no-webui", "-lv", "4",
                   # Non-thinking protocol by default: answers land in the reply (not a hidden reasoning field), which
                   # outside harnesses like lm-eval grade. Requests can still turn thinking on explicitly.
                   "--chat-template-kwargs", '{"enable_thinking": false}'] + _placement_args(cfg, vram_total)
            cmds["load_and_quality"] = _cmdline(cmd)
            t0 = time.monotonic()
            with open(logfile, "w") as lf:
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=lf, stderr=asyncio.subprocess.STDOUT,
                                                            env={**os.environ, "LD_LIBRARY_PATH": str(ebin)})
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=httpx.Timeout(10, read=600)) as client:
                ready = asyncio.create_task(lab.wait_ready(client, timeout=300))
                died = asyncio.create_task(proc.wait())
                done, _ = await asyncio.wait({ready, died}, return_when=asyncio.FIRST_COMPLETED)
                for t in (ready, died):
                    if not t.done():
                        t.cancel()
                if not (ready in done and ready.result()):
                    tail = logfile.read_text(errors="replace").splitlines()[-3:]
                    raise RuntimeError("The model didn't load with this setup (likely out of memory): " + " | ".join(tail)[-300:])
                res["load_seconds"] = round(time.monotonic() - t0, 1)
                res["memory"] = _parse_load_log(logfile.read_text(errors="replace"))
                await asyncio.sleep(1.5)
                res["memory"]["vram_used_mb"] = (sampler.samples[-1].get("vram_mb") if sampler.samples else None)

                # 2) Quality: MMLU-Pro, 5-shot log-likelihood, same 280 questions for every run of the experiment.
                phase("quality")
                def progress(i, n, k):
                    res["mmlu_progress"] = {"done": i, "total": n, "correct": k}
                    _save(run_id, results=res)
                res["mmlu"] = await mmlu_eval(client, MMLU_SEED, lambda: run_id in _cancel, progress)
                res.pop("mmlu_progress", None)

                # 2b) Long context: needle in a haystack, up to this run's context size.
                phase("longctx")
                def nprog(i, n, k):
                    res["needle_progress"] = {"done": i, "total": n, "correct": k}
                    _save(run_id, results=res)
                res["needle"] = await needle_eval(client, cfg["ctx"], lambda: run_id in _cancel, nprog)
                res.pop("needle_progress", None)

                # 2c) Optional: lm-evaluation-harness tasks chosen for this experiment, against this same server.
                lm = (exp.get("protocol") or {}).get("lm_eval")
                if lm and lm.get("tasks"):
                    phase("lm_eval")
                    res["lm_eval"] = await lm_eval_run(lm, exp_dir / f"run{run_id}-lm-eval", log, run_id)
            await arena._stop(proc)
            proc = None
            await asyncio.sleep(1)

            # 3) Speed: llama-bench with the same setup.
            phase("speed")
            res["speed"] = await _bench(cfg, vram_total, run_id, log, cmds)

            # 4) Fidelity: perplexity, and KL divergence vs the experiment's baseline.
            phase("fidelity")
            text = await quantlab.ensure_text()
            pcmd = [str(ebin / "llama-perplexity"), "-m", str(lab.MODELS_DIR / cfg["model"]), "-f", str(text), "-c", "512",
                    "--chunks", "32", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"]] + _placement_args(cfg, vram_total)
            base_run = next((r for r in exp["runs_list"] if r["status"] == "done" and r["id"] != run_id), None)
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
            elif is_baseline or not base_kld.is_file():
                cmds["fidelity"] = _cmdline(pcmd + ["--kl-divergence-base", str(base_kld)])
                out = await quantlab._proc(pcmd + ["--kl-divergence-base", str(base_kld)], log, run_id)
                res["ppl"] = quantlab._num(r"Final estimate: PPL = ([\d.]+)", out)
                res["ppl_sd"] = quantlab._num(r"Final estimate: PPL = [\d.]+\s*\+/-\s*([\d.]+)", out)
                res["kld"], res["same_top"] = 0.0, 100.0
                res["kld_reference"] = True
            else:
                cmds["fidelity"] = _cmdline(pcmd + ["--kl-divergence-base", str(base_kld), "--kl-divergence"])
                out = await quantlab._proc(pcmd + ["--kl-divergence-base", str(base_kld), "--kl-divergence"], log, run_id)
                res["ppl"] = quantlab._num(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", out)
                res["ppl_sd"] = quantlab._num(r"Mean PPL\(Q\)\s*:\s*[\d.]+\s*±\s*([\d.]+)", out)
                res["kld"] = quantlab._num(r"Mean\s+KLD:\s*(-?[\d.]+(?:e-?\d+)?)", out)
                res["kld_sd"] = quantlab._num(r"Mean\s+KLD:\s*[\d.]+\s*±\s*([\d.]+)", out)
                res["kld_p99"] = quantlab._num(r"99\.0%\s+KLD:\s*([\d.]+)", out)
                res["same_top"] = quantlab._num(r"Same top p:\s*([\d.]+)", out)
                res["same_top_sd"] = quantlab._num(r"Same top p:\s*[\d.]+\s*±\s*([\d.]+)", out)

            # 5) Telemetry + bottleneck analysis.
            res["telemetry"] = {p: sampler.summary(p) for p in ("loading", "quality", "longctx", "lm_eval", "speed", "fidelity")}
            res["series"] = sampler.series()
            res["hw"] = analyze(res, cfg, res.get("hardware") or lim)
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001
            status, error = "error", str(e)[:400]
        finally:
            await sampler.stop()
            await arena._stop(proc)
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
    path = lab.MODELS_DIR / cfg["model"]
    if not path.is_file():
        return {}
    w = _weights(path)
    mem = res.get("memory") or {}
    speed = res.get("speed") or {}
    tel = (res.get("telemetry") or {}).get("speed") or {}
    tg, pp = speed.get("tg"), speed.get("pp")
    bw, ram_bw, peak_tf = lim.get("gpu_bandwidth_gbps"), lim.get("ram_bandwidth_gbps"), lim.get("gpu_tflops")

    gb = w["bytes"] / 1e9
    # Which weights live where, from the load log: host "model" buffers minus the embedding table (always on CPU, only looked up).
    cpu_gb = max(0.0, ((mem.get("ram") or {}).get("model", 0) * 2**20 - w["embedding_bytes"]) / 1e9) if mem else 0.0
    cpu_gb = min(cpu_gb, gb)
    gpu_gb = gb - cpu_gb
    flops = 2 * w["params"] if w["params"] else None

    out: dict = {"weights_gb_per_token": round(gb, 3), "weights_on_gpu_gb": round(gpu_gb, 3), "weights_on_cpu_gb": round(cpu_gb, 3),
                 "params_active_b": round(w["params"] / 1e9, 3) if w["params"] else None,
                 "peaks": {"vram_gbps": bw, "ram_gbps": ram_bw, "tflops": peak_tf, "tflops_kind": lim.get("gpu_tflops_kind"),
                           "sm_clock_mhz": lim.get("sm_clock_max")}}

    if tg:
        t_gpu = gpu_gb / bw * 1000 if bw else None
        t_cpu = cpu_gb / ram_bw * 1000 if (cpu_gb and ram_bw) else 0.0
        t_meas = 1000 / tg
        roof_ms = (t_gpu or 0) + (t_cpu or 0) if t_gpu is not None else None
        out["decode"] = {
            "ms_per_token": round(t_meas, 2),
            "effective_gbps": round(gb * tg, 1),
            "roof_tps": round(1000 / roof_ms, 1) if roof_ms else None,
            "roof_util": round(min(1.0, roof_ms / t_meas), 3) if roof_ms else None,
            "time_split_ms": {"vram_reads": round(t_gpu, 2) if t_gpu is not None else None, "ram_reads": round(t_cpu, 2),
                              "other": round(max(0.0, t_meas - (roof_ms or 0)), 2) if roof_ms else None},
            "achieved_tflops": round(flops * tg / 1e12, 3) if flops else None,
            "arithmetic_intensity": round(flops / w["bytes"], 2) if flops else None,
            "long_context_ratio": round(speed["tg_d4096"] / tg, 3) if speed.get("tg_d4096") else None,
        }
    if pp and flops:
        ai = flops * 512 / w["bytes"]  # one 512-token batch reads the weights once
        roof = min(peak_tf, (bw or 0) * ai / 1000) if (peak_tf and bw) else None
        ach = flops * pp / 1e12
        out["prefill"] = {"achieved_tflops": round(ach, 2), "arithmetic_intensity": round(ai, 1),
                          "roof_tflops": round(roof, 2) if roof else None, "roof_util": round(ach / roof, 3) if roof else None,
                          "long_context_ratio": round(speed["pp_d4096"] / pp, 3) if speed.get("pp_d4096") else None}

    sm, smax = tel.get("sm_clock_avg"), lim.get("sm_clock_max")
    out["clocks"] = {"sm_avg_mhz": sm, "sm_max_mhz": smax, "ratio": round(sm / smax, 3) if (sm and smax) else None,
                     "temp_max_c": tel.get("temp_max"), "gpu_util_avg": tel.get("gpu_avg"), "mem_util_avg": tel.get("mem_bw_avg"),
                     "cpu_util_avg": tel.get("cpu_avg"), "power_avg_w": tel.get("power_avg"), "cpu_power_avg_w": tel.get("cpu_power_avg")}

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
    if cpu_gb > 0.05 and split.get("ram_reads"):
        out["limiter"] = {"what": "System-RAM bandwidth",
                          "evidence": f"{split['ram_reads']:.1f} of {d['ms_per_token']:.1f} ms per token is reading {cpu_gb:.2f} GB of weights from RAM at {ram_bw} GB/s"}
    elif d.get("roof_util") is not None and d["roof_util"] >= 0.6:
        out["limiter"] = {"what": "VRAM bandwidth", "evidence": f"{d['effective_gbps']} GB/s effective of {bw} GB/s peak ({d['roof_util'] * 100:.0f}%)"}
    elif d.get("roof_util") is not None:
        out["limiter"] = {"what": "Software overhead",
                          "evidence": f"only {d['roof_util'] * 100:.0f}% of the VRAM-bandwidth roof; {split.get('other')} ms per token is spent outside weight reads"}
    return out


def compare(base: dict | None, run: dict) -> dict:
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
        if pa["model"]["file"] == pb["model"]["file"] and pa["model"]["sha256"] != pb["model"]["sha256"]:
            out["env_mismatch"].append("model file changed on disk")
        # Different hardware is information, not a problem: it's what you compare when testing devices.
        out["hw_diff"] = [name for name, path in HW_KEYS
                          if _get(pa, path) is not None and _get(pb, path) is not None and _get(pa, path) != _get(pb, path)]
    elif pa or pb:
        out["env_mismatch"] = ["provenance missing"]
    if a.get("answers") and b.get("answers"):
        out["mmlu_p"] = mcnemar_p(a["answers"], b["answers"])
        out["mmlu_flips"] = {"lost": sum(1 for x, y in zip(a["answers"], b["answers"]) if x == "1" and y == "0"),
                             "gained": sum(1 for x, y in zip(a["answers"], b["answers"]) if x == "0" and y == "1")}
    return out


def enrich(exp: dict) -> dict:
    """Recompute analysis and statistics for every run, so old runs benefit from fixes too."""
    lim = exp["limits"]
    base = next((r for r in exp["runs_list"] if r["status"] == "done"), None)
    by_id = {r["id"]: r for r in exp["runs_list"]}
    for r in exp["runs_list"]:
        # Runs made before the tree existed (no parent) were all derived from the baseline.
        if r.get("parent_id") is None and base and r["id"] != base["id"]:
            r["parent_id"] = base["id"]
        if r["status"] == "done":
            # Each run is judged against the hardware it actually ran on (its snapshot), not today's machine.
            r["results"]["hw"] = analyze(r["results"], r["config"], r["results"].get("hardware") or lim)
            r["results"].pop("bottleneck", None)
            r["stats"] = compare(base, r)
            if base:
                r["stats"]["baseline_id"] = base["id"]
            parent = by_id.get(r.get("parent_id"))
            if parent and base and parent["id"] != base["id"] and parent["status"] == "done":
                vp = compare(parent, r)
                r["stats"]["vs_parent"] = {k: vp[k] for k in ("mmlu_p", "mmlu_flips") if k in vp}
    return exp


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
