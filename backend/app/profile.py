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

from . import arena, config, db, evals, gguf, lab, quantlab

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
        c.execute("UPDATE exp_runs SET status = 'error', error = 'Interrupted' WHERE status IN ('queued', 'running')")


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
    return dict(r)


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
        c.execute("DELETE FROM exp_runs WHERE id = ? AND status NOT IN ('queued', 'running')", (run_id,))


def cancel(run_id: int) -> None:
    _cancel.add(run_id)


def _save(run_id: int, **f) -> None:
    sets = ", ".join(f"{k} = ?" for k in f)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()]
    with db.connect() as c:
        c.execute(f"UPDATE exp_runs SET {sets} WHERE id = ?", (*vals, run_id))


DEFAULTS = {"ctx": 8192, "kv": "q8_0", "ngl": "auto", "vram_limit_mb": None, "fa": "auto", "threads": None}


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
    return out


def create_run(exp_id: int, body: dict) -> int:
    exp = get_experiment(exp_id)
    if not exp:
        raise ValueError("Experiment not found")
    cfg = normalize_config(exp, body.get("config") or {})
    label = (body.get("label") or "").strip()[:60] or ("Baseline" if not exp["runs_list"] else describe_change(exp["runs_list"][0]["config"], cfg))
    with db.connect() as c:
        return c.execute("INSERT INTO exp_runs (experiment_id, created_at, status, label, config, results) VALUES (?, ?, 'queued', ?, ?, '{}')",
                         (exp_id, time.time(), label, json.dumps(cfg))).lastrowid


def describe_change(base: dict, cfg: dict) -> str:
    parts = []
    if cfg["model"] != base.get("model"):
        parts.append(gguf.metadata(lab.MODELS_DIR / cfg["model"]).get("quant") or cfg["model"])
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

MMLU_PER_CATEGORY = 20
LETTERS = "ABCDEFGHIJ"


def mmlu_sample(seed: int) -> tuple[list[dict], list[dict]]:
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


async def _bench(cfg: dict, vram_total: float, run_id: int, log: list[str]) -> dict:
    cmd = [str(lab.LLAMA_BIN / "llama-bench"), "-m", str(lab.MODELS_DIR / cfg["model"]), "-p", "512", "-n", "128",
           "-d", "0,4096", "-r", "5", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"], "-o", "jsonl"] + _placement_args(cfg, vram_total, bench=True)
    out = await quantlab._proc(cmd, log, run_id, timeout=900)
    res = {}
    for line in out.splitlines():
        if line.startswith("{"):
            x = json.loads(line)
            key = ("pp" if x.get("n_prompt") and not x.get("n_gen") else "tg") + ("_d4096" if x.get("n_depth") else "")
            # Median of the repetitions: one hiccup (a timer glitch, a throttling spike) can't skew the result.
            samples = sorted(x.get("samples_ts") or [])
            if samples:
                med = samples[len(samples) // 2]
                # Robust spread: 1.4826 × median absolute deviation (≈ σ for normal noise, ignores one outlier rep).
                mad = sorted(abs(s - med) for s in samples)[len(samples) // 2]
                res[key], res[key + "_sd"] = round(med, 2), round(1.4826 * mad, 2)
                res[key + "_samples"] = [round(s, 2) for s in x["samples_ts"]]
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
            await lab.llm_service("stop")
            await asyncio.sleep(1)
            sampler.start()

            # 1) Load: start a server with exactly this configuration, time it, read the memory layout.
            phase("loading")
            logfile = exp_dir / f"run{run_id}-server.log"
            cmd = [str(lab.LLAMA_BIN / "llama-server"), "-m", str(lab.MODELS_DIR / cfg["model"]), "--host", "127.0.0.1",
                   "--port", str(PORT), "-c", str(cfg["ctx"]), "-np", "1", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"],
                   "--jinja", "--no-webui", "-lv", "4"] + _placement_args(cfg, vram_total)
            t0 = time.monotonic()
            with open(logfile, "w") as lf:
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=lf, stderr=asyncio.subprocess.STDOUT,
                                                            env={**os.environ, "LD_LIBRARY_PATH": str(lab.LLAMA_BIN)})
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
                res["mmlu"] = await mmlu_eval(client, exp["seed"], lambda: run_id in _cancel, progress)
                res.pop("mmlu_progress", None)
            await arena._stop(proc)
            proc = None
            await asyncio.sleep(1)

            # 3) Speed: llama-bench with the same setup.
            phase("speed")
            res["speed"] = await _bench(cfg, vram_total, run_id, log)

            # 4) Fidelity: perplexity, and KL divergence vs the experiment's baseline.
            phase("fidelity")
            text = await quantlab.ensure_text()
            pcmd = [str(lab.LLAMA_BIN / "llama-perplexity"), "-m", str(lab.MODELS_DIR / cfg["model"]), "-f", str(text), "-c", "512",
                    "--chunks", "32", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"]] + _placement_args(cfg, vram_total)
            if is_baseline or not base_kld.is_file():
                out = await quantlab._proc(pcmd + ["--kl-divergence-base", str(base_kld)], log, run_id)
                res["ppl"] = quantlab._num(r"Final estimate: PPL = ([\d.]+)", out)
                res["ppl_sd"] = quantlab._num(r"Final estimate: PPL = [\d.]+\s*\+/-\s*([\d.]+)", out)
                res["kld"], res["same_top"] = 0.0, 100.0
                res["kld_reference"] = True
            else:
                out = await quantlab._proc(pcmd + ["--kl-divergence-base", str(base_kld), "--kl-divergence"], log, run_id)
                res["ppl"] = quantlab._num(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", out)
                res["ppl_sd"] = quantlab._num(r"Mean PPL\(Q\)\s*:\s*[\d.]+\s*±\s*([\d.]+)", out)
                res["kld"] = quantlab._num(r"Mean\s+KLD:\s*([\d.]+)", out)
                res["kld_sd"] = quantlab._num(r"Mean\s+KLD:\s*[\d.]+\s*±\s*([\d.]+)", out)
                res["kld_p99"] = quantlab._num(r"99\.0%\s+KLD:\s*([\d.]+)", out)
                res["same_top"] = quantlab._num(r"Same top p:\s*([\d.]+)", out)
                res["same_top_sd"] = quantlab._num(r"Same top p:\s*[\d.]+\s*±\s*([\d.]+)", out)

            # 5) Telemetry + bottleneck analysis.
            res["telemetry"] = {p: sampler.summary(p) for p in ("loading", "quality", "speed", "fidelity")}
            res["series"] = sampler.series()
            res["hw"] = analyze(res, cfg, lim)
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
            try:
                await lab.llm_service("start")
                await lab.wait_ready(llm, timeout=300)
            except Exception:  # noqa: BLE001
                pass
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
        # nvidia-smi's "used" includes other programs (e.g. the Windows desktop); with a simulated limit only our buffers count.
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
    if a.get("answers") and b.get("answers"):
        out["mmlu_p"] = mcnemar_p(a["answers"], b["answers"])
        out["mmlu_flips"] = {"lost": sum(1 for x, y in zip(a["answers"], b["answers"]) if x == "1" and y == "0"),
                             "gained": sum(1 for x, y in zip(a["answers"], b["answers"]) if x == "0" and y == "1")}
    return out


def enrich(exp: dict) -> dict:
    """Recompute analysis and statistics for every run, so old runs benefit from fixes too."""
    lim = exp["limits"]
    base = next((r for r in exp["runs_list"] if r["status"] == "done"), None)
    for r in exp["runs_list"]:
        if r["status"] == "done":
            r["results"]["hw"] = analyze(r["results"], r["config"], lim)
            r["results"].pop("bottleneck", None)
            r["stats"] = compare(base, r)
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
