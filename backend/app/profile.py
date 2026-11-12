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

from . import arena, config, db, evals, fresh, generate, gguf, lab, quantlab

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

def gpu_bandwidth(name: str | None) -> float | None:
    if not name:
        return None
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.lower() in name.lower():
            return GPU_BANDWIDTH[key]
    return None


_ram_bw: float | None = None


def ram_bandwidth() -> float | None:
    """Measured system-RAM copy bandwidth (GB/s), cached. The ceiling for layers running on the CPU."""
    global _ram_bw
    cache = DIR / "ram_bandwidth.json"
    if _ram_bw is None and cache.is_file():
        _ram_bw = json.loads(cache.read_text()).get("gbps")
    if _ram_bw is None:
        code = ("import numpy as np, time\na = np.ones(64*2**20, dtype=np.float64); b = np.empty_like(a); best = 0\n"
                "for _ in range(5):\n t = time.perf_counter(); np.copyto(b, a); dt = time.perf_counter() - t\n"
                " best = max(best, 2 * a.nbytes / dt / 1e9)\nprint(best)")
        try:
            out = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=60).stdout.strip()
            _ram_bw = round(float(out), 1)
            cache.write_text(json.dumps({"gbps": _ram_bw, "measured_at": time.time()}))
        except (ValueError, OSError, subprocess.SubprocessError):
            return None
    return _ram_bw


def limits() -> dict:
    hw = lab.hardware()
    g = hw["gpus"][0] if hw["gpus"] else None
    return {"gpu": g["name"] if g else None, "gpu_bandwidth_gbps": gpu_bandwidth(g["name"] if g else None),
            "vram_mb": g["vram_mb"] if g else 0, "power_limit_w": g.get("power_limit_w") if g else None,
            "ram_bandwidth_gbps": ram_bandwidth(), "ram_mb": hw["ram_mb"], "cpu": hw["cpu"], "cpu_threads": hw["cpu_threads"]}


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
    return {**_exp(e), "runs_list": runs, "limits": limits()}


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


class Sampler:
    """GPU/CPU/RAM telemetry every 0.5 s, with phase tags so each phase can be summarized."""

    def __init__(self):
        self.samples: list[dict] = []
        self.phase = "idle"
        self._stop = asyncio.Event()
        self._task = None

    async def _loop(self):
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            s = {"t": time.monotonic(), "phase": self.phase, "cpu": psutil.cpu_percent(interval=None),
                 "ram_mb": (psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**20}
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
                "sm_clock_avg": avg("sm_clock"), "seconds": round(ss[-1]["t"] - ss[0]["t"], 1)}

    def series(self, every: int = 2) -> list[dict]:
        t0 = self.samples[0]["t"] if self.samples else 0
        return [{"t": round(s["t"] - t0, 1), "phase": s["phase"], **{k: s.get(k) for k in ("gpu", "mem_bw", "power", "temp", "vram_mb", "cpu")}}
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


async def _bench(cfg: dict, vram_total: float, run_id: int, log: list[str]) -> dict:
    cmd = [str(lab.LLAMA_BIN / "llama-bench"), "-m", str(lab.MODELS_DIR / cfg["model"]), "-p", "512", "-n", "128",
           "-d", "0,4096", "-r", "3", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"], "-o", "jsonl"] + _placement_args(cfg, vram_total, bench=True)
    out = await quantlab._proc(cmd, log, run_id, timeout=900)
    res = {}
    for line in out.splitlines():
        if line.startswith("{"):
            x = json.loads(line)
            key = ("pp" if x.get("n_prompt") and not x.get("n_gen") else "tg") + ("_d4096" if x.get("n_depth") else "")
            res[key] = round(x.get("avg_ts") or 0, 2)
            res[key + "_sd"] = round(x.get("stddev_ts") or 0, 2)
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

                # 2) Quality: the experiment's fixed fresh questions (identical for every run).
                phase("quality")
                quality, tps, ttft = {}, [], []
                for kind, n in (("math", 15), ("logic", 15), ("code", 10)):
                    qs = fresh.generate(kind, n, exp["seed"])
                    ok = 0
                    for q in qs:
                        if run_id in _cancel:
                            raise asyncio.CancelledError
                        info = await generate.complete(client, [{"role": "user", "content": evals.prompt_for(q)}], think=False,
                                                       sampling={"temperature": 0.0}, seed=1, max_tokens=1024,
                                                       should_stop=lambda: run_id in _cancel)
                        g = await evals.score(q, info["content"])
                        ok += g["correct"]
                        t = info["timings"] or {}
                        if t.get("predicted_per_second") and (t.get("predicted_n") or 0) > 20:
                            tps.append(t["predicted_per_second"])
                        if info.get("ttft_ms"):
                            ttft.append(info["ttft_ms"])
                    quality[kind] = {"correct": ok, "total": n, "score": ok / n}
                    res["quality"] = quality
                    _save(run_id, results=res)
                res["quality_score"] = round(sum(v["score"] for v in quality.values()) / len(quality), 4)
                res["chat_tps"] = round(sum(tps) / len(tps), 2) if tps else None
                res["ttft_ms"] = round(sum(ttft) / len(ttft)) if ttft else None
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
                    "--chunks", "8", "-ctk", cfg["kv"], "-ctv", cfg["kv"], "-fa", cfg["fa"]] + _placement_args(cfg, vram_total)
            if is_baseline or not base_kld.is_file():
                out = await quantlab._proc(pcmd + ["--kl-divergence-base", str(base_kld)], log, run_id)
                res["ppl"] = quantlab._num(r"Final estimate: PPL = ([\d.]+)", out)
                res["kld"], res["same_top"] = 0.0, 100.0
                res["kld_reference"] = True
            else:
                out = await quantlab._proc(pcmd + ["--kl-divergence-base", str(base_kld), "--kl-divergence"], log, run_id)
                res["ppl"] = quantlab._num(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", out)
                res["kld"] = quantlab._num(r"Mean\s+KLD:\s*([\d.]+)", out)
                res["same_top"] = quantlab._num(r"Same top p:\s*([\d.]+)", out)

            # 5) Telemetry + bottleneck analysis.
            res["telemetry"] = {p: sampler.summary(p) for p in ("loading", "quality", "speed", "fidelity")}
            res["series"] = sampler.series()
            res["bottleneck"] = analyze(res, cfg, lim)
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


# ---------- Bottleneck analysis ----------

def analyze(res: dict, cfg: dict, lim: dict) -> dict:
    """What limits this setup? Compares measured speed with the hardware's physical ceilings."""
    findings = []
    mem = res.get("memory") or {}
    layers = mem.get("layers") or {}
    speed = res.get("speed") or {}
    tel = (res.get("telemetry") or {}).get("speed") or {}
    tg = speed.get("tg")
    path = lab.MODELS_DIR / cfg["model"]

    # Bytes read per generated token ≈ all weights on the hot path (MoE: only the active experts).
    try:
        an = gguf.anatomy(path)
        per_token = an["total_bytes"] - an["embedding"]["bytes"]
        if an.get("experts") and an.get("experts_used"):
            exp_bytes = sum(L["parts"].get("experts", {}).get("bytes", 0) for L in an["layers"])
            per_token -= exp_bytes * (1 - an["experts_used"] / an["experts"])
    except (OSError, ValueError):
        per_token = path.stat().st_size
    gb_per_token = per_token / 1e9

    bw = lim.get("gpu_bandwidth_gbps")
    ram_bw = lim.get("ram_bandwidth_gbps")
    on_cpu = (layers.get("total", 0) - layers.get("gpu", 0)) if layers else 0
    ceiling = None
    if bw and gb_per_token:
        if on_cpu and layers.get("total"):
            frac_cpu = on_cpu / layers["total"]
            # time per token = GPU part / GPU bandwidth + CPU part / RAM bandwidth
            t = gb_per_token * (1 - frac_cpu) / bw + gb_per_token * frac_cpu / (ram_bw or 30)
            ceiling = 1 / t
        else:
            ceiling = bw / gb_per_token
    eff = (tg / ceiling) if (tg and ceiling) else None

    if on_cpu:
        findings.append({"kind": "ram", "severity": "major",
                         "title": f"{on_cpu} of {layers['total']} layers run from system RAM",
                         "text": f"They didn't fit in GPU memory. System RAM is ~{(bw or 0) / (ram_bw or 30):.0f}× slower than VRAM here, so those layers dominate the time per word. "
                                 "Fixes: a smaller quantization, a shorter context, q4_0 KV cache, or a GPU with more VRAM."})
    if eff is not None:
        if eff >= 0.5:
            findings.append({"kind": "bandwidth", "severity": "info",
                             "title": f"Memory-bandwidth bound at {eff * 100:.0f}% of the hardware ceiling",
                             "text": f"Writing each word means reading ~{gb_per_token:.2f} GB of weights. At {bw} GB/s the physical limit is ~{ceiling:.0f} tok/s; you get {tg:.0f}. "
                                     "That's normal and close to optimal: only a smaller model/quantization or faster memory can make it much faster."})
        else:
            findings.append({"kind": "overhead", "severity": "minor",
                             "title": f"Only {eff * 100:.0f}% of the memory-speed ceiling reached",
                             "text": f"The hardware could do ~{ceiling:.0f} tok/s for this model size but you get {tg:.0f}. Overheads (small kernels, CPU work, "
                                     "hybrid layers, KV-cache precision) cost speed. Try flash attention on, f16 KV cache, or all layers on the GPU."})
    if tel.get("power_max") and lim.get("power_limit_w") and tel["power_max"] >= 0.95 * lim["power_limit_w"]:
        findings.append({"kind": "power", "severity": "minor", "title": "Hitting the GPU power limit",
                         "text": f"Peaked at {tel['power_max']:.0f} W of {lim['power_limit_w']:.0f} W. The card slows its clocks to stay under it."})
    if tel.get("temp_max") and tel["temp_max"] >= 83:
        findings.append({"kind": "thermal", "severity": "minor", "title": f"Running hot ({tel['temp_max']:.0f}°C)",
                         "text": "Close to where GPUs throttle. Better airflow keeps clocks (and speed) up."})
    if tel.get("cpu_avg") and tel["cpu_avg"] > 85 and (tel.get("gpu_avg") or 0) < 60:
        findings.append({"kind": "cpu", "severity": "major", "title": "CPU is the bottleneck",
                         "text": f"CPU at {tel['cpu_avg']:.0f}% while the GPU idles at {tel.get('gpu_avg', 0):.0f}%."})
    free = (lim.get("vram_mb") or 0) - ((res.get("telemetry") or {}).get("quality", {}).get("vram_max_mb") or 0)
    ctx_mb = (mem.get("gpu", {}).get("context") or 0)
    if free > 800 and not on_cpu and ctx_mb and cfg["ctx"]:
        more = int(free * 0.7 / (ctx_mb / cfg["ctx"])) // 1024 * 1024
        findings.append({"kind": "headroom", "severity": "info", "title": f"{free / 1024:.1f} GB of VRAM unused",
                         "text": f"Room for roughly {more // 1024}k more tokens of context, or a bigger/higher-precision model."})
    main = next((f for f in findings if f["severity"] == "major"), None) or (findings[0] if findings else None)
    return {"findings": findings, "primary": main["kind"] if main else None, "ceiling_tps": round(ceiling, 1) if ceiling else None,
            "efficiency": round(eff, 3) if eff else None, "gb_per_token": round(gb_per_token, 3),
            "gpu_bandwidth_gbps": bw, "ram_bandwidth_gbps": ram_bw}
