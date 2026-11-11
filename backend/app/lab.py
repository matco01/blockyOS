"""Lab: model library, runtime settings, downloads and benchmarks.

Everything a researcher needs to test local models on real hardware:
  - model files with real metadata (read from the GGUF header)
  - "will it fit?" estimates for this machine
  - switching models / context / KV cache type / GPU layers
  - Hugging Face downloads with progress
  - llama-bench runs with GPU power sampling -> tokens per joule
"""
import asyncio
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import psutil

from . import config, db, gguf, system

log = logging.getLogger("blocky.lab")

MODELS_DIR = Path(os.environ.get("BLOCKY_MODELS_DIR", "/opt/blocky/models"))
LLAMA_BIN = Path(os.environ.get("BLOCKY_LLAMA_BIN", "/opt/blocky/llama.cpp/bin"))
RUNTIME_FILE = config.DATA_DIR / "llm.env"   # read by blocky-llm.service

KV_TYPES = ("f16", "q8_0", "q4_0")
RUNTIME_DEFAULTS = {"model": None, "ctx": 16384, "kv": "q8_0", "ngl": "auto", "threads": "auto"}

# Curated starting points. Sizes are the Q4 files; fit is computed per machine.
CATALOG = [
    {"repo": "unsloth/Qwen3.5-4B-GGUF", "file": "Qwen3.5-4B-Q4_K_M.gguf", "size": 2_740_000_000,
     "title": "Qwen3.5 4B", "note": "Small and quick. Runs on 4 GB GPUs.", "moe": False},
    {"repo": "unsloth/Qwen3.5-9B-GGUF", "file": "Qwen3.5-9B-Q4_K_M.gguf", "size": 5_680_000_000,
     "title": "Qwen3.5 9B", "note": "Best all-rounder for 8 GB GPUs. Native reasoning.", "moe": False},
    {"repo": "unsloth/Qwen3.6-27B-GGUF", "file": "Qwen3.6-27B-Q4_K_M.gguf", "size": 16_820_000_000,
     "title": "Qwen3.6 27B", "note": "Dense and strong at coding. Needs 24 GB VRAM to be fast.", "moe": False},
    {"repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "file": "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", "size": 20_890_000_000,
     "title": "Qwen3.6 35B-A3B", "note": "MoE: 35B knowledge, 3B active. Fast even when partly in RAM.", "moe": True},
]

# One heavy operation at a time (switching model or benchmarking).
_lock = asyncio.Lock()
busy: str | None = None
_bench_procs: dict[int, asyncio.subprocess.Process] = {}
downloads: dict[str, dict] = {}


# ---------- Hardware ----------

_static_hw: dict | None = None


def _llama_version() -> str | None:
    try:
        out = subprocess.run([str(LLAMA_BIN / "llama-server"), "--version"], capture_output=True, text=True,
                             timeout=10, env={**os.environ, "LD_LIBRARY_PATH": str(LLAMA_BIN)})
        text = out.stdout + out.stderr
        m = re.search(r"version:\s*(\S+).*?commit\s+(\w+)", text) or re.search(r"version:\s*(\S+)\s*\((\w+)\)", text)
        return f"{m.group(1)} ({m.group(2)})" if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware() -> dict:
    """Fingerprint of this machine, saved with every benchmark so results are comparable."""
    global _static_hw
    if _static_hw is None:
        cpu = "unknown"
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        os_name = platform.system()
        try:
            for line in Path("/etc/os-release").read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip('"')
        except OSError:
            pass
        gpus = []
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,pcie.link.gen.max,pcie.link.width.max",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    name, driver, mem, plimit, gen, width = [p.strip() for p in line.split(",")]
                    gpus.append({"name": name, "driver": driver, "vram_mb": float(mem),
                                 "power_limit_w": system._num(plimit), "pcie": f"Gen{gen} x{width}" if gen.isdigit() else None})
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
        _static_hw = {
            "cpu": cpu,
            "cpu_threads": psutil.cpu_count(logical=True),
            "cpu_cores": psutil.cpu_count(logical=False),
            "ram_mb": psutil.virtual_memory().total / 2**20,
            "gpus": gpus,
            "os": os_name,
            "kernel": platform.release(),
            "wsl": "microsoft" in platform.release().lower(),
            "llama_cpp": _llama_version(),
        }
    return _static_hw


def fit(size_bytes: int, info: dict | None, ctx: int, kv: str, moe: bool = False, kv_mb: float | None = None) -> dict:
    """Estimate where a model would run on this machine. Pass kv_mb when the real value is known."""
    hw = hardware()
    vram = hw["gpus"][0]["vram_mb"] if hw["gpus"] else 0
    ram = hw["ram_mb"]
    measured = kv_mb is not None
    if kv_mb is None:
        kv_mb = (gguf.kv_bytes_per_token(info or {}, kv) * ctx) / 2**20
    need = size_bytes / 2**20 + kv_mb + 700
    usable_vram = max(0.0, vram - 600)
    usable_ram = ram * 0.8
    is_moe = moe or bool(info and info.get("experts"))
    if vram and need <= usable_vram:
        verdict, label = "gpu", "Fits in GPU"
    elif vram and need <= usable_vram + usable_ram:
        verdict, label = ("split_moe", "GPU + RAM · MoE, good speed") if is_moe else ("split", "GPU + RAM · slower")
    elif not vram and need <= usable_ram:
        verdict, label = "cpu", "CPU only · slow"
    else:
        verdict, label = "too_big", f"Too big for this machine ({(usable_vram + usable_ram) / 1024:.0f} GB usable)"
    return {"verdict": verdict, "label": label, "need_mb": round(need), "kv_mb": round(kv_mb), "measured": measured}


# ---------- Runtime settings ----------

def read_runtime() -> dict:
    cfg = dict(RUNTIME_DEFAULTS)
    cfg["model"] = os.environ.get("BLOCKY_MODEL")
    try:
        for line in RUNTIME_FILE.read_text().splitlines():
            key, _, value = line.partition("=")
            k = {"BLOCKY_MODEL": "model", "BLOCKY_CTX": "ctx", "BLOCKY_KV": "kv",
                 "BLOCKY_NGL": "ngl", "BLOCKY_THREADS": "threads"}.get(key.strip())
            if k:
                cfg[k] = value.strip()
    except OSError:
        pass
    cfg["ctx"] = int(cfg["ctx"]) if str(cfg["ctx"]).isdigit() else RUNTIME_DEFAULTS["ctx"]
    if cfg["model"]:
        cfg["model"] = Path(cfg["model"]).name
    return cfg


def write_runtime(cfg: dict) -> None:
    lines = [
        "# Written by the Blocky web app (Models page). Overrides /etc/blocky/blocky.env.",
        f"BLOCKY_MODEL={MODELS_DIR / cfg['model']}",
        f"BLOCKY_CTX={int(cfg['ctx'])}",
        f"BLOCKY_KV={cfg['kv']}",
        f"BLOCKY_NGL={cfg['ngl']}",
        f"BLOCKY_THREADS={cfg['threads']}",
    ]
    tmp = RUNTIME_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(RUNTIME_FILE)


def validate_runtime(body: dict) -> dict:
    cfg = {**read_runtime(), **{k: v for k, v in body.items() if v is not None}}
    model = Path(str(cfg.get("model") or "")).name
    if not model.endswith(".gguf") or not (MODELS_DIR / model).is_file():
        raise ValueError("Model file not found")
    ctx = int(cfg["ctx"])
    if not 512 <= ctx <= 1_048_576:
        raise ValueError("Context must be between 512 and 1,048,576")
    if cfg["kv"] not in KV_TYPES:
        raise ValueError(f"KV cache type must be one of {', '.join(KV_TYPES)}")
    ngl = str(cfg["ngl"])
    if ngl != "auto" and not (ngl.isdigit() and int(ngl) <= 999):
        raise ValueError("GPU layers must be 'auto' or 0–999")
    threads = str(cfg["threads"])
    if threads != "auto" and not (threads.isdigit() and 1 <= int(threads) <= 256):
        raise ValueError("Threads must be 'auto' or 1–256")
    return {"model": model, "ctx": ctx, "kv": cfg["kv"], "ngl": ngl, "threads": threads}


async def _run(*cmd: str, timeout: float = 120) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return -1, "timed out"
    return proc.returncode or 0, out.decode(errors="replace")


async def llm_service(action: str) -> None:
    """start/stop/restart the model server (allowed via a narrow sudoers rule)."""
    code, out = await _run("sudo", "-n", "systemctl", action, "blocky-llm")
    if code != 0:
        raise RuntimeError(f"systemctl {action} failed: {out.strip()[:300]}")


async def wait_ready(llm: httpx.AsyncClient, timeout: float = 300) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = await llm.get("/health", timeout=2)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    return False


last_switch_error: str | None = None


async def apply_runtime(llm: httpx.AsyncClient, cfg: dict) -> None:
    """Write new settings and restart the model server with them."""
    global busy, last_switch_error
    async with _lock:
        busy = "switching"
        last_switch_error = None
        previous = read_runtime()
        try:
            write_runtime(cfg)
            await llm_service("restart")
            if not await wait_ready(llm, timeout=300):
                raise RuntimeError("Model didn't start. It may not fit with these settings; see the log on the System page.")
        except Exception as e:  # roll back to the last working settings
            last_switch_error = str(e)
            log.warning("runtime switch failed: %s", e)
            if previous.get("model") and previous != cfg:
                write_runtime(previous)
                try:
                    await llm_service("restart")
                    await wait_ready(llm, timeout=300)
                except Exception as e2:  # noqa: BLE001
                    log.error("rollback failed: %s", e2)
        finally:
            busy = None


# ---------- Model library ----------

def list_models() -> list[dict]:
    rt = read_runtime()
    out = []
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        if re.search(r"-0000[2-9]-of-|mmproj", p.name):
            continue  # later parts of split files, vision projectors
        info = gguf.metadata(p)
        info["active"] = p.name == rt["model"]
        # For the running model we know the real context cost per token: use it.
        real = _measured.get("kv_per_token") if (_measured.get("model") == p.name and _measured.get("kv") == rt["kv"]) else None
        if real:
            info["kv_mb_per_token"] = real
        if real:
            info["fixed_state_mb"] = _measured.get("fixed_mb", 0)
        info["fit"] = fit(info["size_bytes"], info, rt["ctx"], rt["kv"],
                          kv_mb=real * rt["ctx"] + _measured.get("fixed_mb", 0) if real else None)
        out.append(info)
    return out


def delete_model(name: str) -> None:
    name = Path(name).name
    if name == read_runtime()["model"]:
        raise ValueError("Can't delete the model that's currently running. Switch to another one first.")
    path = MODELS_DIR / name
    if not path.is_file() or path.suffix != ".gguf":
        raise ValueError("Model not found")
    path.unlink()


def catalog() -> list[dict]:
    rt = read_runtime()
    have = {p.name for p in MODELS_DIR.glob("*.gguf")}
    return [{**c, "downloaded": c["file"] in have, "fit": fit(c["size"], None, rt["ctx"], rt["kv"], c["moe"])} for c in CATALOG]


# ---------- Hugging Face ----------

_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
HF_HEADERS = {"User-Agent": "blockyOS/0.1"}


async def hf_files(repo: str) -> list[dict]:
    if not _REPO.match(repo):
        raise ValueError("Use the form owner/repo, e.g. unsloth/Qwen3.5-9B-GGUF")
    async with httpx.AsyncClient(timeout=20, headers=HF_HEADERS) as c:
        r = await c.get(f"https://huggingface.co/api/models/{repo}/tree/main", params={"recursive": "1"})
    if r.status_code in (401, 403, 404):
        raise ValueError("Repository not found (or it requires a login).")
    r.raise_for_status()
    rt = read_runtime()
    files = []
    for f in r.json():
        path = f.get("path", "")
        if not path.endswith(".gguf") or "mmproj" in path.lower():
            continue
        size = (f.get("lfs") or {}).get("size") or f.get("size") or 0
        split = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
        if split and split.group(1) != "00001":
            continue
        files.append({"path": path, "size": size, "split": bool(split),
                      "fit": fit(size, None, rt["ctx"], rt["kv"], "A3B" in path or "-A" in path)})
    return sorted(files, key=lambda f: f["size"])


def start_download(repo: str, path: str) -> str:
    if not _REPO.match(repo) or not path.endswith(".gguf") or ".." in path:
        raise ValueError("Invalid repository or file")
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", path):
        raise ValueError("Split (multi-part) models aren't supported yet")
    name = Path(path).name
    if (MODELS_DIR / name).exists():
        raise ValueError("You already have this model")
    if any(d["file"] == name and d["status"] == "downloading" for d in downloads.values()):
        raise ValueError("Already downloading")
    dl_id = f"{int(time.time() * 1000)}"
    downloads[dl_id] = {"id": dl_id, "repo": repo, "file": name, "total": 0, "done": 0,
                        "status": "downloading", "error": None, "started": time.time(), "speed": 0}
    asyncio.get_running_loop().create_task(_download(dl_id, repo, path))
    return dl_id


async def _download(dl_id: str, repo: str, path: str) -> None:
    d = downloads[dl_id]
    final = MODELS_DIR / d["file"]
    part = final.with_suffix(".gguf.part")
    try:
        url = f"https://huggingface.co/{repo}/resolve/main/{path}"
        async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30, read=120), headers=HF_HEADERS) as c:
            async with c.stream("GET", url) as r:
                r.raise_for_status()
                d["total"] = int(r.headers.get("content-length") or 0)
                free = shutil.disk_usage(MODELS_DIR).free
                if d["total"] and d["total"] > free - 2 * 2**30:
                    raise RuntimeError(f"Not enough disk space ({free / 2**30:.1f} GB free)")
                window_t, window_b = time.monotonic(), 0
                with open(part, "wb") as f:
                    async for chunk in r.aiter_bytes(1 << 20):
                        if d["status"] == "cancelled":
                            raise asyncio.CancelledError
                        f.write(chunk)
                        d["done"] += len(chunk)
                        window_b += len(chunk)
                        now = time.monotonic()
                        if now - window_t >= 1:
                            d["speed"] = window_b / (now - window_t)
                            window_t, window_b = now, 0
        part.replace(final)
        d["status"] = "done"
    except asyncio.CancelledError:
        d["status"] = "cancelled"
        part.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        d["status"], d["error"] = "error", str(e)[:300]
        part.unlink(missing_ok=True)


# ---------- Benchmarks ----------

def _ints(values, lo, hi, name) -> str:
    vals = [int(v) for v in values]
    if not vals or len(vals) > 8 or any(v < lo or v > hi for v in vals):
        raise ValueError(f"{name}: 1–8 values between {lo} and {hi}")
    return ",".join(str(v) for v in vals)


def bench_command(p: dict) -> tuple[list[str], str]:
    model = Path(p["model"]).name
    path = MODELS_DIR / model
    if not path.is_file():
        raise ValueError("Model not found")
    cmd = [str(LLAMA_BIN / "llama-bench"), "-m", str(path),
           "-p", _ints(p.get("n_prompt", [512]), 0, 65536, "Prompt sizes"),
           "-n", _ints(p.get("n_gen", [128]), 0, 4096, "Generation sizes"),
           "-d", _ints(p.get("depth", [0]), 0, 262144, "Depths"),
           "-r", str(max(1, min(20, int(p.get("repetitions", 3))))),
           "-o", "jsonl", "--progress"]
    kv = p.get("kv", "q8_0")
    if kv not in KV_TYPES:
        raise ValueError("Invalid KV cache type")
    cmd += ["-ctk", kv, "-ctv", kv]
    fa = p.get("flash_attn", "auto")
    if fa not in ("auto", "on", "off"):
        raise ValueError("Invalid flash attention setting")
    cmd += ["-fa", fa]
    ngl = str(p.get("ngl", "auto"))
    if ngl == "auto":
        cmd += ["-fitt", "512"]
    elif ngl.isdigit():
        cmd += ["-ngl", ngl]
    else:
        raise ValueError("GPU layers must be 'auto' or a number")
    if p.get("threads"):
        cmd += ["-t", str(max(1, min(256, int(p["threads"]))))]
    if p.get("n_cpu_moe"):
        cmd += ["-ncmoe", str(max(0, min(999, int(p["n_cpu_moe"]))))]
    return cmd, model


def _bench_update(run_id: int, **fields) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    with db.connect() as c:
        c.execute(f"UPDATE bench_runs SET {sets} WHERE id = ?", (*vals, run_id))


def create_bench(params: dict) -> int:
    cmd, model = bench_command(params)  # validates
    with db.connect() as c:
        cur = c.execute(
            "INSERT INTO bench_runs (created_at, status, model, params, hardware, results, log) VALUES (?, 'queued', ?, ?, ?, '[]', '')",
            (time.time(), model, json.dumps(params), json.dumps(hardware())),
        )
        return cur.lastrowid


def list_bench() -> list[dict]:
    with db.connect() as c:
        rows = c.execute("SELECT * FROM bench_runs ORDER BY id DESC LIMIT 100").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("params", "hardware", "results"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append(d)
    return out


def get_bench(run_id: int) -> dict | None:
    return next((r for r in list_bench() if r["id"] == run_id), None)


def delete_bench(run_id: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM bench_runs WHERE id = ? AND status NOT IN ('running', 'queued')", (run_id,))


def cancel_bench(run_id: int) -> None:
    proc = _bench_procs.get(run_id)
    if proc and proc.returncode is None:
        proc.terminate()


async def _sample_gpu(samples: list, stop: asyncio.Event) -> None:
    if not shutil.which("nvidia-smi"):
        return
    while not stop.is_set():
        code, out = await _run("nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used",
                               "--format=csv,noheader,nounits", timeout=5)
        if code == 0 and out.strip():
            p, u, m = (system._num(x.strip()) for x in out.strip().splitlines()[0].split(","))
            samples.append((time.monotonic(), p, u, m))
        try:
            await asyncio.wait_for(stop.wait(), 0.5)
        except asyncio.TimeoutError:
            pass


def _test_name(r: dict) -> str:
    if r.get("n_prompt") and r.get("n_gen"):
        name = f"pp{r['n_prompt']}+tg{r['n_gen']}"
    elif r.get("n_prompt"):
        name = f"pp{r['n_prompt']}"
    else:
        name = f"tg{r.get('n_gen', 0)}"
    return name + (f" @ d{r['n_depth']}" if r.get("n_depth") else "")


async def run_bench(run_id: int, llm: httpx.AsyncClient) -> None:
    global busy
    run = get_bench(run_id)
    cmd, _ = bench_command(run["params"])
    async with _lock:
        busy = "benchmark"
        results, log_lines = [], []
        stop = asyncio.Event()
        samples: list = []
        sampler = None
        status, error = "done", None
        try:
            _bench_update(run_id, status="running", log="Stopping chat model to free GPU memory…")
            await llm_service("stop")
            await asyncio.sleep(1)
            sampler = asyncio.create_task(_sample_gpu(samples, stop))
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "LD_LIBRARY_PATH": str(LLAMA_BIN)})
            _bench_procs[run_id] = proc
            window_start = time.monotonic()

            async def read_err():
                async for raw in proc.stderr:
                    line = raw.decode(errors="replace").rstrip()
                    if line:
                        log_lines.append(line)
                        del log_lines[:-300]

            err_task = asyncio.create_task(read_err())
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                now = time.monotonic()
                win = [s for s in samples if window_start <= s[0] <= now]
                active = [s for s in win if (s[2] or 0) >= 20] or win
                power = sum(s[1] for s in active if s[1] is not None) / len(active) if active and active[0][1] is not None else None
                peak_vram = max((s[3] or 0) for s in win) if win else None
                window_start = now
                ts = r.get("avg_ts")
                results.append({
                    "test": _test_name(r),
                    "n_prompt": r.get("n_prompt"), "n_gen": r.get("n_gen"), "n_depth": r.get("n_depth", 0),
                    "avg_ts": ts, "stddev_ts": r.get("stddev_ts"), "samples_ts": r.get("samples_ts"),
                    "power_w": round(power, 1) if power else None,
                    "tokens_per_joule": round(ts / power, 3) if power and ts else None,
                    "peak_vram_mb": peak_vram,
                    "n_gpu_layers": r.get("n_gpu_layers"), "n_cpu_moe": r.get("n_cpu_moe"),
                    "type_k": r.get("type_k"), "flash_attn": r.get("flash_attn"), "n_threads": r.get("n_threads"),
                    "n_batch": r.get("n_batch"), "n_ubatch": r.get("n_ubatch"),
                    "model_type": r.get("model_type"), "model_size": r.get("model_size"),
                    "model_n_params": r.get("model_n_params"), "backends": r.get("backends"),
                    "build": f"{r.get('build_number')} ({r.get('build_commit')})",
                })
                _bench_update(run_id, results=results, log="\n".join(log_lines[-40:]))
            await proc.wait()
            await err_task
            if proc.returncode not in (0, None):
                if proc.returncode in (-15, 143):
                    status = "cancelled"
                else:
                    status, error = "error", (log_lines[-1] if log_lines else f"llama-bench exited with {proc.returncode}")
        except Exception as e:  # noqa: BLE001
            status, error = "error", str(e)[:500]
        finally:
            stop.set()
            if sampler:
                await sampler
            _bench_procs.pop(run_id, None)
            _bench_update(run_id, status=status, error=error, results=results,
                          log="\n".join(log_lines[-200:]), finished_at=time.time())
            try:
                await llm_service("start")
                await wait_ready(llm, timeout=300)
            except Exception as e:  # noqa: BLE001
                log.error("couldn't restart chat model after benchmark: %s", e)
            busy = None


# ---------- Memory map ----------
# llama.cpp logs exactly where it put everything when the model loads
# (needs --log-verbosity 4, set in start-llm.sh). We parse the latest load.

_BUF = re.compile(r"(\S+)\s+(model|KV|RS|compute|output|rotation) buffer size =\s*([\d.]+) MiB")
_OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_KVINFO = re.compile(r"llama_kv_cache: size =\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers")


def _is_gpu(device: str) -> bool:
    return device.startswith(("CUDA", "Vulkan", "ROCm", "Metal")) and not device.endswith("_Host")


async def memory_map() -> dict:
    code, out = await _run("journalctl", "-u", "blocky-llm", "-n", "800", "--no-pager", "-o", "cat", timeout=10)
    lines = out.splitlines() if code == 0 else []
    start = max((i for i, l in enumerate(lines) if l.startswith("blocky-llm: model=")), default=-1)
    lines = lines[start + 1:] if start >= 0 else []

    gpu = {"model": 0.0, "context": 0.0, "compute": 0.0}
    host = {"model": 0.0, "context": 0.0, "compute": 0.0}
    layers = None
    kv_info = None
    kv_only = 0.0     # grows with context size
    fixed_state = 0.0  # recurrent state (hybrid models): fixed, doesn't grow with context
    for l in lines:
        if m := _BUF.search(l):
            device, kind, mib = m.group(1), m.group(2), float(m.group(3))
            slot = {"model": "model", "KV": "context", "RS": "context"}.get(kind, "compute")
            (gpu if _is_gpu(device) else host)[slot] += mib
            if kind == "KV":
                kv_only += mib
            elif kind == "RS":
                fixed_state += mib
        elif m := _OFFLOAD.search(l):
            layers = {"gpu": int(m.group(1)), "total": int(m.group(2))}
        elif m := _KVINFO.search(l):
            kv_info = {"mib": float(m.group(1)), "cells": int(m.group(2)), "attention_layers": int(m.group(3))}

    snap = system.snapshot()
    g = snap["gpus"][0] if snap["gpus"] else None
    gpu_out = None
    if g and g.get("vram_total_mb"):
        ours = sum(gpu.values())
        used = g.get("vram_used_mb") or 0
        gpu_out = {
            "name": g["name"], "total": g["vram_total_mb"], "used": used,
            **{k: round(v) for k, v in gpu.items()},
            "other": round(max(0.0, used - ours)),
            "free": round(max(0.0, g["vram_total_mb"] - used)),
        }
    mem = psutil.virtual_memory()
    ram_used = (mem.total - mem.available) / 2**20
    ours_ram = sum(host.values())
    ram_out = {
        "total": round(mem.total / 2**20), "used": round(ram_used),
        **{k: round(v) for k, v in host.items()},
        "other": round(max(0.0, ram_used - ours_ram)),
        "free": round(mem.available / 2**20),
    }
    return {"parsed": bool(lines and (layers or any(gpu.values()) or any(host.values()))),
            "gpu": gpu_out, "ram": ram_out, "layers": layers, "kv": kv_info,
            "kv_mb": round(kv_only), "fixed_state_mb": round(fixed_state), "runtime": read_runtime()}


def measured_kv_mb_per_token() -> float | None:
    """Context-cache cost per token from the last load, if available (exact, beats the estimate)."""
    return _measured.get("kv_per_token")


_measured: dict = {}


async def refresh_measured() -> None:
    mm = await memory_map()
    ctx = mm["runtime"]["ctx"]
    if mm["parsed"] and ctx and mm["kv_mb"]:
        _measured.update({"model": mm["runtime"]["model"], "kv": mm["runtime"]["kv"],
                          "kv_per_token": mm["kv_mb"] / ctx, "fixed_mb": mm["fixed_state_mb"]})


# ---------- Logs ----------

async def llm_log(lines: int = 300) -> str:
    code, out = await _run("journalctl", "-u", "blocky-llm", "-n", str(max(10, min(2000, lines))),
                           "--no-pager", "-o", "short-iso", timeout=10)
    return out if code == 0 else f"Couldn't read the log: {out.strip()[:300]}"
