"""Quantization lab: what does compressing a model cost?

For several quantizations of the SAME model (e.g. Q3_K_M, Q4_K_M, Q6_K, Q8_0)
we measure, on identical text:
  - perplexity (how well it predicts real Wikipedia text; lower = better)
  - KL divergence vs the most precise version you have (how far its word
    probabilities drift; 0 = identical) and "same top word" %
  - speed (prompt + generation), file size
Uses llama.cpp's own llama-perplexity (the standard tool researchers use)
and llama-bench. Also: make new quantizations locally with llama-quantize.
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

from . import benchmarks, config, db, gguf, lab

TEXT = config.DATA_DIR / "benchmarks" / "wikitext2-test.txt"
WORK = config.DATA_DIR / "quantlab"
_cancel: set[int] = set()
jobs: dict[str, dict] = {}  # local quantize jobs

QUANT_TYPES = ["Q2_K", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS quant_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    status      TEXT NOT NULL,
    models      TEXT NOT NULL,
    reference   TEXT NOT NULL,
    results     TEXT NOT NULL,
    log         TEXT,
    error       TEXT
);
"""


def init() -> None:
    with db.connect() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE quant_runs SET status = 'error', error = 'Interrupted' WHERE status IN ('queued', 'running')")


def _row(r) -> dict:
    d = dict(r)
    d["models"] = json.loads(d["models"]); d["results"] = json.loads(d["results"])
    return d


def list_runs() -> list[dict]:
    with db.connect() as c:
        return [_row(r) for r in c.execute("SELECT * FROM quant_runs ORDER BY id DESC LIMIT 50")]


def get(run_id: int) -> dict | None:
    with db.connect() as c:
        r = c.execute("SELECT * FROM quant_runs WHERE id = ?", (run_id,)).fetchone()
    return _row(r) if r else None


def _save(run_id: int, **f) -> None:
    sets = ", ".join(f"{k} = ?" for k in f)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()]
    with db.connect() as c:
        c.execute(f"UPDATE quant_runs SET {sets} WHERE id = ?", (*vals, run_id))


def delete(run_id: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM quant_runs WHERE id = ? AND status NOT IN ('queued', 'running')", (run_id,))


def cancel(run_id: int) -> None:
    _cancel.add(run_id)


def _bits(file: str) -> float:
    """Approximate precision of a quant type, to pick the reference (most precise) automatically."""
    q = (gguf.metadata(lab.MODELS_DIR / file).get("quant") or "").upper()
    table = {"F32": 32, "BF16": 16, "F16": 16, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q5_0": 5.5,
             "Q4_K_M": 4.8, "Q4_K_S": 4.6, "Q4_0": 4.5, "IQ4_XS": 4.3, "IQ4_NL": 4.5, "MXFP4_MOE": 4.3, "Q3_K_L": 4.3,
             "Q3_K_M": 3.9, "Q3_K_S": 3.5, "IQ3_M": 3.7, "IQ3_XXS": 3.1, "Q2_K": 3.0, "IQ2_M": 2.7, "IQ2_XS": 2.3, "IQ1_M": 1.8}
    return table.get(q.replace("UD-", ""), 4.0)


# ---------- Text corpus (WikiText-2 test, the standard perplexity text) ----------

async def ensure_text() -> Path:
    if TEXT.is_file() and TEXT.stat().st_size > 100_000:
        return TEXT
    prog = {"done": 0, "total": 1}
    async with httpx.AsyncClient(timeout=60, headers=benchmarks.HEADERS) as c:
        rows = await benchmarks._fetch_all(c, "Salesforce/wikitext", "wikitext-2-raw-v1", "test", prog)
    TEXT.parent.mkdir(parents=True, exist_ok=True)
    TEXT.write_text("".join(r.get("text", "") for r in rows))
    return TEXT


# ---------- Running ----------

def create(body: dict) -> int:
    files = [Path(str(f)).name for f in body.get("models") or []]
    files = [f for f in dict.fromkeys(files) if (lab.MODELS_DIR / f).is_file()]
    if len(files) < 2:
        raise ValueError("Pick at least two versions of the same model")
    archs = {gguf.metadata(lab.MODELS_DIR / f).get("architecture") for f in files}
    layers = {gguf.metadata(lab.MODELS_DIR / f).get("layers") for f in files}
    if len(archs) > 1 or len(layers) > 1:
        raise ValueError("These aren't versions of the same model (different architecture). Compare quantizations of one model.")
    ref = body.get("reference") or max(files, key=_bits)
    with db.connect() as c:
        return c.execute("INSERT INTO quant_runs (created_at, status, models, reference, results, log) VALUES (?, 'queued', ?, ?, '[]', '')",
                         (time.time(), json.dumps(files), ref)).lastrowid


async def _proc(cmd: list[str], log: list[str], run_id: int, timeout: float = 1800) -> str:
    # Each llama.cpp build loads the libraries next to its own binary (custom engines must not pick up the stock .so files).
    libdir = Path(cmd[0]).parent if Path(cmd[0]).is_absolute() else lab.LLAMA_BIN
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                env={**os.environ, "LD_LIBRARY_PATH": str(libdir)})
    out = []
    start = time.monotonic()
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        out.append(line)
        if line.strip():
            log.append(line[-200:])
            del log[:-60]
        if run_id in _cancel or time.monotonic() - start > timeout:
            proc.kill()
            raise asyncio.CancelledError
    await proc.wait()
    return "\n".join(out)


def _num(pattern: str, text: str, group: int = 1) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(group)) if m else None


async def run(run_id: int, llm: httpx.AsyncClient) -> None:
    rec = get(run_id)
    WORK.mkdir(parents=True, exist_ok=True)
    base = WORK / f"run{run_id}.kld"
    results: list[dict] = []
    log: list[str] = []
    status, error = "done", None
    chunks = "8"  # 8 × 512 tokens: enough for stable numbers, small reference file
    async with lab._lock:
        lab.busy = "quantlab"
        _save(run_id, status="running")
        try:
            text = await ensure_text()
            await lab.llm_service("stop")
            order = [rec["reference"]] + [m for m in sorted(rec["models"], key=_bits, reverse=True) if m != rec["reference"]]
            for i, m in enumerate(order):
                path = lab.MODELS_DIR / m
                r = {"model": m, "quant": gguf.metadata(path).get("quant"), "size_bytes": path.stat().st_size,
                     "bits": _bits(m), "reference": i == 0, "state": "measuring quality"}
                results.append(r)
                _save(run_id, results=results, log="\n".join(log))
                common = [str(lab.LLAMA_BIN / "llama-perplexity"), "-m", str(path), "-f", str(text), "-c", "512",
                          "--chunks", chunks, "-fit", "on", "-fa", "auto"]
                if i == 0:
                    out = await _proc(common + ["--kl-divergence-base", str(base)], log, run_id)
                    r["ppl"] = _num(r"Final estimate: PPL = ([\d.]+)", out)
                    r["ppl_err"] = _num(r"Final estimate: PPL = [\d.]+ \+/- ([\d.]+)", out)
                    r["kld"], r["same_top"] = 0.0, 100.0
                else:
                    out = await _proc(common + ["--kl-divergence-base", str(base), "--kl-divergence"], log, run_id)
                    r["ppl"] = _num(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", out) or _num(r"Final estimate: PPL = ([\d.]+)", out)
                    r["ppl_err"] = _num(r"Mean PPL\(Q\)\s*:\s*[\d.]+\s*±\s*([\d.]+)", out)
                    r["kld"] = _num(r"Mean\s+KLD:\s*(-?[\d.]+(?:e-?\d+)?)", out)
                    r["kld_err"] = _num(r"Mean\s+KLD:\s*[\d.]+\s*±\s*([\d.]+)", out)
                    r["kld_99"] = _num(r"99\.0%\s+KLD:\s*([\d.]+)", out)
                    r["same_top"] = _num(r"Same top p:\s*([\d.]+)", out)
                if r.get("ppl") is None:
                    raise RuntimeError(f"Couldn't read the perplexity result for {m}: {(log[-1] if log else '')[:200]}")
                r["state"] = "measuring speed"
                _save(run_id, results=results, log="\n".join(log))
                out = await _proc([str(lab.LLAMA_BIN / "llama-bench"), "-m", str(path), "-p", "512", "-n", "128", "-r", "2",
                                   "-fitt", "512", "-o", "jsonl"], log, run_id, timeout=600)
                for line in out.splitlines():
                    if line.startswith("{"):
                        x = json.loads(line)
                        if x.get("n_prompt") and not x.get("n_gen"):
                            r["pp_tps"] = x.get("avg_ts")
                        elif x.get("n_gen") and not x.get("n_prompt"):
                            r["tg_tps"] = x.get("avg_ts")
                r["state"] = "done"
                _save(run_id, results=results, log="\n".join(log))
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001
            status, error = "error", str(e)[:400]
        finally:
            base.unlink(missing_ok=True)
            _save(run_id, status=status, error=error, results=results, log="\n".join(log[-60:]))
            _cancel.discard(run_id)
            try:
                await lab.llm_service("start")
                await lab.wait_ready(llm, timeout=300)
            except Exception:  # noqa: BLE001
                pass
            lab.busy = None


# ---------- Make a quantization locally ----------

def start_quantize(source: str, qtype: str) -> str:
    src = lab.MODELS_DIR / Path(source).name
    if not src.is_file():
        raise ValueError("Source model not found")
    if qtype not in QUANT_TYPES:
        raise ValueError(f"Type must be one of {', '.join(QUANT_TYPES)}")
    src_q = (gguf.metadata(src).get("quant") or "").upper()
    if _bits(src.name) <= 8.5 and src_q not in ("F16", "BF16", "F32"):
        note = "requantizing from an already-compressed file loses extra quality; a F16/BF16 source is better"
    else:
        note = None
    stem = re.sub(r"[-.](UD-)?(IQ\d_\w+|Q\d_\w+|BF16|F16|F32|MXFP4\w*)$", "", src.stem, flags=re.I)
    out = lab.MODELS_DIR / f"{stem}-{qtype}.gguf"
    if out.exists():
        raise ValueError(f"{out.name} already exists")
    job_id = str(int(time.time() * 1000))
    jobs[job_id] = {"id": job_id, "source": src.name, "type": qtype, "output": out.name, "status": "running", "error": None, "note": note,
                    "started": time.time(), "progress": 0}
    asyncio.get_running_loop().create_task(_quantize(job_id, src, out, qtype, note is not None))
    return job_id


async def _quantize(job_id: str, src: Path, out: Path, qtype: str, requant: bool) -> None:
    j = jobs[job_id]
    tmp = out.with_suffix(".gguf.part")
    cmd = [str(lab.LLAMA_BIN / "llama-quantize")] + (["--allow-requantize"] if requant else []) + [str(src), str(tmp), qtype, str(max(1, (os.cpu_count() or 2) // 2))]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                    env={**os.environ, "LD_LIBRARY_PATH": str(lab.LLAMA_BIN)})
        total = None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace")
            m = re.search(r"\[\s*(\d+)/\s*(\d+)\]", line)
            if m:
                total = int(m.group(2))
                j["progress"] = int(m.group(1)) / total
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"llama-quantize exited with {proc.returncode}")
        tmp.replace(out)
        j["status"], j["progress"] = "done", 1.0
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        j["status"], j["error"] = "error", str(e)[:300]
