"""More lab tools:

Throughput: how many people can use this machine at once? llama-batched-bench
  runs 1, 2, 4, 8… conversations in parallel and measures total and per-user
  speed. More users = more total output, but each user's answer gets slower.

Speculative decoding: a tiny "draft" model from the same family guesses the
  next few words, and the big model checks them all in one step. If the guesses
  are good, the big model writes several words per step. We measure the real
  speed-up on this machine with the same prompts, with and without a draft.
"""
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from . import arena, db, gguf, lab

SCHEMA = """
CREATE TABLE IF NOT EXISTS lab_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    params      TEXT NOT NULL,
    results     TEXT NOT NULL,
    log         TEXT,
    error       TEXT
);
"""
_cancel: set[int] = set()

SPEC_PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
    "Explain in 5 sentences how a refrigerator works.",
    "List the planets of the solar system with one fact about each.",
    "Write a short, friendly email asking a colleague to move a meeting to Thursday.",
]


def init() -> None:
    with db.connect() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE lab_runs SET status = 'error', error = 'Interrupted' WHERE status IN ('queued', 'running')")


def _row(r) -> dict:
    d = dict(r)
    d["params"] = json.loads(d["params"]); d["results"] = json.loads(d["results"])
    return d


def list_runs(kind: str) -> list[dict]:
    with db.connect() as c:
        return [_row(r) for r in c.execute("SELECT * FROM lab_runs WHERE kind = ? ORDER BY id DESC LIMIT 50", (kind,))]


def _save(run_id: int, **f) -> None:
    sets = ", ".join(f"{k} = ?" for k in f)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()]
    with db.connect() as c:
        c.execute(f"UPDATE lab_runs SET {sets} WHERE id = ?", (*vals, run_id))


def delete(run_id: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM lab_runs WHERE id = ? AND status NOT IN ('queued', 'running')", (run_id,))


def cancel(run_id: int) -> None:
    _cancel.add(run_id)


def _model(name) -> str:
    n = Path(str(name or "")).name
    if not n.endswith(".gguf") or not (lab.MODELS_DIR / n).is_file():
        raise ValueError(f"Model not found: {n or '(none)'}")
    return n


def create(kind: str, body: dict) -> int:
    if kind == "throughput":
        users = sorted({max(1, min(64, int(u))) for u in (body.get("users") or [1, 2, 4, 8, 16])})
        params = {"model": _model(body.get("model")), "users": users,
                  "prompt": max(16, min(8192, int(body.get("prompt") or 512))), "gen": max(16, min(2048, int(body.get("gen") or 128)))}
    elif kind == "speculative":
        target, draft = _model(body.get("model")), _model(body.get("draft"))
        if target == draft:
            raise ValueError("Pick a smaller model of the same family as the draft")
        t, d = gguf.metadata(lab.MODELS_DIR / target), gguf.metadata(lab.MODELS_DIR / draft)
        if t.get("architecture") != d.get("architecture"):
            raise ValueError("The draft model must be from the same family (same vocabulary) as the main model")
        params = {"model": target, "draft": draft, "draft_max": max(2, min(32, int(body.get("draft_max") or 16)))}
    else:
        raise ValueError("Unknown tool")
    with db.connect() as c:
        return c.execute("INSERT INTO lab_runs (created_at, kind, status, params, results, log) VALUES (?, ?, 'queued', ?, '[]', '')",
                         (time.time(), kind, json.dumps(params))).lastrowid


async def run(run_id: int, llm: httpx.AsyncClient) -> None:
    with db.connect() as c:
        rec = _row(c.execute("SELECT * FROM lab_runs WHERE id = ?", (run_id,)).fetchone())
    status, error, results, log = "done", None, [], []
    async with lab._lock:
        lab.busy = "throughput" if rec["kind"] == "throughput" else "quantlab"
        _save(run_id, status="running")
        try:
            await lab.llm_service("stop")
            if rec["kind"] == "throughput":
                results = await _throughput(run_id, rec["params"], log)
            else:
                results = await _speculative(run_id, rec["params"], log)
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001
            status, error = "error", str(e)[:400]
        finally:
            _save(run_id, status=status, error=error, results=results, log="\n".join(log[-60:]))
            _cancel.discard(run_id)
            try:
                await lab.llm_service("start")
                await lab.wait_ready(llm, timeout=300)
            except Exception:  # noqa: BLE001
                pass
            lab.busy = None


async def _throughput(run_id: int, p: dict, log: list[str]) -> list[dict]:
    users = p["users"]
    ctx = (p["prompt"] + p["gen"]) * max(users) + 256
    cmd = [str(lab.LLAMA_BIN / "llama-batched-bench"), "-m", str(lab.MODELS_DIR / p["model"]), "-c", str(ctx),
           "-b", "2048", "-ub", "512", "-npp", str(p["prompt"]), "-ntg", str(p["gen"]), "-npl", ",".join(map(str, users)),
           "-fit", "on", "-fa", "auto", "-ctk", "q8_0", "-ctv", "q8_0"]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                env={**os.environ, "LD_LIBRARY_PATH": str(lab.LLAMA_BIN)})
    samples: list = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(lab._sample_gpu(samples, stop))
    rows = []
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line.strip():
                log.append(line[-200:])
            if run_id in _cancel:
                proc.kill()
                raise asyncio.CancelledError
            # | PP | TG | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 10 and cells[0].isdigit():
                pp, tg, b = int(cells[0]), int(cells[1]), int(cells[2])
                s_tg = float(cells[7])
                rows.append({"users": b, "prompt": pp, "gen": tg, "prompt_tps": float(cells[5]),
                             "total_gen_tps": s_tg, "per_user_tps": s_tg / b, "total_s": float(cells[8])})
                _save(run_id, results=rows, log="\n".join(log[-40:]))
        await proc.wait()
    finally:
        stop.set()
        await sampler
    if not rows:
        raise RuntimeError(log[-1] if log else "llama-batched-bench produced no results")
    watts = [s[1] for s in samples if s[1] and (s[2] or 0) >= 20]
    if watts:
        for r in rows:
            r["power_w"] = round(sum(watts) / len(watts), 1)
    return rows


async def _speculative(run_id: int, p: dict, log: list[str]) -> list[dict]:
    """Same prompts, greedy, with and without the draft model; compare real tokens/second."""
    results = []
    for mode in ("without draft", "with draft"):
        cmd = [str(lab.LLAMA_BIN / "llama-server"), "-m", str(lab.MODELS_DIR / p["model"]), "--host", "127.0.0.1",
               "--port", str(arena.TEMP_PORT), "-c", "4096", "-np", "1", "--fit-target", str(lab.fit_margin_mb()), "-fa", "auto", "--jinja", "--no-webui"]
        if mode == "with draft":
            cmd += ["-md", str(lab.MODELS_DIR / p["draft"]), "-ngld", "999", "--spec-draft-n-max", str(p["draft_max"]), "--spec-draft-n-min", "1"]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                                                    env={**os.environ, "LD_LIBRARY_PATH": str(lab.LLAMA_BIN)})
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{arena.TEMP_PORT}", timeout=httpx.Timeout(10, read=300)) as c:
                if not await lab.wait_ready(c, timeout=240):
                    raise RuntimeError(f"Couldn't start the model {mode} (does it fit in memory?)")
                tokens = secs = 0.0
                accepted = drafted = 0
                for prompt in SPEC_PROMPTS:
                    if run_id in _cancel:
                        raise asyncio.CancelledError
                    t0 = time.monotonic()
                    r = await c.post("/v1/chat/completions", json={
                        "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0, "seed": 1,
                        "chat_template_kwargs": {"enable_thinking": False}})
                    r.raise_for_status()
                    d = r.json()
                    t = d.get("timings") or {}
                    tokens += t.get("predicted_n") or 0
                    secs += (t.get("predicted_ms") or (time.monotonic() - t0) * 1000) / 1000
                    accepted += t.get("draft_n_accepted") or 0
                    drafted += t.get("draft_n") or 0
                    log.append(f"{mode}: {t.get('predicted_n')} tokens at {t.get('predicted_per_second', 0):.1f} tok/s")
                res = {"mode": mode, "tokens": int(tokens), "tps": tokens / secs if secs else None}
                if drafted:
                    res["acceptance"] = accepted / drafted
                results.append(res)
                _save(run_id, results=results, log="\n".join(log[-40:]))
        finally:
            await arena._stop(proc)
    if len(results) == 2 and results[0]["tps"] and results[1]["tps"]:
        results[1]["speedup"] = results[1]["tps"] / results[0]["tps"]
    return results
