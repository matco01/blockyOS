"""Arena: one prompt, two contestants (models or settings), answers side by side.

Contestants run one after another so each gets the GPU to itself: timings and
energy are fair. If every contestant uses the running chat model we use the
live server; otherwise we stop it, load each model in a temporary llama-server,
and restore chat afterwards. Same seed + neutral system prompt for both.
"""
import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path

import httpx

from . import db, generate, lab

log = logging.getLogger("blocky.arena")

TEMP_PORT = 8081
NEUTRAL_PROMPT = "You are a helpful assistant. Answer clearly and accurately. Use Markdown when it helps."
_cancel: set[int] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS arena_matches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   REAL NOT NULL,
    status       TEXT NOT NULL,
    prompt       TEXT NOT NULL,
    contestants  TEXT NOT NULL,
    results      TEXT NOT NULL,
    display      TEXT NOT NULL,
    blind        INTEGER NOT NULL DEFAULT 0,
    vote         TEXT,
    error        TEXT
);
"""


def init() -> None:
    with db.connect() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE arena_matches SET status = 'error', error = 'Interrupted' WHERE status IN ('queued', 'running')")


def _row(r) -> dict:
    d = dict(r)
    for k in ("contestants", "results", "display"):
        d[k] = json.loads(d[k])
    d["blind"] = bool(d["blind"])
    return d


def get(match_id: int) -> dict | None:
    with db.connect() as c:
        r = c.execute("SELECT * FROM arena_matches WHERE id = ?", (match_id,)).fetchone()
    return _row(r) if r else None


def list_matches(limit: int = 50) -> list[dict]:
    with db.connect() as c:
        rows = c.execute("SELECT * FROM arena_matches ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def _save(match_id: int, **fields) -> None:
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
    with db.connect() as c:
        c.execute(f"UPDATE arena_matches SET {sets} WHERE id = ?", (*vals, match_id))


SAMPLING_KEYS = {"temperature": (0, 5), "top_p": (0, 1), "top_k": (0, 1000), "min_p": (0, 1)}


def _clean(c: dict) -> dict:
    model = Path(str(c.get("model") or "")).name
    if not model.endswith(".gguf") or not (lab.MODELS_DIR / model).is_file():
        raise ValueError(f"Model not found: {model or '(none)'}")
    out = {"model": model, "think": bool(c.get("think", False)), "sampling": {}}
    for k, (lo, hi) in SAMPLING_KEYS.items():
        v = (c.get("sampling") or {}).get(k)
        if v is not None:
            v = float(v)
            if not lo <= v <= hi:
                raise ValueError(f"{k} must be between {lo} and {hi}")
            out["sampling"][k] = int(v) if k == "top_k" else v
    sp = (c.get("system_prompt") or "").strip()
    if sp:
        out["system_prompt"] = sp[:20_000]
    return out


def create(body: dict) -> int:
    prompt = (body.get("prompt") or "").strip()
    if not prompt or len(prompt) > 20_000:
        raise ValueError("Write a prompt (up to 20,000 characters)")
    raw = body.get("contestants") or []
    if len(raw) != 2:
        raise ValueError("Pick exactly two contestants")
    contestants = [_clean(c) for c in raw]
    max_tokens = int(body.get("max_tokens") or 0) or 2048
    seed = random.randint(1, 2**31 - 1)
    budget = int(body.get("think_budget") or 4096)
    for c in contestants:
        c["max_tokens"] = max(16, min(16384, max_tokens))
        c["seed"] = seed
        c["think_budget"] = max(64, min(65536, budget))
    blind = bool(body.get("blind", False))
    display = [0, 1]
    if blind:
        random.shuffle(display)
    results = [{"state": "waiting", "text": "", "reasoning": "", "stats": None, "error": None} for _ in contestants]
    with db.connect() as c:
        cur = c.execute(
            "INSERT INTO arena_matches (created_at, status, prompt, contestants, results, display, blind) VALUES (?, 'queued', ?, ?, ?, ?, ?)",
            (time.time(), prompt, json.dumps(contestants), json.dumps(results), json.dumps(display), int(blind)),
        )
        return cur.lastrowid


def vote(match_id: int, winner: str) -> None:
    if winner not in ("0", "1", "tie"):
        raise ValueError("winner must be 0, 1 or tie")
    with db.connect() as c:
        c.execute("UPDATE arena_matches SET vote = ? WHERE id = ? AND status = 'done'", (winner, match_id))


def delete(match_id: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM arena_matches WHERE id = ? AND status NOT IN ('queued', 'running')", (match_id,))


def cancel(match_id: int) -> None:
    _cancel.add(match_id)


def _label(c: dict) -> str:
    s = c["model"].replace(".gguf", "") + (" · think" if c["think"] else "")
    if c["sampling"].get("temperature") is not None:
        s += f" · t={c['sampling']['temperature']:g}"
    if c.get("system_prompt"):
        s += " · custom prompt"
    return s


def leaderboard() -> list[dict]:
    board: dict[str, dict] = {}
    for m in list_matches(1000):
        if m["status"] != "done" or not m["vote"]:
            continue
        for i, c in enumerate(m["contestants"]):
            e = board.setdefault(_label(c), {"label": _label(c), "model": c["model"], "matches": 0, "wins": 0, "ties": 0, "losses": 0, "tps": []})
            e["matches"] += 1
            if m["vote"] == "tie":
                e["ties"] += 1
            elif m["vote"] == str(i):
                e["wins"] += 1
            else:
                e["losses"] += 1
            tps = (m["results"][i].get("stats") or {}).get("predicted_per_second")
            if tps:
                e["tps"].append(tps)
    out = []
    for e in board.values():
        tps = e.pop("tps")
        e["avg_tps"] = sum(tps) / len(tps) if tps else None
        e["score"] = (e["wins"] + 0.5 * e["ties"]) / e["matches"]
        out.append(e)
    return sorted(out, key=lambda e: (-e["score"], -e["matches"]))


# ---------- Running ----------

async def _start_temp_server(model: str, kv: str) -> asyncio.subprocess.Process:
    cmd = [str(lab.LLAMA_BIN / "llama-server"), "-m", str(lab.MODELS_DIR / model), "--host", "127.0.0.1",
           "--port", str(TEMP_PORT), "-c", "8192", "--fit", "on", "-fa", "auto", "-ctk", kv, "-ctv", kv,
           "--jinja", "--no-webui", "-np", "1"]
    return await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                                                env={**os.environ, "LD_LIBRARY_PATH": str(lab.LLAMA_BIN)})


async def _stop(proc: asyncio.subprocess.Process | None) -> None:
    if proc and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 15)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


async def _generate(match_id: int, idx: int, c: dict, prompt: str, client: httpx.AsyncClient, results: list) -> None:
    r = results[idx]
    messages = [{"role": "system", "content": c.get("system_prompt") or NEUTRAL_PROMPT}, {"role": "user", "content": prompt}]
    # Same rules for both sides: explicit settings win, else the model's recommended ones.
    sampling = c["sampling"] or generate.recommended_sampling(c["model"], c["think"])
    samples: list = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(lab._sample_gpu(samples, stop))
    started = time.monotonic()
    info: dict = {}
    last_save = 0.0
    r["state"] = "reading"
    try:
        async for kind, val in generate.stream(client, messages, think=c["think"], sampling=sampling, seed=c["seed"],
                                               max_tokens=c["max_tokens"], budget=c.get("think_budget") if c["think"] else None,
                                               answer_tokens=2048, should_stop=lambda: match_id in _cancel):
            if match_id in _cancel:
                raise asyncio.CancelledError
            if kind == "reasoning":
                r["reasoning"] += val
                r["state"] = "thinking"
            elif kind == "content":
                r["text"] += val
                r["state"] = "writing"
            elif kind == "budget":
                r["budget_hit"] = True
            elif kind == "done":
                info = val
            now = time.monotonic()
            if now - last_save > 0.3:
                _save(match_id, results=results)
                last_save = now
    finally:
        stop.set()
        await sampler
    timings = info.get("timings") or {}
    total = time.monotonic() - started
    active = [s for s in samples if (s[2] or 0) >= 20] or samples
    power = sum(s[1] for s in active if s[1] is not None) / len(active) if active and active[0][1] is not None else None
    tokens = timings.get("predicted_n")
    r["stats"] = {
        "ttft_ms": info.get("ttft_ms"),
        "think_ms": info.get("think_ms"),
        "budget_hit": info.get("budget_hit", False),
        "total_ms": round(total * 1000),
        "prompt_n": timings.get("prompt_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": tokens,
        "predicted_per_second": timings.get("predicted_per_second"),
        "power_w": round(power, 1) if power else None,
        "energy_j": round(power * total, 1) if power else None,
        "tokens_per_joule": round(tokens / (power * total), 3) if power and tokens else None,
        "hit_limit": info.get("finish_reason") == "length",
    }
    r["state"] = "done"


async def run(match_id: int, llm: httpx.AsyncClient) -> None:
    m = get(match_id)
    contestants, results = m["contestants"], m["results"]
    rt = lab.read_runtime()
    use_live = all(c["model"] == rt["model"] for c in contestants)
    status, error = "done", None
    async with lab._lock:
        lab.busy = "arena"
        _save(match_id, status="running")
        proc = None
        stopped_main = False
        try:
            if use_live:
                for i, c in enumerate(contestants):
                    await _generate(match_id, i, c, m["prompt"], llm, results)
                    _save(match_id, results=results)
            else:
                await lab.llm_service("stop")
                stopped_main = True
                loaded = None
                async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{TEMP_PORT}", timeout=httpx.Timeout(10, read=600)) as client:
                    for i, c in enumerate(contestants):
                        if c["model"] != loaded:
                            await _stop(proc)
                            results[i]["state"] = "loading"
                            _save(match_id, results=results)
                            proc = await _start_temp_server(c["model"], rt["kv"])
                            ready = asyncio.create_task(lab.wait_ready(client, timeout=240))
                            died = asyncio.create_task(proc.wait())
                            done, _ = await asyncio.wait({ready, died}, return_when=asyncio.FIRST_COMPLETED)
                            ok = ready in done and ready.result()
                            for t in (ready, died):
                                if not t.done():
                                    t.cancel()
                            if not ok:
                                raise RuntimeError(f"Couldn't load {c['model']}. It may not fit in memory.")
                            loaded = c["model"]
                        await _generate(match_id, i, c, m["prompt"], client, results)
                        _save(match_id, results=results)
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001
            status, error = "error", str(e)[:400]
            log.warning("arena %s failed: %s", match_id, e)
        finally:
            await _stop(proc)
            for r in results:
                if r["state"] not in ("done",):
                    r["state"] = "skipped" if status != "done" else r["state"]
            _save(match_id, status=status, error=error, results=results)
            _cancel.discard(match_id)
            if stopped_main:
                try:
                    await lab.llm_service("start")
                    await lab.wait_ready(llm, timeout=300)
                except Exception as e:  # noqa: BLE001
                    log.error("couldn't restart chat model after arena: %s", e)
            lab.busy = None
