"""blockyOS web backend.

One process that:
  - serves the web UI
  - streams chat to/from llama-server
  - stores conversations and long-term memory in SQLite
  - reports hardware stats
"""
import asyncio
import csv
import io
import json
import re
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import arena, benchmarks, config, db, engines, evals, generate, lab, mcp_server, memory, power, profile, quantlab, remote, scans, system, tools, torchlab, weights


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    arena.init()
    evals.init()
    quantlab.init()
    tools.init()
    profile.init()
    weights.init()
    engines.init()
    scans.init()
    torchlab.init()
    remote.init()
    psutil.cpu_percent(interval=None)  # prime the CPU counter so the first reading isn't 0
    app.state.llm = httpx.AsyncClient(base_url=config.LLM_URL, timeout=httpx.Timeout(10, read=300))
    app.state.background = set()  # keep references so tasks aren't garbage-collected
    worker = asyncio.create_task(profile.worker(app.state.llm))
    app.state.background.add(worker)
    async with mcp_server.server.session_manager.run():
        yield
    worker.cancel()
    await app.state.llm.aclose()


app = FastAPI(title="blockyOS", lifespan=lifespan)


class LocalOnly:
    """Only this device may use Blocky, unless its owner turned on remote access (then the token is required).

    Plain ASGI (not BaseHTTPMiddleware) so streaming responses pass through untouched. A browser on another device can
    open http://<box>:8000/?key=<token> once; the key is then kept in a cookie.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        from http.cookies import SimpleCookie
        from urllib.parse import parse_qs
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = (scope.get("client") or ("",))[0]
        key = (parse_qs(scope.get("query_string", b"").decode("latin-1")).get("key") or [""])[0]
        jar = SimpleCookie(headers.get("cookie", ""))
        cookie = jar["blocky_key"].value if "blocky_key" in jar else ""
        if remote.allowed(host, headers.get("authorization", ""), cookie or key):
            if key and host not in ("127.0.0.1", "::1"):
                async def send_with_cookie(msg):
                    if msg["type"] == "http.response.start":
                        msg = {**msg, "headers": [*msg.get("headers", []),
                                                  (b"set-cookie", f"blocky_key={key}; HttpOnly; SameSite=Strict; Max-Age=2592000; Path=/".encode())]}
                    await send(msg)
                return await self.app(scope, receive, send_with_cookie)
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        resp = JSONResponse({"detail": "This Blocky only accepts connections from itself. To use it from another device, turn on "
                                       "remote access in its Settings, or connect through an SSH tunnel."}, status_code=403)
        await resp(scope, receive, send)


app.add_middleware(LocalOnly)


# ---------- Conversations ----------

@app.get("/api/conversations")
def list_conversations():
    return db.list_conversations()


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: int):
    if not db.conversation_exists(conv_id):
        raise HTTPException(404, "Conversation not found")
    return db.get_messages(conv_id)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)


@app.patch("/api/conversations/{conv_id}")
def rename_conversation(conv_id: int, body: RenameIn):
    db.rename_conversation(conv_id, body.title.strip())
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: int):
    db.delete_conversation(conv_id)
    return {"ok": True}


@app.get("/api/search")
def search(q: str):
    return db.search_messages(q)


# ---------- Memory ----------

class MemoryIn(BaseModel):
    content: str = Field(min_length=1, max_length=memory.MAX_MEMORY_CHARS)


@app.get("/api/memories")
def list_memories():
    return db.list_memories()


@app.post("/api/memories")
def add_memory(body: MemoryIn):
    return db.add_memory(body.content.strip(), "user")


@app.patch("/api/memories/{mem_id}")
def update_memory(mem_id: int, body: MemoryIn):
    if not db.update_memory(mem_id, body.content.strip()):
        raise HTTPException(404, "Memory not found")
    return {"ok": True}


@app.delete("/api/memories/{mem_id}")
def delete_memory(mem_id: int):
    db.delete_memory(mem_id)
    return {"ok": True}


@app.delete("/api/memories")
def delete_all_memories():
    db.delete_all_memories()
    return {"ok": True}


# ---------- Settings ----------

@app.get("/api/settings")
def get_settings():
    return db.get_settings()


@app.patch("/api/settings")
def update_settings(body: dict):
    body.pop("remote_token", None)  # only POST /api/remote-access creates or clears it
    try:
        for k in ("device_price", "electricity_price_kwh"):
            if k in body and body[k] not in (None, ""):
                body[k] = float(body[k])
                if body[k] < 0:
                    raise ValueError(k)
            elif k in body:
                body[k] = None
        if "power_meter_url" in body:
            power.meter_config({"power_meter_url": body["power_meter_url"]})
        for k in ("power_meter_field", "currency"):
            if k in body:
                body[k] = str(body[k] or "")[:40]
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e) if "URL" in str(e) else "Prices must be non-negative numbers")
    return db.update_settings(body)


@app.get("/api/device-presets")
def device_presets():
    import os
    return {"presets": profile.DEVICE_PRESETS, "cpu_threads": os.cpu_count()}


# ---------- Remote devices ----------

@app.get("/api/remote-access")
def remote_access_get():
    return {"enabled": bool(remote.access_token()), "token": remote.access_token() or None}


@app.post("/api/remote-access")
def remote_access_set(body: dict):
    return {"enabled": bool(body.get("enabled")), "token": remote.set_access(bool(body.get("enabled"))) or None}


@app.get("/api/devices")
def devices_list():
    return remote.list_devices()


@app.post("/api/devices")
def devices_add(body: dict):
    try:
        return {"id": remote.add_device(body)}
    except ValueError as e:
        _bad(e)


@app.delete("/api/devices/{dev_id}")
def devices_delete(dev_id: int):
    remote.delete_device(dev_id)
    return {"ok": True}


@app.get("/api/devices/{dev_id}/check")
async def devices_check(dev_id: int):
    try:
        return await remote.check(dev_id)
    except ValueError as e:
        _bad(e)


@app.post("/api/experiments/{exp_id}/remote")
async def experiments_remote(exp_id: int, body: dict):
    try:
        return {"job_id": remote.start(exp_id, body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@app.get("/api/remote-jobs")
def remote_jobs(experiment_id: int | None = None):
    return remote.list_jobs(experiment_id)


@app.post("/api/remote-jobs/{job_id}/cancel")
def remote_jobs_cancel(job_id: int):
    remote.cancel(job_id)
    return {"ok": True}


@app.post("/api/power/test")
async def power_test(body: dict):
    """Read the wall-power meter once (to check the address before running experiments)."""
    try:
        meter = power.meter_config(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not meter:
        raise HTTPException(400, "Enter the meter's address")
    async with httpx.AsyncClient() as c:
        w = await power.read_wall(c, *meter)
    if w is None:
        raise HTTPException(502, "No power reading from that address (check the URL, or set the field that holds the watts)")
    return {"watts": w}


# ---------- Chat ----------

class Sampling(BaseModel):
    """Generation parameters a researcher may want to control. None = model/server default."""
    temperature: float | None = Field(None, ge=0, le=5)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=0, le=1000)
    min_p: float | None = Field(None, ge=0, le=1)
    repeat_penalty: float | None = Field(None, ge=0.5, le=3)
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    max_tokens: int | None = Field(None, ge=1, le=262_144)
    seed: int | None = Field(None, ge=-1)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=50_000)
    conversation_id: int | None = None
    think: bool = False  # let the model "reason" first (slower, sometimes smarter)
    sampling: Sampling | None = None
    system_prompt: str | None = Field(None, max_length=20_000)  # replaces the default persona
    think_budget: int | None = Field(4096, ge=64, le=65_536)  # max thinking tokens before forcing an answer
    confidence: bool = False  # stream per-word probabilities (for the confidence view)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    app.state.background.add(task)
    task.add_done_callback(app.state.background.discard)


_token_cache: dict[str, int] = {}


async def _count_tokens(text: str) -> int | None:
    """Token count via llama-server's tokenizer (cached; the system prompt rarely changes)."""
    if text in _token_cache:
        return _token_cache[text]
    try:
        r = await app.state.llm.post("/tokenize", json={"content": text}, timeout=3)
        n = len(r.json().get("tokens", []))
    except (httpx.HTTPError, ValueError):
        return None
    if len(_token_cache) > 64:
        _token_cache.clear()
    _token_cache[text] = n
    return n


def _system_prompt(user_message: str, use_memory: bool, base: str | None = None) -> str:
    prompt = (base.strip() if base and base.strip() else config.SYSTEM_PROMPT) + (
        f"\n\nCurrent date (for your reference; only mention it if asked or relevant): "
        f"{datetime.now():%A, %B %d, %Y}."
    )
    if use_memory:
        prompt += memory.format_for_prompt(memory.recall(user_message))
    return prompt


@app.post("/api/chat")
async def chat(body: ChatIn):
    conv_id = body.conversation_id
    if conv_id is None:
        conv_id = db.create_conversation(body.message.strip()[:60] or "New chat")
    elif not db.conversation_exists(conv_id):
        raise HTTPException(404, "Conversation not found")

    settings = db.get_settings()
    use_memory = settings["memory_enabled"]

    db.add_message(conv_id, "user", body.message)
    history = db.get_messages(conv_id, limit=config.MAX_HISTORY_MESSAGES)
    system_text = _system_prompt(body.message, use_memory, body.system_prompt)
    system_n = await _count_tokens(system_text)
    messages = [{"role": "system", "content": system_text}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]
    sampling = body.sampling.model_dump(exclude_none=True) if body.sampling else {}
    # No explicit settings: use the model's recommended ones (makes small models' thinking far less erratic).
    used_sampling = sampling or generate.recommended_sampling(lab.read_runtime().get("model"), body.think)
    budget = body.think_budget if body.think else None

    async def stream():
        yield _sse({"type": "meta", "conversation_id": conv_id})
        if lab.busy:
            msg = {"benchmark": "A benchmark is running. Chat is paused until it finishes.",
                   "arena": "An Arena match is running. Chat is paused until it finishes.",
                   "eval": "A quality test is running. Chat is paused until it finishes.",
                   "quantlab": "The quantization lab is running. Chat is paused until it finishes.",
                   "throughput": "A throughput test is running. Chat is paused until it finishes.",
                   "experiment": "An experiment profile is running. Chat is paused until it finishes.",
                   "weights": "New model weights are being built. Chat is paused until it finishes.",
                   "engine": "An inference engine is being compiled. Chat is paused until it finishes.",
                   "scan": "A layer sensitivity scan is running. Chat is paused until it finishes.",
                   "torch": "A PyTorch job (activations or fine-tuning) is using the GPU. Chat is paused until it finishes."}.get(
                lab.busy, "The model is being switched. Try again in a moment.")
            yield _sse({"type": "error", "message": msg})
            return
        content, reasoning = [], []
        timings: dict = {}
        started = time.monotonic()
        first_token: float | None = None
        finished = False
        budget_hit = False

        def stats() -> dict:
            s = {
                "ttft_ms": round((first_token - started) * 1000) if first_token else None,
                "budget_hit": budget_hit, "think_budget": budget,
                "total_ms": round((time.monotonic() - started) * 1000),
                "prompt_n": timings.get("prompt_n"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "cache_n": timings.get("cache_n"),
                "predicted_n": timings.get("predicted_n"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "think": body.think,
                "system_n": system_n,  # system prompt + memories (for the context meter)
            }
            if sampling:
                s["sampling"] = sampling
            elif used_sampling:
                s["sampling_recommended"] = used_sampling
            if body.system_prompt:
                s["custom_system_prompt"] = True
            return s

        try:
            extra = {"logprobs": True, "top_logprobs": 5} if body.confidence else None
            async for kind, val in generate.stream(app.state.llm, messages, think=body.think, sampling=used_sampling,
                                                   budget=budget, answer_tokens=4096, extra=extra):
                if kind == "probs":
                    yield _sse({"type": "probs", "items": val})
                    continue
                if kind in ("reasoning", "content") and first_token is None:
                    first_token = time.monotonic()
                if kind == "reasoning":
                    reasoning.append(val)
                    yield _sse({"type": "reasoning", "text": val})
                elif kind == "content":
                    content.append(val)
                    yield _sse({"type": "token", "text": val})
                elif kind == "budget":
                    budget_hit = True
                    yield _sse({"type": "budget", "tokens": val})
                elif kind == "done":
                    timings = val["timings"] or {}
            finished = True
        except httpx.ConnectError:
            yield _sse({"type": "error", "message": "The AI model isn't running. Is blocky-llm started?"})
            return
        except httpx.HTTPStatusError as e:
            msg = "The model is still loading. Try again in a few seconds." if "503" in str(e) else str(e)[:300]
            yield _sse({"type": "error", "message": msg})
            return
        except httpx.HTTPError as e:
            yield _sse({"type": "error", "message": f"Lost connection to the model: {e.__class__.__name__}"})
            return
        finally:
            # Save whatever we got, even if the user navigated away mid-answer.
            if content or reasoning:
                db.add_message(
                    conv_id, "assistant", "".join(content),
                    reasoning="".join(reasoning) or None,
                    tokens=timings.get("predicted_n"),
                    tokens_per_sec=timings.get("predicted_per_second"),
                    stats=stats(),
                )

        yield _sse({
            "type": "done",
            "tokens": timings.get("predicted_n"),
            "tokens_per_sec": timings.get("predicted_per_second"),
            "seconds": round(time.monotonic() - started, 2),
            "stats": stats(),
        })

        # Learn from this exchange. Runs as its own task so it completes even if
        # the browser disconnects; we just stop waiting for it in that case.
        if finished and use_memory and content and not lab.busy:
            task = asyncio.create_task(
                memory.extract(app.state.llm, body.message, "".join(content), conv_id)
            )
            app.state.background.add(task)
            task.add_done_callback(app.state.background.discard)
            added = await asyncio.shield(task)
            if added:
                yield _sse({"type": "memory", "added": added})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------- System ----------

@app.get("/api/system")
def system_stats():
    return system.snapshot()


@app.get("/api/status")
async def status():
    """Is the model loaded, and which one? Also reports Lab activity."""
    base = {"busy": lab.busy, "switch_error": lab.last_switch_error}
    try:
        r = await app.state.llm.get("/props", timeout=2)
        props = r.json()
        model_path = props.get("model_path", "")
        return {
            **base,
            "model_ready": r.status_code == 200,
            "model": Path(model_path).stem if model_path else "unknown",
            "context_size": (props.get("default_generation_settings") or {}).get("n_ctx"),
        }
    except (httpx.HTTPError, ValueError):
        return {**base, "model_ready": False, "model": None, "context_size": None}


# ---------- Lab: models ----------

def _bad(e: Exception):
    raise HTTPException(400, str(e))


@app.get("/api/hardware")
def hardware():
    return lab.hardware()


@app.get("/api/memory-map")
async def memory_map():
    mm = await lab.memory_map()
    await lab.refresh_measured()
    return mm


@app.get("/api/stats/recent")
def recent_stats(limit: int = 30):
    with db.connect() as c:
        rows = c.execute(
            "SELECT m.id, m.created_at, m.stats, m.conversation_id FROM messages m "
            "WHERE m.role = 'assistant' AND m.stats IS NOT NULL ORDER BY m.id DESC LIMIT ?",
            (max(1, min(500, limit)),),
        ).fetchall()
    return [{"id": r["id"], "created_at": r["created_at"], "conversation_id": r["conversation_id"],
             **json.loads(r["stats"])} for r in reversed(rows)]


@app.get("/api/models")
async def models():
    await lab.refresh_measured()
    return {"models": lab.list_models(), "runtime": lab.read_runtime(), "catalog": lab.catalog(),
            "hardware": lab.hardware(), "kv_types": lab.KV_TYPES}


@app.get("/api/models/{name}/anatomy")
def model_anatomy(name: str):
    from . import gguf
    path = lab.MODELS_DIR / Path(name).name
    if not path.is_file() or path.suffix != ".gguf":
        raise HTTPException(404, "Model not found")
    try:
        return gguf.anatomy(path)
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"Couldn't read the model file: {e}")


@app.post("/api/models/runtime")
async def set_runtime(body: dict):
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        cfg = lab.validate_runtime(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(lab.apply_runtime(app.state.llm, cfg))
    return {"ok": True, "runtime": cfg}


@app.delete("/api/models/{name}")
def delete_model(name: str):
    try:
        lab.delete_model(name)
    except ValueError as e:
        _bad(e)
    return {"ok": True}


@app.get("/api/hf/files")
async def hf_files(repo: str):
    try:
        return await lab.hf_files(repo.strip())
    except ValueError as e:
        _bad(e)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Couldn't reach Hugging Face: {e.__class__.__name__}")


class DownloadIn(BaseModel):
    repo: str
    path: str


@app.post("/api/downloads")
async def start_download(body: DownloadIn):  # async: starts a background task on the event loop
    try:
        return {"id": lab.start_download(body.repo.strip(), body.path.strip())}
    except ValueError as e:
        _bad(e)


@app.get("/api/downloads")
def list_downloads():
    return sorted(lab.downloads.values(), key=lambda d: d["started"], reverse=True)


@app.post("/api/downloads/{dl_id}/cancel")
def cancel_download(dl_id: str):
    if dl_id in lab.downloads and lab.downloads[dl_id]["status"] == "downloading":
        lab.downloads[dl_id]["status"] = "cancelled"
    return {"ok": True}


# ---------- Lab: benchmarks ----------

@app.get("/api/bench")
def list_bench():
    return lab.list_bench()


@app.post("/api/bench")
async def start_bench(body: dict):  # async: starts a background task on the event loop
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        run_id = lab.create_bench(body)
    except (ValueError, TypeError, KeyError) as e:
        _bad(e)
    _spawn(lab.run_bench(run_id, app.state.llm))
    return {"id": run_id}


@app.get("/api/bench/export")
def export_bench(format: str = "csv"):
    runs = [r for r in lab.list_bench() if r["results"]]
    if format == "json":
        return JSONResponse(runs, headers={"Content-Disposition": 'attachment; filename="blocky-benchmarks.json"'})
    cols = ["run_id", "date", "model", "test", "n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts", "power_w",
            "tokens_per_joule", "peak_vram_mb", "n_gpu_layers", "n_cpu_moe", "type_k", "flash_attn", "n_threads",
            "gpu", "driver", "cpu", "ram_mb", "os", "wsl", "llama_cpp"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for run in runs:
        hw = run["hardware"] or {}
        gpu = (hw.get("gpus") or [{}])[0]
        for r in run["results"]:
            w.writerow([run["id"], datetime.fromtimestamp(run["created_at"]).isoformat(timespec="seconds"), run["model"],
                        r.get("test"), r.get("n_prompt"), r.get("n_gen"), r.get("n_depth"), r.get("avg_ts"), r.get("stddev_ts"),
                        r.get("power_w"), r.get("tokens_per_joule"), r.get("peak_vram_mb"), r.get("n_gpu_layers"),
                        r.get("n_cpu_moe"), r.get("type_k"), r.get("flash_attn"), r.get("n_threads"),
                        gpu.get("name"), gpu.get("driver"), hw.get("cpu"), round(hw.get("ram_mb") or 0),
                        hw.get("os"), hw.get("wsl"), hw.get("llama_cpp")])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="blocky-benchmarks.csv"'})


@app.get("/api/bench/{run_id}")
def get_bench(run_id: int):
    run = lab.get_bench(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.post("/api/bench/{run_id}/cancel")
def cancel_bench(run_id: int):
    lab.cancel_bench(run_id)
    return {"ok": True}


@app.delete("/api/bench/{run_id}")
def delete_bench(run_id: int):
    lab.delete_bench(run_id)
    return {"ok": True}


# ---------- Lab: established benchmarks ----------

@app.get("/api/benchmarks")
def bench_catalog():
    return benchmarks.summary()


@app.post("/api/benchmarks/{key}/download")
async def bench_download(key: str):  # async: starts a background task on the event loop
    if key not in benchmarks.catalog():
        raise HTTPException(404, "Unknown benchmark")
    if (benchmarks.downloads.get(key) or {}).get("status") == "downloading":
        return {"ok": True}
    _spawn(benchmarks.download(key))
    return {"ok": True}


@app.delete("/api/benchmarks/{key}")
def bench_delete(key: str):
    benchmarks.delete(key)
    return {"ok": True}


@app.get("/api/hf/dataset")
async def hf_dataset_preview(repo: str):
    try:
        return await benchmarks.hf_preview(repo.strip())
    except ValueError as e:
        _bad(e)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Couldn't reach Hugging Face: {e.__class__.__name__}")


@app.post("/api/benchmarks/custom")
async def bench_add_custom(body: dict):  # async: starts the download task
    try:
        key = benchmarks.add_custom(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(benchmarks.download(key))
    return {"key": key}


# ---------- OpenAI-compatible API (for outside tools: lm-eval, EvalPlus, SWE-bench agents…) ----------

@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def openai_proxy(path: str, request: Request):
    """Pass-through to llama-server's OpenAI-compatible API, so external harnesses can test the local model."""
    if path not in ("models", "chat/completions", "completions", "embeddings"):
        raise HTTPException(404, "Not an OpenAI endpoint Blocky exposes")
    if lab.busy:
        raise HTTPException(503, f"Blocky is busy ({lab.busy}). Try again when it's finished.")
    body = await request.body()
    req = app.state.llm.build_request(request.method, f"/v1/{path}", content=body,
                                      headers={"Content-Type": request.headers.get("content-type", "application/json")},
                                      timeout=httpx.Timeout(10, read=900))
    upstream = await app.state.llm.send(req, stream=True)

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code,
                             media_type=upstream.headers.get("content-type", "application/json"))


# ---------- Lab: quality tests ----------

@app.get("/api/evals/suites")
def eval_suites():
    return evals.suites_summary()


@app.get("/api/evals/fresh")
def eval_fresh_sets():
    from . import fresh
    return [{"key": f"fresh:{k}", "id": k, **v} for k, v in fresh.SETS.items()]


@app.get("/api/evals/fresh/{kind}/preview")
def eval_fresh_preview(kind: str, n: int = 3):
    """A few freshly generated example questions (new every call), so users can see what they look like."""
    from . import fresh
    if kind not in fresh.SETS:
        raise HTTPException(404, "Unknown fresh set")
    return [{"q": q["q"], "kind": q["kind"]} for q in fresh.generate(kind, max(1, min(10, n)), random.randint(1, 2**31))]


@app.get("/api/evals/suites/{key}")
def eval_suite(key: str):
    s = evals.all_suites().get(key)
    if not s:
        raise HTTPException(404, "Test set not found")
    return s


class CustomSuiteIn(BaseModel):
    name: str
    questions: list[dict]


@app.post("/api/evals/suites")
def eval_create_suite(body: CustomSuiteIn):
    try:
        return {"key": f"custom:{evals.create_custom(body.name, body.questions)}"}
    except ValueError as e:
        _bad(e)


@app.delete("/api/evals/suites/custom:{suite_id}")
def eval_delete_suite(suite_id: int):
    evals.delete_custom(suite_id)
    return {"ok": True}


@app.get("/api/evals")
def eval_list():
    return evals.list_runs()


@app.post("/api/evals")
async def eval_start(body: dict):  # async: starts a background task on the event loop
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        run_id = evals.create(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(evals.run(run_id, app.state.llm))
    return {"id": run_id}


@app.get("/api/evals/{run_id}")
def eval_get(run_id: int):
    r = evals.get(run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    return r


@app.post("/api/evals/{run_id}/cancel")
def eval_cancel(run_id: int):
    evals.cancel(run_id)
    return {"ok": True}


@app.delete("/api/evals/{run_id}")
def eval_delete(run_id: int):
    evals.delete(run_id)
    return {"ok": True}


# ---------- Lab: quantization ----------

@app.get("/api/quantlab")
def quant_runs():
    return {"runs": quantlab.list_runs(), "jobs": sorted(quantlab.jobs.values(), key=lambda j: j["started"], reverse=True),
            "types": quantlab.QUANT_TYPES}


@app.post("/api/quantlab")
async def quant_start(body: dict):  # async: starts a background task
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        run_id = quantlab.create(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(quantlab.run(run_id, app.state.llm))
    return {"id": run_id}


@app.post("/api/quantlab/{run_id}/cancel")
def quant_cancel(run_id: int):
    quantlab.cancel(run_id)
    return {"ok": True}


@app.delete("/api/quantlab/{run_id}")
def quant_delete(run_id: int):
    quantlab.delete(run_id)
    return {"ok": True}


class QuantizeIn(BaseModel):
    source: str
    type: str


@app.post("/api/quantlab/quantize")
async def quant_make(body: QuantizeIn):  # async: starts a background task
    try:
        return {"id": quantlab.start_quantize(body.source, body.type)}
    except ValueError as e:
        _bad(e)


# ---------- Experiments (research workflow) ----------

@app.get("/api/experiments")
def experiments_list():
    return profile.list_experiments()


class ExperimentIn(BaseModel):
    name: str = ""
    model: str


@app.post("/api/experiments")
def experiments_create(body: ExperimentIn):
    try:
        return {"id": profile.create_experiment(body.name, body.model)}
    except ValueError as e:
        _bad(e)


@app.get("/api/experiments/{exp_id}")
def experiments_get(exp_id: int):
    e = profile.get_experiment(exp_id)
    if not e:
        raise HTTPException(404, "Experiment not found")
    return e


@app.get("/api/experiments/{exp_id}/export.{fmt}")
def experiments_export(exp_id: int, fmt: str):
    e = profile.get_experiment(exp_id)
    if not e:
        raise HTTPException(404, "Experiment not found")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", e["name"]).strip("-") or f"experiment-{exp_id}"
    if fmt == "csv":
        return Response(profile.export_csv(e), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
    if fmt == "json":
        return Response(json.dumps(e, indent=1), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
    raise HTTPException(400, "Use csv or json")


@app.post("/api/experiments/import")
async def experiments_import(request: Request, into: int | None = None):
    """Import an experiment exported as JSON from another Blocky device (new experiment, or merged into `into`)."""
    try:
        data = json.loads(await request.body())
        return profile.import_experiment(data, into)
    except (ValueError, TypeError, KeyError) as e:
        _bad(e)


@app.delete("/api/experiments/{exp_id}")
def experiments_delete(exp_id: int):
    profile.delete_experiment(exp_id)
    return {"ok": True}


@app.post("/api/experiments/{exp_id}/runs")
def experiments_run(exp_id: int, body: dict):
    """Queues a profile; the experiments worker runs queued profiles in order."""
    try:
        run_id = profile.create_run(exp_id, body)
    except (ValueError, TypeError) as e:
        _bad(e)
    return {"id": run_id}


@app.post("/api/experiments/{exp_id}/sweep")
def experiments_sweep(exp_id: int, body: dict):
    try:
        ids = profile.create_sweep(exp_id, body)
    except (ValueError, TypeError) as e:
        _bad(e)
    return {"ids": ids}


# ---------- Inference engines (llama.cpp builds) ----------

@app.get("/api/engines")
def engines_list():
    return engines.list_engines()


@app.post("/api/engines")
async def engines_add(body: dict):  # async: starts a background build
    try:
        return {"id": engines.add(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@app.get("/api/engines/{engine_id}/log")
def engines_log(engine_id: str):
    return {"log": engines.get_log(engine_id)}


@app.post("/api/engines/{engine_id}/cancel")
def engines_cancel(engine_id: str):
    engines.cancel(engine_id)
    return {"ok": True}


@app.delete("/api/engines/{engine_id}")
def engines_delete(engine_id: str):
    try:
        engines.delete(engine_id)
    except ValueError as e:
        _bad(e)
    return {"ok": True}


# ---------- PyTorch lab: checkpoints, activations, LoRA ----------

@app.post("/api/torch/quantsim")
async def torch_quantsim(body: dict):
    try:
        return {"id": torchlab.start_quantsim(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@app.get("/api/torch/status")
async def torch_status():
    return await asyncio.to_thread(torchlab.status)


@app.get("/api/torch/jobs")
def torch_jobs(kind: str | None = None):
    return torchlab.jobs(kind)


@app.get("/api/torch/jobs/{job_id}")
def torch_job(job_id: int):
    j = torchlab.job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


@app.post("/api/torch/jobs/{job_id}/cancel")
def torch_cancel(job_id: int):
    torchlab.cancel(job_id)
    return {"ok": True}


@app.post("/api/torch/download")
async def torch_download(body: dict):
    try:
        return {"id": torchlab.start_download(str(body.get("repo") or ""))}
    except (ValueError, TypeError) as e:
        _bad(e)


@app.post("/api/torch/activations")
async def torch_activations(body: dict):
    try:
        return {"id": torchlab.start_activations(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@app.post("/api/torch/lora")
async def torch_lora(body: dict):
    try:
        return {"id": torchlab.start_lora(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


# ---------- Notebooks (JupyterLab on the box) + Python client ----------

@app.get("/api/notebooks")
async def notebooks_status():
    token_file = config.DATA_DIR / "notebook-token"
    installed = token_file.is_file()
    running = False
    if installed:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                running = (await c.get("http://127.0.0.1:8888/api/status", params={"token": token_file.read_text().strip()})).status_code == 200
        except httpx.HTTPError:
            pass
    return {"installed": installed, "running": running,
            "url": f"http://localhost:8888/lab?token={token_file.read_text().strip()}" if installed else None,
            "folder": str(config.DATA_DIR / "notebooks")}


@app.get("/api/client.py")
def python_client():
    return FileResponse(Path(__file__).resolve().parents[2] / "client" / "blocky.py", media_type="text/x-python", filename="blocky.py")


# ---------- Layer sensitivity scans ----------

@app.get("/api/scans")
def scans_list(model: str | None = None):
    return scans.list_scans(model)


@app.post("/api/scans")
async def scans_start(body: dict):  # async: starts a background task
    try:
        return {"id": scans.start(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@app.post("/api/scans/{scan_id}/cancel")
def scans_cancel(scan_id: int):
    scans.cancel(scan_id)
    return {"ok": True}


@app.delete("/api/scans/{scan_id}")
def scans_delete(scan_id: int):
    scans.delete(scan_id)
    return {"ok": True}


# ---------- Weights workshop ----------

@app.get("/api/weights/family")
def weights_family(model: str):
    try:
        return {"files": weights.family(model), "base_types": weights.BASE_TYPES, "tensor_types": weights.TENSOR_TYPES,
                "needs_imatrix": sorted(weights.NEEDS_IMATRIX)}
    except (OSError, ValueError) as e:
        _bad(e)


@app.get("/api/weights/tensors")
def weights_tensors(model: str):
    try:
        return weights.tensor_groups(model)
    except (OSError, ValueError) as e:
        _bad(e)


@app.post("/api/weights/variants")
async def weights_create(body: dict):  # async: starts a background task
    try:
        return {"id": weights.start(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@app.get("/api/weights/jobs")
def weights_jobs():
    return weights.list_jobs()


@app.post("/api/weights/jobs/{job_id}/cancel")
def weights_cancel(job_id: str):
    weights.cancel(job_id)
    return {"ok": True}


@app.put("/api/experiments/{exp_id}/protocol")
def experiments_protocol(exp_id: int, body: dict):
    try:
        return profile.set_protocol(exp_id, body)
    except (ValueError, TypeError) as e:
        _bad(e)


@app.put("/api/experiments/{exp_id}/canvas")
def experiments_canvas(exp_id: int, body: dict):
    try:
        profile.save_canvas(exp_id, body)
    except ValueError as e:
        _bad(e)
    return {"ok": True}


@app.post("/api/experiments/{exp_id}/cancel-queue")
def experiments_cancel_queue(exp_id: int):
    return {"cancelled": profile.cancel_queue(exp_id)}


@app.post("/api/experiments/runs/{run_id}/cancel")
def experiments_cancel(run_id: int):
    profile.cancel(run_id)
    return {"ok": True}


@app.delete("/api/experiments/runs/{run_id}")
def experiments_delete_run(run_id: int):
    profile.delete_run(run_id)
    return {"ok": True}


@app.get("/api/hardware/limits")
def hardware_limits():
    return profile.limits()


# ---------- Lab: throughput & speculative decoding ----------

@app.get("/api/tools/{kind}")
def tool_runs(kind: str):
    return tools.list_runs(kind)


@app.post("/api/tools/{kind}")
async def tool_start(kind: str, body: dict):  # async: starts a background task
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        run_id = tools.create(kind, body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(tools.run(run_id, app.state.llm))
    return {"id": run_id}


@app.post("/api/tools/run/{run_id}/cancel")
def tool_cancel(run_id: int):
    tools.cancel(run_id)
    return {"ok": True}


@app.delete("/api/tools/run/{run_id}")
def tool_delete(run_id: int):
    tools.delete(run_id)
    return {"ok": True}


# ---------- Lab: arena ----------

@app.get("/api/arena")
def arena_list():
    return arena.list_matches()


@app.get("/api/arena/leaderboard")
def arena_leaderboard():
    return arena.leaderboard()


@app.post("/api/arena")
async def arena_start(body: dict):  # async: starts a background task on the event loop
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        match_id = arena.create(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(arena.run(match_id, app.state.llm))
    return {"id": match_id}


@app.get("/api/arena/{match_id}")
def arena_get(match_id: int):
    m = arena.get(match_id)
    if not m:
        raise HTTPException(404, "Match not found")
    return m


class VoteIn(BaseModel):
    winner: str


@app.post("/api/arena/{match_id}/vote")
def arena_vote(match_id: int, body: VoteIn):
    try:
        arena.vote(match_id, body.winner)
    except ValueError as e:
        _bad(e)
    return arena.get(match_id)


@app.post("/api/arena/{match_id}/cancel")
def arena_cancel(match_id: int):
    arena.cancel(match_id)
    return {"ok": True}


@app.delete("/api/arena/{match_id}")
def arena_delete(match_id: int):
    arena.delete(match_id)
    return {"ok": True}


@app.get("/api/logs")
async def logs(lines: int = 300):
    return {"text": await lab.llm_log(lines)}


# ---------- MCP: AI assistants (Claude Code, Claude Desktop, other MCP clients) drive the lab at /mcp ----------

# Stateless Streamable HTTP; Host/Origin checks against DNS rebinding come from the SDK (localhost only).
app.router.routes.extend(mcp_server.server.streamable_http_app(streamable_http_path="/mcp", stateless_http=True, json_response=True).routes)


@app.get("/api/mcp")
async def mcp_info():
    return {"url": "http://localhost:8000/mcp", "tools": [{"name": t.name, "description": t.description} for t in await mcp_server.server.list_tools()]}


# ---------- Web UI (must be last: catches all other paths) ----------

if config.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        file = config.FRONTEND_DIST / path
        if path and file.is_file() and config.FRONTEND_DIST in file.resolve().parents:
            return FileResponse(file)
        return FileResponse(config.FRONTEND_DIST / "index.html")
