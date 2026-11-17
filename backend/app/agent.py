"""Agent: a chat inside Blocky that can use the whole lab.

The agent gets every MCP tool (experiments, runs, sweeps, weights, scans, engines, device presets, quantization
simulation) plus Python in its own workspace folder, and works in a loop: think, call tools, read results, continue,
until it answers. The person watches every step in the UI and approves the risky ones (running code).

Brains:
  local   the chat model on this device (llama-server, OpenAI-compatible tool calling). Private; weaker at long tasks.
  claude  Claude through the Anthropic API with the user's own key (set on this device). Much stronger, but messages
          and tool results are sent to Anthropic.
"""
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import httpx

from . import config, db, mcp_server

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    title      TEXT NOT NULL DEFAULT 'New chat',
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    messages   TEXT NOT NULL DEFAULT '[]',
    display    TEXT NOT NULL DEFAULT '[]'
);
"""
WORK = config.DATA_DIR / "agent"
PYTHONS = {"analysis": Path("/opt/blocky/notebook-venv/bin/python"), "torch": Path("/opt/blocky/ml-venv/bin/python")}
CLAUDE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
MAX_STEPS = 40
MAX_RESULT = 30_000
NEEDS_APPROVAL = {"run_python", "build_engine"}

_pending: dict[str, tuple[int, str, asyncio.Future]] = {}  # tool call id -> (session, tool name, the person's decision)
_always: dict[int, set[str]] = {}            # session -> tools approved for the rest of the chat
_stop: set[int] = set()
_busy: set[int] = set()

SYSTEM = """You are the research agent built into Blocky, a local-AI research lab running on the user's own hardware.
You help the user make local models faster, smaller and better on inexpensive devices, and invent and test new
techniques. You can use the whole lab through your tools, and run Python in your own workspace folder.

{lab}

Python (run_python): your working directory is your workspace (files persist across calls in this chat).
- env "analysis": pandas, numpy, matplotlib, and the Blocky client: `import blocky; b = blocky.Client();
  df = b.experiment("name").results()` gives one row per run with every metric.
- env "torch": PyTorch + CUDA, transformers, peft. Hugging Face checkpoints are in /opt/blocky/hf/<owner>--<name>.
  Don't use the GPU while a profile runs (get_status -> busy): it ruins both measurements.
- Print what you need to see; long outputs are cut. Save plots as PNG files in the workspace and mention their names.

How to work: look at existing data first; change one variable per run; respect noise (spreads, confidence intervals,
p-values); explain your plan briefly before queuing anything long (each run takes minutes, CPU-only runs much longer);
report numbers with their uncertainty and say plainly when a difference isn't significant. Never delete anything.
Answer in the user's language, concisely."""


def init() -> None:
    with db.connect() as c:
        c.executescript(SCHEMA)
    WORK.mkdir(parents=True, exist_ok=True)


# ---------- Settings ----------

def api_key() -> str:
    return db.get_settings().get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")


def status() -> dict:
    s = db.get_settings()
    return {"claude": bool(api_key()), "claude_model": s.get("agent_model") or CLAUDE_MODELS[0], "claude_models": CLAUDE_MODELS}


# ---------- Sessions ----------

def _row(r) -> dict:
    d = dict(r)
    d["messages"], d["display"] = json.loads(d["messages"]), json.loads(d["display"])
    return d


def list_sessions() -> list[dict]:
    with db.connect() as c:
        return [dict(r) for r in c.execute("SELECT id, created_at, updated_at, title, provider, model FROM agent_sessions ORDER BY updated_at DESC")]


def get_session(sid: int) -> dict | None:
    with db.connect() as c:
        r = c.execute("SELECT * FROM agent_sessions WHERE id = ?", (sid,)).fetchone()
    if not r:
        return None
    d = _row(r)
    d.pop("messages")
    d["running"] = sid in _busy
    return d


def create_session(provider: str) -> int:
    if provider not in ("local", "claude"):
        raise ValueError("Provider must be local or claude")
    if provider == "claude" and not api_key():
        raise ValueError("Add an Anthropic API key first (Settings → AI agent)")
    model = (db.get_settings().get("agent_model") or CLAUDE_MODELS[0]) if provider == "claude" else "local"
    now = time.time()
    with db.connect() as c:
        return c.execute("INSERT INTO agent_sessions (created_at, updated_at, provider, model) VALUES (?, ?, ?, ?)", (now, now, provider, model)).lastrowid


def delete_session(sid: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM agent_sessions WHERE id = ?", (sid,))
    shutil.rmtree(WORK / str(sid), ignore_errors=True)


def _save(sid: int, **f) -> None:
    vals = [json.dumps(v) if isinstance(v, (list, dict)) else v for v in f.values()]
    with db.connect() as c:
        c.execute(f"UPDATE agent_sessions SET {', '.join(f'{k} = ?' for k in f)}, updated_at = ? WHERE id = ?", (*vals, time.time(), sid))


def decide(tool_id: str, allow: bool, always: bool, sid: int, name: str = "") -> bool:
    """Answer one pending approval. The session and tool name come from the pending call itself, never from the
    request, so approving one call can't whitelist a different tool or answer another chat's question."""
    pending = _pending.get(tool_id)
    if not pending:
        return False
    owner, tool, fut = pending
    if owner != sid or fut.done():
        return False
    if allow and always:
        _always.setdefault(sid, set()).add(tool)
    fut.set_result(allow)
    return True


def stop(sid: int) -> None:
    _stop.add(sid)
    for owner, _, f in list(_pending.values()):
        if owner == sid and not f.done():
            f.set_result(False)


# ---------- Tools ----------

RUN_PYTHON = {
    "name": "run_python",
    "description": "Run a Python script in your workspace folder and get its output (stdout + stderr, last 20k characters). "
                   "env: 'analysis' (pandas, matplotlib, blocky client) or 'torch' (PyTorch + CUDA, transformers). "
                   "timeout_s: up to 3600. The user approves each run.",
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}, "env": {"type": "string", "enum": ["analysis", "torch"]},
                                                      "timeout_s": {"type": "integer"}}, "required": ["code"]},
}


async def tools() -> list[dict]:
    out = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema if hasattr(t, "inputSchema") else t.input_schema}
           for t in await mcp_server.server.list_tools()]
    return out + [RUN_PYTHON]


async def _run_python(sid: int, args: dict) -> tuple[str, bool]:
    py = PYTHONS.get(args.get("env") or "analysis")
    if not py or not py.is_file():
        return f"That environment isn't installed ({args.get('env')}).", True
    ws = WORK / str(sid)
    ws.mkdir(parents=True, exist_ok=True)
    script = ws / f"_step_{int(time.time() * 1000)}.py"
    script.write_text(str(args.get("code") or ""))
    timeout = max(5, min(3600, int(args.get("timeout_s") or 600)))
    proc = await asyncio.create_subprocess_exec(str(py), script.name, cwd=ws, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                env={**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}, start_new_session=True)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        code = proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        out, code = b"", -1
        return f"Timed out after {timeout} s.", True
    finally:
        script.unlink(missing_ok=True)
    text = out.decode(errors="replace")
    text = text[-20_000:] if len(text) > 20_000 else text
    files = sorted(p.name for p in ws.iterdir() if p.is_file() and not p.name.startswith("_step_"))
    return f"exit code {code}\n{text}\n[workspace files: {', '.join(files) or 'none'}]", code != 0


async def call_tool(sid: int, name: str, args: dict) -> tuple[str, bool]:
    if name == "run_python":
        return await _run_python(sid, args)
    try:
        r = await mcp_server.server.call_tool(name, args)
    except Exception as e:  # noqa: BLE001  (unknown tool, bad arguments, tool crash)
        return f"Error: {e}", True
    parts = getattr(r, "content", None) or []
    text = "\n".join(getattr(p, "text", "") for p in parts) or json.dumps(getattr(r, "structured_content", None) or "")
    err = bool(getattr(r, "is_error", False) or getattr(r, "isError", False))
    try:
        parsed = json.loads(text)
        err = err or (isinstance(parsed, dict) and "error" in parsed)  # Blocky's tools report problems as {"error": ...}
    except ValueError:
        pass
    return text[:MAX_RESULT], err


# ---------- The loop ----------

async def run_turn(sid: int, text: str):
    """Async generator of UI events for one user message: the agent works until it answers."""
    with db.connect() as c:
        r = c.execute("SELECT * FROM agent_sessions WHERE id = ?", (sid,)).fetchone()
    if not r:
        yield {"type": "error", "message": "Chat not found"}
        return
    if sid in _busy:
        yield {"type": "error", "message": "The agent is still working on the previous message"}
        return
    s = _row(r)
    _busy.add(sid)
    _stop.discard(sid)
    display = s["display"]
    display.append({"role": "user", "text": text})
    if s["title"] == "New chat":
        s["title"] = text.strip().splitlines()[0][:60] or "New chat"
        _save(sid, title=s["title"])
    try:
        brain = _claude if s["provider"] == "claude" else _local
        async for ev in brain(sid, s, text, display):
            yield ev
    except Exception as e:  # noqa: BLE001
        msg = str(e)[:400] or type(e).__name__
        display.append({"role": "error", "text": msg})
        yield {"type": "error", "message": msg}
    finally:
        _busy.discard(sid)
        _save(sid, display=display)
        yield {"type": "done"}


async def _tool_step(sid: int, display: list, tid: str, name: str, args: dict):
    """Run one tool call (asking the person first when needed). Yields UI events; the last one carries the result."""
    item = {"role": "tool", "id": tid, "name": name, "input": args, "status": "running"}
    display.append(item)
    risky = name in NEEDS_APPROVAL or (name == "simulate_quantization" and args.get("method") == "custom")
    if risky and name not in _always.get(sid, set()):
        item["status"] = "approval"
        fut = asyncio.get_running_loop().create_future()
        _pending[tid] = (sid, name, fut)
        _save(sid, display=display)  # a page opened later still shows the question
        yield {"type": "approval", "id": tid, "name": name, "input": args}
        try:
            allowed = await fut
        finally:
            _pending.pop(tid, None)
        if not allowed:
            item.update(status="denied", result="The user declined this action.")
            yield {"type": "tool_result", "id": tid, "content": item["result"], "is_error": True, "status": "denied"}
            yield {"type": "_result", "content": "The user declined this action. Ask them what to do instead.", "is_error": True}
            return
        item["status"] = "running"
    yield {"type": "tool_call", "id": tid, "name": name, "input": args}
    result, err = await call_tool(sid, name, args)
    item.update(status="error" if err else "done", result=result[:4000])
    yield {"type": "tool_result", "id": tid, "content": result[:4000], "is_error": err, "status": item["status"]}
    yield {"type": "_result", "content": result, "is_error": err}


async def _claude(sid: int, s: dict, text: str, display: list):
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key())
    messages = s["messages"] + [{"role": "user", "content": text}]
    tl = [{**t, "eager_input_streaming": True} for t in await tools()]
    system = SYSTEM.format(lab=mcp_server.INSTRUCTIONS)
    json_retries = 0
    for _ in range(MAX_STEPS):
        if sid in _stop:
            break
        reply = {"role": "assistant", "text": ""}
        display.append(reply)
        try:
            async with client.messages.stream(model=s["model"], max_tokens=32000, system=system, tools=tl, messages=messages,
                                              thinking={"type": "adaptive", "display": "summarized"}) as stream:
                async for ev in stream:
                    if ev.type == "text":
                        reply["text"] += ev.text
                        yield {"type": "text", "text": ev.text}
                    elif ev.type == "thinking":
                        yield {"type": "thinking", "text": ev.thinking}
                msg = await stream.get_final_message()
            json_retries = 0
        except ValueError:
            # Streamed tool input that couldn't be parsed at all: no tool_use id to answer, so re-issue the turn.
            display.remove(reply)
            json_retries += 1
            if json_retries > 2:
                raise
            continue
        if not reply["text"]:
            display.remove(reply)
        messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]})
        uses = [b for b in msg.content if b.type == "tool_use"]
        if sid in _stop and uses:  # every tool call needs an answer, or the next message is rejected
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u.id, "content": "Stopped by the user.", "is_error": True} for u in uses]})
            break
        if sid in _stop or msg.stop_reason in ("refusal", "end_turn") or not uses:
            if msg.stop_reason == "refusal":
                display.append({"role": "error", "text": "Claude declined to continue."})
            break
        if msg.stop_reason == "max_tokens":  # a tool call may be cut off: don't run it
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u.id, "content": "Output was cut off; try again with less.",
                                                          "is_error": True} for u in uses]})
            continue
        results = []
        for u in uses:
            res = {"content": "", "is_error": True}
            args = u.input if isinstance(u.input, dict) else {}
            async for ev in _tool_step(sid, display, u.id, u.name, args):
                if ev["type"] == "_result":
                    res = ev
                else:
                    yield ev
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": res["content"] or "(no output)", "is_error": res["is_error"]})
        messages.append({"role": "user", "content": results})
        _save(sid, messages=messages, display=display)
    else:
        display.append({"role": "error", "text": f"Stopped after {MAX_STEPS} steps."})
    _save(sid, messages=messages, display=display)


async def _local(sid: int, s: dict, text: str, display: list):
    messages = s["messages"] or [{"role": "system", "content": SYSTEM.format(lab=mcp_server.INSTRUCTIONS)}]
    messages.append({"role": "user", "content": text})
    tl = [{"type": "function", "function": {"name": t["name"], "description": t["description"][:1000], "parameters": t["input_schema"]}} for t in await tools()]
    async with httpx.AsyncClient(base_url=config.LLM_URL, timeout=httpx.Timeout(10, read=900)) as c:
        for _ in range(MAX_STEPS):
            if sid in _stop:
                break
            yield {"type": "thinking", "text": ""}
            body = {"model": "blocky", "messages": messages, "tools": tl, "temperature": 0.3}
            r = await c.post("/v1/chat/completions", json=body)
            if r.status_code >= 400:
                raise RuntimeError(f"The local model failed: {r.text[:300]}")
            m = r.json()["choices"][0]["message"]
            if m.get("reasoning_content"):
                yield {"type": "thinking", "text": m["reasoning_content"]}
            if not (m.get("content") or "").strip() and not m.get("tool_calls"):
                # Thinking models sometimes end with reasoning only: ask again for the answer itself.
                r = await c.post("/v1/chat/completions", json={**body, "chat_template_kwargs": {"enable_thinking": False}})
                if r.status_code < 400:
                    m = r.json()["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            messages.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": calls} if calls else {})})
            if m.get("content"):
                display.append({"role": "assistant", "text": m["content"]})
                yield {"type": "text", "text": m["content"]}
            if not calls:
                break
            for call in calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = None
                res = {"content": "Error: arguments weren't valid JSON", "is_error": True}
                if isinstance(args, dict):
                    async for ev in _tool_step(sid, display, call.get("id") or f"call{time.time()}", fn.get("name", ""), args):
                        if ev["type"] == "_result":
                            res = ev
                        else:
                            yield ev
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": res["content"]})
            _save(sid, messages=messages, display=display)
        else:
            display.append({"role": "error", "text": f"Stopped after {MAX_STEPS} steps."})
    _save(sid, messages=messages, display=display)
