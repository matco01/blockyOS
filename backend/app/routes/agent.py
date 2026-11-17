"""The research agent inside the UI."""
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import agent, db
from .common import _bad, _sse, state

router = APIRouter()


@router.get("/api/agent/status")
def agent_status():
    return agent.status()


@router.post("/api/agent/key")
def agent_key(body: dict):
    key = str(body.get("key") or "").strip()
    if key and not key.startswith("sk-"):
        raise HTTPException(400, "That doesn't look like an Anthropic API key (they start with sk-)")
    db.update_settings({"anthropic_api_key": key})
    return agent.status()


@router.get("/api/agent/sessions")
def agent_sessions():
    return agent.list_sessions()


@router.post("/api/agent/sessions")
def agent_new(body: dict):
    try:
        return {"id": agent.create_session(str(body.get("provider") or "local"))}
    except ValueError as e:
        _bad(e)


@router.get("/api/agent/sessions/{sid}")
def agent_get(sid: int):
    s = agent.get_session(sid)
    if not s:
        raise HTTPException(404, "Chat not found")
    return s


@router.delete("/api/agent/sessions/{sid}")
def agent_delete(sid: int):
    agent.stop(sid)
    agent.delete_session(sid)
    return {"ok": True}


@router.post("/api/agent/sessions/{sid}/message")
async def agent_message(sid: int, body: dict):
    """Streams the agent's work on one message. The work runs as a background task, so leaving the page doesn't stop it."""
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Empty message")
    q: asyncio.Queue = asyncio.Queue()

    async def work():
        try:
            async for ev in agent.run_turn(sid, text):
                await q.put(ev)
        finally:
            await q.put(None)

    task = asyncio.create_task(work())
    state.background.add(task)
    task.add_done_callback(state.background.discard)

    async def stream():
        while (ev := await q.get()) is not None:
            yield _sse(ev)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/api/agent/sessions/{sid}/approve")
def agent_approve(sid: int, body: dict):
    ok = agent.decide(str(body.get("id") or ""), bool(body.get("allow")), bool(body.get("always")), sid, str(body.get("name") or ""))
    return {"ok": ok}


@router.post("/api/agent/sessions/{sid}/stop")
def agent_stop(sid: int):
    agent.stop(sid)
    return {"ok": True}
