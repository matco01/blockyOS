"""Live system status, logs, notebooks, the Python client and MCP info."""
import json
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import FileResponse

from .. import config, db, lab, mcp_server, profile, system
from .common import state

router = APIRouter()


@router.get("/api/system")
def system_stats():
    return system.snapshot()


@router.get("/api/status")
async def status():
    """Is the model loaded, and which one? Also reports Lab activity."""
    from .. import gpuguard
    base = {"busy": lab.busy, "switch_error": lab.last_switch_error, "gpu_pause": gpuguard.paused(),
            "gpu_waiting": gpuguard.state["waiting"]}
    try:
        r = await state.llm.get("/props", timeout=2)
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


@router.get("/api/hardware")
def hardware():
    return lab.hardware()


@router.get("/api/memory-map")
async def memory_map():
    mm = await lab.memory_map()
    await lab.refresh_measured()
    return mm


@router.get("/api/stats/recent")
def recent_stats(limit: int = 30):
    with db.connect() as c:
        rows = c.execute(
            "SELECT m.id, m.created_at, m.stats, m.conversation_id FROM messages m "
            "WHERE m.role = 'assistant' AND m.stats IS NOT NULL ORDER BY m.id DESC LIMIT ?",
            (max(1, min(500, limit)),),
        ).fetchall()
    return [{"id": r["id"], "created_at": r["created_at"], "conversation_id": r["conversation_id"],
             **json.loads(r["stats"])} for r in reversed(rows)]


@router.get("/api/notebooks")
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


@router.get("/api/client.py")
def python_client():
    return FileResponse(Path(__file__).resolve().parents[3] / "client" / "blocky.py", media_type="text/x-python", filename="blocky.py")


@router.get("/api/hardware/limits")
def hardware_limits():
    return profile.limits()


@router.get("/api/logs")
async def logs(lines: int = 300):
    return {"text": await lab.llm_log(lines)}


@router.get("/api/mcp")
async def mcp_info():
    return {"url": "http://localhost:8000/mcp", "tools": [{"name": t.name, "description": t.description} for t in await mcp_server.server.list_tools()]}


@router.get("/api/gpu-safety")
def gpu_safety():
    """GPU safety: current temperature, the limits in use, whether GPU work is paused and why, recent events."""
    from .. import gpuguard
    return gpuguard.report()


@router.post("/api/gpu-safety/resume")
def gpu_safety_resume():
    """Lets GPU work start again after a safety pause (owner only: not available to AI assistants)."""
    from .. import gpuguard
    gpuguard.resume()
    return gpuguard.report()
