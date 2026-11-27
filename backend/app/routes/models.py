"""Model files, runtime settings, Hugging Face downloads."""
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .. import lab
from .common import _bad, _spawn, state

router = APIRouter()


@router.get("/api/models")
async def models():
    await lab.refresh_measured()
    return {"models": lab.list_models(), "runtime": lab.read_runtime(), "catalog": lab.catalog(),
            "hardware": lab.hardware(), "kv_types": lab.KV_TYPES}


@router.get("/api/models/{name}/anatomy")
def model_anatomy(name: str):
    from .. import gguf
    path = lab.MODELS_DIR / Path(name).name
    if not path.is_file() or path.suffix != ".gguf":
        raise HTTPException(404, "Model not found")
    try:
        return gguf.anatomy(path)
    except (OSError, ValueError) as e:
        raise HTTPException(400, f"Couldn't read the model file: {e}")


@router.post("/api/models/runtime")
async def set_runtime(body: dict):
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        cfg = lab.validate_runtime(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(lab.apply_runtime(state.llm, cfg))
    return {"ok": True, "runtime": cfg}


@router.delete("/api/models/{name}")
def delete_model(name: str):
    try:
        lab.delete_model(name)
    except ValueError as e:
        _bad(e)
    return {"ok": True}


@router.get("/api/hf/files")
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


@router.post("/api/downloads")
async def start_download(body: DownloadIn):  # async: starts a background task on the event loop
    try:
        return {"id": lab.start_download(body.repo.strip(), body.path.strip())}
    except ValueError as e:
        _bad(e)


@router.get("/api/downloads")
def list_downloads():
    return sorted(lab.downloads.values(), key=lambda d: d["started"], reverse=True)


@router.post("/api/downloads/{dl_id}/cancel")
def cancel_download(dl_id: str):
    if dl_id in lab.downloads and lab.downloads[dl_id]["status"] == "downloading":
        lab.downloads[dl_id]["status"] = "cancelled"
    return {"ok": True}


def _size(request) -> int | None:
    try:
        return int(request.headers.get("content-length") or "") or None
    except ValueError:
        return None


@router.put("/api/models/upload")
async def models_upload(request: Request, name: str):
    """A .gguf model file from your computer: the raw file is the request body (streamed to disk)."""
    from .. import imports
    try:
        return await imports.model(request.stream(), name, _size(request))
    except ValueError as e:
        _bad(e)
