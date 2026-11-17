"""Inference engines (llama.cpp builds, vLLM, SGLang)."""
from fastapi import APIRouter

from .. import engines
from .common import _bad

router = APIRouter()


@router.get("/api/engines")
def engines_list():
    return engines.list_engines()


@router.post("/api/engines")
async def engines_add(body: dict):  # async: starts a background build
    try:
        return {"id": engines.add(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.get("/api/engines/{engine_id}/log")
def engines_log(engine_id: str):
    return {"log": engines.get_log(engine_id)}


@router.post("/api/engines/{engine_id}/cancel")
def engines_cancel(engine_id: str):
    engines.cancel(engine_id)
    return {"ok": True}


@router.delete("/api/engines/{engine_id}")
def engines_delete(engine_id: str):
    try:
        engines.delete(engine_id)
    except ValueError as e:
        _bad(e)
    return {"ok": True}
