"""Compression lab: quantizations and their fidelity."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import lab, quantlab
from .common import _bad, _spawn, state

router = APIRouter()


@router.get("/api/quantlab")
def quant_runs():
    return {"runs": quantlab.list_runs(), "jobs": sorted(quantlab.jobs.values(), key=lambda j: j["started"], reverse=True),
            "types": quantlab.QUANT_TYPES}


@router.post("/api/quantlab")
async def quant_start(body: dict):  # async: starts a background task
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        run_id = quantlab.create(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(quantlab.run(run_id, state.llm))
    return {"id": run_id}


@router.post("/api/quantlab/{run_id}/cancel")
def quant_cancel(run_id: int):
    quantlab.cancel(run_id)
    return {"ok": True}


@router.delete("/api/quantlab/{run_id}")
def quant_delete(run_id: int):
    quantlab.delete(run_id)
    return {"ok": True}


class QuantizeIn(BaseModel):
    source: str
    type: str


@router.post("/api/quantlab/quantize")
async def quant_make(body: QuantizeIn):  # async: starts a background task
    try:
        return {"id": quantlab.start_quantize(body.source, body.type)}
    except ValueError as e:
        _bad(e)
