"""Throughput / speculative-decoding tools and the model arena."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import arena, lab, tools
from .common import _bad, _spawn, state

router = APIRouter()


@router.get("/api/tools/{kind}")
def tool_runs(kind: str):
    return tools.list_runs(kind)


@router.post("/api/tools/{kind}")
async def tool_start(kind: str, body: dict):  # async: starts a background task
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        run_id = tools.create(kind, body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(tools.run(run_id, state.llm))
    return {"id": run_id}


@router.post("/api/tools/run/{run_id}/cancel")
def tool_cancel(run_id: int):
    tools.cancel(run_id)
    return {"ok": True}


@router.delete("/api/tools/run/{run_id}")
def tool_delete(run_id: int):
    tools.delete(run_id)
    return {"ok": True}


@router.get("/api/arena")
def arena_list():
    return arena.list_matches()


@router.get("/api/arena/leaderboard")
def arena_leaderboard():
    return arena.leaderboard()


@router.post("/api/arena")
async def arena_start(body: dict):  # async: starts a background task on the event loop
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        match_id = arena.create(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(arena.run(match_id, state.llm))
    return {"id": match_id}


@router.get("/api/arena/{match_id}")
def arena_get(match_id: int):
    m = arena.get(match_id)
    if not m:
        raise HTTPException(404, "Match not found")
    return m


class VoteIn(BaseModel):
    winner: str


@router.post("/api/arena/{match_id}/vote")
def arena_vote(match_id: int, body: VoteIn):
    try:
        arena.vote(match_id, body.winner)
    except ValueError as e:
        _bad(e)
    return arena.get(match_id)


@router.post("/api/arena/{match_id}/cancel")
def arena_cancel(match_id: int):
    arena.cancel(match_id)
    return {"ok": True}


@router.delete("/api/arena/{match_id}")
def arena_delete(match_id: int):
    arena.delete(match_id)
    return {"ok": True}
