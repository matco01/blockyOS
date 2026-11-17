"""Quality tests."""
import random

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import evals, lab
from .common import _bad, _spawn, state

router = APIRouter()


@router.get("/api/evals/suites")
def eval_suites():
    return evals.suites_summary()


@router.get("/api/evals/fresh")
def eval_fresh_sets():
    from .. import fresh
    return [{"key": f"fresh:{k}", "id": k, **v} for k, v in fresh.SETS.items()]


@router.get("/api/evals/fresh/{kind}/preview")
def eval_fresh_preview(kind: str, n: int = 3):
    """A few freshly generated example questions (new every call), so users can see what they look like."""
    from .. import fresh
    if kind not in fresh.SETS:
        raise HTTPException(404, "Unknown fresh set")
    return [{"q": q["q"], "kind": q["kind"]} for q in fresh.generate(kind, max(1, min(10, n)), random.randint(1, 2**31))]


@router.get("/api/evals/suites/{key}")
def eval_suite(key: str):
    s = evals.all_suites().get(key)
    if not s:
        raise HTTPException(404, "Test set not found")
    return s


class CustomSuiteIn(BaseModel):
    name: str
    questions: list[dict]


@router.post("/api/evals/suites")
def eval_create_suite(body: CustomSuiteIn):
    try:
        return {"key": f"custom:{evals.create_custom(body.name, body.questions)}"}
    except ValueError as e:
        _bad(e)


@router.delete("/api/evals/suites/custom:{suite_id}")
def eval_delete_suite(suite_id: int):
    evals.delete_custom(suite_id)
    return {"ok": True}


@router.get("/api/evals")
def eval_list():
    return evals.list_runs()


@router.post("/api/evals")
async def eval_start(body: dict):  # async: starts a background task on the event loop
    if lab.busy:
        raise HTTPException(409, f"Busy: {lab.busy}")
    try:
        run_id = evals.create(body)
    except (ValueError, TypeError) as e:
        _bad(e)
    _spawn(evals.run(run_id, state.llm))
    return {"id": run_id}


@router.get("/api/evals/{run_id}")
def eval_get(run_id: int):
    r = evals.get(run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    return r


@router.post("/api/evals/{run_id}/cancel")
def eval_cancel(run_id: int):
    evals.cancel(run_id)
    return {"ok": True}


@router.delete("/api/evals/{run_id}")
def eval_delete(run_id: int):
    evals.delete(run_id)
    return {"ok": True}
