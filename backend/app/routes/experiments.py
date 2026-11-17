"""Experiments: runs, sweeps, validation, instrument checks, export/import."""
import json
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from .. import profile
from .common import _bad

router = APIRouter()


@router.get("/api/experiments")
def experiments_list():
    return profile.list_experiments()


class ExperimentIn(BaseModel):
    name: str = ""
    model: str


@router.post("/api/experiments")
def experiments_create(body: ExperimentIn):
    try:
        return {"id": profile.create_experiment(body.name, body.model)}
    except ValueError as e:
        _bad(e)


@router.get("/api/experiments/{exp_id}")
def experiments_get(exp_id: int):
    e = profile.get_experiment(exp_id)
    if not e:
        raise HTTPException(404, "Experiment not found")
    return e


@router.get("/api/experiments/{exp_id}/export.{fmt}")
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


@router.post("/api/experiments/import")
async def experiments_import(request: Request, into: int | None = None):
    """Import an experiment exported as JSON from another Blocky device (new experiment, or merged into `into`)."""
    try:
        data = json.loads(await request.body())
        return profile.import_experiment(data, into)
    except (ValueError, TypeError, KeyError) as e:
        _bad(e)


@router.delete("/api/experiments/{exp_id}")
def experiments_delete(exp_id: int):
    profile.delete_experiment(exp_id)
    return {"ok": True}


@router.post("/api/experiments/{exp_id}/runs")
def experiments_run(exp_id: int, body: dict):
    """Queues a profile; the experiments worker runs queued profiles in order."""
    try:
        run_id = profile.create_run(exp_id, body)
    except (ValueError, TypeError) as e:
        _bad(e)
    return {"id": run_id}


@router.post("/api/experiments/{exp_id}/sweep")
def experiments_sweep(exp_id: int, body: dict):
    try:
        ids = profile.create_sweep(exp_id, body)
    except (ValueError, TypeError) as e:
        _bad(e)
    return {"ids": ids}


@router.post("/api/experiments/{exp_id}/validate")
def experiments_validate(exp_id: int, body: dict):
    from .. import validation
    try:
        return validation.create(exp_id, body)
    except (ValueError, TypeError) as e:
        _bad(e)


@router.get("/api/experiments/{exp_id}/instrument-check")
def experiments_instrument_check(exp_id: int):
    from .. import instrument
    e = profile.get_experiment(exp_id)
    if not e:
        raise HTTPException(404, "Experiment not found")
    return instrument.check(e)


@router.get("/api/experiments/runs/{run_id}/evidence")
def experiments_evidence(run_id: int):
    try:
        return profile.evidence_outputs(run_id)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/api/experiments/{exp_id}/validations/{vid}/cancel")
def experiments_validation_cancel(exp_id: int, vid: int):
    from .. import validation
    try:
        validation.cancel(exp_id, vid)
        return {"ok": True}
    except ValueError as e:
        _bad(e)


@router.put("/api/experiments/{exp_id}/protocol")
def experiments_protocol(exp_id: int, body: dict):
    try:
        return profile.set_protocol(exp_id, body)
    except (ValueError, TypeError) as e:
        _bad(e)


@router.put("/api/experiments/{exp_id}/canvas")
def experiments_canvas(exp_id: int, body: dict):
    try:
        profile.save_canvas(exp_id, body)
    except ValueError as e:
        _bad(e)
    return {"ok": True}


@router.post("/api/experiments/{exp_id}/cancel-queue")
def experiments_cancel_queue(exp_id: int):
    return {"cancelled": profile.cancel_queue(exp_id)}


@router.post("/api/experiments/runs/{run_id}/cancel")
def experiments_cancel(run_id: int):
    profile.cancel(run_id)
    return {"ok": True}


@router.delete("/api/experiments/runs/{run_id}")
def experiments_delete_run(run_id: int):
    profile.delete_run(run_id)
    return {"ok": True}
