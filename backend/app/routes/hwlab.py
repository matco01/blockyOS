"""Hardware lab: models on simulated chips (level 1: the chip's arithmetic)."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import hwlab, torchlab
from .common import _bad

router = APIRouter()


def _csv(text: str, name: str) -> Response:
    return Response(text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/api/hwlab/presets")
def hwlab_presets():
    return hwlab.catalog()


@router.get("/api/hwlab/runs")
def hwlab_runs():
    return hwlab.runs()


@router.post("/api/hwlab/runs")
async def hwlab_run(body: dict):
    try:
        return {"id": hwlab.start(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.post("/api/hwlab/sweeps")
async def hwlab_sweep(body: dict):
    try:
        return hwlab.start_sweep(body)
    except (ValueError, TypeError) as e:
        _bad(e)


@router.get("/api/hwlab/runs/{job_id}/csv")
def hwlab_run_csv(job_id: int):
    j = torchlab.job(job_id)
    if not j or j["kind"] != "hwsim":
        raise HTTPException(404, "Run not found")
    return _csv(hwlab.run_csv(j), f"hardware-lab-run-{job_id}.csv")


@router.get("/api/hwlab/sweeps/{sweep_id}/csv")
def hwlab_sweep_csv(sweep_id: str):
    jobs = hwlab.sweep_runs(sweep_id)
    if not jobs:
        raise HTTPException(404, "Sweep not found")
    return _csv(hwlab.sweep_csv(jobs), f"hardware-lab-sweep-{sweep_id}.csv")
