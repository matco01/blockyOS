"""Experiments: runs, sweeps, validation, instrument checks, export/import."""
import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from .. import profile
from .common import _bad

router = APIRouter()


# Request bodies. Extra fields are accepted and passed through (older clients and remote devices keep working), and only
# the fields a client actually sent reach the experiment code, which still applies its own validation and defaults.
class _Body(BaseModel):
    model_config = ConfigDict(extra="allow")

    def sent(self) -> dict:
        return self.model_dump(exclude_unset=True)


class RunIn(_Body):
    """One profiled run: the full setup (missing settings come from the experiment's defaults) and the run it branches from."""
    config: dict[str, Any] = Field(default_factory=dict, description="Setup: model, engine, kv, ctx, ngl, fa, threads, device, gpus, split_mode, rpc, pp, ...")
    parent_id: int | None = Field(None, description="Run this one branches from; omit for the baseline")
    label: str | None = Field(None, max_length=60)


class SweepIn(_Body):
    grid: dict[str, list[Any]] = Field(description="Setting -> values; every combination is queued (max 32 runs)")
    parent_id: int | None = None
    repeats: int = Field(1, ge=1, le=5)


class LmEvalIn(BaseModel):
    tasks: str
    limit: int = 100


class RulerIn(BaseModel):
    samples: int = 8


class ProtocolIn(_Body):
    """Tests and measurement settings every run of the experiment gets (replaces the previous choice)."""
    lm_eval: LmEvalIn | None = None
    kernels: bool = False
    mmlu_per_category: int | None = Field(None, description="40, 100 or 250 questions per subject (560, 1400, 3500)")
    ruler: RulerIn | None = None
    ppl_chunks: int | None = Field(None, description="Perplexity/KLD sample: 32, 64, 128 or 256 chunks of 512 tokens")
    fidelity_text: str | None = Field(None, description="Absolute path of a .txt to score perplexity/KLD on (default WikiText-2 test)")
    equivalence_margin_pts: float | None = Field(None, description="Accuracy margin for equivalence (TOST) verdicts")
    lock_clocks_mhz: int | None = Field(None, description="Lock the GPU clock during runs (needs the install.sh helper)")
    cooldown_s: int | None = Field(None, description="Pause before each run")


class ValidateIn(_Body):
    """Counterbalanced repeats of two setups for one preselected metric."""
    reference_id: int
    candidate_id: int
    pairs: int = Field(6, ge=3, le=12)
    metric: str = "tg"
    min_effect_pct: float = Field(3, ge=0, le=100)


class PreregistrationIn(_Body):
    hypothesis: str = Field(min_length=10, max_length=4000)
    primary_metric: str
    claim: str = "difference"
    direction: str = "either"
    alpha: float = 0.05
    min_effect: float | None = None
    equivalence_margin_pts: float | None = None
    analysis: str = ""
    secondary_metrics: list[str] = Field(default_factory=list)


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
def experiments_run(exp_id: int, body: RunIn):
    """Queues a profile; the experiments worker runs queued profiles in order."""
    try:
        run_id = profile.create_run(exp_id, body.sent())
    except (ValueError, TypeError) as e:
        _bad(e)
    return {"id": run_id}


@router.post("/api/experiments/{exp_id}/sweep")
def experiments_sweep(exp_id: int, body: SweepIn):
    try:
        ids = profile.create_sweep(exp_id, body.sent())
    except (ValueError, TypeError) as e:
        _bad(e)
    return {"ids": ids}


@router.post("/api/experiments/{exp_id}/validate")
def experiments_validate(exp_id: int, body: ValidateIn):
    from .. import validation
    try:
        return validation.create(exp_id, body.sent())
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


@router.get("/api/experiments/{exp_id}/bundle.zip")
def experiments_bundle(exp_id: int):
    from .. import bundle
    try:
        name, data = bundle.build(exp_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.post("/api/experiments/{exp_id}/preregistrations")
def experiments_preregister(exp_id: int, body: PreregistrationIn):
    try:
        return profile.preregister(exp_id, body.sent())
    except (ValueError, TypeError) as e:
        _bad(e)


@router.put("/api/experiments/{exp_id}/protocol")
def experiments_protocol(exp_id: int, body: ProtocolIn):
    try:
        return profile.set_protocol(exp_id, body.model_dump(exclude_unset=True, mode="json"))
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
