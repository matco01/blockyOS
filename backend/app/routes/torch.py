"""PyTorch lab, layer scans and the weights workshop."""
import asyncio

from fastapi import APIRouter, HTTPException, Request

from .. import scans, torchlab, weights
from .common import _bad

router = APIRouter()


@router.post("/api/torch/quantsim/{sim_id}/export")
async def torch_quantsim_export(sim_id: int, body: dict):
    try:
        return {"id": torchlab.start_export(sim_id, body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.post("/api/torch/quantsim")
async def torch_quantsim(body: dict):
    try:
        return {"id": torchlab.start_quantsim(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.get("/api/torch/status")
async def torch_status():
    return await asyncio.to_thread(torchlab.status)


@router.get("/api/torch/jobs")
def torch_jobs(kind: str | None = None):
    return torchlab.jobs(kind)


@router.get("/api/torch/jobs/{job_id}")
def torch_job(job_id: int):
    j = torchlab.job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j


@router.post("/api/torch/jobs/{job_id}/cancel")
def torch_cancel(job_id: int):
    torchlab.cancel(job_id)
    return {"ok": True}


@router.post("/api/torch/download")
async def torch_download(body: dict):
    try:
        return {"id": torchlab.start_download(str(body.get("repo") or ""))}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.post("/api/torch/activations")
async def torch_activations(body: dict):
    try:
        return {"id": torchlab.start_activations(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.post("/api/torch/train")
async def torch_train(body: dict):
    try:
        return {"id": torchlab.start_train(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.post("/api/torch/lora")
async def torch_lora(body: dict):
    try:
        return {"id": torchlab.start_lora(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.get("/api/scans")
def scans_list(model: str | None = None):
    return scans.list_scans(model)


@router.post("/api/scans")
async def scans_start(body: dict):  # async: starts a background task
    try:
        return {"id": scans.start(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.post("/api/scans/{scan_id}/cancel")
def scans_cancel(scan_id: int):
    scans.cancel(scan_id)
    return {"ok": True}


@router.delete("/api/scans/{scan_id}")
def scans_delete(scan_id: int):
    scans.delete(scan_id)
    return {"ok": True}


@router.get("/api/weights/family")
def weights_family(model: str):
    try:
        return {"files": weights.family(model), "base_types": weights.BASE_TYPES, "tensor_types": weights.TENSOR_TYPES,
                "needs_imatrix": sorted(weights.NEEDS_IMATRIX)}
    except (OSError, ValueError) as e:
        _bad(e)


@router.get("/api/weights/tensors")
def weights_tensors(model: str):
    try:
        return weights.tensor_groups(model)
    except (OSError, ValueError) as e:
        _bad(e)


@router.post("/api/weights/variants")
async def weights_create(body: dict):  # async: starts a background task
    try:
        return {"id": weights.start(body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.get("/api/weights/jobs")
def weights_jobs():
    return weights.list_jobs()


@router.post("/api/weights/jobs/{job_id}/cancel")
def weights_cancel(job_id: str):
    weights.cancel(job_id)
    return {"ok": True}


@router.put("/api/torch/checkpoints/upload")
async def torch_checkpoint_upload(request: Request, name: str, path: str):
    """One file of a Hugging Face-format model folder from your computer (raw body). Call finish when all are sent."""
    from .. import imports
    from .models import _size
    try:
        return await imports.checkpoint_file(request.stream(), name, path, _size(request))
    except ValueError as e:
        _bad(e)


@router.post("/api/torch/checkpoints/finish")
def torch_checkpoint_finish(body: dict):
    """Checks the uploaded folder (config.json and weights) and adds it to the trainable models."""
    from .. import imports
    try:
        return imports.checkpoint_finish(str(body.get("name") or ""))
    except ValueError as e:
        _bad(e)
