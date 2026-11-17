"""Remote access, other Blocky devices, and the wall-power meter."""
import httpx
from fastapi import APIRouter, HTTPException

from .. import power, remote
from .common import _bad

router = APIRouter()


@router.get("/api/remote-access")
def remote_access_get():
    return {"enabled": bool(remote.access_token()), "token": remote.access_token() or None}


@router.post("/api/remote-access")
def remote_access_set(body: dict):
    return {"enabled": bool(body.get("enabled")), "token": remote.set_access(bool(body.get("enabled"))) or None}


@router.get("/api/devices")
def devices_list():
    return remote.list_devices()


@router.post("/api/devices")
def devices_add(body: dict):
    try:
        return {"id": remote.add_device(body)}
    except ValueError as e:
        _bad(e)


@router.delete("/api/devices/{dev_id}")
def devices_delete(dev_id: int):
    remote.delete_device(dev_id)
    return {"ok": True}


@router.get("/api/devices/{dev_id}/check")
async def devices_check(dev_id: int):
    try:
        return await remote.check(dev_id)
    except ValueError as e:
        _bad(e)


@router.post("/api/experiments/{exp_id}/remote")
async def experiments_remote(exp_id: int, body: dict):
    try:
        return {"job_id": remote.start(exp_id, body)}
    except (ValueError, TypeError) as e:
        _bad(e)


@router.get("/api/remote-jobs")
def remote_jobs(experiment_id: int | None = None):
    return remote.list_jobs(experiment_id)


@router.post("/api/remote-jobs/{job_id}/cancel")
def remote_jobs_cancel(job_id: int):
    remote.cancel(job_id)
    return {"ok": True}


@router.get("/api/rpc-worker")
def rpc_worker_get():
    from .. import rpcworker
    return rpcworker.status()


@router.post("/api/rpc-worker")
def rpc_worker_set(body: dict):
    from .. import rpcworker
    try:
        return rpcworker.configure(body)
    except (ValueError, TypeError, OSError) as e:
        raise HTTPException(400, str(e))


@router.post("/api/power/test")
async def power_test(body: dict):
    """Read the wall-power meter once (to check the address before running experiments)."""
    try:
        meter = power.meter_config(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not meter:
        raise HTTPException(400, "Enter the meter's address")
    async with httpx.AsyncClient() as c:
        w = await power.read_wall(c, *meter)
    if w is None:
        raise HTTPException(502, "No power reading from that address (check the URL, or set the field that holds the watts)")
    return {"watts": w}
