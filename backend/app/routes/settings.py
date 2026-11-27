"""Device settings and presets."""
import os

from fastapi import APIRouter, HTTPException

from .. import agent, db, lab, power, profile

router = APIRouter()


@router.get("/api/settings")
def get_settings():
    s = db.get_settings()
    key = s.pop("anthropic_api_key", "")
    s.pop("ai_pairing_code", None)  # Settings → AI assistants shows it (GET /api/ai-access)
    return {**s, "anthropic_api_key_set": bool(key)}  # the key itself never leaves the backend


@router.patch("/api/settings")
def update_settings(body: dict):
    body.pop("remote_token", None)  # only POST /api/remote-access creates or clears it
    body.pop("rpc_worker", None)  # only POST /api/rpc-worker changes it (and starts or stops the worker)
    body.pop("anthropic_api_key", None)  # only POST /api/agent/key sets it
    body.pop("ai_access", None)  # only /api/ai-access changes these
    body.pop("ai_pairing_code", None)
    if "agent_model" in body and body["agent_model"] not in agent.CLAUDE_MODELS:
        raise HTTPException(400, "Unknown model")
    try:
        for k in ("device_price", "electricity_price_kwh"):
            if k in body and body[k] not in (None, ""):
                body[k] = float(body[k])
                if body[k] < 0:
                    raise ValueError(k)
            elif k in body:
                body[k] = None
        if "power_meter_url" in body:
            power.meter_config({"power_meter_url": body["power_meter_url"]})
        for k in ("power_meter_field", "currency"):
            if k in body:
                body[k] = str(body[k] or "")[:40]
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e) if "URL" in str(e) else "Prices must be non-negative numbers")
    if "gpu_guard" in body:
        body["gpu_guard"] = bool(body["gpu_guard"])
    for k in ("gpu_max_temp_c", "gpu_resume_temp_c"):
        if k in body:
            if body[k] in (None, ""):
                body[k] = None
                continue
            try:
                body[k] = int(body[k])
            except (TypeError, ValueError):
                raise HTTPException(400, "Temperatures must be whole numbers of °C") from None
            if not 40 <= body[k] <= 100:
                raise HTTPException(400, "Temperatures must be between 40 and 100 °C")
    current = db.get_settings()
    limit = body.get("gpu_max_temp_c", current.get("gpu_max_temp_c"))
    resume = body.get("gpu_resume_temp_c", current.get("gpu_resume_temp_c"))
    if limit and resume and resume >= limit:
        raise HTTPException(400, "The resume temperature must be below the limit")
    return db.update_settings(body)


@router.get("/api/device-presets")
def device_presets():
    return {"presets": profile.DEVICE_PRESETS, "cpu_threads": os.cpu_count(), "gpu_count": len(lab.hardware().get("gpus") or [])}
