"""OpenAI-compatible pass-through for outside harnesses."""
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import lab
from .common import state

router = APIRouter()


@router.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def openai_proxy(path: str, request: Request):
    """Pass-through to llama-server's OpenAI-compatible API, so external harnesses can test the local model."""
    if path not in ("models", "chat/completions", "completions", "embeddings"):
        raise HTTPException(404, "Not an OpenAI endpoint Blocky exposes")
    if lab.busy:
        raise HTTPException(503, f"Blocky is busy ({lab.busy}). Try again when it's finished.")
    body = await request.body()
    req = state.llm.build_request(request.method, f"/v1/{path}", content=body,
                                      headers={"Content-Type": request.headers.get("content-type", "application/json")},
                                      timeout=httpx.Timeout(10, read=900))
    upstream = await state.llm.send(req, stream=True)

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code,
                             media_type=upstream.headers.get("content-type", "application/json"))
