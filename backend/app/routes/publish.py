"""Ship: model cards, license checks, model file downloads and publishing to Hugging Face (owner only, not via MCP)."""
from fastapi import APIRouter
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .. import publish
from .common import _bad

router = APIRouter()


@router.get("/api/publish/license")
def publish_license(experiment_id: int, run_id: int):
    """The license the version's model file declares, the choices, and whether sharing needs confirming."""
    try:
        return publish.license_info(experiment_id, run_id)
    except ValueError as e:
        _bad(e)


@router.get("/api/publish/card", response_class=PlainTextResponse)
def publish_card(experiment_id: int, run_id: int, license: str = "", base_model: str = "", title: str = ""):
    """The model card (README.md) written from this version's measurements."""
    try:
        return publish.card(experiment_id, run_id, license, base_model, title)
    except ValueError as e:
        _bad(e)


@router.get("/api/publish/file")
def publish_file(experiment_id: int, run_id: int):
    """Downloads the version's model file (.gguf)."""
    try:
        _, run, _ = publish._version(experiment_id, run_id)
        path = publish.model_file(run)
    except ValueError as e:
        _bad(e)
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@router.get("/api/publish/account")
def publish_account():
    return publish.account()


class TokenIn(BaseModel):
    token: str = Field(max_length=400)


@router.post("/api/publish/account")
def publish_connect(body: TokenIn):
    """Connects a Hugging Face account with an access token ('write' access). The token is never shown again."""
    try:
        return publish.connect(body.token)
    except ValueError as e:
        _bad(e)


@router.delete("/api/publish/account")
def publish_disconnect():
    return publish.disconnect()


class PublishIn(BaseModel):
    experiment_id: int
    run_id: int
    repo: str = Field(max_length=200, description="owner/name on Hugging Face")
    private: bool = True
    license: str = Field(max_length=60)
    confirm_license: bool = False
    base_model: str = Field("", max_length=200)
    title: str = Field("", max_length=200)
    include_bundle: bool = True


@router.post("/api/publish/hf")
async def publish_hf(body: PublishIn):  # async: starts a background task
    """Creates (or reuses) the repository and uploads the model card, the reproducibility bundle and the model file."""
    try:
        return {"id": publish.start_hf(body.model_dump())}
    except ValueError as e:
        _bad(e)


@router.get("/api/publish/jobs")
def publish_jobs():
    return sorted(publish.jobs.values(), key=lambda j: j["started"], reverse=True)
