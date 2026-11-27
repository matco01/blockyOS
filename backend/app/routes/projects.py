"""Projects: name a piece of work, then keep its experiments and code together."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import projects
from .common import _bad

router = APIRouter()


class ProjectIn(BaseModel):
    name: str = Field(max_length=200)
    goal: str = Field("", max_length=2000)


class ProjectPatch(BaseModel):
    name: str | None = Field(None, max_length=200)
    goal: str | None = Field(None, max_length=2000)


class AssignIn(BaseModel):
    kind: str = Field(description="experiment or code")
    id: int
    project_id: int | None = Field(None, description="The project to put it in; null takes it out of its project")


@router.get("/api/projects")
def projects_list():
    """Every project with its experiments and code, plus what isn't in a project yet."""
    return projects.list_all()


@router.post("/api/projects")
def projects_create(body: ProjectIn):
    try:
        return {"id": projects.create(body.name, body.goal)}
    except ValueError as e:
        _bad(e)


@router.put("/api/projects/assign")
def projects_assign(body: AssignIn):
    try:
        projects.assign(body.kind, body.id, body.project_id)
    except ValueError as e:
        _bad(e)
    return {"ok": True}


@router.get("/api/projects/{pid}")
def projects_get(pid: int):
    p = projects.get(pid)
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@router.patch("/api/projects/{pid}")
def projects_update(pid: int, body: ProjectPatch):
    try:
        return projects.update(pid, body.model_dump(exclude_unset=True))
    except ValueError as e:
        _bad(e)


@router.delete("/api/projects/{pid}")
def projects_delete(pid: int):
    """Removes the project only: its experiments and code stay, outside any project."""
    projects.delete(pid)
    return {"ok": True}


def _project(pid: int) -> None:
    if not projects.get(pid):
        raise HTTPException(404, "Project not found")


@router.get("/api/projects/{pid}/data")
def projects_data(pid: int):
    """This project's training and test files (.jsonl, .txt), with their paths for the training forms."""
    from .. import imports
    _project(pid)
    return {"datasets": imports.datasets(pid)}


@router.put("/api/projects/{pid}/data")
async def projects_data_upload(pid: int, request: Request, name: str):
    """A .jsonl or .txt file from your computer: the raw file is the request body."""
    from .. import imports
    from .models import _size
    _project(pid)
    try:
        return await imports.dataset(request.stream(), pid, name, _size(request))
    except ValueError as e:
        _bad(e)


@router.delete("/api/projects/{pid}/data")
def projects_data_delete(pid: int, name: str):
    """Deletes one uploaded data file of this project (the owner's explicit choice)."""
    from .. import imports
    _project(pid)
    try:
        imports.delete_dataset(pid, name)
    except ValueError as e:
        _bad(e)
    return {"ok": True}
