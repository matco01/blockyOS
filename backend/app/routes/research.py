"""Research projects: custom programs, jobs, registered methods."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .. import methods, research_jobs, torchlab

router = APIRouter()


@router.get("/api/research/projects")
def research_projects():
    python = __import__("sys").executable
    return {"projects": research_jobs.projects(), "python": python,
            "environments": [{"label": label, "python": str(path)} for label, path in
                             (("Standard Python (CPU projects)", Path(python)), ("ML environment (PyTorch / Transformers)", torchlab.PY)) if path.is_file()]}


@router.post("/api/research/jobs/{jid}/experiment")
def research_promote(jid: int, body: dict):
    try:
        return methods.promote(jid, body.get("name"))
    except (ValueError, TypeError, OSError) as e:
        raise HTTPException(400, str(e))


@router.get("/api/research/methods")
def research_methods():
    return methods.list_methods()


@router.post("/api/research/jobs/{jid}/register")
def research_register(jid: int, body: dict):
    try:
        return methods.register(jid, body.get("name"))
    except (ValueError, TypeError, OSError) as e:
        raise HTTPException(400, str(e))


@router.post("/api/research/projects")
def research_create(body: dict):
    try:
        return research_jobs.create(body.get("name"), body.get("template", "byte"))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@router.get("/api/research/projects/{pid}/files")
def research_files(pid: int):
    try:
        return research_jobs.files(pid)
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/api/research/projects/{pid}/file")
def research_file(pid: int, name: str):
    try:
        return {"content": research_jobs.read_file(pid, name)}
    except (ValueError, OSError, UnicodeError) as e:
        raise HTTPException(400, str(e))


@router.put("/api/research/projects/{pid}/file")
def research_write(pid: int, body: dict):
    try:
        research_jobs.write_file(pid, body["name"], body["content"])
        return {"ok": True}
    except (ValueError, KeyError, TypeError, OSError) as e:
        raise HTTPException(400, str(e))


@router.get("/api/research/projects/{pid}/jobs")
def research_job_list(pid: int):
    return research_jobs.jobs(pid)


@router.post("/api/research/projects/{pid}/jobs")
def research_queue(pid: int, body: dict):
    try:
        return research_jobs.queue(pid, body)
    except (ValueError, TypeError, OSError) as e:
        raise HTTPException(400, str(e))


@router.post("/api/research/jobs/{jid}/cancel")
def research_cancel(jid: int):
    try:
        research_jobs.cancel(jid)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/api/research/jobs/{jid}/artifact")
def research_artifact(jid: int, name: str):
    try:
        path = research_jobs.artifact(jid, name)
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.get("/api/research/compare")
def research_compare(ids: str):
    try:
        return research_jobs.compare(ids.split(","))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@router.get("/api/research/profile-runs")
def research_profile_runs():
    return research_jobs.profile_runs()


@router.put("/api/research/projects/{pid}/upload")
async def research_upload(pid: int, request: Request, path: str):
    """One file of your code at `path` inside the workspace (folders are created). The raw file is the request body."""
    from .. import imports
    from .models import _size
    try:
        return await imports.code_file(request.stream(), pid, path, _size(request))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.put("/api/research/projects/{pid}/import-zip")
async def research_import_zip(pid: int, request: Request):
    """A .zip of your code, unpacked into the workspace. The raw archive is the request body."""
    from .. import imports
    from .models import _size
    try:
        return await imports.code_zip(request.stream(), pid, _size(request))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/research/projects/{pid}/entries")
def research_entries(pid: int):
    """Python files that could start the program, most likely first."""
    from .. import imports
    try:
        return {"entries": imports.entry_candidates(pid)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/api/research/projects/{pid}/import-git")
async def research_import_git(pid: int, body: dict):
    """Your code from a public Git repository link (shallow clone, no history; Git LFS files are not downloaded)."""
    from .. import imports
    try:
        return await imports.code_git(pid, str(body.get("url") or ""), str(body.get("ref") or ""))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/research/projects/{pid}/env")
def research_env(pid: int):
    """This workspace's own Python environment: whether it exists, what the code asks for, the install log."""
    from .. import envs
    try:
        return envs.status(pid)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/api/research/projects/{pid}/env/install")
async def research_env_install(pid: int, body: dict | None = None):  # async: starts a background task
    """Creates the environment if needed and installs requirements.txt (or the pyproject/setup.py package).
    Downloads from the Python package index; only when the owner asks."""
    from .. import envs
    try:
        return envs.install(pid, bool((body or {}).get("reuse_ml", True)))
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/api/research/projects/{pid}/env")
def research_env_delete(pid: int):
    """Deletes this workspace's environment; the code is untouched."""
    from .. import envs
    try:
        envs.remove(pid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}
