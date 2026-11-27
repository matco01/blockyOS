"""A Python environment per code workspace, so uploaded code gets the packages it needs with one button.

The environment lives next to the workspace's source (`<workspace>/env`), outside the code that runs are snapshotted
from. When the ML environment exists and uses the same Python version, its packages (PyTorch, Transformers, ...) are
reused through a .pth file instead of being downloaded again; packages the code asks for that are missing, or in other
versions, are installed into the workspace's own environment and take precedence.

Installing downloads packages from the Python package index: it only ever happens when the owner presses the button.
"""
import asyncio
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

from . import research_jobs, torchlab

state: dict[int, dict] = {}   # workspace id -> {"status": "installing"|"ready"|"error", "error", "started"}


def _base(rid: int) -> Path:
    return research_jobs.folder(rid)


def python(rid: int) -> Path:
    return _base(rid) / "env" / "bin" / "python"


def _requirements(rid: int) -> dict:
    src = _base(rid) / "source"
    for name in ("requirements.txt", "pyproject.toml", "setup.py"):
        if (src / name).is_file():
            packages = []
            if name == "requirements.txt":
                for line in (src / name).read_text(errors="replace").splitlines():
                    line = line.split("#", 1)[0].strip()
                    if line and not line.startswith("-"):
                        packages.append(line)
            return {"file": name, "packages": packages[:200], "sha256": hashlib.sha256((src / name).read_bytes()).hexdigest()}
    return {"file": None, "packages": [], "sha256": None}


def _log(rid: int) -> Path:
    return _base(rid) / "env-install.log"


def status(rid: int) -> dict:
    req = _requirements(rid)
    env = _base(rid) / "env"
    marker = env / ".blocky-installed"
    installed = marker.read_text().strip() if marker.is_file() else None
    s = state.get(rid, {})
    current = s.get("status") or ("ready" if installed else "none")
    log = _log(rid).read_text(errors="replace")[-6000:] if _log(rid).is_file() else ""
    return {"status": current, "error": s.get("error"), "python": str(python(rid)) if python(rid).is_file() else None,
            "requirements": req, "up_to_date": bool(installed) and installed == (req["sha256"] or "none"),
            "reuses_ml": (env / "blocky-ml.pth").is_file() or any(env.glob("lib/python*/site-packages/blocky-ml.pth")),
            "ml_available": torchlab.PY.is_file(), "log": log}


def install(rid: int, reuse_ml: bool = True) -> dict:
    if state.get(rid, {}).get("status") == "installing":
        raise ValueError("Already installing")
    req = _requirements(rid)
    state[rid] = {"status": "installing", "error": None, "started": time.time()}
    asyncio.get_running_loop().create_task(_install(rid, reuse_ml and torchlab.PY.is_file(), req))
    return status(rid)


async def _run(cmd: list[str], log, timeout: float, cwd: Path | None = None) -> int:
    log.write(("$ " + " ".join(cmd) + "\n").encode())
    log.flush()
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1", "PYTHONUNBUFFERED": "1"}
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=log, stderr=asyncio.subprocess.STDOUT, cwd=cwd, env=env)
    try:
        return await asyncio.wait_for(proc.wait(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"Took longer than {timeout / 60:.0f} minutes") from None


async def _out(cmd: list[str]) -> str:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    return out.decode().strip() if proc.returncode == 0 else ""


async def _install(rid: int, reuse_ml: bool, req: dict) -> None:
    base = _base(rid)
    env, src = base / "env", base / "source"
    try:
        with open(_log(rid), "wb") as log:
            if not python(rid).is_file():
                if await _run([sys.executable, "-m", "venv", str(env)], log, 600) != 0:
                    raise RuntimeError("Couldn't create the environment (is python3-venv installed?)")
            site = await _out([str(python(rid)), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
            pth = Path(site) / "blocky-ml.pth" if site else None
            if pth is not None:
                if reuse_ml:
                    ours = await _out([str(python(rid)), "-c", "import sys; print(sys.version_info[:2])"])
                    theirs = await _out([str(torchlab.PY), "-c", "import sys; print(sys.version_info[:2])"])
                    ml_site = await _out([str(torchlab.PY), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
                    if ours and ours == theirs and ml_site:
                        pth.write_text(ml_site + "\n")
                        log.write(f"Reusing the ML environment's packages from {ml_site}\n".encode())
                    else:
                        pth.unlink(missing_ok=True)
                        log.write(b"The ML environment uses another Python version: not reused\n")
                else:
                    pth.unlink(missing_ok=True)
            if req["file"] == "requirements.txt":
                cmd = [str(python(rid)), "-m", "pip", "install", "-r", str(src / "requirements.txt")]
            elif req["file"]:
                cmd = [str(python(rid)), "-m", "pip", "install", str(src)]
            else:
                cmd = None
                log.write(b"No requirements.txt, pyproject.toml or setup.py: the environment has only the standard library"
                          + (b" and the ML packages" if reuse_ml else b"") + b".\n")
            if cmd and await _run(cmd, log, 7200, cwd=src) != 0:
                raise RuntimeError("Installing the packages failed: see the log")
        (env / ".blocky-installed").write_text(req["sha256"] or "none")
        state[rid] = {"status": "ready", "error": None}
    except Exception as e:  # noqa: BLE001
        state[rid] = {"status": "error", "error": str(e)[:300]}


def remove(rid: int) -> None:
    """Deletes the workspace's environment (the owner's choice); the code is untouched."""
    if state.get(rid, {}).get("status") == "installing":
        raise ValueError("Wait for the installation to finish")
    shutil.rmtree(_base(rid) / "env", ignore_errors=True)
    _log(rid).unlink(missing_ok=True)
    state.pop(rid, None)
