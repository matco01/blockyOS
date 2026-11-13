"""Inference engines: the llama.cpp build itself as an experiment variable.

"stock" is the build Blocky installed. Researchers can add any git repository + ref (a branch, tag, commit or PR head
of llama.cpp or a fork): Blocky clones it, builds the tools experiments need with the same CUDA settings, and the
engine becomes selectable in Experiments next to weights, KV cache and so on. A/B a new kernel or quantization format
against stock with identical measurements and provenance.

Building runs code from that repository on this machine (as the unprivileged `blocky` user). Only add repositories
you trust.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import config, db, lab

ROOT = config.DATA_DIR.parent / "engines"          # /opt/blocky/engines/<id>/bin
SRC = ROOT / "src"
TARGETS = ["llama-server", "llama-bench", "llama-perplexity"]
_REPO = re.compile(r"^https://[A-Za-z0-9.-]+(:\d+)?/[A-Za-z0-9._/~-]+?(\.git)?$")
_REF = re.compile(r"^[A-Za-z0-9._/+-]{1,120}$")
_NAME = re.compile(r"[^A-Za-z0-9._-]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS engines (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    name        TEXT NOT NULL,
    repo        TEXT NOT NULL,
    ref         TEXT NOT NULL,
    commit_sha  TEXT,
    status      TEXT NOT NULL,
    step        TEXT,
    error       TEXT,
    log         TEXT,
    cmake_args  TEXT,
    built_at    REAL
);
"""
_cancel: set[str] = set()


def init() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with db.connect() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE engines SET status = 'error', error = 'Interrupted by a restart' WHERE status IN ('queued', 'building')")


def bin_dir(engine: str | None) -> Path:
    """Where an engine's binaries live. 'stock' (or nothing) = the build Blocky installed."""
    if not engine or engine == "stock":
        return lab.LLAMA_BIN
    if not re.fullmatch(r"[a-z0-9-]{1,40}", engine):
        raise ValueError("Unknown engine")
    d = ROOT / engine / "bin"
    if not (d / "llama-server").is_file():
        raise ValueError(f"Engine '{engine}' isn't built")
    return d


def _row(r) -> dict:
    d = dict(r)
    d.pop("log", None)
    return d


def list_engines() -> list[dict]:
    with db.connect() as c:
        rows = [_row(r) for r in c.execute("SELECT * FROM engines ORDER BY created_at")]
    stock = {"id": "stock", "name": "stock", "repo": "https://github.com/ggml-org/llama.cpp", "ref": "installed",
             "commit_sha": _stock_commit(), "status": "ready", "built_at": None}
    return [stock] + rows


def get_log(engine_id: str) -> str:
    with db.connect() as c:
        r = c.execute("SELECT log FROM engines WHERE id = ?", (engine_id,)).fetchone()
    return (r["log"] if r else "") or ""


_stock: str | None = None


def _stock_commit() -> str | None:
    global _stock
    if _stock is None:
        try:
            out = subprocess.run([str(lab.LLAMA_BIN / "llama-server"), "--version"], capture_output=True, text=True, timeout=20,
                                 env={**os.environ, "LD_LIBRARY_PATH": str(lab.LLAMA_BIN)})
            m = re.search(r"commit\s*([0-9a-f]{6,})|\(([0-9a-f]{6,})\)", out.stdout + out.stderr)
            _stock = (m.group(1) or m.group(2)) if m else ""
        except (OSError, subprocess.SubprocessError):
            _stock = ""
    return _stock or None


def _cuda_arch() -> str | None:
    """Compute capability of the installed GPU, e.g. '75' (builds for just this GPU are much faster)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        cc = out.strip().splitlines()[0].strip()
        return cc.replace(".", "") if re.fullmatch(r"\d+\.\d+", cc) else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def add(body: dict) -> str:
    repo = str(body.get("repo") or "").strip()
    ref = str(body.get("ref") or "").strip() or "master"
    if not _REPO.match(repo):
        raise ValueError("Use an https git URL, e.g. https://github.com/ggml-org/llama.cpp")
    if not _REF.match(ref) or ref.startswith("-") or ".." in ref:
        raise ValueError("Use a branch, tag or commit, e.g. master, b6000 or a commit hash")
    name = _NAME.sub("-", str(body.get("name") or "").strip()).strip("-")[:40] or f"{repo.rstrip('/').split('/')[-1].removesuffix('.git')}-{ref}"[:40]
    engine_id = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:30] or "engine"
    with db.connect() as c:
        existing = {r["id"] for r in c.execute("SELECT id FROM engines")}
    base, n = engine_id, 2
    while engine_id in existing or engine_id == "stock":
        engine_id, n = f"{base}-{n}", n + 1
    arch = _cuda_arch()
    args = ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"]
    if Path("/usr/local/cuda/bin/nvcc").is_file():
        args += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch or '61;75;86;89'}"]
    with db.connect() as c:
        c.execute("INSERT INTO engines (id, created_at, name, repo, ref, status, cmake_args) VALUES (?, ?, ?, ?, ?, 'queued', ?)",
                  (engine_id, time.time(), name, repo, ref, json.dumps(args)))
    asyncio.get_running_loop().create_task(_build(engine_id))
    return engine_id


def cancel(engine_id: str) -> None:
    _cancel.add(engine_id)


def delete(engine_id: str) -> None:
    if engine_id == "stock":
        raise ValueError("The stock engine can't be removed")
    with db.connect() as c:
        r = c.execute("SELECT status FROM engines WHERE id = ?", (engine_id,)).fetchone()
        if r and r["status"] in ("queued", "building"):
            raise ValueError("Stop the build first")
        c.execute("DELETE FROM engines WHERE id = ?", (engine_id,))
    shutil.rmtree(ROOT / engine_id, ignore_errors=True)
    shutil.rmtree(SRC / engine_id, ignore_errors=True)


def _set(engine_id: str, **f) -> None:
    with db.connect() as c:
        c.execute(f"UPDATE engines SET {', '.join(f'{k} = ?' for k in f)} WHERE id = ?", (*f.values(), engine_id))


async def _sh(engine_id: str, cmd: list[str], log: list[str], cwd: Path | None = None, timeout: float = 3600) -> int:
    env = {**os.environ, "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}", "GIT_TERMINAL_PROMPT": "0"}
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(cwd) if cwd else None, env=env,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    start = time.monotonic()
    async for raw in proc.stdout:
        log.append(raw.decode(errors="replace").rstrip()[:400])
        del log[:-400]
        if engine_id in _cancel or time.monotonic() - start > timeout:
            proc.kill()
            raise asyncio.CancelledError
        if len(log) % 25 == 0:
            _set(engine_id, log="\n".join(log[-120:]))
    return await proc.wait()


async def _build(engine_id: str) -> None:
    with db.connect() as c:
        e = dict(c.execute("SELECT * FROM engines WHERE id = ?", (engine_id,)).fetchone())
    log: list[str] = []
    src = SRC / engine_id
    # Builds are CPU-heavy: never overlap with a measurement.
    async with lab._lock:
        lab.busy = "engine"
        _set(engine_id, status="building")
        try:
            shutil.rmtree(src, ignore_errors=True)
            src.parent.mkdir(parents=True, exist_ok=True)
            _set(engine_id, step="downloading source")
            if await _sh(engine_id, ["git", "clone", "--filter=blob:none", "--no-checkout", e["repo"], str(src)], log, timeout=1200):
                raise RuntimeError("git clone failed")
            if await _sh(engine_id, ["git", "-c", "advice.detachedHead=false", "checkout", e["ref"]], log, cwd=src):
                # A PR head or a commit that isn't on a default branch: fetch it explicitly.
                if await _sh(engine_id, ["git", "fetch", "origin", e["ref"]], log, cwd=src) or \
                   await _sh(engine_id, ["git", "-c", "advice.detachedHead=false", "checkout", "FETCH_HEAD"], log, cwd=src):
                    raise RuntimeError(f"Couldn't find '{e['ref']}' in that repository")
            sha = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            _set(engine_id, commit_sha=sha[:12], step="configuring")
            if await _sh(engine_id, ["cmake", "-S", str(src), "-B", str(src / "build"), *json.loads(e["cmake_args"])], log):
                raise RuntimeError("cmake configure failed (see log)")
            _set(engine_id, step="compiling (10–20 min with CUDA)")
            if await _sh(engine_id, ["cmake", "--build", str(src / "build"), "--config", "Release", "-j", str(os.cpu_count() or 4),
                                     "--target", *TARGETS], log, timeout=7200):
                raise RuntimeError("Build failed (see log)")
            out = ROOT / engine_id / "bin"
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            for f in (src / "build" / "bin").iterdir():
                if f.is_file() and (f.name in TARGETS or ".so" in f.name):
                    shutil.copy2(f, out / f.name, follow_symlinks=True)
            missing = [t for t in TARGETS if not (out / t).is_file()]
            if missing:
                raise RuntimeError(f"Build finished but {', '.join(missing)} is missing")
            shutil.rmtree(src / "build", ignore_errors=True)  # keep the source for reference, drop the ~GB build tree
            _set(engine_id, status="ready", step=None, built_at=time.time(), log="\n".join(log[-120:]))
        except asyncio.CancelledError:
            _set(engine_id, status="cancelled", step=None, log="\n".join(log[-120:]))
        except Exception as ex:  # noqa: BLE001
            _set(engine_id, status="error", step=None, error=str(ex)[:300], log="\n".join(log[-120:]))
        finally:
            _cancel.discard(engine_id)
            lab.busy = None
