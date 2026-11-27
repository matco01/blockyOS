"""Bringing your own files in: model files, training data and code, uploaded from the browser.

Uploads arrive as the raw request body (no multipart parsing) and are streamed to disk, so multi-GB model files never
sit in memory. Every file is written under a temporary name and renamed only when complete and checked.
"""
import os
import re
import shutil
import zipfile
from pathlib import Path

from . import config, lab, research_jobs

DATASETS = config.DATA_DIR / "datasets"
DATA_SUFFIXES = (".jsonl", ".txt")
MAX_DATA = 20 * 2**30          # one training file
MAX_CODE_FILE = 2 * 2**30      # one file of uploaded code (may include small data or weights)
MAX_ZIP_TOTAL = 20 * 2**30     # everything a code archive unpacks to
MAX_ZIP_FILES = 20_000
DISK_RESERVE = 2 * 2**30       # never fill the disk completely


def safe_name(name: str, suffixes: tuple[str, ...]) -> str:
    """A plain file name: no folders, only letters, digits, dot, dash and underscore."""
    base = Path(str(name or "")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-.")
    if not base or not base.lower().endswith(suffixes):
        raise ValueError(f"The file must end in {' or '.join(suffixes)}")
    return base[:120]


def _check_space(folder: Path, size: int | None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if size is not None and size + DISK_RESERVE > free:
        raise ValueError(f"Not enough disk space: the file needs {size / 2**30:.1f} GB and {free / 2**30:.1f} GB is free")


async def receive(stream, dest: Path, max_bytes: int, size: int | None = None, magic: bytes | None = None) -> int:
    """Streams a request body to `dest` (via a temporary file). Refuses existing targets, oversize bodies and files
    that don't start with `magic`."""
    if dest.exists():
        raise ValueError(f"{dest.name} already exists. Rename the file or delete the old one first.")
    if size is not None and size > max_bytes:
        raise ValueError(f"The file is larger than the {max_bytes / 2**30:.0f} GB limit")
    _check_space(dest.parent, size)
    part = dest.with_name(dest.name + ".part")
    written, head = 0, b""
    try:
        with open(part, "wb") as f:
            async for chunk in stream:
                if not chunk:
                    continue
                if magic and len(head) < len(magic):
                    head += chunk[:len(magic) - len(head)]
                    if not magic.startswith(head):
                        raise ValueError("This doesn't look like the right kind of file")
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError(f"The file is larger than the {max_bytes / 2**30:.0f} GB limit")
                f.write(chunk)
        if written == 0:
            raise ValueError("The file is empty")
        if magic and head != magic:
            raise ValueError("This doesn't look like the right kind of file")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    return written


# ---------- Models ----------

async def model(stream, name: str, size: int | None) -> dict:
    """A GGUF model file from your computer, into the models folder (then it is like any downloaded model)."""
    file = safe_name(name, (".gguf",))
    written = await receive(stream, lab.MODELS_DIR / file, max_bytes=2**40, size=size, magic=b"GGUF")
    return {"file": file, "bytes": written}


# ---------- Training data (per project) ----------

def data_dir(pid: int) -> Path:
    return DATASETS / f"project-{int(pid)}"


def datasets(pid: int) -> list[dict]:
    folder = data_dir(pid)
    out = []
    for p in sorted(folder.glob("*")) if folder.is_dir() else []:
        if p.is_file() and p.suffix.lower() in DATA_SUFFIXES:
            rows = None
            if p.stat().st_size <= 200 * 2**20:
                with open(p, "rb") as f:
                    rows = sum(1 for line in f if line.strip())
            out.append({"name": p.name, "path": str(p), "bytes": p.stat().st_size, "rows": rows, "kind": p.suffix.lower()[1:]})
    return out


async def dataset(stream, pid: int, name: str, size: int | None) -> dict:
    file = safe_name(name, DATA_SUFFIXES)
    dest = data_dir(pid) / file
    written = await receive(stream, dest, max_bytes=MAX_DATA, size=size)
    if file.lower().endswith(".jsonl"):
        _check_jsonl(dest)
    return {"name": file, "path": str(dest), "bytes": written}


def _check_jsonl(path: Path) -> None:
    """The first lines must be JSON objects, or training fails much later with a confusing error."""
    import json
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            if not line.strip():
                continue
            try:
                if not isinstance(json.loads(line), dict):
                    raise ValueError
            except ValueError:
                path.unlink(missing_ok=True)
                raise ValueError(f"Line {i + 1} isn't a JSON object. Each line of a .jsonl file must look like "
                                 '{"instruction": "...", "output": "..."}') from None


def delete_dataset(pid: int, name: str) -> None:
    file = safe_name(name, DATA_SUFFIXES)
    (data_dir(pid) / file).unlink(missing_ok=True)


# ---------- Code ----------

async def code_file(stream, rid: int, rel: str, size: int | None) -> dict:
    """One file of your code, at its path inside the code workspace (folders are created)."""
    root = research_jobs.folder(rid) / "source"
    rel = str(rel or "").replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or any(p in (".git", "__pycache__", ".venv", "node_modules") for p in parts):
        raise ValueError(f"Skipped {rel or 'a file without a name'}")
    dest = research_jobs.confined(root, "/".join(parts))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)  # uploading the same folder again replaces its files
    written = await receive(stream, dest, max_bytes=MAX_CODE_FILE, size=size)
    return {"path": "/".join(parts), "bytes": written}


async def code_zip(stream, rid: int, size: int | None) -> dict:
    """A .zip of your code, unpacked into the workspace. A single top folder in the archive is stripped."""
    root = research_jobs.folder(rid) / "source"
    tmp = root.parent / "upload.zip"
    tmp.unlink(missing_ok=True)
    await receive(stream, tmp, max_bytes=MAX_ZIP_TOTAL, size=size, magic=b"PK")
    try:
        with zipfile.ZipFile(tmp) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            if len(infos) > MAX_ZIP_FILES:
                raise ValueError(f"The archive has more than {MAX_ZIP_FILES} files")
            if sum(i.file_size for i in infos) > MAX_ZIP_TOTAL:
                raise ValueError("The archive unpacks to more than 20 GB")
            names = [i.filename.replace("\\", "/") for i in infos]
            tops = {n.split("/", 1)[0] for n in names}
            strip = len(tops) == 1 and all("/" in n for n in names)
            _check_space(root, sum(i.file_size for i in infos))
            count = 0
            for info, name in zip(infos, names):
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"{name} is a symbolic link; links aren't supported")
                rel = name.split("/", 1)[1] if strip else name
                parts = [p for p in rel.split("/") if p not in ("", ".")]
                if not parts or any(p in (".git", "__pycache__", ".venv", "node_modules", "__MACOSX") for p in parts):
                    continue
                if any(p == ".." for p in parts) or rel.startswith("/"):
                    raise ValueError(f"{name} points outside the archive")
                dest = research_jobs.confined(root, "/".join(parts))
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.unlink(missing_ok=True)  # replace, never edit in place (run snapshots may share the old file)
                with z.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                count += 1
    except zipfile.BadZipFile:
        raise ValueError("This isn't a valid .zip file") from None
    finally:
        tmp.unlink(missing_ok=True)
    return {"files": count, "entry_candidates": entry_candidates(rid)}


_GIT_URL = re.compile(r"^https://[A-Za-z0-9.-]+(:\d{1,5})?/[A-Za-z0-9._~/%+-]+$")
_GIT_REF = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
SKIP_DIRS = (".git", "__pycache__", ".venv", "venv", "node_modules", "__MACOSX")


async def code_git(rid: int, url: str, ref: str = "") -> dict:
    """Your code from a public Git repository link (GitHub, GitLab, ...): a shallow clone, without its history."""
    import asyncio
    import tempfile
    url, ref = url.strip().removesuffix("/"), ref.strip()
    if not _GIT_URL.match(url):
        raise ValueError("Paste the repository's https:// link, e.g. https://github.com/owner/project")
    if ref and (not _GIT_REF.match(ref) or ref.startswith("-") or ".." in ref):
        raise ValueError("Branch or tag: letters, digits, dot, dash, underscore and slash only")
    root = research_jobs.folder(rid) / "source"
    _check_space(root, None)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix="clone-") as tmp:
        cmd = ["git", "-c", "protocol.allow=never", "-c", "protocol.https.allow=always", "-c", "core.hooksPath=/dev/null",
               "clone", "--depth", "1", "--single-branch", "--no-tags"] + (["--branch", ref] if ref else []) + ["--", url, f"{tmp}/repo"]
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/false", "GIT_LFS_SKIP_SMUDGE": "1"}
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=1800)
        except asyncio.TimeoutError:
            proc.kill()
            raise ValueError("Cloning took longer than 30 minutes") from None
        if proc.returncode != 0:
            text = out.decode(errors="replace")
            if "could not read Username" in text or "Authentication failed" in text or "not found" in text.lower():
                raise ValueError("Couldn't open that repository. Check the link; private repositories aren't supported yet (download a .zip instead)")
            raise ValueError("git clone failed: " + text.strip().splitlines()[-1][:300] if text.strip() else "git clone failed")
        repo = Path(tmp) / "repo"
        files = [p for p in repo.rglob("*") if p.is_file() and not any(x in p.relative_to(repo).parts for x in SKIP_DIRS)]
        if any(p.is_symlink() for p in files):
            files = [p for p in files if not p.is_symlink()]  # links can point anywhere: left out
        if sum(p.stat().st_size for p in files) > MAX_ZIP_TOTAL:
            raise ValueError("The repository is larger than 20 GB")
        lfs = 0
        for p in files:
            rel = p.relative_to(repo).as_posix()
            if p.stat().st_size < 200:
                with open(p, "rb") as f:
                    lfs += f.read(40).startswith(b"version https://git-lfs.github.com/spec")
            dest = research_jobs.confined(root, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.unlink(missing_ok=True)
            os.replace(p, dest)
    return {"files": len(files), "lfs_pointers": lfs, "entry_candidates": entry_candidates(rid)}


# ---------- Trainable models (Hugging Face format) from your computer ----------

def _checkpoint_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "")).strip("-.")[:80]
    if not slug:
        raise ValueError("Give the model a name")
    return slug


def _hf() -> Path:
    from . import torchlab
    return torchlab.HF


async def checkpoint_file(stream, name: str, rel: str, size: int | None) -> dict:
    """One file of a model folder (config.json, *.safetensors, tokenizer files). It stays hidden until `finish`."""
    staging = _hf() / f".upload-{_checkpoint_slug(name)}"
    parts = [p for p in str(rel or "").replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or p.startswith(".") for p in parts):
        raise ValueError(f"Skipped {rel}")
    dest = (staging / "/".join(parts)).resolve()
    if not dest.is_relative_to(staging.resolve()):
        raise ValueError(f"Skipped {rel}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    written = await receive(stream, dest, max_bytes=2**40, size=size)
    return {"path": "/".join(parts), "bytes": written}


def checkpoint_finish(name: str) -> dict:
    """Checks the uploaded folder is a Hugging Face model and makes it available for training and conversion."""
    slug = _checkpoint_slug(name)
    staging = _hf() / f".upload-{slug}"
    if not staging.is_dir():
        raise ValueError("Nothing was uploaded for this model")
    # A folder chosen in the browser may arrive inside its own top folder: use the folder that holds config.json.
    configs = sorted(staging.rglob("config.json"), key=lambda p: len(p.parts))
    if not configs:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("No config.json found. Choose the folder that contains config.json and the .safetensors files")
    top = configs[0].parent
    if not any(top.glob("*.safetensors")) and not any(top.glob("*.bin")):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("No weights found (.safetensors or .bin) next to config.json")
    final = _hf() / f"local--{slug}"
    if final.exists():
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"A model named {slug} already exists")
    os.replace(top, final)
    shutil.rmtree(staging, ignore_errors=True)
    return {"id": final.name, "repo": f"local/{slug}"}


def entry_candidates(rid: int) -> list[str]:
    """Python files that could start the program, most likely first."""
    not_programs = ("setup.py", "conftest.py", "__init__.py", "__main__.py", "noxfile.py")
    py = [f for f in research_jobs.files(rid) if f.endswith(".py") and Path(f).name not in not_programs
          and not Path(f).name.startswith("test_") and not any(p in ("tests", "test", "docs") for p in Path(f).parts[:-1])]
    likely = ("main.py", "train.py", "run.py", "app.py", "evaluate.py", "eval.py")
    return sorted(py, key=lambda f: (f.count("/"), likely.index(Path(f).name) if Path(f).name in likely else len(likely), f))
