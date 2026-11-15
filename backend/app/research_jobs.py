"""General local research programs, isolated source snapshots and declared evaluation contracts."""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time

from . import config, db, lab

ROOT = config.DATA_DIR / "research-projects"
SDK = Path(__file__).resolve().parents[2] / "client"
_cancel = set()
SCHEMA = """
CREATE TABLE IF NOT EXISTS research_projects (
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS research_jobs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER NOT NULL REFERENCES research_projects(id),
 parent_id INTEGER REFERENCES research_jobs(id), created_at REAL NOT NULL, finished_at REAL,
 status TEXT NOT NULL, spec TEXT NOT NULL, provenance TEXT NOT NULL DEFAULT '{}', error TEXT
);
"""


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Non-finite report value")
    return value


def init():
    ROOT.mkdir(parents=True, exist_ok=True)
    with db.connect() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE research_jobs SET status='error',error='Interrupted while preparing source snapshot' WHERE status='preparing'")
        c.execute("UPDATE research_jobs SET status='error', error='Interrupted by backend restart', finished_at=? WHERE status='running'", (time.time(),))


def folder(pid):
    with db.connect() as c:
        if not c.execute("SELECT 1 FROM research_projects WHERE id=?", (pid,)).fetchone():
            raise ValueError("Project not found")
    return ROOT / str(pid)


def confined(root, name):
    path = (root / str(name)).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Path must be inside the project")
    # Symlinks are excluded even when their current target happens to be inside the project.
    candidate = root / str(name)
    if any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root.parent):
        raise ValueError("Symlinks are not supported")
    return path


def projects():
    with db.connect() as c:
        return [{**dict(r), "path": str(ROOT / str(r["id"]) / "source")} for r in c.execute("SELECT * FROM research_projects ORDER BY id DESC")]


def create(name, template="byte"):
    if template not in ("byte", "transformer", "blank"):
        raise ValueError("Unknown template")
    with db.connect() as c:
        pid = c.execute("INSERT INTO research_projects (name,created_at) VALUES (?,?)", (str(name or "").strip()[:100] or "Research project", time.time())).lastrowid
    source = ROOT / str(pid) / "source"
    source.mkdir(parents=True)
    templates = Path(__file__).resolve().parents[1] / "research_templates"
    shutil.copy(templates / (template + ".py"), source / "main.py")
    # Deliberately tiny, identified inputs for checking the pipeline, not a benchmark dataset.
    (source / "train.txt").write_text("The river flows past the old bridge. A small boat moves with the current.\n" * 10, encoding="utf-8")
    (source / "eval.txt").write_text("The boat passes under the bridge.\n", encoding="utf-8")
    (source / "requirements.txt").write_text("torch\ntransformers\n" if template == "transformer" else "", encoding="utf-8")
    (source / "README.md").write_text("Replace these tiny smoke-test inputs with your training and held-out evaluation data. main.py is editable here or on disk. Dependencies are installed by you in your chosen environment. Programs run with the Blocky account's permissions.\n", encoding="utf-8")
    return {"id": pid, "path": str(source)}


def files(pid):
    root = folder(pid) / "source"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not p.is_symlink() and not any(x in p.parts for x in (".venv", ".git", "__pycache__")))[:500]


def read_file(pid, name):
    path = confined(folder(pid) / "source", name)
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("Choose a text file smaller than 1 MB")
    return path.read_text(encoding="utf-8")


def write_file(pid, name, content):
    path = confined(folder(pid) / "source", name)
    if not isinstance(content, str) or len(content.encode()) > 1_000_000:
        raise ValueError("Files edited here must be smaller than 1 MB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def job(jid):
    with db.connect() as c:
        row = c.execute("SELECT * FROM research_jobs WHERE id=?", (jid,)).fetchone()
    if not row:
        raise ValueError("Job not found")
    r = dict(row)
    for key in ("spec", "provenance"):
        r[key] = json.loads(r[key])
    root = ROOT / str(r["project_id"]) / "runs" / str(jid)
    events = []
    path = root / "events.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as f:
            for _ in range(10000):
                line = f.readline(256001)
                if not line:
                    break
                if len(line) > 256000:
                    break
                if len(events) >= 10000:
                    break
                try:
                    event = json.loads(line, parse_float=_finite_float, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                    if isinstance(event, dict):
                        samples = event.get("samples")
                        if isinstance(samples, list) and samples and all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in samples):
                            event["spread"] = {"n": len(samples), "sd": statistics.stdev(samples) if len(samples) > 1 else None,
                                               "note": "Descriptive sample SD; independence and evaluator validity are not established"}
                        events.append(event)
                except (ValueError, OverflowError):
                    pass  # a concurrently written last line may not yet be complete
    log = root / "stdout.log"
    if log.is_file():
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 32000))
            r["log"] = f.read().decode(errors="replace")
    else:
        r["log"] = ""
    r.update({"events": events, "directory": str(root)})
    return r


def jobs(pid):
    with db.connect() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM research_jobs WHERE project_id=? ORDER BY id DESC LIMIT 100", (pid,))]
    return [job(jid) for jid in ids]


def queue(pid, body, *, source_root=None, profile_run_id=None):
    source = Path(source_root) if source_root is not None else folder(pid) / "source"
    argv = body.get("argv", ["{python}", "main.py"])
    if not isinstance(argv, list) or not 1 <= len(argv) <= 100 or any(not isinstance(x, str) or not x or "\0" in x for x in argv):
        raise ValueError("Command must be a nonempty list of arguments")
    if sum(map(len, argv)) > 16000:
        raise ValueError("Command is too long")
    python = str(body.get("python") or sys.executable)
    if not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError("Choose an existing absolute Python interpreter path")
    cfg = body.get("config", {})
    if not isinstance(cfg, dict) or len(json.dumps(cfg, allow_nan=False)) > 64000:
        raise ValueError("Config must be a JSON object smaller than 64 KB")
    parent = body.get("parent_id")
    if parent is not None and job(int(parent))["project_id"] != pid:
        raise ValueError("Parent must be in this project")
    timeout = int(body.get("timeout_s") or 3600)
    if not 1 <= timeout <= 7 * 86400:
        raise ValueError("Timeout must be between 1 second and 7 days")
    spec = {"argv": argv, "python": python, "config": cfg, "gpu": bool(body.get("gpu")), "timeout_s": timeout}
    if profile_run_id is not None:
        spec["profile_run_id"] = profile_run_id
    candidates = [p for p in source.rglob("*") if p.is_file() and not any(x in p.relative_to(source).parts for x in (".git", ".venv", "__pycache__"))]
    if len(candidates) > 500 or sum(p.stat().st_size for p in candidates) > 20_000_000:
        raise ValueError("Source snapshot limit is 500 files / 20 MB. Keep datasets/checkpoints outside source and reference their absolute paths")
    for p in candidates:
        confined(source, str(p.relative_to(source)))
    with db.connect() as c:
        jid = c.execute("INSERT INTO research_jobs (project_id,parent_id,created_at,status,spec) VALUES (?,?,?,'preparing',?)", (pid, parent, time.time(), json.dumps(spec))).lastrowid
    try:
        root = ROOT / str(pid) / "runs" / str(jid)
        snapshot = root / "source"
        snapshot.mkdir(parents=True)
        hashes = {}
        for path in candidates:
            relative = path.relative_to(source)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
        if source_root is None or not (snapshot / "blocky_research.py").is_file():
            shutil.copyfile(SDK / "blocky_research.py", snapshot / "blocky_research.py")
        hashes["blocky_research.py"] = hashlib.sha256((snapshot / "blocky_research.py").read_bytes()).hexdigest()
        provenance = {"files": hashes, "snapshot_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "runner": "blocky-research/1"}
        with db.connect() as c:
            c.execute("UPDATE research_jobs SET status='queued', provenance=? WHERE id=?", (json.dumps(provenance), jid))
    except Exception as e:
        with db.connect() as c:
            c.execute("UPDATE research_jobs SET status='error',error=? WHERE id=?", (str(e)[:500], jid))
        raise
    return {"id": jid}


def cancel(jid):
    r = job(jid)
    with db.connect() as c:
        c.execute("UPDATE research_jobs SET status='cancelled',finished_at=? WHERE id=? AND status='queued'", (time.time(), jid))
    if r["status"] == "running":
        _cancel.add(jid)


async def _stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        await asyncio.wait_for(proc.wait(), 3)
    except asyncio.TimeoutError:
        os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()
    except ProcessLookupError:
        pass
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def execute(jid, llm):
    r = job(jid)
    spec = r["spec"]
    root = Path(r["directory"])
    proc = None
    status, error = "done", None
    async with lab._lock:
        # Recheck after waiting for a profile or another heavy job.
        if job(jid)["status"] != "queued":
            return
        lab.busy = "research"
        with db.connect() as c:
            c.execute("UPDATE research_jobs SET status='running' WHERE id=?", (jid,))
        try:
            if spec["gpu"]:
                await lab.llm_service("stop")
            env = {**os.environ, "BLOCKY_RUN_DIR": str(root), "BLOCKY_CONFIG": json.dumps(spec["config"]),
                   "PYTHONPATH": str(root / "source"), "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline"}
            if not spec["gpu"]:
                env["CUDA_VISIBLE_DEVICES"] = ""
            argv = [spec["python"] if x == "{python}" else x for x in spec["argv"]]
            probe = await asyncio.create_subprocess_exec(spec["python"], "-m", "pip", "freeze", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            try:
                packages, _ = await asyncio.wait_for(probe.communicate(), 20)
            except asyncio.TimeoutError:
                probe.kill(); await probe.wait(); packages = b"Environment capture timed out"
            (root / "environment.txt").write_bytes(packages)
            provenance = {**r["provenance"], "hardware": await asyncio.to_thread(lab.hardware), "argv": argv,
                          "environment_sha256": hashlib.sha256(packages).hexdigest()}
            with db.connect() as c:
                c.execute("UPDATE research_jobs SET provenance=? WHERE id=?", (json.dumps(provenance), jid))
            with (root / "stdout.log").open("wb") as log:
                proc = await asyncio.create_subprocess_exec(*argv, cwd=root / "source", env=env, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                while proc.returncode is None:
                    if jid in _cancel:
                        status = "cancelled"; break
                    if time.monotonic() - started > spec["timeout_s"]:
                        status, error = "error", "Job exceeded its timeout"; break
                    try:
                        await asyncio.wait_for(proc.wait(), 0.25)
                    except asyncio.TimeoutError:
                        pass
                if status == "done" and proc.returncode != 0:
                    status, error = "error", f"Program exited with code {proc.returncode}"
        except asyncio.CancelledError:
            status, error = "error", "Interrupted by backend shutdown"
            raise
        except Exception as e:
            status, error = "error", str(e)[:1000]
        finally:
            if proc:
                await _stop(proc)  # also terminate surviving children after the parent exits
            if spec["gpu"]:
                try:
                    await lab.llm_service("start")
                    await lab.wait_ready(llm, timeout=300)
                except Exception as e:
                    status = "error"
                    error = (error or "") + f"; chat restore failed: {e}"
            with db.connect() as c:
                c.execute("UPDATE research_jobs SET status=?,error=?,finished_at=? WHERE id=?", (status, error, time.time(), jid))
            _cancel.discard(jid)
            lab.busy = None


async def worker(llm):
    while True:
        await asyncio.sleep(1)
        try:
            with db.connect() as c:
                # Existing profile queues keep priority.
                profile_waiting = c.execute("SELECT 1 FROM exp_runs WHERE status IN ('queued','running') LIMIT 1").fetchone()
                row = c.execute("SELECT id FROM research_jobs WHERE status='queued' AND json_extract(spec,'$.profile_run_id') IS NULL ORDER BY id LIMIT 1").fetchone()
            if row and not profile_waiting and not lab.busy and not lab._lock.locked():
                await execute(row[0], llm)
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(2)


def artifact(jid, name):
    root = Path(job(jid)["directory"])
    path = confined(root, name)
    if not path.is_file():
        raise ValueError("Artifact not found")
    return path


def profile_runs():
    with db.connect() as c:
        return [dict(r) for r in c.execute("SELECT r.id,r.label,e.name AS experiment FROM exp_runs r JOIN experiments e ON e.id=r.experiment_id WHERE r.status='done' ORDER BY r.id DESC LIMIT 200")]


def _profile_job(rid):
    with db.connect() as c:
        row = c.execute("SELECT * FROM exp_runs WHERE id=?", (rid,)).fetchone()
    if not row:
        raise ValueError("Profile run not found")
    res = json.loads(row["results"])
    return {"id": f"profile:{rid}", "project_id": None, "status": row["status"], "events": list(res["research_metrics"].values()) if res.get("research_metrics") else native_metrics(res),
            "provenance": {"hardware": res.get("hardware") or (res.get("research_provenance") or {}).get("hardware")}}


def native_metrics(res):
    prov = res.get("provenance") or {}
    events = []
    mmlu = res.get("mmlu") or {}
    dataset = (prov.get("data") or {}).get("mmlu_pro_sha256")
    if mmlu.get("acc") is not None:
        events.append({"type": "metric", "name": "mmlu_accuracy", "value": mmlu["acc"], "unit": "fraction", "kind": "measured", "better": "higher",
                       "protocol": prov.get("protocol"), "task": "MMLU-Pro", "dataset_sha256": dataset,
                       "scope": json.dumps({"scoring": mmlu.get("protocol"), "n": mmlu.get("n"), "seed": (prov.get("data") or {}).get("mmlu_seed")}, sort_keys=True),
                       "samples": [int(x) for x in mmlu.get("answers", "") if x in "01"]})
    for key in ("tg", "pp"):
        value = (res.get("speed") or {}).get(key)
        if value is not None:
            events.append({"type": "metric", "name": "decode_tokens_per_second" if key == "tg" else "prefill_tokens_per_second", "value": value,
                           "unit": "token/s", "kind": "measured", "better": "higher", "protocol": prov.get("protocol"), "task": "llama-bench",
                           "dataset_sha256": "llama-bench-generated-input", "scope": "token units specific to model " + str((prov.get("model") or {}).get("sha256")),
                           "samples": (res.get("speed") or {}).get(key + "_samples")})
    return events


def compare(ids):
    if not 2 <= len(ids) <= 8:
        raise ValueError("Choose 2–8 jobs")
    runs = [_profile_job(int(str(jid).split(":")[1])) if str(jid).startswith("profile:") else job(int(jid)) for jid in ids]
    metrics = []
    for r in runs:
        latest = {}
        for event in r["events"]:
            if event.get("type") == "metric" and isinstance(event.get("name"), str):
                latest[event["name"]] = event
        metrics.append(latest)
    rows = []
    for name in sorted(set().union(*(set(m) for m in metrics))):
        values = [m.get(name) for m in metrics]
        reasons = []
        keys = ("unit", "protocol", "task", "dataset_sha256", "scope", "kind", "better")
        if any(r["status"] != "done" for r in runs):
            reasons.append("Some jobs did not finish successfully")
        if any(v is None for v in values):
            reasons.append("Metric missing from some jobs")
        present = [v for v in values if v is not None]
        for key in keys:
            if any(not isinstance(v.get(key), (str, type(None))) for v in present) or len({v.get(key) for v in present}) != 1 or key != "better" and any(not v.get(key) for v in present):
                reasons.append(f"Different or missing {key}")
        for v in present:
            value = v.get("value")
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
                reasons.append("Invalid numeric measurement")
        if any("token" in str(v.get("unit", "")).lower() for v in present):
            tokens = [v.get("tokenizer_sha256") for v in present]
            if not all(isinstance(t, str) and t for t in tokens) or len(set(tokens)) != 1:
                reasons.append("Different or unknown tokenizers; token units are not interchangeable")
        if any(r["provenance"].get("hardware") != runs[0]["provenance"].get("hardware") for r in runs):
            reasons.append("Different hardware; performance values describe different devices")
        rows.append({"name": name, "values": values, "compatible": not reasons, "reasons": sorted(set(reasons))})
    return {"runs": [{"id": r["id"], "project_id": r["project_id"], "status": r["status"]} for r in runs], "metrics": rows,
            "note": "Compatibility checks declared evaluation contracts, not evaluator correctness. Raw differences do not establish significance. Training data and compute budgets may differ."}
