"""Remote devices: one Blocky drives others (e.g. a few cheap boxes) and collects their results.

Security model: every Blocky answers only itself (loopback) unless its owner turns on remote access, which creates a
token. Another device must send that token (Authorization: Bearer ...) on every request. Over an untrusted network,
use an SSH tunnel instead (ssh -L 8001:localhost:8000 user@box, then add http://localhost:8001 as the device).

Running on a device: the controller recreates the chosen runs' setups there (same model file, same settings), lets
that device profile them with the same protocol, and imports the finished results into the local experiment. Each run
keeps its own hardware snapshot, so the comparison shows which device it ran on.
"""
import asyncio
import hmac
import json
import re
import secrets
import time

import httpx

from . import db, profile

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    name       TEXT NOT NULL,
    url        TEXT NOT NULL,
    token      TEXT NOT NULL DEFAULT ''
);
"""
jobs: dict[int, dict] = {}
_cancel: set[int] = set()
POLL_S = 15
_URL = re.compile(r"^https?://[A-Za-z0-9.\-\[\]:]+(:\d+)?/?$")


def init() -> None:
    from . import jobstore
    with db.connect() as c:
        c.executescript(SCHEMA)
    jobstore.persist("remote", jobs, int)
    # The other device kept profiling while this backend was down: pick the job up again and keep following it.
    for job in jobs.values():
        rexp = (job.get("remote") or {}).get("experiment_id")
        if job.get("interrupted_status") == "running" and rexp:
            try:
                dev = _device(job["device_id"])
            except ValueError:
                continue
            job.update(status="running", error=None, step="reconnecting after a restart")
            job.pop("interrupted_status", None)
            asyncio.get_running_loop().create_task(_follow(job["id"], dev, rexp))


# ---------- This device: accepting remote control ----------

def access_token() -> str:
    return db.get_settings().get("remote_token") or ""


def set_access(enabled: bool) -> str:
    token = secrets.token_urlsafe(24) if enabled else ""
    db.update_settings({"remote_token": token})
    return token


LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}


def host_name(host_header: str) -> str:
    """'localhost:8000' -> 'localhost', '[::1]:8000' -> '::1' (lowercase, no port)."""
    h = (host_header or "").strip().lower()
    if h.startswith("["):
        return h[1:h.index("]")] if "]" in h else h[1:]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def effective_client(client_host: str, host_header: str) -> str:
    """A loopback connection only counts as this device when it also names this device in Host.

    DNS rebinding: a web page on attacker.example can make the browser resolve its own name to 127.0.0.1. The request
    then arrives from loopback but carries Host: attacker.example. Treating it as a remote client means it needs the
    remote-access token, which the page doesn't have. SSH tunnels and the Vite dev proxy send Host: localhost[:port].
    """
    if client_host in LOOPBACK_NAMES and host_name(host_header) not in LOOPBACK_NAMES:
        return "rebound"
    return client_host


def same_origin(origin: str, host_header: str) -> bool:
    """A browser request sent from another site carries that site's Origin. No Origin (curl, MCP clients, the Python
    client, same-origin GET) is fine; a different or opaque ('null') origin is not."""
    if not origin:
        return True
    from urllib.parse import urlsplit
    try:
        o = urlsplit(origin)
    except ValueError:
        return False
    return bool(o.netloc) and o.netloc.lower() == (host_header or "").strip().lower()


def allowed(client_host: str, auth_header: str, cookie: str) -> bool:
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return True
    token = access_token()
    if not token:
        return False
    given = auth_header[7:] if auth_header.startswith("Bearer ") else cookie
    return bool(given) and hmac.compare_digest(given, token)


# ---------- Other devices ----------

def _client(dev: dict) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {dev['token']}"} if dev.get("token") else {}
    return httpx.AsyncClient(base_url=dev["url"].rstrip("/"), headers=headers, timeout=httpx.Timeout(20, read=120))


def list_devices() -> list[dict]:
    with db.connect() as c:
        return [{**dict(r), "token": bool(r["token"])} for r in c.execute("SELECT * FROM devices ORDER BY id")]


def _device(dev_id: int) -> dict:
    with db.connect() as c:
        r = c.execute("SELECT * FROM devices WHERE id = ?", (dev_id,)).fetchone()
    if not r:
        raise ValueError("Device not found")
    return dict(r)


def add_device(body: dict) -> int:
    url = str(body.get("url") or "").strip().rstrip("/")
    if not _URL.match(url):
        raise ValueError("Use the device's address, e.g. http://192.168.1.40:8000 (or http://localhost:8001 through an SSH tunnel)")
    name = str(body.get("name") or "").strip()[:40] or url.split("//", 1)[1]
    with db.connect() as c:
        return c.execute("INSERT INTO devices (created_at, name, url, token) VALUES (?, ?, ?, ?)",
                         (time.time(), name, url, str(body.get("token") or "").strip()[:200])).lastrowid


def delete_device(dev_id: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM devices WHERE id = ?", (dev_id,))


async def check(dev_id: int) -> dict:
    """Reach the device and describe it: hardware and model files."""
    dev = _device(dev_id)
    try:
        async with _client(dev) as c:
            r = await c.get("/api/hardware/limits")
            if r.status_code == 403:
                return {"ok": False, "error": "The device refused: turn on remote access there (Settings → Remote access) and paste its token here"}
            r.raise_for_status()
            models = (await c.get("/api/models")).json().get("models") or []
            ts = await c.get("/api/torch/status")
            ckpts = (ts.json().get("checkpoints") or []) if ts.status_code == 200 else []
            files = [m["file"] for m in models if isinstance(m, dict) and m.get("file")] + [f"hf:{k['id']}" for k in ckpts if isinstance(k, dict) and k.get("id")]
            return {"ok": True, "hardware": r.json(), "models": files}
    except (httpx.HTTPError, ValueError) as e:
        return {"ok": False, "error": f"Can't reach {dev['url']}: {str(e)[:160] or type(e).__name__}"}


def start(exp_id: int, body: dict) -> int:
    dev = _device(int(body.get("device_id") or 0))
    exp = profile.get_experiment(exp_id)
    if not exp:
        raise ValueError("Experiment not found")
    ids = [int(i) for i in body.get("run_ids") or []]
    runs = [r for r in exp["runs_list"] if r["id"] in ids and not r["results"].get("imported_from")]
    if not runs:
        raise ValueError("Pick the runs (setups) to profile on that device")
    job_id = int(time.time() * 1000)
    jobs[job_id] = {"id": job_id, "experiment_id": exp_id, "device": dev["name"], "device_id": dev["id"], "status": "starting", "step": None, "error": None,
                    "configs": [r["config"] for r in sorted(runs, key=lambda r: r["id"])], "remote": {}, "started": time.time()}
    asyncio.get_running_loop().create_task(_run(job_id, dev, exp))
    return job_id


def cancel(job_id: int) -> None:
    _cancel.add(job_id)


def list_jobs(exp_id: int | None = None) -> list[dict]:
    return sorted((j for j in jobs.values() if exp_id is None or j["experiment_id"] == exp_id), key=lambda j: -j["started"])


async def _run(job_id: int, dev: dict, exp: dict) -> None:
    job = jobs[job_id]
    rexp = None
    try:
        async with _client(dev) as c:
            info = await check(dev["id"])
            if not info["ok"]:
                raise RuntimeError(info["error"])
            missing = sorted({cfg["model"] for cfg in job["configs"]} - set(info["models"]))
            if missing:
                raise RuntimeError(f"Not on {dev['name']}: {', '.join(missing)}. Download the same model files there first.")
            job["step"] = "creating the experiment there"
            r = await c.post("/api/experiments", json={"name": f"{exp['name']} (from controller)"[:80], "model": job["configs"][0]["model"]})
            r.raise_for_status()
            rexp = r.json()["id"]
            job["remote"]["experiment_id"] = rexp
            r = await c.put(f"/api/experiments/{rexp}/protocol", json=exp.get("protocol") or {})
            r.raise_for_status()
            root = None
            for cfg in job["configs"]:
                r = await c.post(f"/api/experiments/{rexp}/runs", json={"config": cfg, "parent_id": root})
                if r.status_code >= 400:
                    raise RuntimeError(f"{dev['name']} rejected a setup: {r.json().get('detail', r.text)[:200]}")
                root = root or r.json()["id"]
            job["status"] = "running"
    except Exception as e:  # noqa: BLE001
        job["status"], job["error"] = "error", str(e)[:400]
        _cancel.discard(job_id)
        return
    await _follow(job_id, dev, rexp)


async def _follow(job_id: int, dev: dict, rexp: int) -> None:
    """Wait for the device to finish the runs, then import them into the local experiment."""
    job = jobs[job_id]
    try:
        async with _client(dev) as c:
            while True:
                if job_id in _cancel:
                    await c.post(f"/api/experiments/{rexp}/cancel-queue")
                    for rr in (await c.get(f"/api/experiments/{rexp}")).json()["runs_list"]:
                        if rr["status"] == "running":
                            await c.post(f"/api/experiments/runs/{rr['id']}/cancel")
                    job["status"] = "cancelled"
                    return
                e = (await c.get(f"/api/experiments/{rexp}")).json()
                states = [x["status"] for x in e["runs_list"]]
                running = next((x for x in e["runs_list"] if x["status"] == "running"), None)
                job["step"] = f"{sum(s in ('done', 'error', 'cancelled') for s in states)} of {len(states)} runs finished" + (f" · now: {running['phase']}" if running else "")
                if all(s in ("done", "error", "cancelled") for s in states):
                    break
                await asyncio.sleep(POLL_S)
            job["step"] = "importing results"
            data = (await c.get(f"/api/experiments/{rexp}/export.json")).json()
            for x in data.get("runs_list") or []:
                if isinstance(x.get("results"), dict):  # lets the canvas hang these runs under this device
                    x["results"]["remote_device"] = {"id": dev["id"], "name": dev["name"], "url": dev["url"]}
        n = profile.import_experiment(data, into=job["experiment_id"])
        job["status"], job["step"] = "done", None
        job["imported"] = n
        errors = [x.get("error") for x in data.get("runs_list", []) if x.get("status") == "error"]
        if errors:
            job["error"] = f"{len(errors)} run(s) failed there: {errors[0]}"
    except Exception as e:  # noqa: BLE001
        job["status"], job["error"] = "error", str(e)[:400]
    finally:
        _cancel.discard(job_id)


def summary_json(job: dict) -> str:
    return json.dumps({k: job[k] for k in ("id", "device", "status", "step", "error")})
