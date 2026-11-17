"""Background-job records that survive a backend restart.

Several modules keep their jobs (downloads, weight builds, quantizations, remote-device runs) in plain dicts that the
job code mutates in place. `persist(name, jobs)` registers such a dict: its contents are saved to SQLite every few
seconds and at shutdown, and restored at startup. A job that was still active when the backend stopped can't still
be running (its process died with the backend), so it comes back as an error that says so, unless the module
registers a `resume` callback that can pick it up again (remote-device jobs keep polling the other device).
"""
import asyncio
import json
import time

from . import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS bg_jobs (
    store      TEXT NOT NULL,
    job_key    TEXT NOT NULL,
    updated_at REAL NOT NULL,
    data       TEXT NOT NULL,
    PRIMARY KEY (store, job_key)
);
"""
ACTIVE = {"queued", "starting", "running", "downloading", "building"}
KEEP = 200  # finished jobs kept per store
_stores: dict[str, tuple[dict, type]] = {}
_last: dict[tuple[str, str], str] = {}


def init() -> None:
    with db.connect() as c:
        c.executescript(SCHEMA)


def persist(name: str, jobs: dict, key_type: type = str) -> None:
    """Restore `jobs` from the database and keep it saved from now on."""
    _stores[name] = (jobs, key_type)
    with db.connect() as c:
        c.executescript(SCHEMA)
        rows = c.execute("SELECT job_key, data FROM bg_jobs WHERE store = ? ORDER BY updated_at", (name,)).fetchall()
    for r in rows:
        try:
            job = json.loads(r["data"])
            key = key_type(r["job_key"])
        except (ValueError, TypeError):
            continue
        if isinstance(job, dict) and key not in jobs:
            if job.get("status") in ACTIVE:
                job["interrupted_status"] = job["status"]
                job["status"] = "error"
                job["error"] = "Interrupted: the backend restarted while this was running"
            jobs[key] = job
            _last[(name, str(key))] = json.dumps(job, default=str)


def save() -> None:
    """Write every changed job (cheap: unchanged jobs are skipped) and drop old finished ones."""
    now = time.time()
    with db.connect() as c:
        c.executescript(SCHEMA)
        for name, (jobs, _) in _stores.items():
            for key, job in list(jobs.items()):
                try:
                    data = json.dumps(job, default=str)
                except (TypeError, ValueError):
                    continue
                if _last.get((name, str(key))) != data:
                    c.execute("INSERT OR REPLACE INTO bg_jobs (store, job_key, updated_at, data) VALUES (?, ?, ?, ?)",
                              (name, str(key), now, data))
                    _last[(name, str(key))] = data
            live = {str(k) for k in jobs}
            for (store, key) in [k for k in _last if k[0] == name and k[1] not in live]:
                c.execute("DELETE FROM bg_jobs WHERE store = ? AND job_key = ?", (store, key))
                _last.pop((store, key), None)
            c.execute("DELETE FROM bg_jobs WHERE store = ? AND job_key NOT IN (SELECT job_key FROM bg_jobs WHERE store = ? "
                      "ORDER BY updated_at DESC LIMIT ?)", (name, name, KEEP))


async def saver(interval: float = 3.0) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            save()  # on the event loop, so job dicts can't change mid-save
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  (a failed save is retried on the next tick)
            pass
