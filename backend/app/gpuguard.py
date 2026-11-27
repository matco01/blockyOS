"""GPU safety: stop long jobs before an overheating or failing GPU takes the computer down, and never restart heavy work
by itself after a crash.

- A monitor reads the GPU every few seconds. If it stays above its limit while Blocky has GPU work running, or stops
  responding (driver error), all running GPU work is cancelled and GPU work is paused.
- A pause is stored, so it survives restarts. Blocky also pauses by itself when it starts and finds GPU work that was
  interrupted (a crash, power loss or reboot). Nothing GPU-heavy starts until the owner resumes.
- Before each queued run, Blocky waits until the GPU has cooled to the resume temperature, so back-to-back runs don't
  heat the card up step by step.

Limits come from the GPU itself (its maximum operating temperature) unless the owner sets them in Settings.
"""
import asyncio
import json
import logging
import re
import subprocess
import time

from . import db

log = logging.getLogger("blocky.gpuguard")

POLL_S = 5
HOT_SAMPLES = 3          # this many readings in a row at or above the limit (15 s) stop the work
LOST_SAMPLES = 2         # a GPU that stops answering twice in a row is treated as lost
MAX_COOL_WAIT_S = 900    # a GPU that doesn't cool below the resume temperature in 15 minutes with nothing running pauses work

state = {"temp_c": None, "gpu_seen": False, "waiting": None, "hot": 0, "lost": 0, "checked_at": None}
_limits_cache: dict | None = None

# Tables of jobs that use the GPU, with the function that stops a running one.
_JOB_TABLES = (("exp_runs", "profile", "cancel"), ("research_jobs", "research_jobs", "cancel"), ("torch_jobs", "torchlab", "cancel"),
               ("bench_runs", "lab", "cancel_bench"), ("quant_runs", "quantlab", "cancel"), ("layer_scans", "scans", "cancel"),
               ("arena_matches", "arena", "cancel"), ("lab_runs", "tools", "cancel"), ("eval_runs", "evals", "cancel"))


# ---------- Reading the GPU ----------

def _query() -> tuple[list[int] | None, str | None]:
    """GPU temperatures, or (None, None) without an NVIDIA GPU/tool, or (None, error) when the driver doesn't answer."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return None, None
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"nvidia-smi failed ({e.__class__.__name__})"
    lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    if r.returncode != 0 or not lines or not all(x.isdigit() for x in lines):
        msg = (r.stdout + r.stderr).strip().splitlines()
        return None, (msg[0] if msg else f"nvidia-smi exit code {r.returncode}")[:200]
    return [int(x) for x in lines], None


def auto_limits() -> dict:
    """The GPU's own limits: stop 3 °C under its maximum operating temperature, resume 15 °C under the stop limit."""
    global _limits_cache
    if _limits_cache is not None:
        return _limits_cache
    max_op = slowdown = None
    try:
        out = subprocess.run(["nvidia-smi", "-q", "-d", "TEMPERATURE"], capture_output=True, text=True, timeout=10).stdout
        ops = [int(x) for x in re.findall(r"Max Operating Temp\s*:\s*(\d+)\s*C", out)]
        slows = [int(x) for x in re.findall(r"Slowdown Temp\s*:\s*(\d+)\s*C", out)]
        max_op, slowdown = (min(ops) if ops else None), (min(slows) if slows else None)
    except (OSError, subprocess.SubprocessError):
        pass
    limit = max_op - 3 if max_op else slowdown - 8 if slowdown else 85
    limit = max(60, min(95, limit))
    _limits_cache = {"limit_c": limit, "resume_c": limit - 15, "max_operating_c": max_op, "slowdown_c": slowdown}
    return _limits_cache


def settings() -> dict:
    s = db.get_settings()
    auto = auto_limits()
    limit = s.get("gpu_max_temp_c") or auto["limit_c"]
    resume = s.get("gpu_resume_temp_c") or min(auto["resume_c"], limit - 5)
    return {"enabled": bool(s.get("gpu_guard", True)), "limit_c": int(limit), "resume_c": int(min(resume, limit - 1)), "auto": auto}


# ---------- Pause (stored, survives restarts) ----------

def _store(key: str, value) -> None:
    with db.connect() as c:
        c.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                  (key, json.dumps(value)))


def _load(key: str, default=None):
    with db.connect() as c:
        r = c.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return json.loads(r["value"]) if r else default


def paused() -> dict | None:
    return _load("gpu_pause")


def _event(kind: str, text: str) -> None:
    events = (_load("gpu_events", []) or [])[-19:]
    events.append({"at": time.time(), "kind": kind, "text": text})
    _store("gpu_events", events)


def pause(reason: str, kind: str = "manual") -> None:
    if not paused():
        _store("gpu_pause", {"reason": reason, "kind": kind, "since": time.time()})
        _event(kind, reason)
        log.warning("GPU work paused: %s", reason)


def resume() -> None:
    if paused():
        _store("gpu_pause", None)
        _event("resume", "GPU work resumed by the owner")


def check_start(what: str = "This job") -> None:
    """Refuse to start GPU work while it is paused."""
    p = paused()
    if p:
        raise ValueError(f"{what} can't start: GPU work is paused ({p['reason']}). Resume it in Settings → GPU safety.")


# ---------- Stopping running work ----------

def running() -> list[tuple[str, int]]:
    out = []
    with db.connect() as c:
        for table, _, _ in _JOB_TABLES:
            try:
                out += [(table, r[0]) for r in c.execute(f"SELECT id FROM {table} WHERE status = 'running'")]
            except Exception:  # noqa: BLE001 - a table that doesn't exist yet
                pass
    from . import weights
    out += [("weights", int(j)) for j, x in weights.jobs.items() if x.get("status") == "running" and str(j).isdigit()]
    return out


def stop_gpu_work() -> int:
    """Cancels every running GPU job of every kind. Returns how many were stopped."""
    import importlib
    stopped = 0
    for table, ident in running():
        try:
            if table == "weights":
                importlib.import_module(".weights", __package__).cancel(str(ident))
            else:
                module, fn = next((m, f) for t, m, f in _JOB_TABLES if t == table)
                getattr(importlib.import_module(f".{module}", __package__), fn)(ident)
            stopped += 1
        except Exception:  # noqa: BLE001 - stop as much as possible
            log.exception("Couldn't stop %s %s", table, ident)
    return stopped


def trip(reason: str, kind: str) -> None:
    pause(reason, kind)
    n = stop_gpu_work()
    if n:
        _event("stopped", f"Stopped {n} running job{'s' if n != 1 else ''}")


# ---------- Startup, monitor and cool-down ----------

def init() -> None:
    """After a crash, power loss or reboot, don't restart heavy GPU work by itself. Run before the job modules reset
    interrupted work, so it can still be seen."""
    interrupted = 0
    with db.connect() as c:
        for table in ("exp_runs", "research_jobs", "torch_jobs"):
            try:
                interrupted += c.execute(f"SELECT COUNT(*) FROM {table} WHERE status = 'running'").fetchone()[0]
            except Exception:  # noqa: BLE001 - first start: the table doesn't exist yet
                pass
    if interrupted:
        pause("Blocky restarted while GPU work was running (a crash, power loss or reboot). Check the computer, then resume",
              "restart")


async def check_once() -> str:
    """One reading. Returns what happened: off, no-gpu, ok, hot, lost or tripped."""
    cfg = settings()
    if not cfg["enabled"]:
        state.update(hot=0, lost=0)
        return "off"
    temps, err = await asyncio.to_thread(_query)
    state["checked_at"] = time.time()
    if temps is None and err is None:      # no NVIDIA GPU on this device
        return "no-gpu"
    if err:
        state["lost"] += 1
        if state["gpu_seen"] and state["lost"] >= LOST_SAMPLES and not paused():
            trip(f"The GPU stopped responding ({err}). Running GPU work was stopped. If the screen misbehaves, restart "
                 "the computer, then resume", "lost")
            return "tripped"
        return "lost"
    state.update(gpu_seen=True, lost=0, temp_c=max(temps))
    state["hot"] = state["hot"] + 1 if state["temp_c"] >= cfg["limit_c"] else 0
    if state["hot"] >= HOT_SAMPLES and running() and not paused():
        trip(f"The GPU reached {state['temp_c']} °C (limit {cfg['limit_c']} °C). Running GPU work was stopped to protect "
             "the computer. Let it cool and check its fans and airflow, then resume", "heat")
        return "tripped"
    return "hot" if state["hot"] else "ok"


async def monitor() -> None:
    while True:
        try:
            await asyncio.sleep(POLL_S)
            if await check_once() == "no-gpu":
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the monitor must never die
            log.exception("GPU monitor")


async def wait_until_cool(should_stop=lambda: False) -> float:
    """Before a queued run: waits until the GPU is at or below the resume temperature. Returns the seconds waited."""
    cfg = settings()
    if not cfg["enabled"]:
        return 0.0
    t0 = time.monotonic()
    try:
        while True:
            temps, err = await asyncio.to_thread(_query)
            if not temps:
                return round(time.monotonic() - t0, 1)
            if max(temps) <= cfg["resume_c"] or should_stop():
                return round(time.monotonic() - t0, 1)
            if time.monotonic() - t0 > MAX_COOL_WAIT_S:
                pause(f"The GPU stayed above {cfg['resume_c']} °C for 15 minutes with nothing running. Check its fans and "
                      "airflow, then resume", "heat")
                raise RuntimeError(f"The GPU didn't cool below {cfg['resume_c']} °C; GPU work is paused (Settings → GPU safety)")
            state["waiting"] = {"temp_c": max(temps), "target_c": cfg["resume_c"], "since": t0}
            await asyncio.sleep(POLL_S)
    finally:
        state["waiting"] = None


def report() -> dict:
    cfg = settings()
    return {**cfg, "temp_c": state["temp_c"], "gpu_seen": state["gpu_seen"], "waiting": state["waiting"],
            "paused": paused(), "events": list(reversed(_load("gpu_events", []) or []))[:10]}
