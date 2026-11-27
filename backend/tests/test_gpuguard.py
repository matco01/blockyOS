"""GPU safety: stop work on overheating or a lost GPU, pause after a crash, wait for cool-down before queued runs."""
import asyncio

import pytest

from app import db, gpuguard, profile


@pytest.fixture
def guard(fresh_db, monkeypatch):
    monkeypatch.setattr(gpuguard, "_limits_cache", {"limit_c": 86, "resume_c": 71, "max_operating_c": 89, "slowdown_c": 97})
    monkeypatch.setattr(gpuguard, "POLL_S", 0)
    gpuguard.state.update(temp_c=None, gpu_seen=False, waiting=None, hot=0, lost=0)
    return gpuguard


def _readings(monkeypatch, values):
    """Each call to the GPU returns the next value: an int (°C) or a string (driver error)."""
    it = iter(values)

    def query():
        v = next(it)
        return ([v], None) if isinstance(v, int) else (None, v)
    monkeypatch.setattr(gpuguard, "_query", query)


def _running_run() -> int:
    with db.connect() as c:
        eid = c.execute("INSERT INTO experiments (created_at, name, model, seed) VALUES (1, 'e', 'm.gguf', 1)").lastrowid
        return c.execute("INSERT INTO exp_runs (experiment_id, created_at, status, label, config, results) VALUES (?, 1, 'running', 'r', '{}', '{}')",
                         (eid,)).lastrowid


def test_pause_blocks_new_work_and_survives(guard):
    assert guard.paused() is None
    guard.check_start()
    guard.pause("test", "manual")
    with pytest.raises(ValueError, match="paused"):
        guard.check_start("A benchmark")
    assert guard.paused()["reason"] == "test"          # stored in the database: survives a restart
    guard.resume()
    assert guard.paused() is None and guard.report()["events"][0]["kind"] == "resume"


def test_overheating_stops_running_work(guard, monkeypatch):
    rid = _running_run()
    _readings(monkeypatch, [80, 87, 88, 88])
    results = [asyncio.run(guard.check_once()) for _ in range(4)]
    assert results == ["ok", "hot", "hot", "tripped"]   # three readings in a row at or above 86 °C
    assert guard.paused()["kind"] == "heat" and rid in profile._cancel
    profile._cancel.discard(rid)


def test_heat_without_blocky_work_doesnt_pause(guard, monkeypatch):
    _readings(monkeypatch, [90, 90, 90, 90])
    assert [asyncio.run(guard.check_once()) for _ in range(4)][-1] == "hot"
    assert guard.paused() is None   # a hot GPU from a game is the next queued run's cool-down problem, not a stop


def test_lost_gpu_stops_work(guard, monkeypatch):
    rid = _running_run()
    _readings(monkeypatch, [60, "Failed to initialize NVML: N/A", "Failed to initialize NVML: N/A"])
    assert [asyncio.run(guard.check_once()) for _ in range(3)] == ["ok", "lost", "tripped"]
    assert guard.paused()["kind"] == "lost" and rid in profile._cancel
    profile._cancel.discard(rid)


def test_no_gpu_means_no_guard(guard, monkeypatch):
    monkeypatch.setattr(gpuguard, "_query", lambda: (None, None))
    assert asyncio.run(guard.check_once()) == "no-gpu" and guard.paused() is None


def test_restart_after_crash_pauses(guard):
    _running_run()
    guard.init()
    assert guard.paused()["kind"] == "restart"


def test_waits_for_cool_down_before_a_run(guard, monkeypatch):
    _readings(monkeypatch, [80, 75, 70])
    assert asyncio.run(guard.wait_until_cool()) >= 0 and guard.state["waiting"] is None
    _readings(monkeypatch, [60])
    assert asyncio.run(guard.wait_until_cool()) == 0


def test_settings_validation(guard, http_client):
    assert http_client.patch("/api/settings", json={"gpu_max_temp_c": 120}).status_code == 400
    assert http_client.patch("/api/settings", json={"gpu_max_temp_c": 80, "gpu_resume_temp_c": 85}).status_code == 400
    assert http_client.patch("/api/settings", json={"gpu_max_temp_c": 82, "gpu_resume_temp_c": 65}).status_code == 200
    r = http_client.get("/api/gpu-safety").json()
    assert (r["limit_c"], r["resume_c"]) == (82, 65)
    http_client.patch("/api/settings", json={"gpu_max_temp_c": None, "gpu_resume_temp_c": None})
    assert http_client.get("/api/gpu-safety").json()["limit_c"] == 86   # back to the GPU's own limit
    guard.pause("x")
    assert http_client.get("/api/status").json()["gpu_pause"]["reason"] == "x"
    assert http_client.post("/api/gpu-safety/resume").json()["paused"] is None
