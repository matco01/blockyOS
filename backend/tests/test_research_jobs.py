import asyncio
import json
from pathlib import Path
import sys

import pytest

from app import research_jobs as research


@pytest.fixture
def project(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setattr(research, "ROOT", tmp_path / "projects")
    research.init()
    return research.create("Test project", "byte")["id"]


def test_real_cpu_job_records_evidence_and_snapshots(project):
    pid = project
    original = research.read_file(pid, "main.py")
    queued = research.queue(pid, {"python": sys.executable, "config": {"reps": 2}})["id"]
    research.write_file(pid, "main.py", "raise RuntimeError('changed after queue')")
    asyncio.run(research.execute(queued, None))
    r = research.job(queued)
    assert r["status"] == "done", r["error"]
    assert (Path(r["directory"]) / "source/main.py").read_text() == original
    metrics = [e for e in r["events"] if e.get("type") == "metric"]
    assert {m["name"] for m in metrics} == {"bits_per_byte", "evaluation_bytes_per_second"}
    assert all(m["value"] > 0 and m["samples"] for m in metrics)
    assert r["provenance"]["environment_sha256"]
    assert research.artifact(queued, "byte-model.json").is_file()


def test_timeout_and_cancel_silent_process(project):
    research.write_file(project, "main.py", "import time\ntime.sleep(60)")
    jid = research.queue(project, {"python": sys.executable, "timeout_s": 1})["id"]
    asyncio.run(research.execute(jid, None))
    assert research.job(jid)["status"] == "error"
    assert "timeout" in research.job(jid)["error"]
    jid = research.queue(project, {"python": sys.executable})["id"]
    async def check():
        task = asyncio.create_task(research.execute(jid, None))
        for _ in range(100):
            if research.job(jid)["status"] == "running":
                break
            await asyncio.sleep(.01)
        research.cancel(jid)
        await task
    asyncio.run(check())
    assert research.job(jid)["status"] == "cancelled"


def test_compare_refuses_changed_contracts_and_accepts_shared_bytes(project):
    a = research.queue(project, {"python": sys.executable})["id"]
    b = research.queue(project, {"python": sys.executable, "parent_id": a, "config": {"smoothing": 2}})["id"]
    asyncio.run(research.execute(a, None)); asyncio.run(research.execute(b, None))
    assert all(m["compatible"] for m in research.compare([a, b])["metrics"])
    path = Path(research.job(b)["directory"]) / "events.jsonl"
    events = [json.loads(s) for s in path.read_text().splitlines()]
    for e in events:
        if e.get("type") == "metric":
            e["dataset_sha256"] = "different input"
    path.write_text("\n".join(json.dumps(e) for e in events))
    assert all(not m["compatible"] for m in research.compare([a, b])["metrics"])


def test_paths_and_queue_validation(project, tmp_path):
    for path in ("../outside.py", "/etc/passwd"):
        with pytest.raises(ValueError):
            research.read_file(project, path)
    outside = tmp_path / "outside.txt"; outside.write_text("secret")
    symlink = research.folder(project) / "source/link.txt"
    symlink.symlink_to(outside)
    with pytest.raises(ValueError):
        research.read_file(project, "link.txt")
    symlink.unlink()
    for body in ({"argv": "python main.py"}, {"config": []}, {"timeout_s": -1}, {"python": "relative"}):
        with pytest.raises(ValueError):
            research.queue(project, body)


def test_failure_and_queued_cancel(project):
    research.write_file(project, "main.py", "raise RuntimeError('test failure')")
    jid = research.queue(project, {})["id"]
    asyncio.run(research.execute(jid, None))
    assert research.job(jid)["status"] == "error"
    assert "test failure" in research.job(jid)["log"]
    jid = research.queue(project, {})["id"]
    research.cancel(jid)
    asyncio.run(research.execute(jid, None))
    assert research.job(jid)["status"] == "cancelled"


def test_gpu_restores_chat_after_error(project, monkeypatch):
    calls = []
    async def service(action): calls.append(action)
    async def ready(*args, **kwargs): return True
    monkeypatch.setattr(research.lab, "llm_service", service)
    monkeypatch.setattr(research.lab, "wait_ready", ready)
    research.write_file(project, "main.py", "raise RuntimeError('test')")
    jid = research.queue(project, {"gpu": True})["id"]
    asyncio.run(research.execute(jid, None))
    assert calls == ["stop", "start"]
    assert research.lab.busy is None


def test_profile_comparison_retains_unavailable_and_tokenizer_limits(project, make_gguf):
    from app import profile
    make_gguf("compare.gguf")
    eid = profile.create_experiment("Profile", "compare.gguf")
    rid = profile.create_run(eid, {"config": {}})
    profile._save(rid, status="done", results={"speed": {"tg": 20}})
    jid = research.queue(project, {})["id"]
    asyncio.run(research.execute(jid, None))
    compared = research.compare([jid, f"profile:{rid}"])
    assert all(not row["compatible"] for row in compared["metrics"])
    token_row = next(row for row in compared["metrics"] if row["name"] == "decode_tokens_per_second")
    assert any("tokenizers" in s for s in token_row["reasons"])
    assert token_row["values"][0] is None


def test_invalid_report_numbers_do_not_break_result_reads(project):
    jid = research.queue(project, {})["id"]
    path = Path(research.job(jid)["directory"]) / "events.jsonl"
    path.write_text('{"type":"metric","value":1e400}\n{"type":"progress","message":"valid"}\n')
    assert research.job(jid)["events"] == [{"type": "progress", "message": "valid"}]
