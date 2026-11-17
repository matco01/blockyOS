import asyncio
from pathlib import Path

import pytest

from app import methods, profile, research_jobs as research, db, validation


@pytest.fixture
def registered(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setattr(research, "ROOT", tmp_path / "projects")
    monkeypatch.setattr(profile, "limits", lambda: {"vram_mb": 0, "ram_mb": 8192})
    research.init()
    pid = research.create("Byte method", "byte")["id"]
    jid = research.queue(pid, {})["id"]
    asyncio.run(research.execute(jid, None))
    method = methods.register(jid, "Byte baseline")
    return pid, jid, method


def test_registered_method_runs_in_experiment_and_preserves_snapshot(registered):
    pid, jid, method = registered
    eid = profile.create_experiment("Shared evaluation", f"method:{method['id']}")
    research.write_file(pid, "main.py", "raise RuntimeError('new untested project code')")
    a = profile.create_run(eid, {"config": {}})
    asyncio.run(profile.run(a, None))
    baseline = profile.get_experiment(eid)["runs_list"][0]
    assert baseline["status"] == "done", baseline["error"]
    assert "bits_per_byte" in baseline["results"]["research_metrics"]
    assert "speed" not in baseline["results"] and "mmlu" not in baseline["results"]
    assert baseline["results"]["research_job_id"] != jid
    assert baseline["results"]["research_artifacts"]
    b = profile.create_run(eid, {"parent_id": a, "config": {"model": f"method:{method['id']}", "method_config": {"smoothing": 2}}})
    asyncio.run(profile.run(b, None))
    rows = profile.get_experiment(eid)["runs_list"]
    assert rows[1]["parent_id"] == a and rows[1]["status"] == "done"
    assert rows[0]["results"]["research_metrics"]["bits_per_byte"]["value"] != rows[1]["results"]["research_metrics"]["bits_per_byte"]["value"]


def test_cancel_method_run_cancels_its_job(registered):
    _, _, method = registered
    eid = profile.create_experiment("Cancel", f"method:{method['id']}")
    rid = profile.create_run(eid, {"config": {}})
    row = profile.get_experiment(eid)["runs_list"][0]
    assert row["id"] == rid
    profile.cancel_queue(eid)
    assert research.job(row["results"]["research_job_id"])["status"] == "cancelled"
    assert profile.get_experiment(eid)["runs_list"][0]["status"] == "cancelled"


def test_shared_metric_validation_is_linked_and_alternating(registered):
    _, _, method = registered
    eid = profile.create_experiment("Repeat", f"method:{method['id']}")
    a = profile.create_run(eid, {"config": {}})
    asyncio.run(profile.run(a, None))
    b = profile.create_run(eid, {"config": {"method_config": {"smoothing": 2}}, "parent_id": a})
    asyncio.run(profile.run(b, None))
    batch = validation.create(eid, {"reference_id": a, "candidate_id": b, "metric": "research:bits_per_byte", "pairs": 3})
    exp = profile.get_experiment(eid)
    queued = exp["runs_list"][2:]
    assert [r["parent_id"] for r in queued] == [a, b] * 3
    assert all(r["results"]["research_job_id"] for r in queued)
    validation.cancel(eid, batch["id"])
    assert profile.get_experiment(eid)["validations"][0]["summary"]["conclusion"] == "insufficient"


def test_fixed_dataset_quality_repeats_never_invent_significance(registered):
    _, _, method = registered
    eid = profile.create_experiment("Descriptive quality repeats", f"method:{method['id']}")
    a = profile.create_run(eid, {"config": {}})
    asyncio.run(profile.run(a, None))
    b = profile.create_run(eid, {"config": {"method_config": {"smoothing": 2}}, "parent_id": a})
    asyncio.run(profile.run(b, None))
    batch = validation.create(eid, {"reference_id": a, "candidate_id": b, "metric": "research:bits_per_byte", "pairs": 3})
    for rid in batch["run_ids"]:
        asyncio.run(profile.run(rid, None))
    summary = profile.get_experiment(eid)["validations"][0]["summary"]
    assert summary["stats"]["n"] == 3
    assert summary["stats"]["p"] is None and summary["stats"]["ci95_pct"] is None
    assert summary["conclusion"] == "insufficient"


def test_custom_settings_do_not_silently_apply_native_flags(registered):
    _, _, method = registered
    with pytest.raises(ValueError, match="does not implement native setting"):
        methods.normalize(f"method:{method['id']}", {"ctx": 32768})


def test_promote_reuses_tested_result_without_launching_another_job(registered):
    _, jid, _ = registered
    with db.connect() as c:
        count = c.execute("SELECT COUNT(*) FROM research_jobs").fetchone()[0]
    promoted = methods.promote(jid, "Connected workflow")
    assert methods.promote(jid, "Repeated click") == promoted
    exp = profile.get_experiment(promoted["experiment_id"])
    assert len(exp["runs_list"]) == 1
    baseline = exp["runs_list"][0]
    assert baseline["id"] == promoted["run_id"] and baseline["status"] == "done"
    assert baseline["results"]["research_job_id"] == jid
    assert baseline["results"]["promoted_from_job_id"] == jid
    assert baseline["results"]["research_provenance"] == research.job(jid)["provenance"]
    assert "bits_per_byte" in baseline["results"]["research_metrics"]
    assert "speed" not in baseline["results"]
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM research_jobs").fetchone()[0] == count
    # A promoted baseline remains a real starting point for new evaluation branches.
    rid = profile.create_run(exp["id"], {"parent_id": baseline["id"], "config": {"method_config": {"smoothing": 2}}})
    asyncio.run(profile.run(rid, None))
    assert profile.get_experiment(exp["id"])["runs_list"][1]["status"] == "done"


def test_unfinished_job_cannot_be_promoted(registered):
    pid, _, _ = registered
    jid = research.queue(pid, {})["id"]
    with pytest.raises(ValueError, match="successfully completed"):
        methods.promote(jid)


def test_registration_rejects_unfinished_or_modified_source(registered):
    pid, jid, method = registered
    queued = research.queue(pid, {})["id"]
    with pytest.raises(ValueError, match="successfully completed"):
        methods.register(queued, "not tested")
    source = Path(research.job(jid)["directory"]) / "source/main.py"
    source.write_text("print('tampered')")
    with pytest.raises(ValueError, match="changed"):
        methods.register(jid, "changed")
    eid = profile.create_experiment("Pinned", f"method:{method['id']}")
    with pytest.raises(ValueError, match="snapshot changed"):
        profile.create_run(eid, {"config": {}})


def test_native_and_registered_runs_coexist(registered, make_gguf):
    _, _, method = registered
    make_gguf("native.gguf")
    eid = profile.create_experiment("Mixed methods", "native.gguf")
    native = profile.create_run(eid, {"config": {}})
    profile._save(native, status="done", results={"speed": {"tg": 20}})
    custom = profile.create_run(eid, {"config": {"model": f"method:{method['id']}"}, "parent_id": native})
    asyncio.run(profile.run(custom, None))
    rows = profile.get_experiment(eid)["runs_list"]
    assert "decode_tokens_per_second" in rows[0]["results"]["research_metrics"]
    assert "bits_per_byte" in rows[1]["results"]["research_metrics"]
    assert "bits_per_byte" not in rows[0]["results"]["research_metrics"]
