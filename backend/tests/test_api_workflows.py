"""Exercise user workflows through HTTP with temporary data and no inference processes."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app import db, main, profile, research_jobs


@pytest.fixture(scope="module")
def http_client():
    async def idle_worker(client):
        await asyncio.Event().wait()

    # The MCP SDK's session manager supports one lifespan per process.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(profile, "worker", idle_worker)
        patch.setattr(research_jobs, "worker", idle_worker)
        patch.setattr(profile, "limits", lambda: {"gpu": None, "vram_mb": 0})
        with TestClient(main.app, client=("127.0.0.1", 12345)) as client:
            yield client


@pytest.fixture
def api(http_client, fresh_db):
    return http_client


def test_experiment_branch_sweep_cancel_export_import(api, make_gguf):
    make_gguf("workflow.gguf")
    response = api.post("/api/experiments", json={"name": "Workflow", "model": "workflow.gguf"})
    assert response.status_code == 200
    eid = response.json()["id"]
    protocol = {"mmlu_per_category": profile.MMLU_SIZES[-1], "ruler": {"samples": 3}}
    assert api.put(f"/api/experiments/{eid}/protocol", json=protocol).status_code == 200
    baseline = api.post(f"/api/experiments/{eid}/runs", json={"config": {}}).json()["id"]
    cfg = api.get(f"/api/experiments/{eid}").json()["runs_list"][0]["config"]
    branch = api.post(f"/api/experiments/{eid}/runs", json={"parent_id": baseline, "config": {**cfg, "kv": "q4_0"}})
    assert branch.status_code == 200
    assert api.post(f"/api/experiments/{eid}/sweep", json={"parent_id": 9999, "grid": {"ctx": [4096]}}).status_code == 400
    assert len(api.get(f"/api/experiments/{eid}").json()["runs_list"]) == 2
    assert api.post(f"/api/experiments/{eid}/cancel-queue").status_code == 200
    assert {r["status"] for r in api.get(f"/api/experiments/{eid}").json()["runs_list"]} == {"cancelled"}
    # An explicitly marked test result simulates a completed evaluator; the API/storage path remains real.
    with db.connect() as c:
        c.execute("UPDATE exp_runs SET status = 'done', results = ? WHERE id = ?", (json.dumps({"fixture": True}), baseline))
    exported = api.get(f"/api/experiments/{eid}/export.json")
    assert exported.status_code == 200
    imported = api.post("/api/experiments/import", json=exported.json())
    assert imported.status_code == 200
    restored = api.get(f"/api/experiments/{imported.json()['experiment_id']}").json()
    assert restored["protocol"] == protocol
    assert restored["runs_list"][0]["results"]["fixture"] is True


def test_settings_memory_and_mcp_catalog(api):
    assert api.get("/api/settings").status_code == 200
    added = api.post("/api/memories", json={"content": "API test fact"})
    assert added.status_code == 200
    assert any(m["content"] == "API test fact" for m in api.get("/api/memories").json())
    catalog = api.get("/api/mcp")
    assert catalog.status_code == 200
    names = {t["name"] for t in catalog.json()["tools"]}
    assert {"queue_run", "queue_sweep", "get_experiment"} <= names
    assert not any("delete" in name for name in names)


def test_validation_http_queue_and_evidence(api, make_gguf):
    make_gguf("http-validation.gguf")
    eid = api.post("/api/experiments", json={"name": "Validate", "model": "http-validation.gguf"}).json()["id"]
    ids = [api.post(f"/api/experiments/{eid}/runs", json={"config": cfg}).json()["id"] for cfg in ({}, {"kv": "q4_0"})]
    for rid in ids:
        profile._save(rid, status="done", results={"provenance": {"protocol": profile.PROTOCOL, "model": {"file": "http-validation.gguf", "sha256": "fixture"}}})
    response = api.post(f"/api/experiments/{eid}/validate", json={"reference_id": ids[0], "candidate_id": ids[1], "pairs": 3})
    assert response.status_code == 200
    batch = response.json()
    assert len(batch["run_ids"]) == 6
    assert api.get(f"/api/experiments/{eid}").json()["validations"][0]["summary"]["conclusion"] == "pending"
    assert api.post(f"/api/experiments/{eid}/validations/{batch['id']}/cancel").status_code == 200
    assert api.get(f"/api/experiments/runs/{ids[0]}/evidence").status_code == 200
    assert api.get("/api/experiments/runs/999999/evidence").status_code == 404


def test_remote_client_is_denied_without_token(fresh_db):
    client = TestClient(main.app, client=("192.0.2.1", 12345))
    assert client.get("/api/settings").status_code == 403


def test_research_http_project_and_job(api, tmp_path, monkeypatch):
    monkeypatch.setattr(research_jobs, "ROOT", tmp_path / "research")
    research_jobs.init()
    project = api.post("/api/research/projects", json={"name": "HTTP test", "template": "byte"})
    assert project.status_code == 200
    pid = project.json()["id"]
    assert "main.py" in api.get(f"/api/research/projects/{pid}/files").json()
    assert api.get(f"/api/research/projects/{pid}/file", params={"name": "../outside"}).status_code == 400
    assert api.get(f"/api/research/projects/{pid}/file", params={"name": "main.py"}).status_code == 200
    queued = api.post(f"/api/research/projects/{pid}/jobs", json={})
    assert queued.status_code == 200
    jid = queued.json()["id"]
    assert api.get(f"/api/research/projects/{pid}/jobs").json()[0]["status"] == "queued"
    assert api.post(f"/api/research/jobs/{jid}/cancel").status_code == 200
    assert api.get(f"/api/research/jobs/{jid}/artifact", params={"name": "source/main.py"}).status_code == 200
    assert api.get(f"/api/research/jobs/{jid}/artifact", params={"name": "../../../../etc/passwd"}).status_code == 404
