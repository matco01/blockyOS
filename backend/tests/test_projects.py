"""Projects group experiments and code; they never own or delete them."""
import pytest

from app import db


@pytest.fixture
def api(http_client, fresh_db):
    return http_client


def test_project_lifecycle(api, make_gguf):
    make_gguf("proj.gguf")
    assert api.post("/api/projects", json={"name": "   "}).status_code == 400
    pid = api.post("/api/projects", json={"name": "  Support  bot ", "goal": "Czech support on a 4 GB GPU"}).json()["id"]

    # A new experiment can go straight into the project; an old one starts outside any project.
    loose = api.post("/api/experiments", json={"name": "Loose", "model": "proj.gguf"}).json()["id"]
    inside = api.post("/api/experiments", json={"name": "Inside", "model": "proj.gguf", "project_id": pid}).json()["id"]
    assert api.post("/api/experiments", json={"name": "X", "model": "proj.gguf", "project_id": 999}).status_code == 400

    p = api.get(f"/api/projects/{pid}").json()
    assert p["name"] == "Support bot" and p["goal"] == "Czech support on a 4 GB GPU"
    assert [e["id"] for e in p["experiments"]] == [inside]
    assert p["experiments"][0]["runs"] == 0 and p["experiments"][0]["baseline"] is None
    everything = api.get("/api/projects").json()
    assert [e["id"] for e in everything["unassigned"]["experiments"]] == [loose]

    # Move the loose experiment in, rename, then delete: the experiments survive, outside any project.
    assert api.put("/api/projects/assign", json={"kind": "experiment", "id": loose, "project_id": pid}).status_code == 200
    assert api.put("/api/projects/assign", json={"kind": "model", "id": loose, "project_id": pid}).status_code == 400
    assert api.put("/api/projects/assign", json={"kind": "experiment", "id": 999, "project_id": pid}).status_code == 400
    assert {e["id"] for e in api.get(f"/api/projects/{pid}").json()["experiments"]} == {loose, inside}
    assert api.patch(f"/api/projects/{pid}", json={"name": "Support bot v2"}).json()["goal"] == "Czech support on a 4 GB GPU"
    assert api.delete(f"/api/projects/{pid}").status_code == 200
    assert api.get(f"/api/projects/{pid}").status_code == 404
    assert {e["id"] for e in api.get("/api/projects").json()["unassigned"]["experiments"]} == {loose, inside}
    assert api.get(f"/api/experiments/{inside}").status_code == 200


def test_project_card_shows_baseline_and_latest_results(api, make_gguf):
    make_gguf("cards.gguf")
    pid = api.post("/api/projects", json={"name": "Cards"}).json()["id"]
    eid = api.post("/api/experiments", json={"name": "E", "model": "cards.gguf", "project_id": pid}).json()["id"]
    with db.connect() as c:
        for label, results in (("Baseline", '{"mmlu": {"acc": 0.6}, "speed": {"tg": 50.0}, "ppl": 7.8}'), ("Q4", '{"speed": {"tg": 70.0}}')):
            c.execute("INSERT INTO exp_runs (experiment_id, created_at, status, label, config, results) VALUES (?, 1, 'done', ?, '{}', ?)",
                      (eid, label, results))
    e = api.get(f"/api/projects/{pid}").json()["experiments"][0]
    assert e["done"] == 2 and e["baseline"] == {"label": "Baseline", "results": {"mmlu": 0.6, "tg": 50.0, "ppl": 7.8}}
    assert e["latest"]["results"] == {"mmlu": None, "tg": 70.0, "ppl": None}


def test_code_can_join_a_project(api):
    pid = api.post("/api/projects", json={"name": "With code"}).json()["id"]
    with db.connect() as c:
        rid = c.execute("INSERT INTO research_projects (name, created_at) VALUES ('my-arch', 1)").lastrowid
    assert api.put("/api/projects/assign", json={"kind": "code", "id": rid, "project_id": pid}).status_code == 200
    assert [x["name"] for x in api.get(f"/api/projects/{pid}").json()["code"]] == ["my-arch"]
    assert api.put("/api/projects/assign", json={"kind": "code", "id": rid, "project_id": None}).status_code == 200
    assert api.get(f"/api/projects/{pid}").json()["code"] == []
