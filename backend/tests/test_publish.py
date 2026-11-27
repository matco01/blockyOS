"""Ship: a model card from the measurements, the license check, and publishing (against a fake Hugging Face)."""
import json
import time

import pytest

from app import db


@pytest.fixture
def version(http_client, fresh_db, make_gguf):
    make_gguf("Ship-F16.gguf", ftype=1)
    make_gguf("Ship-Q4_K_M.gguf")
    eid = http_client.post("/api/experiments", json={"name": "Ship", "model": "Ship-F16.gguf"}).json()["id"]
    base = {"mmlu": {"acc": 0.6, "ci95": [0.56, 0.64], "n": 560}, "speed": {"tg": 50.0, "tg_sd": 1.0, "pp": 1500.0, "pp_sd": 20.0},
            "ppl": 7.8, "ppl_sd": 0.1, "kld": 0.0, "same_top": 100.0, "needle": {"acc": 1.0},
            "provenance": {"protocol": "blocky-profile/4 (…)", "llama_cpp": {"version": "0.9", "commit": "abc1234"},
                           "gpu": {"name": "Test GPU", "driver": "1.0"}, "model": {"sha256": "aa" * 32}}}
    q4 = {**base, "speed": {"tg": 80.0, "tg_sd": 2.0, "pp": 1600.0, "pp_sd": 30.0}, "kld": 0.016, "same_top": 94.5,
          "provenance": {**base["provenance"], "model": {"sha256": "bb" * 32, "recipe": {"source": "Ship-F16.gguf", "qtype": "Q4_K_M",
                                                                                             "source_sha256": "aa" * 32}}}}
    with db.connect() as c:
        b = c.execute("INSERT INTO exp_runs (experiment_id, created_at, status, label, config, results) VALUES (?, 1, 'done', 'Baseline', ?, ?)",
                      (eid, json.dumps({"model": "Ship-F16.gguf"}), json.dumps(base))).lastrowid
        v = c.execute("INSERT INTO exp_runs (experiment_id, created_at, status, label, config, results, parent_id) VALUES (?, 2, 'done', 'Q4_K_M', ?, ?, ?)",
                      (eid, json.dumps({"model": "Ship-Q4_K_M.gguf"}), json.dumps(q4), b)).lastrowid
    return http_client, eid, v


def test_model_card_is_written_from_measurements(version):
    api, eid, v = version
    text = api.get("/api/publish/card", params={"experiment_id": eid, "run_id": v, "license": "apache-2.0", "base_model": "Org/Base"}).text
    assert text.startswith("---\nlicense: apache-2.0") and "- Org/Base" in text
    assert "quantized to **Q4_K_M**" in text and "Ship-F16.gguf" in text
    assert "| 80.0 ± 2.0 |" in text and "60.0% [56.0, 64.0]" in text and "Test GPU" in text
    assert "blocky-profile/4" in text and "not for leaderboards" in text
    assert "bbbb" in text  # the file's own hash


def test_license_check(version):
    api, eid, v = version
    info = api.get("/api/publish/license", params={"experiment_id": eid, "run_id": v}).json()
    assert info["detected"] is None and info["permissive"] is False and info["file"] == "Ship-Q4_K_M.gguf"
    assert any(o["id"] == "llama3.1" for o in info["options"])


def test_publishing(version, monkeypatch):
    api, eid, v = version
    body = {"experiment_id": eid, "run_id": v, "repo": "me/ship-q4", "license": "apache-2.0"}
    assert "Connect" in api.post("/api/publish/hf", json=body).json()["detail"]
    assert api.post("/api/publish/account", json={"token": "not-a-token"}).status_code == 400

    calls = []

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self, token=None):
            return {"name": "me"}

        def create_repo(self, repo_id, **kw):
            calls.append(("create", repo_id, kw.get("private")))
            return f"https://huggingface.co/{repo_id}"

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, **kw):
            calls.append(("upload", path_in_repo))

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert api.post("/api/publish/account", json={"token": "hf_" + "a" * 30}).json() == {"connected": True, "user": "me"}
    assert "hf_token" not in api.get("/api/settings").json()            # the token is never returned
    # Conditional licenses need the owner's confirmation.
    assert api.post("/api/publish/hf", json={**body, "license": "llama3.1"}).status_code == 400
    assert api.post("/api/publish/hf", json={**body, "repo": "bad repo"}).status_code == 400
    assert api.post("/api/publish/hf", json=body).status_code == 200
    for _ in range(100):
        job = api.get("/api/publish/jobs").json()[0]
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["url"] == "https://huggingface.co/me/ship-q4"
    assert calls == [("create", "me/ship-q4", True), ("upload", "README.md"), ("upload", "reproducibility-bundle.zip"),
                     ("upload", "Ship-Q4_K_M.gguf")]
    assert api.delete("/api/publish/account").json() == {"connected": False, "user": None}


def test_assistants_cant_publish():
    from app import mcp_server
    assert all(not e["path"].startswith("/api/publish") for e in mcp_server._endpoints())
