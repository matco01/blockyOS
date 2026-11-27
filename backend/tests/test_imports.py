"""Uploading your own model files, training data and code from the browser."""
import io
import zipfile

import pytest

from app import lab


@pytest.fixture
def api(http_client, fresh_db):
    return http_client


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_model_upload(api):
    body = b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 100
    r = api.put("/api/models/upload", params={"name": "../../My Model.gguf"}, content=body)
    assert r.status_code == 200 and r.json() == {"file": "My-Model.gguf", "bytes": len(body)}
    assert (lab.MODELS_DIR / "My-Model.gguf").read_bytes() == body
    # Never overwritten, never a non-GGUF file, never another extension; nothing half-written is left behind.
    assert api.put("/api/models/upload", params={"name": "My Model.gguf"}, content=body).status_code == 400
    assert "right kind of file" in api.put("/api/models/upload", params={"name": "fake.gguf"}, content=b"<html>" * 20).json()["detail"]
    assert api.put("/api/models/upload", params={"name": "x.bin"}, content=body).status_code == 400
    assert not list(lab.MODELS_DIR.glob("*.part")) and not (lab.MODELS_DIR / "fake.gguf").exists()


def test_project_data_upload_list_delete(api):
    pid = api.post("/api/projects", json={"name": "Data"}).json()["id"]
    good = b'{"instruction": "Hi", "output": "Hello"}\n{"instruction": "Bye", "output": "See you"}\n'
    r = api.put(f"/api/projects/{pid}/data", params={"name": "support chats.jsonl"}, content=good)
    assert r.status_code == 200 and r.json()["name"] == "support-chats.jsonl"
    assert api.put(f"/api/projects/{pid}/data", params={"name": "notes.txt"}, content=b"line one\nline two\n").status_code == 200
    listed = {d["name"]: d for d in api.get(f"/api/projects/{pid}/data").json()["datasets"]}
    assert listed["support-chats.jsonl"]["rows"] == 2 and listed["notes.txt"]["kind"] == "txt"
    assert listed["support-chats.jsonl"]["path"].endswith(f"project-{pid}/support-chats.jsonl")
    # A broken .jsonl is refused with an example of the right shape, and not kept.
    bad = api.put(f"/api/projects/{pid}/data", params={"name": "bad.jsonl"}, content=b"just some text\n")
    assert bad.status_code == 400 and "instruction" in bad.json()["detail"]
    assert "bad.jsonl" not in {d["name"] for d in api.get(f"/api/projects/{pid}/data").json()["datasets"]}
    assert api.put(f"/api/projects/{pid}/data", params={"name": "x.csv"}, content=b"a,b\n").status_code == 400
    assert api.put("/api/projects/999/data", params={"name": "a.txt"}, content=b"x").status_code == 404
    assert api.delete(f"/api/projects/{pid}/data", params={"name": "notes.txt"}).status_code == 200
    assert [d["name"] for d in api.get(f"/api/projects/{pid}/data").json()["datasets"]] == ["support-chats.jsonl"]


def test_code_zip_and_folder_upload(api):
    rid = api.post("/api/research/projects", json={"name": "Mine", "template": "import"}).json()["id"]
    assert api.get(f"/api/research/projects/{rid}/files").json() == []
    # A zip with one top folder (as GitHub makes them) is unpacked without it; junk folders are skipped.
    archive = _zip({"repo-main/src/model.py": b"x = 1\n", "repo-main/train.py": b"print('hi')\n",
                    "repo-main/.git/config": b"[core]\n", "repo-main/data/train.txt": b"text\n"})
    r = api.put(f"/api/research/projects/{rid}/import-zip", content=archive)
    assert r.status_code == 200 and r.json()["files"] == 3
    assert r.json()["entry_candidates"][0] == "train.py"
    assert set(api.get(f"/api/research/projects/{rid}/files").json()) == {"src/model.py", "train.py", "data/train.txt"}
    # Archives can't write outside the workspace.
    evil = api.put(f"/api/research/projects/{rid}/import-zip", content=_zip({"../../escape.py": b"x"}))
    assert evil.status_code == 400
    assert api.put(f"/api/research/projects/{rid}/import-zip", content=b"not a zip").status_code == 400
    # A folder chosen in the browser arrives file by file, with its relative paths.
    assert api.put(f"/api/research/projects/{rid}/upload", params={"path": "pkg/main.py"}, content=b"print(1)\n").status_code == 200
    assert api.put(f"/api/research/projects/{rid}/upload", params={"path": "pkg/main.py"}, content=b"print(2)\n").status_code == 200
    assert api.get(f"/api/research/projects/{rid}/file", params={"name": "pkg/main.py"}).json()["content"] == "print(2)\n"
    assert api.put(f"/api/research/projects/{rid}/upload", params={"path": "../x.py"}, content=b"x").status_code == 400
    assert "pkg/main.py" in api.get(f"/api/research/projects/{rid}/entries").json()["entries"]
