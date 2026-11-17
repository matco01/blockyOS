"""Remote access: local-only by default, token required from other devices; device registry validation."""
import pytest
import asyncio
import httpx
import json

from app import remote


def test_remote_run_copies_protocol_before_queuing(monkeypatch):
    calls = []
    exp = {"id": 1, "name": "Source", "protocol": {"mmlu_per_category": 100, "ruler": {"samples": 3}}}
    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.method == "POST":
            return httpx.Response(200, json={"id": 20})
        if request.url.path.endswith("export.json"):
            return httpx.Response(200, json={"runs_list": []})
        if request.method == "GET":
            return httpx.Response(200, json={"runs_list": [{"status": "done"}]})
        return httpx.Response(200, json={"ok": True})
    async def check(dev_id):
        return {"ok": True, "models": ["m.gguf"]}
    monkeypatch.setattr(remote, "check", check)
    monkeypatch.setattr(remote, "_client", lambda dev: httpx.AsyncClient(base_url="http://other", transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(remote.profile, "import_experiment", lambda data, into: {"runs": 1})
    monkeypatch.setattr(remote, "jobs", {1: {"configs": [{"model": "m.gguf"}], "remote": {}, "experiment_id": 1}})
    asyncio.run(remote._run(1, {"id": 1, "name": "Other"}, exp))
    assert calls[1] == ("PUT", "/api/experiments/20/protocol", exp["protocol"])
    assert calls[2][1].endswith("/runs")
    assert remote.jobs[1]["status"] == "done"


def test_local_only_by_default(fresh_db):
    remote.init()
    assert remote.allowed("127.0.0.1", "", "")
    assert remote.allowed("::1", "", "")
    assert not remote.allowed("192.168.1.20", "", "")
    assert not remote.allowed("192.168.1.20", "Bearer anything", "")


def test_token_when_remote_access_is_on(fresh_db):
    remote.init()
    token = remote.set_access(True)
    assert len(token) >= 24
    assert remote.allowed("192.168.1.20", f"Bearer {token}", "")
    assert remote.allowed("192.168.1.20", "", token)               # browser cookie
    assert not remote.allowed("192.168.1.20", "Bearer wrong", "")
    assert not remote.allowed("192.168.1.20", f"Basic {token}", "")
    remote.set_access(False)
    assert not remote.allowed("192.168.1.20", f"Bearer {token}", "")


def test_device_registry(fresh_db):
    remote.init()
    dev = remote.add_device({"url": "http://192.168.1.40:8000/", "token": "abc"})
    assert remote.list_devices() == [{**remote.list_devices()[0], "id": dev, "url": "http://192.168.1.40:8000", "token": True}]
    for bad in ("ftp://x", "http://a b", "javascript:alert(1)", "http://host/path?x=1"):
        with pytest.raises(ValueError):
            remote.add_device({"url": bad})
    remote.delete_device(dev)
    assert remote.list_devices() == []


def test_host_and_origin_helpers():
    assert remote.host_name("localhost:8000") == "localhost"
    assert remote.host_name("[::1]:8000") == "::1"
    assert remote.host_name("127.0.0.1") == "127.0.0.1"
    assert remote.effective_client("127.0.0.1", "localhost:8000") == "127.0.0.1"
    assert remote.effective_client("127.0.0.1", "evil.example") == "rebound"
    assert remote.effective_client("192.168.1.20", "192.168.1.40:8000") == "192.168.1.20"
    assert remote.same_origin("", "localhost:8000")
    assert remote.same_origin("http://localhost:8000", "localhost:8000")
    assert not remote.same_origin("http://localhost:5173", "localhost:8000")
    assert not remote.same_origin("null", "localhost:8000")
