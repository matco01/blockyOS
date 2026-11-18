"""The MCP server: tools are registered, read the same data as the UI, and queue runs through the normal validation."""
import asyncio
import json

import pytest

from app import mcp_server, profile


@pytest.fixture
def no_hardware(monkeypatch):
    lim = {"gpu": "Test GPU", "gpu_bandwidth_gbps": 500, "vram_mb": 8192, "ram_mb": 16384, "cpu": "Test CPU", "cpu_threads": 8}
    monkeypatch.setattr(profile, "limits", lambda: lim)


def test_tools_registered():
    names = {t.name for t in asyncio.run(mcp_server.server.list_tools())}
    assert {"get_status", "get_experiment", "create_experiment", "queue_run", "queue_sweep", "build_weight_variant", "start_layer_scan"} <= names
    tools = {t.name: t for t in asyncio.run(mcp_server.server.list_tools())}
    assert tools["get_status"].annotations.read_only_hint and tools["query_database"].annotations.read_only_hint
    assert not tools["queue_run"].annotations.read_only_hint and not tools["queue_run"].annotations.destructive_hint
    assert tools["api_delete"].annotations.destructive_hint  # deleting needs the 'full' level and is marked for confirmation


def test_experiment_and_branch(fresh_db, make_gguf, no_hardware):
    make_gguf("M-Q8_0.gguf")
    make_gguf("M-Q4_K_M.gguf", ftype=15)
    eid = json.loads(mcp_server.create_experiment("M-Q8_0.gguf"))["experiment_id"]
    base = json.loads(mcp_server.queue_run(eid))["run_id"]
    child = json.loads(mcp_server.queue_run(eid, {"kv": "q4_0"}, parent_run_id=base))["run_id"]
    e = json.loads(mcp_server.get_experiment(eid))
    runs = {r["id"]: r for r in e["runs"]}
    assert runs[child]["parent_id"] == base and runs[child]["config"]["kv"] == "q4_0"
    assert runs[child]["config"]["model"] == "M-Q8_0.gguf"  # inherited from the parent
    assert [r["id"] for r in mcp_server._active_runs()] == [base, child]


def test_errors_are_returned_not_raised(fresh_db, make_gguf, no_hardware):
    assert "error" in json.loads(mcp_server.get_experiment(12345))
    assert "error" in json.loads(mcp_server.create_experiment("missing.gguf"))
    make_gguf("M-Q8_0.gguf")
    eid = json.loads(mcp_server.create_experiment("M-Q8_0.gguf"))["experiment_id"]
    assert "error" in json.loads(mcp_server.queue_run(eid, {"model": "missing.gguf"}))
