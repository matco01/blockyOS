"""The in-app agent: tools, sessions, approvals, and the API key never leaving the backend."""
import asyncio

import pytest

from app import agent, db


@pytest.fixture
def adb(fresh_db):
    agent.init()


def test_tools_include_lab_and_python():
    names = {t["name"] for t in asyncio.run(agent.tools())}
    assert {"get_status", "queue_run", "simulate_quantization", "run_python"} <= names
    assert all(t["input_schema"]["type"] == "object" for t in asyncio.run(agent.tools()))


def test_sessions(adb):
    sid = agent.create_session("local")
    assert agent.get_session(sid)["provider"] == "local" and agent.get_session(sid)["display"] == []
    with pytest.raises(ValueError):
        agent.create_session("gpt")
    with pytest.raises(ValueError):
        agent.create_session("claude")  # no API key yet
    db.update_settings({"anthropic_api_key": "sk-ant-test"})
    assert agent.status()["claude"]
    assert agent.get_session(agent.create_session("claude"))["model"] == "claude-opus-5-5"
    agent.delete_session(sid)
    assert agent.get_session(sid) is None


def test_settings_endpoint_hides_the_key(adb):
    from app.routes import settings as main
    db.update_settings({"anthropic_api_key": "sk-ant-secret"})
    s = main.get_settings()
    assert "anthropic_api_key" not in s and s["anthropic_api_key_set"] is True
    main.update_settings({"anthropic_api_key": "sk-ant-overwrite"})   # PATCH can't set it
    assert db.get_settings()["anthropic_api_key"] == "sk-ant-secret"


def test_approval_and_always(adb):
    async def flow():
        sid = agent.create_session("local")
        display: list = []
        gen = agent._tool_step(sid, display, "t1", "run_python", {"code": "print(1)"})
        ev = await gen.__anext__()
        assert ev["type"] == "approval" and display[0]["status"] == "approval"
        assert agent.decide("t1", False, False, sid, "run_python")
        rest = [e async for e in gen]
        assert rest[-1]["is_error"] and display[0]["status"] == "denied"
        # "Allow for this chat": the next call doesn't ask again
        agent._always.setdefault(sid, set()).add("run_python")
        gen2 = agent._tool_step(sid, display, "t2", "get_jobs", {})
        assert (await gen2.__anext__())["type"] == "tool_call"   # safe tools never ask
    asyncio.run(flow())


def test_approval_is_bound_to_its_own_chat_and_tool(adb):
    async def flow():
        a, b = agent.create_session("local"), agent.create_session("local")
        agent._always.pop(a, None); agent._always.pop(b, None)  # session ids restart with each fresh test database
        gen_a = agent._tool_step(a, [], "ta", "run_python", {"code": "print(1)"})
        gen_b = agent._tool_step(b, [], "tb", "build_engine", {"repo": "https://example.org/x"})
        await gen_a.__anext__(); await gen_b.__anext__()
        assert not agent.decide("ta", True, True, b, "run_python")       # another chat can't answer it
        agent.stop(b)                                                      # stopping chat b leaves chat a's question open
        assert not agent._pending["ta"][2].done()
        assert agent.decide("ta", False, True, a, "get_jobs")              # the claimed name is ignored
        assert "get_jobs" not in agent._always.get(a, set())
        [e async for e in gen_a]; [e async for e in gen_b]
        gen_c = agent._tool_step(a, [], "tc", "run_python", {"code": "print(1)"})
        await gen_c.__anext__()
        agent.decide("tc", True, True, a, "something_else")
        assert agent._always[a] == {"run_python"}
        await gen_c.aclose()
    asyncio.run(flow())
