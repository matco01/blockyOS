"""Without systemd (containers), Blocky runs the chat model server itself, like blocky-llm.service would."""
import asyncio

from app import lab


def test_env_files_are_read_like_systemd(tmp_path):
    f = tmp_path / "blocky.env"
    f.write_text("# comment\n\nBLOCKY_CTX=4096\nexport BLOCKY_KV='q4_0'\nBLOCKY_MODEL=\"/m/a b.gguf\"\nnot a line\n1BAD=x\n")
    assert lab._env_file(f) == {"BLOCKY_CTX": "4096", "BLOCKY_KV": "q4_0", "BLOCKY_MODEL": "/m/a b.gguf"}
    assert lab._env_file(tmp_path / "missing.env") == {}


def test_supervised_chat_model_starts_logs_and_stops(tmp_path, monkeypatch):
    script = tmp_path / "start-llm.sh"
    script.write_text('#!/bin/sh\necho "blocky-llm: model=fake.gguf ctx=$BLOCKY_CTX"\nexec sleep 60\n')
    script.chmod(0o755)
    runtime = tmp_path / "llm.env"
    runtime.write_text("BLOCKY_CTX=2048\n")
    monkeypatch.setattr(lab, "SUPERVISE", True)
    monkeypatch.setattr(lab, "START_LLM", script)
    monkeypatch.setattr(lab, "LLM_LOG", tmp_path / "llm.log")
    monkeypatch.setattr(lab, "RUNTIME_FILE", runtime)
    monkeypatch.setattr(lab, "_llm", {"proc": None, "want": False, "started": None})

    async def scenario():
        await lab.llm_service("start")
        proc = lab._llm["proc"]
        await asyncio.sleep(0.5)
        assert proc.returncode is None and lab._llm["want"]
        log = await lab.llm_log(50)
        assert "blocky-llm: model=fake.gguf ctx=2048" in log      # the Models page's settings file was applied
        await lab.llm_service("restart")
        assert lab._llm["proc"] is not proc and proc.returncode is not None
        await lab.llm_service("stop")
        assert lab._llm["proc"].returncode is not None and not lab._llm["want"]
    asyncio.run(scenario())
