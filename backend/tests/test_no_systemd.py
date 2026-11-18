"""Machines without systemd (WSL1, containers): measurements must not fail on pausing a chat service that doesn't exist."""
import asyncio

import pytest

from app import lab


def test_pause_and_resume_are_noops_without_systemd(monkeypatch):
    calls = []

    async def run(*cmd, **kw):
        calls.append(cmd)
        return 1, "System has not been booted with systemd"

    monkeypatch.setattr(lab, "has_systemd", lambda: False)
    monkeypatch.setattr(lab, "_run", run)
    asyncio.run(lab.llm_service("stop"))
    asyncio.run(lab.llm_service("start"))
    assert calls == []
    with pytest.raises(RuntimeError, match="needs systemd"):
        asyncio.run(lab.llm_service("restart"))


def test_systemd_failures_still_reported(monkeypatch):
    async def run(*cmd, **kw):
        return 5, "Unit blocky-llm.service not found."

    monkeypatch.setattr(lab, "has_systemd", lambda: True)
    monkeypatch.setattr(lab, "_run", run)
    with pytest.raises(RuntimeError, match="systemctl stop failed"):
        asyncio.run(lab.llm_service("stop"))
