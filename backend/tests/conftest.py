"""Test setup: every path Blocky uses points into a temporary folder, so tests never touch a real device.

`make_gguf` writes tiny but valid GGUF files (header + metadata, no weights) so code that reads model metadata
(architecture, layers, tokenizer, quant type) can be tested without multi-GB models.
"""
import os
import struct
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="blocky-tests-"))
os.environ["BLOCKY_DATA_DIR"] = str(_TMP / "data")
os.environ["BLOCKY_MODELS_DIR"] = str(_TMP / "models")
os.environ["BLOCKY_LLAMA_BIN"] = str(_TMP / "llama-bin")
for d in ("data", "models", "llama-bin"):
    (_TMP / d).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _s(v: str) -> bytes:
    b = v.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    out = _s(key)
    if isinstance(value, str):
        return out + struct.pack("<I", 8) + _s(value)
    if isinstance(value, list):  # array of strings
        return out + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + b"".join(_s(x) for x in value)
    return out + struct.pack("<I", 4) + struct.pack("<I", int(value))


@pytest.fixture
def make_gguf():
    from app import lab

    def make(name: str, arch="qwen35", model_name="Test-Model", layers=4, ftype=15, vocab=32, pre="qwen35", ctx=8192):
        kvs = {"general.architecture": arch, "general.name": model_name, "general.file_type": ftype,
               f"{arch}.block_count": layers, f"{arch}.context_length": ctx, f"{arch}.embedding_length": 64,
               f"{arch}.attention.head_count": 4, "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": pre,
               "tokenizer.ggml.tokens": [f"t{i}" for i in range(vocab)]}
        body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs)) + b"".join(_kv(k, v) for k, v in kvs.items())
        p = lab.MODELS_DIR / name
        p.write_bytes(body)
        return p

    return make


@pytest.fixture(scope="session")
def http_client():
    """The whole app over HTTP from this device (loopback, Host: localhost). One per test session: the MCP SDK's session
    manager supports a single lifespan per process."""
    import asyncio
    from fastapi.testclient import TestClient
    from app import main, profile, research_jobs

    async def idle_worker(client):
        await asyncio.Event().wait()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(profile, "worker", idle_worker)
        patch.setattr(research_jobs, "worker", idle_worker)
        patch.setattr(profile, "limits", lambda: {"gpu": None, "vram_mb": 0})
        with TestClient(main.app, client=("127.0.0.1", 12345), base_url="http://localhost:8000") as client:
            yield client


@pytest.fixture
def fresh_db():
    """An empty database with every table Blocky creates."""
    from app import aiaccess, arena, config, db, engines, evals, jobstore, profile, projects, quantlab, research_jobs, scans, tools, torchlab, weights
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    db.init()
    for module in (aiaccess, jobstore, arena, evals, quantlab, tools, profile, weights, engines, scans, torchlab, research_jobs, projects):
        module.init()
    return db
