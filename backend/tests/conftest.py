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


@pytest.fixture
def fresh_db():
    """An empty database with every table Blocky creates."""
    from app import config, db, profile
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    db.init()
    profile.init()
    return db
