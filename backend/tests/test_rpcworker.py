"""This device as a llama.cpp RPC worker: off by default, localhost unless exposed on purpose, started and stopped cleanly."""
import sys

import pytest

from app import db, rpcworker


@pytest.fixture
def fake_rpc_server(tmp_path, monkeypatch, fresh_db):
    exe = tmp_path / "rpc-server"
    exe.write_text(f"#!{sys.executable}\nimport sys, time\nopen(sys.argv[0] + '.args', 'w').write(' '.join(sys.argv[1:]))\ntime.sleep(60)\n")
    exe.chmod(0o755)
    monkeypatch.setattr(rpcworker, "binary", lambda: exe)
    yield exe
    rpcworker.stop()


def test_off_by_default(fresh_db):
    s = rpcworker.status()
    assert s["enabled"] is False and s["running"] is False and s["host"] == "127.0.0.1"


def test_start_on_localhost_then_stop(fake_rpc_server):
    import time
    s = rpcworker.configure({"enabled": True, "port": 50060})
    assert s["running"] and s["host"] == "127.0.0.1"
    for _ in range(50):
        if fake_rpc_server.with_name("rpc-server.args").exists():
            break
        time.sleep(0.1)
    assert fake_rpc_server.with_name("rpc-server.args").read_text() == "-H 127.0.0.1 -p 50060"
    assert db.get_settings()["rpc_worker"]["enabled"] is True
    s = rpcworker.configure({"enabled": False})
    assert not s["running"]


def test_network_listening_is_explicit(fake_rpc_server):
    assert rpcworker.configure({"enabled": True, "expose": True})["host"] == "0.0.0.0"
    with pytest.raises(ValueError):
        rpcworker.configure({"enabled": True, "port": 80})


def test_finds_the_worker_under_either_name(tmp_path, monkeypatch):
    """llama.cpp renamed rpc-server to ggml-rpc-server: builds from before and after the rename both work."""
    from app import engines
    monkeypatch.setattr(engines, "bin_dir", lambda engine: tmp_path)
    assert rpcworker.binary() == tmp_path / "ggml-rpc-server"  # neither built: the current name, reported as missing
    (tmp_path / "rpc-server").write_text("")
    assert rpcworker.binary() == tmp_path / "rpc-server"
    (tmp_path / "ggml-rpc-server").write_text("")
    assert rpcworker.binary() == tmp_path / "ggml-rpc-server"


def test_missing_binary_is_explained(tmp_path, monkeypatch, fresh_db):
    monkeypatch.setattr(rpcworker, "binary", lambda: tmp_path / "nope")
    with pytest.raises(ValueError, match="install.sh"):
        rpcworker.configure({"enabled": True})
