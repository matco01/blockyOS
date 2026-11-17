"""This device as a llama.cpp RPC worker: another machine's llama.cpp can then place part of a model on this GPU.

The controller runs llama-server / llama-bench / llama-perplexity with `--rpc host:port,...` (an experiment's "RPC
workers" setting); each worker runs llama.cpp's `rpc-server`, which executes the tensors it is sent.

Security: rpc-server has no authentication and no encryption. Anyone who can reach its port can run computations on
this GPU and send it arbitrary data. So it's off by default and listens on localhost (reach it through an SSH tunnel:
`ssh -L 50052:localhost:50052 user@worker`); listening on the network is a separate, explicit choice for trusted
networks only. Blocky doesn't schedule the worker's GPU: a model loaded through RPC shares it with this device's own work.
"""
import subprocess

from . import db, engines

DEFAULT = {"enabled": False, "expose": False, "port": 50052}
_proc: subprocess.Popen | None = None
_error: str | None = None


def settings() -> dict:
    stored = db.get_settings().get("rpc_worker") or {}
    return {**DEFAULT, **(stored if isinstance(stored, dict) else {})}


def binary():
    return engines.bin_dir("stock") / "rpc-server"


def status() -> dict:
    s = settings()
    running = _proc is not None and _proc.poll() is None
    return {**s, "running": running, "host": "0.0.0.0" if s["expose"] else "127.0.0.1", "installed": binary().is_file(),
            "error": None if running or not s["enabled"] else _error}


def _start(s: dict) -> None:
    global _proc, _error
    stop()
    exe = binary()
    if not exe.is_file():
        _error = "rpc-server isn't built: re-run scripts/install.sh (it now builds llama.cpp with the RPC backend)"
        raise ValueError(_error)
    host = "0.0.0.0" if s["expose"] else "127.0.0.1"
    _proc = subprocess.Popen([str(exe), "-H", host, "-p", str(s["port"])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env={"LD_LIBRARY_PATH": str(exe.parent)}, start_new_session=True)
    _error = None


def stop() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def configure(body: dict) -> dict:
    port = int(body.get("port") or DEFAULT["port"])
    if not 1024 <= port <= 65535:
        raise ValueError("Port must be 1024-65535")
    s = {"enabled": bool(body.get("enabled")), "expose": bool(body.get("expose")), "port": port}
    db.update_settings({"rpc_worker": s})
    if s["enabled"]:
        _start(s)
    else:
        stop()
    return status()


def init() -> None:
    """Start the worker at boot if its owner turned it on."""
    global _error
    s = settings()
    if s["enabled"]:
        try:
            _start(s)
        except (ValueError, OSError) as e:
            _error = str(e)
