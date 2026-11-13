"""Layer sensitivity scan: which layers of a model can be squeezed, and which must be protected?

For each layer, build a copy of the model where ONLY that layer is changed, and measure how far its token
predictions drift from the original (KL divergence on WikiText-2):

  quantize  that layer's tensors at a low precision (e.g. q2_k), everything else kept at the source precision
  prune     that layer removed entirely

A "control" copy (same re-quantization, nothing changed) gives the noise floor, so each layer's number is the
drift caused by that layer alone. The result guides mixed-precision recipes (protect the sensitive layers, squeeze
the rest) and layer pruning, and can be turned into a weights-workshop recipe in one click.
"""
import asyncio
import json
import time
from pathlib import Path

from . import config, db, gguf, lab, quantlab

DIR = config.DATA_DIR / "scans"
CHUNKS = 12                  # 12 × 512 tokens of WikiText-2 per measurement
LOW_TYPES = ["q2_k", "q3_k", "q4_k", "iq4_xs"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS layer_scans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    model       TEXT NOT NULL,
    mode        TEXT NOT NULL,
    qtype       TEXT,
    status      TEXT NOT NULL,
    step        TEXT,
    results     TEXT NOT NULL,
    error       TEXT
);
"""
_cancel: set[int] = set()


def init() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    with db.connect() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE layer_scans SET status = 'error', error = 'Interrupted by a restart' WHERE status IN ('queued', 'running')")


def _row(r) -> dict:
    d = dict(r)
    d["results"] = json.loads(d["results"])
    return d


def list_scans(model: str | None = None) -> list[dict]:
    with db.connect() as c:
        q, a = ("SELECT * FROM layer_scans WHERE model = ? ORDER BY id DESC", (model,)) if model else ("SELECT * FROM layer_scans ORDER BY id DESC", ())
        return [_row(r) for r in c.execute(q, a)]


def _save(scan_id: int, **f) -> None:
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()]
    with db.connect() as c:
        c.execute(f"UPDATE layer_scans SET {', '.join(f'{k} = ?' for k in f)} WHERE id = ?", (*vals, scan_id))


def start(body: dict) -> int:
    model = Path(str(body.get("model") or "")).name
    if not (lab.MODELS_DIR / model).is_file():
        raise ValueError("Model not found")
    mode = body.get("mode") or "quantize"
    if mode not in ("quantize", "prune"):
        raise ValueError("Mode must be quantize or prune")
    qtype = (body.get("qtype") or "q2_k").lower()
    if mode == "quantize" and qtype not in LOW_TYPES:
        raise ValueError(f"Type must be one of {', '.join(LOW_TYPES)}")
    if not gguf.metadata(lab.MODELS_DIR / model).get("layers"):
        raise ValueError("Couldn't read the number of layers")
    with db.connect() as c:
        scan_id = c.execute("INSERT INTO layer_scans (created_at, model, mode, qtype, status, results) VALUES (?, ?, ?, ?, 'queued', '{}')",
                            (time.time(), model, mode, qtype if mode == "quantize" else None)).lastrowid
    asyncio.get_running_loop().create_task(_run(scan_id))
    return scan_id


def cancel(scan_id: int) -> None:
    _cancel.add(scan_id)
    quantlab._cancel.add(-1_000_000 - scan_id)


def delete(scan_id: int) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM layer_scans WHERE id = ? AND status NOT IN ('queued', 'running')", (scan_id,))


def _source_type(src: Path) -> str:
    """Re-quantization target for the untouched tensors: the source's own type (Q8_0 stays Q8_0)."""
    q = (gguf.metadata(src).get("quant") or "Q8_0").upper()
    return q if q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "F16", "BF16") else "Q8_0"


async def _kld(src_model: Path, base: Path, text: Path, log: list[str], pid: int, make_base: bool = False) -> dict:
    cmd = [str(lab.LLAMA_BIN / "llama-perplexity"), "-m", str(src_model), "-f", str(text), "-c", "512", "--chunks", str(CHUNKS), "--fit", "on"]
    cmd += ["--kl-divergence-base", str(base)] + ([] if make_base else ["--kl-divergence"])
    out = await quantlab._proc(cmd, log, pid, timeout=1800)
    if make_base:
        return {"ppl": quantlab._num(r"Final estimate: PPL = ([\d.]+)", out)}
    return {"kld": quantlab._num(r"Mean\s+KLD:\s*(-?[\d.]+(?:e-?\d+)?)", out), "kld_p99": quantlab._num(r"99\.0%\s+KLD:\s*([\d.]+)", out),
            "same_top": quantlab._num(r"Same top p:\s*([\d.]+)", out), "ppl": quantlab._num(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", out)}


async def _variant(src: Path, out: Path, qtype_src: str, layer: int | None, mode: str, low: str | None, log: list[str], pid: int) -> None:
    cmd = [str(lab.LLAMA_BIN / "llama-quantize"), "--allow-requantize"]
    if layer is not None and mode == "quantize":
        cmd += ["--tensor-type", f"blk\\.{layer}\\.={low}"]
    if layer is not None and mode == "prune":
        cmd += ["--prune-layers", str(layer)]
    cmd += [str(src), str(out), qtype_src, "8"]
    await quantlab._proc(cmd, log, pid, timeout=1800)
    if not out.is_file() or out.stat().st_size < 1_000_000:
        raise RuntimeError("llama-quantize failed: " + " | ".join(log[-3:]))


async def _run(scan_id: int) -> None:
    with db.connect() as c:
        s = _row(c.execute("SELECT * FROM layer_scans WHERE id = ?", (scan_id,)).fetchone())
    src = lab.MODELS_DIR / s["model"]
    n_layers = gguf.metadata(src).get("layers") or 0
    qsrc = _source_type(src)
    work = DIR / str(scan_id)
    work.mkdir(parents=True, exist_ok=True)
    tmp, base = work / "variant.gguf", work / "base.kld"
    pid = -1_000_000 - scan_id
    log: list[str] = []
    res: dict = {"layers": [], "n_layers": n_layers, "source_type": qsrc, "chunks": CHUNKS, "started_at": time.time()}
    async with lab._lock:
        lab.busy = "scan"
        _save(scan_id, status="running")
        try:
            await lab.llm_service("stop")
            await asyncio.sleep(1)
            text = await quantlab.ensure_text()
            _save(scan_id, step="reference logits from the original model")
            res["reference"] = await _kld(src, base, text, log, pid, make_base=True)
            # Control: the same re-quantization with nothing changed = the noise floor every layer is measured against.
            _save(scan_id, step="control (nothing changed)")
            await _variant(src, tmp, qsrc, None, s["mode"], None, log, pid)
            res["control"] = await _kld(tmp, base, text, log, pid)
            res["variant_bytes_control"] = tmp.stat().st_size
            tmp.unlink(missing_ok=True)
            _save(scan_id, results=res)
            for layer in range(n_layers):
                if scan_id in _cancel:
                    raise asyncio.CancelledError
                _save(scan_id, step=f"layer {layer + 1} of {n_layers}")
                t0 = time.monotonic()
                await _variant(src, tmp, qsrc, layer, s["mode"], s["qtype"], log, pid)
                size = tmp.stat().st_size
                m = await _kld(tmp, base, text, log, pid)
                tmp.unlink(missing_ok=True)
                floor = (res["control"].get("kld") or 0)
                res["layers"].append({"layer": layer, **m, "excess_kld": round((m.get("kld") or 0) - floor, 6),
                                      "bytes": size, "seconds": round(time.monotonic() - t0, 1)})
                _save(scan_id, results=res)
            ranked = sorted(res["layers"], key=lambda x: -(x["excess_kld"] or 0))
            res["most_sensitive"] = [x["layer"] for x in ranked[:8]]
            res["least_sensitive"] = [x["layer"] for x in ranked[::-1][:8]]
            res["finished_at"] = time.time()
            _save(scan_id, status="done", step=None, results=res)
        except asyncio.CancelledError:
            _save(scan_id, status="cancelled", step=None, results=res)
        except Exception as e:  # noqa: BLE001
            _save(scan_id, status="error", step=None, error=str(e)[:300], results=res)
        finally:
            tmp.unlink(missing_ok=True)
            base.unlink(missing_ok=True)
            _cancel.discard(scan_id)
            quantlab._cancel.discard(pid)
            try:
                await lab.llm_service("start")
            except Exception:  # noqa: BLE001
                pass
            lab.busy = None
