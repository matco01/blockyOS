"""Weights workshop: make custom versions of a model's weights, then profile them in Experiments.

A variant is built from a high-precision *source* file by llama.cpp's own tools:

  importance matrix  llama-imatrix on calibration text (WikiText-2 *train*, never the test text we score on),
                     so the quantizer spends precision where activations are large
  quantization       llama-quantize to a base type, with optional per-tensor overrides (e.g. keep attention
                     at 8-bit), a different output-head / embedding type, and layer pruning

Every variant gets a recipe file next to it (<name>.recipe.json): source + its SHA-256, imatrix details, the exact
command. Experiments' provenance picks that up, so a result can always be traced back to how its weights were made.
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

from . import benchmarks, config, gguf, lab, quantlab

DIR = config.DATA_DIR / "weights"
CALIB = config.DATA_DIR / "benchmarks" / "wikitext2-train-calib.txt"
CALIB_ROWS = 4000            # ≈ 300k tokens of text; imatrix uses the first IMATRIX_CHUNKS × 512 tokens
IMATRIX_CHUNKS = 128

BASE_TYPES = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
              "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_XXS", "IQ2_M", "IQ2_XS"]
NEEDS_IMATRIX = {"IQ3_XXS", "IQ2_M", "IQ2_XS"}  # llama-quantize refuses these without one
TENSOR_TYPES = ["q2_k", "q3_k", "q4_0", "q4_k", "q5_k", "q6_k", "q8_0", "iq4_xs", "f16", "bf16"]
_PATTERN = re.compile(r"^[A-Za-z0-9_.\\*+?\[\]-]{1,64}$")

jobs: dict[str, dict] = {}
_cancel: set[str] = set()


def init() -> None:
    DIR.mkdir(parents=True, exist_ok=True)


# ---------- What can be built from what ----------

def _bits(name: str) -> float:
    return quantlab._bits(name)


def family(model: str) -> list[dict]:
    """Files on disk that are versions of the same model (same architecture and model name), best precision first."""
    ref = gguf.metadata(lab.MODELS_DIR / Path(model).name)
    out = []
    for p in sorted(lab.MODELS_DIR.glob("*.gguf")):
        try:
            m = gguf.metadata(p)
        except (OSError, ValueError):
            continue
        if m.get("architecture") == ref.get("architecture") and m.get("name") == ref.get("name"):
            recipe = p.with_suffix(".recipe.json")
            out.append({"file": p.name, "quant": m.get("quant"), "bits": _bits(p.name), "size_bytes": p.stat().st_size,
                        "layers": m.get("layers"), "recipe": json.loads(recipe.read_text()) if recipe.is_file() else None})
    return sorted(out, key=lambda f: -f["bits"])


def tensor_groups(model: str) -> list[dict]:
    """Tensor name patterns in this model (layer number stripped), with count, size and current types, for overrides."""
    path = lab.MODELS_DIR / Path(model).name
    t = gguf._tensors(str(path), path.stat().st_mtime)["tensors"]
    groups: dict[str, dict] = {}
    for x in t:
        key = re.sub(r"^blk\.\d+\.", "", x["name"]).replace(".weight", "")
        g = groups.setdefault(key, {"pattern": key, "count": 0, "bytes": 0, "types": set(), "part": gguf._part(x["name"])})
        g["count"] += 1; g["bytes"] += x["bytes"]; g["types"].add(x["type"])
    return sorted(({**g, "types": sorted(g["types"])} for g in groups.values() if not g["pattern"].endswith(".bias")),
                  key=lambda g: -g["bytes"])


# ---------- Calibration text (separate from the evaluation text) ----------

async def ensure_calibration() -> Path:
    if CALIB.is_file() and CALIB.stat().st_size > 200_000:
        return CALIB
    rows: list[dict] = []
    failed = 0
    async with httpx.AsyncClient(timeout=60, headers=benchmarks.HEADERS) as c:
        for off in range(0, CALIB_ROWS, 100):
            for attempt in range(5):  # the datasets server returns occasional 502/429s
                try:
                    r = await c.get("https://datasets-server.huggingface.co/rows",
                                    params={"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "train", "offset": off, "length": 100})
                    r.raise_for_status()
                    rows += [x["row"] for x in r.json().get("rows", [])]
                    break
                except (httpx.HTTPError, ValueError):
                    await asyncio.sleep(2 * (attempt + 1))
            else:
                failed += 1
                if failed > 4:
                    raise RuntimeError("Couldn't download the calibration text from Hugging Face (server errors). Try again later.")
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text("".join(r.get("text", "") for r in rows))
    return CALIB


# ---------- Building a variant ----------

def _validate(spec: dict) -> dict:
    src = Path(str(spec.get("source") or "")).name
    if not (lab.MODELS_DIR / src).is_file():
        raise ValueError("Source file not found")
    qtype = str(spec.get("qtype") or "").upper()
    if qtype not in BASE_TYPES:
        raise ValueError(f"Base type must be one of {', '.join(BASE_TYPES)}")
    use_imatrix = bool(spec.get("imatrix")) or qtype in NEEDS_IMATRIX
    overrides = {}
    for pat, typ in (spec.get("overrides") or {}).items():
        pat, typ = str(pat).strip(), str(typ).strip().lower()
        if not _PATTERN.match(pat) or typ not in TENSOR_TYPES:
            raise ValueError(f"Bad override {pat}={typ}: use a tensor name pattern and one of {', '.join(TENSOR_TYPES)}")
        overrides[pat] = typ
    out_t = (spec.get("output_type") or "").lower() or None
    emb_t = (spec.get("embedding_type") or "").lower() or None
    for t in (out_t, emb_t):
        if t and t not in TENSOR_TYPES:
            raise ValueError(f"Type must be one of {', '.join(TENSOR_TYPES)}")
    layers = gguf.metadata(lab.MODELS_DIR / src).get("layers") or 0
    prune = sorted({int(x) for x in (spec.get("prune") or [])})
    if any(x < 0 or x >= layers for x in prune) or len(prune) >= layers:
        raise ValueError(f"Layers to prune must be between 0 and {layers - 1}, and not all of them")
    stem = re.sub(r"-(UD-)?(I?Q\d\w*|BF16|F16|F32)$", "", src.replace(".gguf", ""), flags=re.I)
    tag = "-imat" if use_imatrix else ""
    if overrides or out_t or emb_t:
        tag += "-custom"
    if prune:
        tag += f"-pruned{len(prune)}"
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(spec.get("name") or "")).strip("-")
    out = f"{name or f'{stem}-{qtype}{tag}'}.gguf"
    if (lab.MODELS_DIR / out).exists():
        raise ValueError(f"{out} already exists. Give the variant a different name.")
    return {"source": src, "qtype": qtype, "imatrix": use_imatrix, "overrides": overrides, "output_type": out_t,
            "embedding_type": emb_t, "prune": prune, "out": out}


def start(spec: dict) -> str:
    s = _validate(spec)
    job_id = str(int(time.time() * 1000))
    jobs[job_id] = {"id": job_id, **s, "status": "queued", "step": None, "progress": None, "log": [], "error": None, "started": time.time()}
    asyncio.get_running_loop().create_task(_build(job_id))
    return job_id


def _pid(job_id: str) -> int:
    return -int(job_id)  # quantlab._proc kills a process when its (numeric) id is in quantlab._cancel


def cancel(job_id: str) -> None:
    _cancel.add(job_id)
    if job_id in jobs:
        quantlab._cancel.add(_pid(job_id))


async def _imatrix(job: dict, src: Path, log: list[str]) -> Path:
    """Importance matrix for a source file (cached: same source + same calibration = same matrix)."""
    from .profile import file_sha256  # local import: profile imports a lot
    out = DIR / f"{src.stem}.imatrix.gguf"
    if out.is_file():
        job["imatrix_cached"] = True
        return out
    calib = await ensure_calibration()
    job["step"] = "importance matrix"
    cmd = [str(lab.LLAMA_BIN / "llama-imatrix"), "-m", str(src), "-f", str(calib), "-o", str(out.with_suffix(".part")),
           "--chunks", str(IMATRIX_CHUNKS), "-c", "512", "--fit", "on", "-ofreq", "1000"]
    job["commands"].append(" ".join(cmd))
    await quantlab._proc(cmd, log, _pid(job["id"]), timeout=3600)
    part = out.with_suffix(".part")
    if not part.is_file():
        raise RuntimeError("llama-imatrix produced no output: " + " | ".join(log[-3:]))
    part.rename(out)
    (DIR / f"{src.stem}.imatrix.json").write_text(json.dumps({"source": src.name, "source_sha256": file_sha256(src),
        "calibration": "WikiText-2 raw, train split, first 4000 rows", "calibration_sha256": file_sha256(calib),
        "chunks": IMATRIX_CHUNKS, "ctx": 512, "created_at": time.time()}))
    return out


async def _build(job_id: str) -> None:
    from .profile import file_sha256
    j = jobs[job_id]
    j["commands"] = []
    src = lab.MODELS_DIR / j["source"]
    out = lab.MODELS_DIR / j["out"]
    tmp = out.with_suffix(".gguf.part")
    log: list[str] = j["log"]
    async with lab._lock:  # GPU (imatrix) and CPU (quantize) heavy: never overlap with measurements
        lab.busy = "weights"
        j["status"] = "running"
        stopped = False
        try:
            imat = None
            if j["imatrix"]:
                await lab.llm_service("stop"); stopped = True
                await asyncio.sleep(1)
                imat = await _imatrix(j, src, log)
            if job_id in _cancel:
                raise asyncio.CancelledError
            j["step"] = "quantizing"
            src_q = (gguf.metadata(src).get("quant") or "").upper()
            cmd = [str(lab.LLAMA_BIN / "llama-quantize")]
            if src_q not in ("F16", "BF16", "F32"):
                cmd.append("--allow-requantize")
            if imat:
                cmd += ["--imatrix", str(imat)]
            for pat, typ in j["overrides"].items():
                cmd += ["--tensor-type", f"{pat}={typ}"]
            if j["output_type"]:
                cmd += ["--output-tensor-type", j["output_type"]]
            if j["embedding_type"]:
                cmd += ["--token-embedding-type", j["embedding_type"]]
            if j["prune"]:
                cmd += ["--prune-layers", ",".join(str(x) for x in j["prune"])]
            cmd += [str(src), str(tmp), j["qtype"], str(max(1, (os.cpu_count() or 2) // 2))]
            j["commands"].append(" ".join(cmd))
            await quantlab._proc(cmd, log, _pid(job_id), timeout=3600)
            if not tmp.is_file() or tmp.stat().st_size < 1_000_000:
                raise RuntimeError("llama-quantize failed: " + " | ".join(log[-3:]))
            tmp.rename(out)
            recipe = {"made_by": "Blocky weights workshop", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "source": j["source"], "source_sha256": file_sha256(src), "qtype": j["qtype"],
                      "imatrix": json.loads((DIR / f"{src.stem}.imatrix.json").read_text()) if imat else None,
                      "overrides": j["overrides"], "output_type": j["output_type"], "embedding_type": j["embedding_type"],
                      "pruned_layers": j["prune"], "commands": j["commands"], "size_bytes": out.stat().st_size}
            out.with_suffix(".recipe.json").write_text(json.dumps(recipe, indent=1))
            j["status"], j["size_bytes"] = "done", out.stat().st_size
        except asyncio.CancelledError:
            j["status"] = "cancelled"
        except Exception as e:  # noqa: BLE001
            j["status"], j["error"] = "error", str(e)[:400]
        finally:
            tmp.unlink(missing_ok=True)
            (DIR / f"{src.stem}.imatrix.part").unlink(missing_ok=True)
            j["step"] = None
            _cancel.discard(job_id)
            quantlab._cancel.discard(_pid(job_id))
            if stopped:
                try:
                    await lab.llm_service("start")
                except Exception:  # noqa: BLE001
                    pass
            lab.busy = None


def list_jobs() -> list[dict]:
    return sorted(({k: v for k, v in j.items() if k != "log"} | {"log_tail": j["log"][-4:]} for j in jobs.values()),
                  key=lambda j: -j["started"])
