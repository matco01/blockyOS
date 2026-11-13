"""The PyTorch side of the lab: Hugging Face checkpoints, activations, LoRA fine-tuning back to GGUF.

Heavy work runs as scripts in a separate environment (/opt/blocky/ml-venv, see scripts/setup-ml.sh) so the
backend itself stays small. GPU jobs take the lab lock and pause the chat model, like every measurement.

  checkpoints   original Hugging Face weights (safetensors) in /opt/blocky/hf/<owner>--<name>
  activations   one prompt through the model: per-layer residual norms, layer deltas, logit lens, attention entropy
  lora          LoRA fine-tune on a dataset -> merge -> llama.cpp convert_hf_to_gguf.py -> a GGUF in the models folder
                (+ recipe.json), so it can be profiled in Experiments like any other weights
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import config, db, lab

ML = config.DATA_DIR.parent / "ml-venv"
PY = ML / "bin" / "python"
HF = config.DATA_DIR.parent / "hf"
WORK = config.DATA_DIR / "torch"
TOOLS = Path(__file__).resolve().parents[1] / "torch_tools"
LLAMA_SRC = Path("/home/blocky/llama.cpp")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_cancel: set[int] = set()

SCHEMA = """
CREATE TABLE IF NOT EXISTS torch_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    step        TEXT,
    params      TEXT NOT NULL,
    results     TEXT NOT NULL,
    error       TEXT,
    log         TEXT
);
"""


def init() -> None:
    for d in (HF, WORK):
        d.mkdir(parents=True, exist_ok=True)
    with db.connect() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE torch_jobs SET status = 'error', error = 'Interrupted by a restart' WHERE status IN ('queued', 'running')")


def status() -> dict:
    ok = PY.is_file()
    info = {"installed": ok, "checkpoints": checkpoints(), "converter": (LLAMA_SRC / "convert_hf_to_gguf.py").is_file()}
    if ok:
        try:
            out = subprocess.run([str(PY), "-c", "import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__)"],
                                 capture_output=True, text=True, timeout=60).stdout.split()
            info.update({"torch": out[0], "cuda": out[1] == "True", "transformers": out[2], "peft": out[3]})
        except (OSError, subprocess.SubprocessError, IndexError):
            info["installed"] = False
    return info


def checkpoints() -> list[dict]:
    out = []
    for d in sorted(HF.iterdir()) if HF.is_dir() else []:
        cfg = d / "config.json"
        if d.is_dir() and cfg.is_file():
            c = json.loads(cfg.read_text())
            size = sum(f.stat().st_size for f in d.glob("*.safetensors"))
            out.append({"id": d.name, "repo": d.name.replace("--", "/", 1), "architecture": (c.get("architectures") or ["?"])[0],
                        "layers": c.get("num_hidden_layers") or (c.get("text_config") or {}).get("num_hidden_layers"), "size_bytes": size})
    return out


def _row(r) -> dict:
    d = dict(r)
    d["params"], d["results"] = json.loads(d["params"]), json.loads(d["results"])
    d.pop("log", None)
    return d


def jobs(kind: str | None = None) -> list[dict]:
    with db.connect() as c:
        q = "SELECT * FROM torch_jobs" + (" WHERE kind = ?" if kind else "") + " ORDER BY id DESC LIMIT 30"
        return [_row(r) for r in c.execute(q, (kind,) if kind else ())]


def job(job_id: int) -> dict | None:
    with db.connect() as c:
        r = c.execute("SELECT * FROM torch_jobs WHERE id = ?", (job_id,)).fetchone()
    if not r:
        return None
    d = _row(r)
    d["log_tail"] = (dict(r)["log"] or "").splitlines()[-20:]
    return d


def _save(job_id: int, **f) -> None:
    vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in f.values()]
    with db.connect() as c:
        c.execute(f"UPDATE torch_jobs SET {', '.join(f'{k} = ?' for k in f)} WHERE id = ?", (*vals, job_id))


def _new(kind: str, params: dict) -> int:
    if not PY.is_file():
        raise ValueError("The PyTorch environment isn't installed yet (scripts/setup-ml.sh)")
    with db.connect() as c:
        return c.execute("INSERT INTO torch_jobs (created_at, kind, status, params, results) VALUES (?, ?, 'queued', ?, '{}')",
                         (time.time(), kind, json.dumps(params))).lastrowid


def cancel(job_id: int) -> None:
    _cancel.add(job_id)


def _ckpt(cid: str) -> Path:
    d = HF / Path(str(cid)).name
    if not (d / "config.json").is_file():
        raise ValueError("Checkpoint not found: download it first")
    return d


async def _py(job_id: int, args: list[str], log: list[str], timeout: float = 6 * 3600, env: dict | None = None) -> int:
    proc = await asyncio.create_subprocess_exec(str(PY), *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1", **(env or {})})
    start = time.monotonic()
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line:
            log.append(line[-300:])
            del log[:-300]
        if job_id in _cancel or time.monotonic() - start > timeout:
            proc.kill()
            raise asyncio.CancelledError
        if len(log) % 10 == 0:
            _save(job_id, log="\n".join(log[-200:]))
    return await proc.wait()


# ---------- Jobs ----------

def start_download(repo: str) -> int:
    repo = repo.strip()
    if not _REPO.match(repo):
        raise ValueError("Use the form owner/name, e.g. Qwen/Qwen3.5-0.8B")
    jid = _new("download", {"repo": repo})
    asyncio.get_running_loop().create_task(_download(jid, repo))
    return jid


async def _download(jid: int, repo: str) -> None:
    log: list[str] = []
    dest = HF / repo.replace("/", "--", 1)
    _save(jid, status="running", step="downloading")
    code = ("import sys; from huggingface_hub import snapshot_download as s; "
            "p = s(sys.argv[1], local_dir=sys.argv[2], allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', 'tokenizer*', '*.tiktoken']); print('done', p)")
    try:
        if await _py(jid, ["-c", code, repo, str(dest)], log, timeout=4 * 3600):
            raise RuntimeError(" | ".join(log[-2:])[-300:] or "download failed")
        if not (dest / "config.json").is_file():
            raise RuntimeError("That repository has no config.json: not a Transformers checkpoint")
        _save(jid, status="done", step=None, results={"checkpoint": dest.name}, log="\n".join(log[-50:]))
    except asyncio.CancelledError:
        _save(jid, status="cancelled", step=None)
    except Exception as e:  # noqa: BLE001
        _save(jid, status="error", step=None, error=str(e)[:300], log="\n".join(log[-50:]))


def start_activations(body: dict) -> int:
    d = _ckpt(body.get("checkpoint", ""))
    prompt = str(body.get("prompt") or "").strip()[:4000]
    if not prompt:
        raise ValueError("Type a prompt")
    jid = _new("activations", {"checkpoint": d.name, "prompt": prompt})
    asyncio.get_running_loop().create_task(_gpu(jid, _activations, d, prompt))
    return jid


async def _gpu(jid: int, fn, *args) -> None:
    """GPU work: take the lab lock, pause the chat model, always bring it back."""
    log: list[str] = []
    async with lab._lock:
        lab.busy = "torch"
        _save(jid, status="running")
        try:
            await lab.llm_service("stop")
            await asyncio.sleep(1)
            res = await fn(jid, log, *args)
            _save(jid, status="done", step=None, results=res, log="\n".join(log[-200:]))
        except asyncio.CancelledError:
            _save(jid, status="cancelled", step=None, log="\n".join(log[-200:]))
        except Exception as e:  # noqa: BLE001
            _save(jid, status="error", step=None, error=str(e)[:400], log="\n".join(log[-200:]))
        finally:
            _cancel.discard(jid)
            try:
                await lab.llm_service("start")
            except Exception:  # noqa: BLE001
                pass
            lab.busy = None


async def _activations(jid: int, log: list[str], d: Path, prompt: str) -> dict:
    _save(jid, step="running the prompt")
    out = WORK / f"act-{jid}.json"
    if await _py(jid, [str(TOOLS / "activations.py"), "--model", str(d), "--prompt", prompt, "--out", str(out), "--timing", "--knockout"], log, timeout=1800):
        raise RuntimeError(" | ".join(log[-3:])[-400:])
    res = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    return res


QUANT_METHODS = ("rtn", "gptq", "awq", "nf4", "ternary", "custom", "none")
EXPORT_TYPES = ("Q4_0", "Q4_1", "Q5_0", "Q8_0", "F16")


def start_quantsim(body: dict) -> int:
    """Technique playground: simulate a quantization method on a checkpoint and measure its effect (fake quantization)."""
    d = _ckpt(body.get("checkpoint", ""))
    method = str(body.get("method") or "rtn")
    if method not in QUANT_METHODS:
        raise ValueError(f"Method must be one of {', '.join(QUANT_METHODS)}")
    grid = "q4_0" if body.get("grid") == "q4_0" else "int"
    p = {"checkpoint": d.name, "method": method, "grid": grid, "bits": max(2, min(8, int(body.get("bits") or 4))),
         "group": int(body.get("group") or 128), "sym": bool(body.get("sym", True)), "include_head": bool(body.get("include_head")),
         "kv_bits": max(0, min(8, int(body.get("kv_bits") or 0))), "chunks": max(2, min(64, int(body.get("chunks") or 16))),
         "code": str(body.get("code") or "")[:20000], "label": str(body.get("label") or "")[:60]}
    if grid == "q4_0" and method not in ("rtn", "gptq", "awq"):
        raise ValueError("The Q4_0 grid works with RTN, GPTQ and AWQ")
    if p["group"] not in (32, 64, 128, 256):
        raise ValueError("Group size must be 32, 64, 128 or 256")
    if method == "custom" and "def quantize" not in p["code"]:
        raise ValueError("Custom method: write a function  def quantize(w, name): ...  that returns the dequantized weight")
    if method == "none" and not p["kv_bits"]:
        raise ValueError("Nothing to simulate: pick a weight method or KV bits")
    jid = _new("quantsim", p)
    asyncio.get_running_loop().create_task(_gpu(jid, _quantsim, d, p))
    return jid


def _qsim_args(jid: int, d: Path, p: dict, text: Path, calib: Path, out: Path) -> list[str]:
    args = [str(TOOLS / "quantsim.py"), "--model", str(d), "--text", str(text), "--calib", str(calib), "--out", str(out), "--method", p["method"],
            "--grid", p.get("grid") or "int", "--bits", str(p["bits"]), "--group", str(p["group"]), "--chunks", str(p["chunks"]), "--kv_bits", str(p["kv_bits"])]
    if p["sym"]:
        args.append("--sym")
    if p["include_head"]:
        args.append("--include_head")
    if p["method"] == "custom":
        code = WORK / f"qsim-{jid}-custom.py"
        code.write_text(p["code"])
        args += ["--custom", str(code)]
    return args


async def _quantsim(jid: int, log: list[str], d: Path, p: dict) -> dict:
    from . import quantlab, weights
    text, calib = await quantlab.ensure_text(), await weights.ensure_calibration()
    out = WORK / f"qsim-{jid}.json"
    args = _qsim_args(jid, d, p, text, calib, out)

    async def follow():
        while True:
            await asyncio.sleep(3)
            last = next((x for x in reversed(log) if x.startswith(("chunk", "quantized", "loaded", "loading", "calibrated", "exported", "layer "))), None)
            if last:
                _save(jid, step=last)

    watcher = asyncio.create_task(follow())
    try:
        _save(jid, step="loading the model")
        if await _py(jid, args, log, timeout=3 * 3600):
            raise RuntimeError(" | ".join(log[-3:])[-400:])
    finally:
        watcher.cancel()
    res = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    return res


def start_export(sim_id: int, body: dict) -> int:
    """Turn a playground method into a real GGUF (same method, applied again and saved), and optionally profile it."""
    src = job(sim_id)
    if not src or src["kind"] != "quantsim" or src["status"] != "done":
        raise ValueError("Pick a finished simulation")
    qtype = str(body.get("type") or "Q4_0").upper()
    if qtype not in EXPORT_TYPES:
        raise ValueError(f"Type must be one of {', '.join(EXPORT_TYPES)}")
    p = {**src["params"], "sim_id": sim_id, "type": qtype, "profile": bool(body.get("profile", True))}
    if p["method"] == "none":
        raise ValueError("Nothing to export: that simulation only changed the KV cache")
    d = _ckpt(p["checkpoint"])
    jid = _new("qexport", p)
    asyncio.get_running_loop().create_task(_gpu(jid, _export, d, p))
    return jid


async def _convert(jid: int, log: list[str], hf_dir: Path, dest: Path) -> None:
    conv = [str(LLAMA_SRC / "convert_hf_to_gguf.py"), str(hf_dir), "--outfile", str(dest.with_suffix(".gguf.part")), "--outtype", "f16"]
    if await _py(jid, conv, log, timeout=3600, env={"PYTHONPATH": str(LLAMA_SRC / "gguf-py")}):
        raise RuntimeError("GGUF conversion failed: " + " | ".join(log[-3:])[-300:])
    dest.with_suffix(".gguf.part").rename(dest)


async def _quantize(jid: int, log: list[str], src: Path, dest: Path, qtype: str) -> list[str]:
    # Every quantized matrix gets exactly the target type (--pure); embeddings and the output head stay at q8_0,
    # close to the fp16 the simulation kept them at.
    cmd = [str(lab.LLAMA_BIN / "llama-quantize"), "--pure", "--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0", str(src), str(dest), qtype]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                env={**os.environ, "LD_LIBRARY_PATH": str(lab.LLAMA_BIN)})
    async for raw in proc.stdout:
        log.append(raw.decode(errors="replace").rstrip()[-300:])
        del log[:-300]
        if jid in _cancel:
            proc.kill()
            raise asyncio.CancelledError
    if await proc.wait():
        raise RuntimeError("llama-quantize failed: " + " | ".join(log[-3:])[-300:])
    return cmd


async def _export(jid: int, log: list[str], d: Path, p: dict) -> dict:
    from . import profile, quantlab, weights
    text, calib = await quantlab.ensure_text(), await weights.ensure_calibration()
    short = d.name.split("--", 1)[-1]
    work = WORK / f"qexp-{jid}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        # 1) The original checkpoint as an F16 GGUF: the reference everything from this checkpoint is compared with.
        base = lab.MODELS_DIR / f"{short}-F16.gguf"
        if not base.exists():
            _save(jid, step="converting the original checkpoint to GGUF (reference)")
            await _convert(jid, log, d, base)
        # 2) Apply the method again and save the quantized weights as a checkpoint.
        _save(jid, step="quantizing with the method")
        out = work / "res.json"
        if await _py(jid, _qsim_args(jid, d, p, text, calib, out) + ["--export", str(work / "hf"), "--no_eval"], log, timeout=3 * 3600):
            raise RuntimeError(" | ".join(log[-3:])[-400:])
        # 3) To GGUF, then to the real type.
        _save(jid, step="converting to GGUF")
        f16 = work / "variant-F16.gguf"
        await _convert(jid, log, work / "hf", f16)
        tag = (p.get("label") or f"{p['method']}{'-q4_0grid' if p.get('grid') == 'q4_0' else f'-{p['bits']}b-g{p['group']}'}").replace(" ", "-")
        name = f"{short}-{tag}-{p['type']}.gguf"
        dest = lab.MODELS_DIR / name
        if dest.exists():
            name = name.replace(".gguf", f"-{jid}.gguf"); dest = lab.MODELS_DIR / name
        cmds = []
        if p["type"] == "F16":
            f16.rename(dest)
        else:
            _save(jid, step=f"quantizing to {p['type']}")
            cmds = await _quantize(jid, log, f16, dest, p["type"])
        exact = p.get("grid") == "q4_0" and p["type"] in ("Q4_0", "F16")
        recipe = {"made_by": "Blocky technique playground", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base_checkpoint": d.name.replace("--", "/", 1),
                  "method": {k: v for k, v in p.items() if k not in ("profile", "type", "sim_id")}, "gguf_type": p["type"],
                  "simulation_job": p["sim_id"], "exact": exact,
                  "note": ("The GGUF stores the simulated values up to fp16 rounding of the scales (same grid)." if exact else
                           f"Re-quantized to {p['type']}: the GGUF's values are close to, but not exactly, the simulated ones."),
                  "commands": [" ".join(cmds)] if cmds else [], "size_bytes": dest.stat().st_size}
        dest.with_suffix(".recipe.json").write_text(json.dumps(recipe, indent=1))
        res = {"gguf": name, "reference": base.name, "size_bytes": dest.stat().st_size, "exact": exact, "note": recipe["note"]}
        # 4) Profile it against the F16 reference, in one experiment per checkpoint.
        if p["profile"]:
            exp_name = f"{short} · playground"
            exp = next((e for e in profile.list_experiments() if e["name"] == exp_name), None)
            eid = exp["id"] if exp else profile.create_experiment(exp_name, base.name)
            runs = (profile.get_experiment(eid) or {}).get("runs_list") or []
            base_run = next((r for r in runs if r["config"].get("model") == base.name and r.get("parent_id") is None), None)
            if not base_run:
                base_run = {"id": profile.create_run(eid, {"config": {"model": base.name}})}
            rid = profile.create_run(eid, {"config": {"model": name}, "parent_id": base_run["id"], "label": f"{tag} → {p['type']}"})
            res.update({"experiment_id": eid, "run_id": rid, "baseline_run_id": base_run["id"]})
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


def start_lora(body: dict) -> int:
    d = _ckpt(body.get("checkpoint", ""))
    data = str(body.get("data") or "").strip()
    if not data or not re.match(r"^[A-Za-z0-9_./:-]+$", data) or ".." in data:
        raise ValueError("Dataset: a Hugging Face id like yahma/alpaca-cleaned (optionally id:config)")
    p = {"checkpoint": d.name, "data": data, "split": str(body.get("split") or "train")[:40],
         "rows": max(10, min(200_000, int(body.get("rows") or 2000))), "steps": max(1, min(20_000, int(body.get("steps") or 200))),
         "lr": float(body.get("lr") or 2e-4), "rank": max(1, min(256, int(body.get("rank") or 16))),
         "batch": max(1, min(64, int(body.get("batch") or 4))), "max_len": max(64, min(8192, int(body.get("max_len") or 512))),
         "outtype": body.get("outtype") if body.get("outtype") in ("q8_0", "f16", "bf16") else "q8_0",
         "name": re.sub(r"[^A-Za-z0-9._-]+", "-", str(body.get("name") or "")).strip("-")[:40],
         "qlora": bool(body.get("qlora"))}
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", p["split"]):
        raise ValueError("Bad split name")
    jid = _new("lora", p)
    asyncio.get_running_loop().create_task(_gpu(jid, _lora, d, p))
    return jid


async def _lora(jid: int, log: list[str], d: Path, p: dict) -> dict:
    out = WORK / f"lora-{jid}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    prog = out / "progress.jsonl"

    async def follow():  # stream the loss curve into the job's results while training
        while True:
            await asyncio.sleep(5)
            if prog.is_file():
                lines = [json.loads(x) for x in prog.read_text().splitlines() if x.strip()]
                steps = [x for x in lines if "step" in x]
                _save(jid, step=f"training · step {steps[-1]['step']} of {p['steps']}" if steps else "loading data",
                      results={"curve": steps[::max(1, len(steps) // 400)], "info": next((x for x in lines if x.get("event") == "start"), None)})

    watcher = asyncio.create_task(follow())
    try:
        _save(jid, step="loading data")
        args = [str(TOOLS / "lora.py"), "--model", str(d), "--data", p["data"], "--split", p["split"], "--rows", str(p["rows"]), "--steps", str(p["steps"]),
                "--lr", str(p["lr"]), "--rank", str(p["rank"]), "--batch", str(p["batch"]), "--max_len", str(p["max_len"]), "--out", str(out)]
        if p["qlora"]:
            args.append("--qlora")
        if await _py(jid, args, log):
            raise RuntimeError(" | ".join(log[-3:])[-400:])
    finally:
        watcher.cancel()
    lines = [json.loads(x) for x in prog.read_text().splitlines() if x.strip()]
    steps = [x for x in lines if "step" in x]

    # Back to GGUF with the same llama.cpp tree the engine was built from.
    _save(jid, step="converting to GGUF")
    tag = p["name"] or f"{'q' if p['qlora'] else ''}lora-r{p['rank']}-{p['steps']}s"
    name = f"{d.name.split('--', 1)[-1]}-{tag}-{p['outtype'].upper()}.gguf"
    dest = lab.MODELS_DIR / name
    if dest.exists():
        name = name.replace(".gguf", f"-{jid}.gguf"); dest = lab.MODELS_DIR / name
    conv = [str(LLAMA_SRC / "convert_hf_to_gguf.py"), str(out / "merged"), "--outfile", str(dest.with_suffix(".gguf.part")), "--outtype", p["outtype"]]
    if await _py(jid, conv, log, timeout=3600, env={"PYTHONPATH": str(LLAMA_SRC / "gguf-py")}):
        raise RuntimeError("GGUF conversion failed: " + " | ".join(log[-3:])[-300:])
    dest.with_suffix(".gguf.part").rename(dest)
    recipe = {"made_by": "Blocky LoRA fine-tune", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base_checkpoint": d.name.replace("--", "/", 1),
              "params": p, "final_loss": steps[-1]["loss"] if steps else None, "first_loss": steps[0]["loss"] if steps else None,
              "commands": ["python " + " ".join(args), "python " + " ".join(conv)], "size_bytes": dest.stat().st_size}
    dest.with_suffix(".recipe.json").write_text(json.dumps(recipe, indent=1))
    shutil.rmtree(out / "merged", ignore_errors=True)  # the GGUF is what we keep
    return {"curve": steps[::max(1, len(steps) // 400)], "info": next((x for x in lines if x.get("event") == "start"), None),
            "gguf": name, "size_bytes": dest.stat().st_size, "final_loss": recipe["final_loss"], "first_loss": recipe["first_loss"]}
