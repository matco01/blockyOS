"""Hardware lab, level 1: run a model on a simulated chip and measure what the chip's arithmetic does to it.

Chips are analog in-memory computing (crossbar tiles of device pairs, converters, device physics) or digital number
formats. The simulation is torch_tools/hwsim.py, run as a PyTorch job (GPU lock, chat model paused, GPU safety). This
module holds the chip presets, checks requests, starts runs and parameter sweeps, and exports CSV.
docs/HARDWARE_LAB.md describes the model and its sources.
"""
import asyncio
import copy
import csv
import io
import json
import secrets
from pathlib import Path

from . import db, torchlab

PCM_SOURCE = ("IBM AIHWKit PCMLikeNoiseModel (MIT licence): programming noise, drift and read noise fitted to measurements "
              "of IBM phase-change memory devices (Nandakumar et al., ICECS 2019)")
MX_SOURCE = "OCP Microscaling Formats (MX) Specification v1.0 (2023)"
FP8_SOURCE = "Micikevicius et al., FP8 Formats for Deep Learning (2022): E4M3"

HOUR, DAY, YEAR = 3600.0, 86400.0, 365.25 * 86400.0
TIMES = [{"label": "right after programming", "seconds": 0.0}, {"label": "1 minute", "seconds": 60.0},
         {"label": "1 hour", "seconds": HOUR}, {"label": "1 day", "seconds": DAY}, {"label": "1 month", "seconds": 30 * DAY},
         {"label": "1 year", "seconds": YEAR}, {"label": "10 years", "seconds": 10 * YEAR}]

ANALOG = {"kind": "analog", "tile_rows": 512, "weight_scaling": "column", "g_max_us": 25.0, "levels": 0,
          "prog_noise": None, "drift": None, "read_noise": None, "t_seconds": 0.0, "drift_compensation": True,
          "stuck": {"zero": 0.0, "max": 0.0}, "dac_bits": 8, "adc_bits": 8, "adc_range": "auto", "adc_percentile": 99.99,
          "out_noise": 0.0}
DIGITAL = {"kind": "digital", "weights": {"format": "int8", "group": 0}, "inputs": {"format": "int8"}}
PCM = {"prog_noise": {"model": "pcm"}, "drift": {"model": "pcm"}, "read_noise": {"model": "pcm"}}
W_FORMATS = ("fp16", "int8", "int4", "fp8_e4m3", "mxfp4")
X_FORMATS = ("fp16", "int8", "fp8_e4m3", "mxfp4")


def _digital(w: str, x: str, group: int = 0) -> dict:
    return {"kind": "digital", "weights": {"format": w, "group": group}, "inputs": {"format": x}}


PRESETS = [
    {"id": "exact", "group": "check", "label": "Exact (check)", "chip": _digital("fp16", "fp16"),
     "about": "Ordinary 16-bit math through the simulator. It must match the model (KL divergence near 0): run it first, it checks the pipeline."},
    {"id": "ideal-analog", "group": "check", "label": "Ideal analog chip (check)",
     "chip": {**ANALOG, "dac_bits": 0, "adc_bits": 0, "adc_range": None},
     "about": "Crossbar tiles, device pairs and scaling, but perfect devices and converters. It must also match the model."},
    {"id": "int8", "group": "digital", "label": "Digital INT8", "chip": _digital("int8", "int8"),
     "about": "8-bit integer weights (one scale per output) and 8-bit inputs (one scale per token): a common accelerator design."},
    {"id": "w4a8", "group": "digital", "label": "Digital INT4 weights, INT8 inputs", "chip": _digital("int4", "int8", 128),
     "about": "4-bit weights in groups of 128 values with one scale each, 8-bit inputs: half the weight memory of INT8."},
    {"id": "fp8", "group": "digital", "label": "Digital FP8 (E4M3)", "chip": _digital("fp8_e4m3", "fp8_e4m3"), "sources": [FP8_SOURCE],
     "about": "8-bit floating point for weights and inputs: 3 mantissa bits, a wide range."},
    {"id": "mxfp4", "group": "digital", "label": "Digital MXFP4", "chip": _digital("mxfp4", "mxfp4"), "sources": [MX_SOURCE],
     "about": "4-bit floating point with a shared power-of-two scale per 32 values (microscaling), for weights and inputs."},
    {"id": "pcm-1h", "group": "analog", "label": "Analog PCM, 1 hour after programming", "chip": {**ANALOG, **PCM, "t_seconds": HOUR},
     "sources": [PCM_SOURCE],
     "about": "Phase-change memory with IBM's measured programming noise, drift and read noise; 8-bit converters, 512-row tiles, drift compensation."},
    {"id": "pcm-1y", "group": "analog", "label": "Analog PCM, 1 year after programming", "chip": {**ANALOG, **PCM, "t_seconds": YEAR},
     "sources": [PCM_SOURCE], "about": "The same chip a year later: drift has had time to act."},
    {"id": "rram-4bit", "group": "analog", "label": "Analog RRAM-like, 16-level cells", "illustrative": True,
     "chip": {**ANALOG, "levels": 16, "prog_noise": {"model": "constant", "sigma": 0.02}, "stuck": {"zero": 0.005, "max": 0.002}},
     "about": "Resistive memory with 16 conductance levels per device, 2% programming noise and a few stuck devices. "
              "Illustrative values, not fitted to a device: replace them with your device's data."},
]

# Settings a sweep can vary (analog chips): key -> (label, integer?, low, high)
SWEEP_KEYS = {
    "adc_bits": ("ADC bits", True, 0, 16), "dac_bits": ("DAC bits", True, 0, 16), "levels": ("Levels per device", True, 0, 1024),
    "tile_rows": ("Tile rows", True, 0, 8192), "t_seconds": ("Time after programming (s)", False, 0, 10 * YEAR),
    "out_noise": ("Output noise", False, 0, 10), "stuck.zero": ("Devices stuck at 0", False, 0, 0.2),
    "prog_noise.sigma": ("Programming noise σ", False, 0, 0.5), "read_noise.sigma": ("Read noise σ", False, 0, 0.5),
    "drift.nu": ("Drift exponent ν", False, 0, 0.5),
}

_tasks: set = set()


def catalog() -> dict:
    return {"presets": PRESETS, "analog_default": ANALOG, "digital_default": DIGITAL, "weight_formats": W_FORMATS,
            "input_formats": X_FORMATS, "times": TIMES,
            "sweep_keys": [{"key": k, "label": v[0], "integer": v[1], "min": v[2], "max": v[3]} for k, v in SWEEP_KEYS.items()]}


# ---------- Checking a chip description ----------

def _num(v, lo: float, hi: float, what: str, integer: bool = False):
    if isinstance(v, bool):
        raise ValueError(f"{what} must be a number")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number") from None
    if integer:
        if x != int(x):
            raise ValueError(f"{what} must be a whole number")
        x = int(x)
    if not lo <= x <= hi:
        raise ValueError(f"{what} must be between {lo:g} and {hi:g}")
    return x


def _noise(v, what: str, params: dict) -> dict | None:
    """A noise model: off, the measured PCM model, or constant parameters (params: name -> (default, low, high))."""
    if v in (None, False, "", "off"):
        return None
    if v == "pcm":
        v = {"model": "pcm"}
    if not isinstance(v, dict) or v.get("model") not in ("pcm", "constant"):
        raise ValueError(f"{what}: off, 'pcm' (measured PCM devices) or {{'model': 'constant', ...}}")
    if v["model"] == "pcm":
        return {"model": "pcm"}
    unknown = set(v) - {"model", *params}
    if unknown:
        raise ValueError(f"{what}: unknown setting {sorted(unknown)[0]}")
    return {"model": "constant", **{k: _num(v.get(k, d), lo, hi, f"{what} {k}") for k, (d, lo, hi) in params.items()}}


def resolve_chip(chip: dict) -> dict:
    """A complete, checked chip description (missing settings take the defaults of its kind)."""
    if not isinstance(chip, dict):
        raise ValueError("The chip must be an object")
    if chip.get("kind") == "digital":
        unknown = set(chip) - set(DIGITAL)
        if unknown:
            raise ValueError(f"Unknown chip setting: {sorted(unknown)[0]}")
        w, x = {**DIGITAL["weights"], **(chip.get("weights") or {})}, {**DIGITAL["inputs"], **(chip.get("inputs") or {})}
        if w["format"] not in W_FORMATS:
            raise ValueError(f"Weight format must be one of {', '.join(W_FORMATS)}")
        if x["format"] not in X_FORMATS:
            raise ValueError(f"Input format must be one of {', '.join(X_FORMATS)}")
        group = _num(w.get("group") or 0, 0, 256, "Weight group size", True)
        if group not in (0, 32, 64, 128, 256):
            raise ValueError("Weight group size must be 0 (one scale per output), 32, 64, 128 or 256")
        if w["format"] in ("fp16", "mxfp4"):          # no scale, or MX's fixed blocks of 32
            group = 0
        return _digital(w["format"], x["format"], group)
    if chip.get("kind") != "analog":
        raise ValueError("Chip kind must be 'analog' or 'digital'")
    unknown = set(chip) - set(ANALOG)
    if unknown:
        raise ValueError(f"Unknown chip setting: {sorted(unknown)[0]}")
    c = {**copy.deepcopy(ANALOG), **chip}
    stuck = {**ANALOG["stuck"], **(c["stuck"] or {})}
    if set(stuck) - {"zero", "max"}:
        raise ValueError("Stuck devices: give 'zero' and 'max' fractions")
    out = {
        "kind": "analog",
        "tile_rows": _num(c["tile_rows"], 0, 8192, "Tile rows", True),
        "weight_scaling": c["weight_scaling"],
        "g_max_us": _num(c["g_max_us"], 0.1, 1000, "Maximum conductance (µS)"),
        "levels": _num(c["levels"], 0, 1024, "Levels per device", True),
        "prog_noise": _noise(c["prog_noise"], "Programming noise", {"sigma": (0.02, 0, 0.5)}),
        "drift": _noise(c["drift"], "Drift", {"nu": (0.05, 0, 0.5), "nu_sd": (0.0, 0, 0.5)}),
        "read_noise": _noise(c["read_noise"], "Read noise", {"sigma": (0.01, 0, 0.5)}),
        "t_seconds": _num(c["t_seconds"], 0, 10 * YEAR, "Time after programming (s)"),
        "drift_compensation": bool(c["drift_compensation"]),
        "stuck": {"zero": _num(stuck["zero"], 0, 0.2, "Devices stuck at 0"), "max": _num(stuck["max"], 0, 0.2, "Devices stuck at the maximum")},
        "dac_bits": _num(c["dac_bits"] or 0, 0, 16, "DAC bits", True),
        "adc_bits": _num(c["adc_bits"] or 0, 0, 16, "ADC bits", True),
        "adc_range": c["adc_range"] if c["adc_range"] in (None, "auto") else _num(c["adc_range"], 0.01, 1e5, "ADC range"),
        "adc_percentile": _num(c["adc_percentile"], 90, 100, "ADC range percentile"),
        "out_noise": _num(c["out_noise"] or 0, 0, 10, "Output noise"),
    }
    if out["weight_scaling"] not in ("column", "tile"):
        raise ValueError("Weight scaling must be 'column' (one scale per output and tile) or 'tile'")
    if out["tile_rows"] and out["tile_rows"] < 16:
        raise ValueError("Tile rows must be 0 (one tile per matrix) or at least 16")
    if out["levels"] == 1:
        raise ValueError("Levels per device must be 0 (continuous) or at least 2")
    for k in ("dac_bits", "adc_bits"):
        if out[k] == 1:
            raise ValueError(f"{k.split('_')[0].upper()} bits must be 0 (ideal) or at least 2")
    if out["adc_bits"] and out["adc_range"] is None:
        raise ValueError("An ADC with bits needs a range: 'auto' (calibrated) or a number")
    if out["stuck"]["zero"] + out["stuck"]["max"] > 0.3:
        raise ValueError("At most 30% of the devices can be stuck")
    return out


def _merge(base: dict, over: dict) -> dict:
    """Overrides on top of a chip: nested groups (stuck, weights, inputs) merge, everything else replaces."""
    if over.get("kind") and over["kind"] != base.get("kind"):
        base = copy.deepcopy(ANALOG if over["kind"] == "analog" else DIGITAL)
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = {**(out.get(k) or {}), **v} if k in ("stuck", "weights", "inputs") and isinstance(v, dict) else v
    return out


def plan(body: dict) -> tuple[Path, dict]:
    d = torchlab._ckpt(body.get("checkpoint", ""))
    preset = str(body.get("preset") or "")
    base = next((p for p in PRESETS if p["id"] == preset), None)
    if preset and not base:
        raise ValueError(f"Unknown preset {preset}")
    over = body.get("chip") or {}
    if not isinstance(over, dict):
        raise ValueError("The chip must be an object")
    if not base and not over:
        raise ValueError("Pick a preset or describe a chip")
    chip = resolve_chip(_merge(copy.deepcopy(base["chip"]) if base else {}, over))
    return d, {"checkpoint": d.name, "preset": base["id"] if base else None,
               "preset_label": (base["label"] if base else "Custom chip") + (" (changed)" if base and over else ""),
               "chip": chip, "trials": _num(body.get("trials", 5), 1, 20, "Trials", True),
               "chunks": _num(body.get("chunks", 16), 2, 64, "Text chunks", True),
               "seed": _num(body.get("seed", 0), 0, 2 ** 31 - 1, "Seed", True), "label": str(body.get("label") or "")[:60]}


# ---------- Runs and sweeps ----------

def _launch(d: Path, p: dict) -> int:
    jid = torchlab._new("hwsim", p)
    task = asyncio.get_running_loop().create_task(torchlab._gpu(jid, _run, d, p))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return jid


def start(body: dict) -> int:
    return _launch(*plan(body))


def _vary(over: dict, key: str, v) -> dict:
    over = copy.deepcopy(over)
    if key == "stuck.zero":
        over["stuck"] = {**(over.get("stuck") or {}), "zero": v}
    elif key in ("prog_noise.sigma", "read_noise.sigma"):
        over[key.split(".")[0]] = {"model": "constant", "sigma": v}
    elif key == "drift.nu":
        over["drift"] = {"model": "constant", "nu": v, "nu_sd": 0.0}
    else:
        over[key] = v
    return over


def start_sweep(body: dict) -> dict:
    """The same model and chip with one setting varied: one run per value, queued one after another."""
    vary = body.get("vary") or {}
    key, values = vary.get("key"), vary.get("values")
    if key not in SWEEP_KEYS:
        raise ValueError(f"A sweep varies one of: {', '.join(SWEEP_KEYS)}")
    label, integer, lo, hi = SWEEP_KEYS[key]
    if not isinstance(values, list) or not 2 <= len(values) <= 12:
        raise ValueError("Give 2 to 12 values")
    values = [_num(v, lo, hi, label, integer) for v in values]
    if len(set(values)) != len(values):
        raise ValueError("The values must differ")
    base = next((p for p in PRESETS if p["id"] == body.get("preset")), None)
    if ((body.get("chip") or {}).get("kind") or (base["chip"]["kind"] if base else "analog")) != "analog":
        raise ValueError("Sweeps vary an analog chip's settings; compare number formats with the digital presets")
    plans = [plan({**body, "chip": _vary(body.get("chip") or {}, key, v)}) for v in values]   # check all before starting any
    sid = secrets.token_hex(4)
    ids = []
    for (d, p), v in zip(plans, values):
        p["sweep"] = {"id": sid, "key": key, "label": label, "value": v, "n": len(values)}
        ids.append(_launch(d, p))
    return {"sweep": sid, "ids": ids}


async def _run(jid: int, log: list[str], d: Path, p: dict) -> dict:
    from . import quantlab, weights
    text = await quantlab.ensure_text()
    calib = await weights.ensure_calibration() if p["chip"].get("adc_range") == "auto" else None
    chip_file, out = torchlab.WORK / f"hwsim-{jid}-chip.json", torchlab.WORK / f"hwsim-{jid}.json"
    chip_file.write_text(json.dumps(p["chip"]))
    args = [str(torchlab.TOOLS / "hwsim.py"), "--model", str(d), "--text", str(text), "--out", str(out), "--chip", str(chip_file),
            "--trials", str(p["trials"]), "--chunks", str(p["chunks"]), "--seed", str(p["seed"])]
    if calib:
        args += ["--calib", str(calib)]

    async def follow():
        while True:
            await asyncio.sleep(3)
            last = next((x for x in reversed(log) if x.startswith(("loading", "layer ", "scored"))), None)
            if last:
                torchlab._save(jid, step=last)

    watcher = asyncio.create_task(follow())
    try:
        torchlab._save(jid, step="loading the model")
        if await torchlab._py(jid, args, log, timeout=6 * 3600):
            raise RuntimeError(" | ".join(log[-3:])[-400:])
    finally:
        watcher.cancel()
        chip_file.unlink(missing_ok=True)
    res = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    return res


def runs(limit: int = 100) -> list[dict]:
    with db.connect() as c:
        rows = c.execute("SELECT * FROM torch_jobs WHERE kind = 'hwsim' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [torchlab._row(r) for r in rows]


def sweep_runs(sid: str) -> list[dict]:
    with db.connect() as c:
        rows = c.execute("SELECT * FROM torch_jobs WHERE kind = 'hwsim' AND json_extract(params, '$.sweep.id') = ? ORDER BY id",
                         (sid,)).fetchall()
    return [torchlab._row(r) for r in rows]


# ---------- CSV ----------

def _stat(r: dict, k: str, f: str):
    s = (r.get("summary") or {}).get(k) or {}
    v = s.get(f)
    return "" if v is None else round(v, 6)


def run_csv(job: dict) -> str:
    """One row per chip (trial), with the chip description, so a run can be reproduced from the file."""
    p, r = job["params"], job["results"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["run", "model", "chip", "trial", "tokens", "kld", "kld_se", "kld_p99", "same_top_pct", "ppl", "ppl_exact",
                "output_error", "weight_error", "seed", "chip_json"])
    for t in r.get("per_trial") or []:
        w.writerow([job["id"], p["checkpoint"], p.get("preset_label"), t["trial"], t["tokens"], t["kld"], t["kld_sd"], t["kld_p99"],
                    t["same_top"], t["ppl"], t["ppl_base"], r.get("output_error"), r.get("weight_error"), p["seed"],
                    json.dumps(p["chip"], separators=(",", ":"))])
    return buf.getvalue()


def sweep_csv(jobs: list[dict]) -> str:
    """One row per value of the swept setting: mean, standard deviation and 95% interval over the chips."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["sweep", "setting", "value", "run", "status", "model", "chips", "kld_mean", "kld_sd", "kld_ci95", "same_top_mean",
                "same_top_sd", "ppl_mean", "ppl_sd", "ppl_exact", "output_error", "weight_error"])
    for j in jobs:
        p, r, s = j["params"], j["results"] or {}, j["params"]["sweep"]
        w.writerow([s["id"], s["key"], s["value"], j["id"], j["status"], p["checkpoint"], r.get("trials", ""),
                    _stat(r, "kld", "mean"), _stat(r, "kld", "sd"), _stat(r, "kld", "ci95"), _stat(r, "same_top", "mean"),
                    _stat(r, "same_top", "sd"), _stat(r, "ppl", "mean"), _stat(r, "ppl", "sd"), r.get("ppl_base", ""),
                    r.get("output_error", ""), r.get("weight_error", "")])
    return buf.getvalue()
