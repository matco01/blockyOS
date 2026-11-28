"""Hardware lab (app/hwlab.py): chip presets and checks, runs and sweeps, the command handed to the simulator, CSV."""
import asyncio
import csv
import io
import json

import pytest

from app import hwlab, torchlab


@pytest.fixture
def ckpt(monkeypatch, tmp_path):
    hf = tmp_path / "hf"
    d = hf / "Org--Tiny"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 2}))
    monkeypatch.setattr(torchlab, "HF", hf)
    return d


@pytest.fixture
def launched(monkeypatch):
    """Record runs instead of starting the simulator."""
    started = []
    monkeypatch.setattr(hwlab, "_launch", lambda d, p: started.append((d, p)) or len(started))
    return started


def test_every_preset_is_a_valid_complete_chip():
    for p in hwlab.PRESETS:
        chip = hwlab.resolve_chip(p["chip"])
        assert hwlab.resolve_chip(chip) == chip, p["id"]           # complete: checking it again changes nothing
        assert p["about"] and p["group"] in ("check", "digital", "analog")
    pcm = hwlab.resolve_chip(next(p for p in hwlab.PRESETS if p["id"] == "pcm-1h")["chip"])
    assert pcm["prog_noise"] == pcm["drift"] == pcm["read_noise"] == {"model": "pcm"} and pcm["t_seconds"] == 3600
    assert all(p.get("sources") for p in hwlab.PRESETS if p["id"].startswith(("pcm", "fp8", "mxfp4")))
    assert next(p for p in hwlab.PRESETS if p["id"] == "rram-4bit")["illustrative"]


@pytest.mark.parametrize("chip,msg", [
    ({"kind": "quantum"}, "kind"),
    ({"kind": "analog", "adc_bitz": 8}, "Unknown chip setting"),
    ({"kind": "analog", "tile_rows": 8}, "Tile rows"),
    ({"kind": "analog", "levels": 1}, "Levels"),
    ({"kind": "analog", "adc_bits": 1}, "ADC bits"),
    ({"kind": "analog", "adc_bits": 8, "adc_range": None}, "needs a range"),
    ({"kind": "analog", "adc_range": -1}, "ADC range"),
    ({"kind": "analog", "prog_noise": {"model": "magic"}}, "Programming noise"),
    ({"kind": "analog", "prog_noise": {"model": "constant", "sigma": 2}}, "sigma"),
    ({"kind": "analog", "drift": {"model": "constant", "nu": 0.05, "speed": 1}}, "unknown setting"),
    ({"kind": "analog", "t_seconds": -1}, "Time"),
    ({"kind": "analog", "stuck": {"zero": 0.2, "max": 0.2}}, "30%"),
    ({"kind": "analog", "dac_bits": True}, "number"),
    ({"kind": "analog", "weight_scaling": "row"}, "scaling"),
    ({"kind": "digital", "weights": {"format": "int3"}}, "Weight format"),
    ({"kind": "digital", "inputs": {"format": "int4"}}, "Input format"),
    ({"kind": "digital", "weights": {"format": "int4", "group": 48}}, "group size"),
])
def test_rejects_impossible_chips(chip, msg):
    with pytest.raises(ValueError, match=msg):
        hwlab.resolve_chip(chip)


def test_constant_noise_models_get_their_defaults():
    chip = hwlab.resolve_chip({"kind": "analog", "prog_noise": {"model": "constant"}, "drift": "pcm", "read_noise": "off"})
    assert chip["prog_noise"] == {"model": "constant", "sigma": 0.02} and chip["drift"] == {"model": "pcm"} and chip["read_noise"] is None
    assert hwlab.resolve_chip({"kind": "digital", "weights": {"format": "mxfp4", "group": 128}})["weights"]["group"] == 0  # MX: fixed blocks


def test_a_run_starts_from_a_preset_with_changes(ckpt, launched):
    hwlab.start({"checkpoint": "Org--Tiny", "preset": "pcm-1h", "chip": {"t_seconds": 86400, "stuck": {"zero": 0.01}}, "trials": 3})
    d, p = launched[0]
    assert d == ckpt and p["checkpoint"] == "Org--Tiny" and p["trials"] == 3 and p["preset_label"].endswith("(changed)")
    assert p["chip"]["t_seconds"] == 86400 and p["chip"]["stuck"] == {"zero": 0.01, "max": 0.0} and p["chip"]["drift"] == {"model": "pcm"}
    hwlab.start({"checkpoint": "Org--Tiny", "preset": "pcm-1h", "chip": {"kind": "digital"}})   # another kind starts from its defaults
    assert launched[1][1]["chip"] == hwlab.resolve_chip(hwlab.DIGITAL)


@pytest.mark.parametrize("body,msg", [
    ({"checkpoint": "Nope", "preset": "int8"}, "Checkpoint not found"),
    ({"checkpoint": "Org--Tiny"}, "Pick a preset"),
    ({"checkpoint": "Org--Tiny", "preset": "gpu-9000"}, "Unknown preset"),
    ({"checkpoint": "Org--Tiny", "preset": "int8", "trials": 50}, "Trials"),
    ({"checkpoint": "Org--Tiny", "preset": "int8", "chunks": 1}, "Text chunks"),
])
def test_rejects_bad_runs(ckpt, launched, body, msg):
    with pytest.raises(ValueError, match=msg):
        hwlab.start(body)
    assert not launched


def test_sweeps(ckpt, launched):
    r = hwlab.start_sweep({"checkpoint": "Org--Tiny", "preset": "pcm-1h", "vary": {"key": "adc_bits", "values": [4, 6, 8]}})
    assert r["ids"] == [1, 2, 3] and [p["chip"]["adc_bits"] for _, p in launched] == [4, 6, 8]
    assert {p["sweep"]["id"] for _, p in launched} == {r["sweep"]} and launched[0][1]["sweep"]["n"] == 3
    hwlab.start_sweep({"checkpoint": "Org--Tiny", "preset": "rram-4bit", "vary": {"key": "prog_noise.sigma", "values": [0.01, 0.05]}})
    assert launched[-1][1]["chip"]["prog_noise"] == {"model": "constant", "sigma": 0.05}


@pytest.mark.parametrize("vary,msg", [
    ({"key": "colour", "values": [1, 2]}, "varies one of"),
    ({"key": "adc_bits", "values": [8]}, "2 to 12"),
    ({"key": "adc_bits", "values": [8, 8]}, "differ"),
    ({"key": "adc_bits", "values": [4, 99]}, "ADC bits"),
    ({"key": "adc_bits", "values": [1, 4]}, "ADC bits must be 0"),     # checked as a chip before anything starts
])
def test_rejects_bad_sweeps(ckpt, launched, vary, msg):
    with pytest.raises(ValueError, match=msg):
        hwlab.start_sweep({"checkpoint": "Org--Tiny", "preset": "pcm-1h", "vary": vary})
    assert not launched


def test_sweeps_are_for_analog_chips(ckpt, launched):
    with pytest.raises(ValueError, match="analog"):
        hwlab.start_sweep({"checkpoint": "Org--Tiny", "preset": "int8", "vary": {"key": "t_seconds", "values": [0, 60]}})


def test_the_simulator_gets_the_chip_and_settings(ckpt, monkeypatch, tmp_path):
    from app import quantlab, weights
    seen = {}

    async def fake_py(jid, args, log, timeout=0, env=None):
        seen["args"] = args
        seen["chip"] = json.loads(open(args[args.index("--chip") + 1]).read())
        open(args[args.index("--out") + 1], "w").write(json.dumps({"summary": {}}))
        return 0

    async def text():
        return tmp_path / "test.txt"

    async def calib():
        return tmp_path / "train.txt"
    monkeypatch.setattr(torchlab, "_py", fake_py)
    monkeypatch.setattr(torchlab, "_save", lambda *a, **k: None)
    monkeypatch.setattr(torchlab, "WORK", tmp_path)
    monkeypatch.setattr(quantlab, "ensure_text", text)
    monkeypatch.setattr(weights, "ensure_calibration", calib)
    d, p = hwlab.plan({"checkpoint": "Org--Tiny", "preset": "pcm-1h", "trials": 4, "seed": 7})
    assert asyncio.run(hwlab._run(1, [], d, p)) == {"summary": {}}
    a = seen["args"]
    assert a[0].endswith("hwsim.py") and a[a.index("--model") + 1] == str(ckpt) and a[a.index("--trials") + 1] == "4"
    assert a[a.index("--seed") + 1] == "7" and a[a.index("--calib") + 1] == str(tmp_path / "train.txt") and seen["chip"] == p["chip"]
    assert not list(tmp_path.glob("hwsim-1-chip.json"))                                    # the chip file is cleaned up
    d, p = hwlab.plan({"checkpoint": "Org--Tiny", "preset": "int8"})
    asyncio.run(hwlab._run(2, [], d, p))
    assert "--calib" not in seen["args"]                                                   # nothing to calibrate


RESULT = {"trials": 2, "ppl_base": 10.0, "output_error": 0.05, "weight_error": 0.04,
          "summary": {"kld": {"mean": 0.02, "sd": 0.005, "ci95": 0.045}, "same_top": {"mean": 90.0, "sd": 1.0}, "ppl": {"mean": 10.4, "sd": 0.1}},
          "per_trial": [{"trial": t, "tokens": 511, "kld": k, "kld_sd": 0.001, "kld_p99": 0.2, "same_top": 90.0, "ppl": 10.4, "ppl_base": 10.0}
                        for t, k in ((1, 0.016), (2, 0.024))]}


def test_csv_exports():
    job = {"id": 5, "status": "done", "params": {"checkpoint": "Org--Tiny", "preset_label": "Analog PCM", "seed": 0,
                                                 "chip": {"kind": "analog"}, "sweep": {"id": "ab12", "key": "adc_bits", "value": 6}},
           "results": RESULT}
    rows = list(csv.DictReader(io.StringIO(hwlab.run_csv(job))))
    assert [r["kld"] for r in rows] == ["0.016", "0.024"] and json.loads(rows[0]["chip_json"]) == {"kind": "analog"}
    rows = list(csv.DictReader(io.StringIO(hwlab.sweep_csv([job, {**job, "id": 6, "status": "running", "results": {}}]))))
    assert rows[0]["value"] == "6" and rows[0]["kld_mean"] == "0.02" and rows[0]["kld_ci95"] == "0.045" and rows[1]["kld_mean"] == ""


def test_api(http_client):
    cat = http_client.get("/api/hwlab/presets").json()
    assert {p["id"] for p in cat["presets"]} >= {"exact", "pcm-1h", "mxfp4"} and any(k["key"] == "adc_bits" for k in cat["sweep_keys"])
    r = http_client.post("/api/hwlab/runs", json={"checkpoint": "Nope", "preset": "int8"})
    assert r.status_code == 400 and "Checkpoint not found" in r.json()["detail"]
    assert http_client.get("/api/hwlab/runs/999999/csv").status_code == 404
    assert http_client.get("/api/hwlab/sweeps/nope/csv").status_code == 404
    assert isinstance(http_client.get("/api/hwlab/runs").json(), list)
