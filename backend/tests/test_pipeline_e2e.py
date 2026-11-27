"""End to end: whole experiments through profile.run() with a fake llama.cpp (tests/fake_llama.py).

Every phase runs for real (server start and load-log parsing, MMLU-Pro scoring, long-context retrieval, serving speed,
energy, llama-bench parsing, perplexity / KL divergence, provenance, hardware analysis, comparison statistics and the
instrument checks); only the inference engine is simulated. The fake model's quality depends on the quantization in its
file name, so the pipeline's results can be checked against known truths.
"""
import asyncio
import json
import socket

import pytest

from fake_llama import accuracy, bits_of, install_fake_llama, kld

LIMITS = {"gpu": "Fake GPU", "gpu_bandwidth_gbps": 272.0, "gpu_tflops": 15.0, "gpu_tflops_kind": "fp16", "sm_clock_max": 2500,
          "vram_mb": 8188, "power_limit_w": 115, "gpu_count": 1, "ram_bandwidth_gbps": 40.0, "ram_mb": 32000, "cpu": "Fake CPU",
          "cpu_threads": 16, "power_source": None, "cpu_power_source": None}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def lab_env(fresh_db, make_gguf, tmp_path, monkeypatch):
    from app import benchmarks, lab, profile, quantlab

    monkeypatch.setattr(lab, "LLAMA_BIN", install_fake_llama(tmp_path / "bin"))

    async def noop(*a, **k):
        return 0, ""
    monkeypatch.setattr(lab, "llm_service", noop)
    monkeypatch.setattr(profile, "_restore_chat", noop)
    monkeypatch.setattr(profile, "ENERGY_IDLE_S", 0.2)
    monkeypatch.setattr(profile, "PORT", _free_port())
    monkeypatch.setattr(profile, "limits", lambda: dict(LIMITS))
    monkeypatch.setattr(profile, "hardware_snapshot", lambda: dict(LIMITS))  # each run is analyzed against its own snapshot
    # The fake GGUF files carry metadata but no tensors: report the size a 1B-parameter model of that type would have.
    monkeypatch.setattr(profile, "_weights", lambda path: {"bytes": 1e9 * bits_of(str(path)) / 8, "params": 1e9, "embedding_bytes": 0})

    # MMLU-Pro stand-in: 14 subjects, the correct letter is written into each question for the fake model to use.
    qs = []
    for c in range(14):
        for i in range(60):
            ans = "ABCDEFGHIJ"[(c * 7 + i * 3) % 10]
            qs.append({"q": f"Question {i} about subject {c} (key {ans})", "options": [f"option {x}" for x in "ABCDEFGHIJ"],
                       "answer": ans, "tag": f"subject{c:02d}"})
    benchmarks.path("mmlu_pro").parent.mkdir(parents=True, exist_ok=True)
    benchmarks.path("mmlu_pro").write_text(json.dumps(qs))
    quantlab.TEXT.parent.mkdir(parents=True, exist_ok=True)
    quantlab.TEXT.write_text(" ".join(f"Sentence number {i} of a long article about rivers, bridges and boats." for i in range(4000)))

    for name in ("Fake-F16.gguf", "Fake-Q4_K_M.gguf", "Fake-Q2_K.gguf"):
        make_gguf(name)
    return profile


def _profile(profile, eid, config, parent=None):
    rid = profile.create_run(eid, {"config": config, "parent_id": parent})
    asyncio.run(profile.run(rid, None))
    run = next(r for r in profile.get_experiment(eid)["runs_list"] if r["id"] == rid)
    assert run["status"] == "done", run.get("error")
    return run


def test_full_experiment_end_to_end(lab_env):
    profile = lab_env
    from app import instrument
    eid = profile.create_experiment("e2e", "Fake-F16.gguf")
    base = _profile(profile, eid, {})
    cfg = base["config"]
    r = base["results"]

    # Provenance and commands: what's needed to reproduce the run.
    assert r["provenance"]["protocol"] == profile.PROTOCOL
    assert r["provenance"]["llama_cpp"]["commit"] == "c0ffee12"
    assert r["provenance"]["model"]["sha256"] and r["provenance"]["extra_tests"]["ppl_chunks"] == 32
    assert {"load_and_quality", "speed", "fidelity"} <= set(r["commands"])
    # Load log parsed into the memory layout.
    assert r["memory"]["gpu"]["model"] == 3000 and r["memory"]["layers"] == {"gpu": 33, "total": 33}
    # One layout per run: the server fits with the margin, the other tools reuse its layer count.
    pl = r["provenance"]["placement"]
    assert pl["fit_margin_mb"] >= 1024 and pl["tool_gpu_layers"] == 33
    assert f"--fit-target {pl['fit_margin_mb']}" in r["commands"]["load_and_quality"]
    assert "-ngl 33" in r["commands"]["speed"] and "-fitt" not in r["commands"]["speed"]
    assert "-ngl 33" in r["commands"]["fidelity"]
    # MMLU-Pro: 560 paired answers, accuracy near the fake model's true rate, with a Wilson interval around it.
    m = r["mmlu"]
    assert m["n"] == 560 and len(m["answers"]) == 560
    assert abs(m["acc"] - accuracy("Fake-F16.gguf")) < 0.06 and m["ci95"][0] < m["acc"] < m["ci95"][1]
    # The fake model always finds the needle: retrieval must score 100% at every length that fits.
    assert r["needle"]["acc"] == 1.0 and r["needle"]["lengths"] == [2048, 4096]
    # Serving speed and energy (no power source on this machine: said so, not guessed).
    assert r["serving"]["reps"] == 5 and r["serving"]["tg"] > 0
    assert r["energy"].get("primary") is None and "No power reading" in r["energy"]["note"]
    # llama-bench parsed: median of 5 and robust spread; decode speed is 600 / bits per weight in the fake.
    assert r["speed"]["tg"] == pytest.approx(600 / 16, rel=0.01) and r["speed"]["tg_sd"] > 0 and len(r["speed"]["tg_samples"]) == 5
    # Fidelity: the baseline is the KL reference.
    assert r["kld"] == 0.0 and r["kld_reference"] is True and r["ppl"] == pytest.approx(6.0)
    # Hardware analysis ran against the stored limits.
    assert r["hw"]["decode"]["ms_per_token"] == pytest.approx(1000 / r["speed"]["tg"], rel=0.01)
    assert profile.evidence_outputs(base["id"])["outputs"].keys() >= {"speed", "fidelity"}

    # Weight variants branched from the baseline: fewer bits, more KL divergence, lower accuracy, faster decode.
    q4 = _profile(profile, eid, {**cfg, "model": "Fake-Q4_K_M.gguf"}, base["id"])
    q2 = _profile(profile, eid, {**cfg, "model": "Fake-Q2_K.gguf"}, base["id"])
    rep = _profile(profile, eid, cfg, base["id"])
    exp = profile.get_experiment(eid)
    by = {x["id"]: x for x in exp["runs_list"]}
    q4, q2, rep = by[q4["id"]], by[q2["id"]], by[rep["id"]]
    assert q2["results"]["kld"] == pytest.approx(kld("Fake-Q2_K.gguf"), rel=1e-3)
    assert q2["results"]["kld"] > q4["results"]["kld"] > 0
    assert q2["results"]["speed"]["tg"] > q4["results"]["speed"]["tg"] > base["results"]["speed"]["tg"]
    assert q2["results"]["mmlu"]["acc"] < base["results"]["mmlu"]["acc"]
    # Paired statistics against the baseline.
    s2 = q2["stats"]
    assert s2["baseline_id"] == base["id"] and s2["env_mismatch"] == []
    assert 0 <= s2["mmlu_p"] <= 1 and s2["mmlu_diff"]["diff"] < 0 and s2["mmlu_diff"]["ci95"][0] <= s2["mmlu_diff"]["diff"]
    # A rerun of the baseline gives identical answers: no change, and the paired test agrees.
    assert rep["results"]["mmlu"]["answers"] == base["results"]["mmlu"]["answers"] and rep["stats"]["mmlu_p"] == 1.0
    assert profile.describe_change(cfg, rep["config"]) == "Repeat of baseline"

    # The instrument checks pass on a correct pipeline.
    report = {c["id"]: c for c in instrument.check(exp, [])["checks"]}
    for check in ("self_kld", "determinism", "speed_repeat", "quant_order", "roofline"):
        assert report[check]["status"] == "pass", (check, report[check])


def test_kld_needs_the_baselines_logits(lab_env):
    profile = lab_env
    eid = profile.create_experiment("e2e-missing-reference", "Fake-F16.gguf")
    base = _profile(profile, eid, {})
    (profile.DIR / str(eid) / "base.kld").unlink()
    rid = profile.create_run(eid, {"config": {**base["config"], "model": "Fake-Q4_K_M.gguf"}, "parent_id": base["id"]})
    asyncio.run(profile.run(rid, None))
    run = next(r for r in profile.get_experiment(eid)["runs_list"] if r["id"] == rid)
    assert run["status"] == "error" and "Baseline logits are missing" in run["error"]


def test_a_server_that_dies_is_reported(lab_env, monkeypatch):
    profile = lab_env
    from app import lab
    (lab.LLAMA_BIN / "llama-server").write_text("#!/bin/sh\necho 'cudaMalloc failed: out of memory'\nexit 1\n")
    eid = profile.create_experiment("e2e-oom", "Fake-F16.gguf")
    rid = profile.create_run(eid, {"config": {}})
    asyncio.run(profile.run(rid, None))
    run = next(r for r in profile.get_experiment(eid)["runs_list"] if r["id"] == rid)
    assert run["status"] == "error" and "didn't load" in run["error"] and "out of memory" in run["error"]


def test_reproducibility_bundle(lab_env):
    import importlib.util
    import io
    import zipfile
    profile = lab_env
    from app import bundle
    eid = profile.create_experiment("bundle me", "Fake-F16.gguf")
    base = _profile(profile, eid, {})
    q4 = _profile(profile, eid, {**base["config"], "model": "Fake-Q4_K_M.gguf"}, base["id"])
    name, data = bundle.build(eid)
    assert name == "bundle-me-bundle.zip"
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    assert {"README.md", "experiment.json", "instrument-check.json", "environment.json", "reproduce.py"} <= names
    for rid in (base["id"], q4["id"]):
        assert {f"runs/{rid}/provenance.json", f"runs/{rid}/commands.txt", f"runs/{rid}/outputs/fidelity.txt"} <= names
    env = json.loads(z.read("environment.json"))
    assert env["protocol"] == profile.PROTOCOL and env["blocky_version"]
    assert json.loads(z.read("experiment.json"))["id"] == eid

    # The bundled script plans the same runs with their parents, and a perfect reproduction compares clean.
    spec = importlib.util.spec_from_loader("reproduce", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = "reproduce.py"
    exec(z.read("reproduce.py").decode(), mod.__dict__)
    exp = json.loads(z.read("experiment.json"))
    steps = mod.plan(exp)
    assert [s["original"] for s in steps] == [base["id"], q4["id"]] and steps[1]["parent"] == base["id"]
    conditions, metrics = mod.compare(base["results"], base["results"])
    assert conditions == [] and all(m["within_interval"] in (True, None) for m in metrics)
    other = json.loads(json.dumps(base["results"]))
    other["provenance"]["model"]["sha256"] = "different"
    other["mmlu"]["acc"] = 0.1
    conditions, metrics = mod.compare(base["results"], other)
    assert conditions[0]["condition"] == "model SHA-256"
    assert next(m for m in metrics if m["metric"] == "MMLU-Pro accuracy")["within_interval"] is True  # the original lies in the old CI
