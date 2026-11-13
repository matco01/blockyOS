"""Hardware analysis: measured speed against physical limits, from stored data only."""
import pytest

from app import lab, profile

LIM = {"gpu_bandwidth_gbps": 500, "ram_bandwidth_gbps": 50, "gpu_tflops": 40.0, "gpu_tflops_kind": "FP16 tensor", "sm_clock_max": 2000, "vram_mb": 8192}


@pytest.fixture
def model(make_gguf, monkeypatch):
    make_gguf("M.gguf")
    # 5 GB of weights read per token, 8B parameters, 0.5 GB embedding table.
    monkeypatch.setattr(profile, "_weights", lambda path: {"bytes": 5e9, "params": 8e9, "embedding_bytes": 0.5e9})
    return {"model": "M.gguf", "ctx": 8192}


def test_decode_roof_and_utilization(model):
    res = {"speed": {"tg": 50.0, "pp": 1600.0}, "memory": {"gpu": {"model": 5000, "context": 200, "compute": 100}, "ram": {"model": 477}},
           "telemetry": {"speed": {"sm_clock_avg": 1900}}}
    hw = profile.analyze(res, model, LIM)
    d = hw["decode"]
    assert d["roof_tps"] == pytest.approx(100.0, rel=0.01)      # 500 GB/s ÷ 5 GB
    assert d["roof_util"] == pytest.approx(0.5, rel=0.01)
    assert d["effective_gbps"] == pytest.approx(250.0, rel=0.01)
    assert hw["limiter"]["what"] == "Software overhead"           # only 50% of the memory roof
    assert hw["clocks"]["ratio"] == pytest.approx(0.95)
    p = hw["prefill"]
    assert p["achieved_tflops"] == pytest.approx(2 * 8e9 * 1600 / 1e12, rel=0.01)
    assert p["roof_tflops"] == 40.0                               # compute-bound at 512-token batches


def test_bandwidth_bound_when_close_to_the_roof(model):
    hw = profile.analyze({"speed": {"tg": 80.0}, "memory": {"gpu": {"model": 5000}, "ram": {"model": 477}}}, model, LIM)
    assert hw["limiter"]["what"] == "VRAM bandwidth"


def test_weights_in_system_ram_dominate(model):
    # 2 GB of weights (beyond the 0.5 GB embedding) live in system RAM.
    res = {"speed": {"tg": 12.0}, "memory": {"gpu": {"model": 3000}, "ram": {"model": (2.5e9) / 2**20}}}
    hw = profile.analyze(res, model, LIM)
    assert hw["weights_on_cpu_gb"] == pytest.approx(2.0, rel=0.01)
    split = hw["decode"]["time_split_ms"]
    assert split["ram_reads"] == pytest.approx(2.0 / 50 * 1000, rel=0.01)   # 40 ms at 50 GB/s
    assert hw["limiter"]["what"] == "System-RAM bandwidth"


def test_each_run_is_judged_against_its_own_hardware(model, fresh_db):
    """A run from a slower GPU keeps its own roof even when today's machine is faster (stored snapshot)."""
    slow = dict(LIM, gpu_bandwidth_gbps=250)
    res = {"speed": {"tg": 50.0}, "memory": {"gpu": {"model": 5000}, "ram": {"model": 477}}, "hardware": slow}
    exp = {"limits": LIM, "runs_list": [{"id": 1, "status": "done", "config": model, "results": res, "parent_id": None}]}
    profile.enrich(exp)
    assert exp["runs_list"][0]["results"]["hw"]["decode"]["roof_tps"] == pytest.approx(50.0, rel=0.01)


def test_missing_model_file_gives_empty_analysis():
    assert profile.analyze({"speed": {"tg": 10}}, {"model": "does-not-exist.gguf", "ctx": 4096}, LIM) == {}


def test_lab_paths_point_to_the_test_folder():
    assert "blocky-tests-" in str(lab.MODELS_DIR)
