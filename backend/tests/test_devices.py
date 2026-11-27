"""Cheap-device simulation: CPU-only runs, core pinning, RAM limits, presets, and the bandwidth projection."""
import os

import pytest

from app import engines, profile

LIM = {"gpu_bandwidth_gbps": 500, "ram_bandwidth_gbps": 40, "gpu_tflops": 40.0, "vram_mb": 8192}


@pytest.fixture(autouse=True)
def sixteen_threads(monkeypatch):
    """Presets pin up to 8 cores; tests must not depend on how many the test machine has (CI runners have 2)."""
    monkeypatch.setattr(profile.os, "cpu_count", lambda: 16)


@pytest.fixture
def exp(make_gguf):
    make_gguf("M-Q4_K_M.gguf")
    return {"model": "M-Q4_K_M.gguf"}


def test_more_cores_than_the_device_has(exp, monkeypatch):
    monkeypatch.setattr(profile.os, "cpu_count", lambda: 2)
    with pytest.raises(ValueError, match="2 CPU threads"):
        profile.normalize_config(exp, {"preset": "minipc-16gb"})


def test_preset_fills_device_settings_and_wins_over_inherited(exp):
    parent = profile.normalize_config(exp, {})
    assert parent["device"] == "gpu" and parent["preset"] is None
    cfg = profile.normalize_config(exp, {**parent, "preset": "sbc-8gb"})
    assert (cfg["device"], cfg["cores"], cfg["threads"], cfg["ram_limit_mb"]) == ("cpu", 4, 4, 8192)
    assert profile.describe_change(parent, cfg) == "Single-board computer, 8 GB"     # one change, not four
    custom = profile.normalize_config(exp, {**cfg, "preset": "", "cores": 2, "threads": 2})
    assert custom["preset"] is None and custom["cores"] == 2 and custom["device"] == "cpu"
    assert profile.describe_change({**cfg, "preset": None}, custom) == "2 threads · 2 cores"


def test_device_validation(exp):
    with pytest.raises(ValueError):
        profile.normalize_config(exp, {"preset": "supercomputer"})
    with pytest.raises(ValueError):
        profile.normalize_config(exp, {"device": "tpu"})
    with pytest.raises(ValueError):
        profile.normalize_config(exp, {"cores": (os.cpu_count() or 1) + 1})
    assert profile.normalize_config(exp, {"ram_limit_mb": 100})["ram_limit_mb"] == 1024


def test_cpu_only_command_line_and_pinning():
    cfg = {"device": "cpu", "ngl": "auto", "cores": 4, "threads": None, "vram_limit_mb": None}
    assert profile._placement_args(cfg, 8192) == ["-dev", "none", "-ngl", "0", "-t", "4"]
    assert profile._pin(["llama-server", "-m", "x"], cfg)[:3] == ["taskset", "-c", "0-3"]
    assert profile._pin(["llama-server"], {"cores": None}) == ["llama-server"]


def test_one_layout_per_run():
    cfg = {"device": "gpu", "ngl": "auto", "cores": None, "threads": None, "vram_limit_mb": None}
    # The server fits with the margin; the tools get the layer count it chose and don't fit again.
    assert profile._placement_args(cfg, 8192, margin_mb=2048) == ["--fit-target", "2048"]
    assert profile._placement_args(cfg, 8192, bench=True, margin_mb=2048) == ["-fitt", "2048"]
    assert profile._placement_args(cfg, 8192, bench=True, margin_mb=2048, gpu_layers=30) == ["-ngl", "30"]
    # A simulated smaller GPU never leaves less free than the margin.
    small = {**cfg, "vram_limit_mb": 4096}
    assert profile._placement_args(small, 8192, margin_mb=2048) == ["--fit-target", "4096"]
    assert profile._placement_args({**cfg, "vram_limit_mb": 7680}, 8192, margin_mb=2048) == ["--fit-target", "2048"]
    # An explicit layer count is the setup itself.
    assert profile._placement_args({**cfg, "ngl": 20}, 8192, margin_mb=2048, gpu_layers=30) == ["-ngl", "20"]


def test_display_reserve(monkeypatch):
    from app import lab

    class Out:
        stdout = "Enabled\n"
    monkeypatch.setattr(lab.subprocess, "run", lambda *a, **k: Out)
    monkeypatch.setattr(lab, "_display", None)
    # A freshly booted machine (CI runners) has a small monotonic clock: "never checked" must still mean "check now".
    monkeypatch.setattr(lab.time, "monotonic", lambda: 5.0)
    assert lab.display_active() and lab.fit_margin_mb() == lab.FIT_MARGIN_MB + lab.DISPLAY_RESERVE_MB
    Out.stdout = "Disabled\n"
    monkeypatch.setattr(lab, "_display", None)
    assert not lab.display_active() and lab.fit_margin_mb() == lab.FIT_MARGIN_MB


def test_different_memory_margin_is_a_mismatch():
    def run(i, margin):
        return {"id": i, "config": {}, "results": {"provenance": {"placement": {"fit_margin_mb": margin}}}}
    assert "GPU memory margin" in profile.compare(run(1, 1024), run(2, 2048))["env_mismatch"]
    assert "GPU memory margin" not in profile.compare(run(1, 2048), run(2, 2048))["env_mismatch"]


def test_cpu_only_analysis_and_projection(exp, monkeypatch):
    monkeypatch.setattr(profile, "_weights", lambda path: {"bytes": 2e9, "params": 3e9, "embedding_bytes": 0.2e9})
    cfg = {**profile.normalize_config(exp, {"preset": "minipc-16gb"}), "ctx": 8192}
    res = {"speed": {"tg": 10.0, "pp": 100.0}, "memory": {"gpu": {}, "ram": {"model": 2e9 / 2**20}}}
    hw = profile.analyze(res, cfg, LIM)
    assert hw["weights_on_cpu_gb"] == pytest.approx(2.0)
    assert hw["limiter"]["what"] == "Possible RAM bandwidth limit"
    assert hw["prefill"]["roof_tflops"] is None          # no GPU compute roof for a CPU-only run
    pr = hw["projection"]
    assert pr["bandwidth_share"] == pytest.approx(0.5)  # 2 GB × 10 tok/s of 40 GB/s
    assert pr["tg"] == pytest.approx(0.5 * 38.4 / 2.0, rel=0.01)


def test_projection_says_why_when_bandwidth_unknown(exp, monkeypatch):
    monkeypatch.setattr(profile, "_weights", lambda path: {"bytes": 2e9, "params": 3e9, "embedding_bytes": 0.2e9})
    cfg = {**profile.normalize_config(exp, {"preset": "sbc-8gb"}), "ctx": 8192}
    res = {"speed": {"tg": 10.0, "pp": 100.0}, "memory": {"gpu": {}, "ram": {"model": 2e9 / 2**20}}}
    pr = profile.analyze(res, cfg, {**LIM, "ram_bandwidth_gbps": None})["projection"]
    assert pr["tg"] is None and "numpy" in pr["unavailable"]  # never a silent gap


@pytest.mark.parametrize("backend,ok", [("cpu", True), ("auto", True), ("opencl", False)])
def test_engine_backends(backend, ok):
    if ok:
        assert isinstance(engines.backend_args(backend), list)
    else:
        with pytest.raises(ValueError):
            engines.backend_args(backend)
